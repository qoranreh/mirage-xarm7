"""Read-only Gate A evidence collection for real or simulated xArm state."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Pose, PoseStamped, Transform
from moveit_msgs.srv import GetPositionFK
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from xarm_msgs.msg import RobotMsg


ARM_DIAGNOSTIC = 'mirage_xarm_state/arm'
GRIPPER_DIAGNOSTIC = 'mirage_xarm_state/gripper'
TF_DIAGNOSTIC = 'mirage_xarm_state/tf'
ROBOT_DIAGNOSTIC = 'mirage_xarm_state/robot'
INPUT_DIAGNOSTIC = 'mirage_xarm_state/input'


@dataclass(frozen=True)
class GateAConfig:
    """Thresholds and required evidence for a Gate A run."""

    arm_joint_names: tuple[str, ...] = tuple(
        f'joint{index}' for index in range(1, 8)
    )
    base_frame: str = 'link_base'
    ee_frame: str = 'link_eef'
    min_arm_samples: int = 20
    min_pose_samples: int = 20
    min_gripper_samples: int = 2
    min_robot_samples: int = 2
    min_arm_rate_hz: float = 8.0
    min_pose_rate_hz: float = 8.0
    min_gripper_rate_hz: float = 1.0
    min_robot_rate_hz: float = 1.0
    max_arm_gap_sec: float = 0.2
    max_pose_gap_sec: float = 0.2
    max_gripper_gap_sec: float = 1.5
    max_robot_gap_sec: float = 1.0
    max_message_age_sec: float = 0.5
    min_stamp_pair_ratio: float = 0.95
    min_tf_success_ratio: float = 0.95
    quaternion_norm_tolerance: float = 0.01
    tf_position_tolerance_m: float = 1e-5
    tf_orientation_tolerance_rad: float = 1e-4
    moveit_position_tolerance_m: float = 1e-4
    moveit_orientation_tolerance_rad: float = 1e-3
    require_gripper: bool = True
    require_robot_state: bool = False
    require_motion: bool = False
    minimum_joint_travel_rad: float = 1e-3
    check_moveit_fk: bool = True
    require_moveit_fk: bool = False


@dataclass
class StreamStats:
    """Timestamp and delivery statistics for one state stream."""

    count: int = 0
    first_stamp_ns: Optional[int] = None
    last_stamp_ns: Optional[int] = None
    max_gap_sec: float = 0.0
    zero_stamps: int = 0
    backwards_stamps: int = 0
    stale_samples: int = 0
    max_age_sec: float = 0.0

    def record(
        self,
        stamp_ns: int,
        now_ns: int,
        max_message_age_sec: float,
    ) -> None:
        self.count += 1
        if stamp_ns <= 0:
            self.zero_stamps += 1
            return

        age_sec = (now_ns - stamp_ns) / 1e9
        self.max_age_sec = max(self.max_age_sec, age_sec)
        if age_sec > max_message_age_sec or age_sec < -0.05:
            self.stale_samples += 1

        if self.first_stamp_ns is None:
            self.first_stamp_ns = stamp_ns
        if self.last_stamp_ns is not None:
            if stamp_ns < self.last_stamp_ns:
                self.backwards_stamps += 1
            else:
                gap_sec = (stamp_ns - self.last_stamp_ns) / 1e9
                self.max_gap_sec = max(self.max_gap_sec, gap_sec)
        self.last_stamp_ns = stamp_ns

    @property
    def rate_hz(self) -> float:
        if (
            self.count < 2
            or self.first_stamp_ns is None
            or self.last_stamp_ns is None
            or self.last_stamp_ns <= self.first_stamp_ns
        ):
            return 0.0
        duration = (self.last_stamp_ns - self.first_stamp_ns) / 1e9
        return (self.count - 1) / duration

    def to_dict(self) -> dict:
        result = asdict(self)
        result['rate_hz'] = self.rate_hz
        return result


@dataclass
class GateAEvidence:
    """ROS-message-independent decision state used by the validator node."""

    config: GateAConfig
    arm: StreamStats = field(default_factory=StreamStats)
    pose: StreamStats = field(default_factory=StreamStats)
    gripper: StreamStats = field(default_factory=StreamStats)
    robot: StreamStats = field(default_factory=StreamStats)
    diagnostics: StreamStats = field(default_factory=StreamStats)
    arm_invalid_messages: int = 0
    pose_invalid_messages: int = 0
    gripper_invalid_messages: int = 0
    robot_errors: int = 0
    robot_warnings: int = 0
    arm_stamps: set[int] = field(default_factory=set)
    pose_stamps: set[int] = field(default_factory=set)
    joint_minimums: Dict[str, float] = field(default_factory=dict)
    joint_maximums: Dict[str, float] = field(default_factory=dict)
    last_diagnostics: Dict[str, dict] = field(default_factory=dict)
    tf_comparisons: int = 0
    tf_lookup_failures: int = 0
    tf_max_position_error_m: float = 0.0
    tf_max_orientation_error_rad: float = 0.0
    moveit_fk_attempted: bool = False
    moveit_fk_available: bool = False
    moveit_fk_success: bool = False
    moveit_fk_message: str = 'not attempted'
    moveit_fk_position_error_m: Optional[float] = None
    moveit_fk_orientation_error_rad: Optional[float] = None

    @staticmethod
    def _stamp_ns(message) -> int:
        return Time.from_msg(message.header.stamp).nanoseconds

    @staticmethod
    def _finite(values: Iterable[float]) -> bool:
        return all(math.isfinite(value) for value in values)

    def record_arm(self, message: JointState, now_ns: int) -> None:
        stamp_ns = self._stamp_ns(message)
        self.arm.record(
            stamp_ns, now_ns, self.config.max_message_age_sec
        )
        self.arm_stamps.add(stamp_ns)
        valid = (
            tuple(message.name) == self.config.arm_joint_names
            and len(message.position) == len(self.config.arm_joint_names)
            and self._finite(message.position)
            and (
                not message.velocity
                or (
                    len(message.velocity) == len(message.name)
                    and self._finite(message.velocity)
                )
            )
            and (
                not message.effort
                or (
                    len(message.effort) == len(message.name)
                    and self._finite(message.effort)
                )
            )
        )
        if not valid:
            self.arm_invalid_messages += 1
            return

        for name, position in zip(message.name, message.position):
            self.joint_minimums[name] = min(
                position, self.joint_minimums.get(name, position)
            )
            self.joint_maximums[name] = max(
                position, self.joint_maximums.get(name, position)
            )

    def record_pose(self, message: PoseStamped, now_ns: int) -> None:
        stamp_ns = self._stamp_ns(message)
        self.pose.record(
            stamp_ns, now_ns, self.config.max_message_age_sec
        )
        self.pose_stamps.add(stamp_ns)
        position = message.pose.position
        orientation = message.pose.orientation
        quaternion = (
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w,
        )
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        valid = (
            message.header.frame_id == self.config.base_frame
            and self._finite(
                (position.x, position.y, position.z, *quaternion)
            )
            and abs(quaternion_norm - 1.0)
            <= self.config.quaternion_norm_tolerance
        )
        if not valid:
            self.pose_invalid_messages += 1

    def record_gripper(self, message: JointState, now_ns: int) -> None:
        stamp_ns = self._stamp_ns(message)
        self.gripper.record(
            stamp_ns, now_ns, self.config.max_message_age_sec
        )
        valid = (
            bool(message.name)
            and len(message.position) == len(message.name)
            and self._finite(message.position)
            and (
                not message.velocity
                or (
                    len(message.velocity) == len(message.name)
                    and self._finite(message.velocity)
                )
            )
            and (
                not message.effort
                or (
                    len(message.effort) == len(message.name)
                    and self._finite(message.effort)
                )
            )
        )
        if not valid:
            self.gripper_invalid_messages += 1

    def record_robot(self, message: RobotMsg, now_ns: int) -> None:
        self.robot.record(
            self._stamp_ns(message),
            now_ns,
            self.config.max_message_age_sec,
        )
        if message.err:
            self.robot_errors += 1
        if message.warn:
            self.robot_warnings += 1

    def record_diagnostics(
        self, message: DiagnosticArray, now_ns: int
    ) -> None:
        self.diagnostics.record(
            self._stamp_ns(message),
            now_ns,
            self.config.max_message_age_sec,
        )
        for status in message.status:
            self.last_diagnostics[status.name] = {
                'level': diagnostic_level(status.level),
                'message': status.message,
                'values': {
                    item.key: item.value for item in status.values
                },
            }

    def record_tf_comparison(
        self, pose: Pose, transform: Transform
    ) -> None:
        self.tf_comparisons += 1
        position_error = math.sqrt(
            (pose.position.x - transform.translation.x) ** 2
            + (pose.position.y - transform.translation.y) ** 2
            + (pose.position.z - transform.translation.z) ** 2
        )
        orientation_error = quaternion_angle(
            (
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ),
            (
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w,
            ),
        )
        self.tf_max_position_error_m = max(
            self.tf_max_position_error_m, position_error
        )
        self.tf_max_orientation_error_rad = max(
            self.tf_max_orientation_error_rad, orientation_error
        )

    def record_tf_failure(self) -> None:
        self.tf_lookup_failures += 1

    def record_moveit_fk(
        self,
        *,
        available: bool,
        success: bool,
        message: str,
        reference_pose: Optional[Pose] = None,
        moveit_pose: Optional[Pose] = None,
    ) -> None:
        self.moveit_fk_attempted = True
        self.moveit_fk_available = available
        self.moveit_fk_success = success
        self.moveit_fk_message = message
        if reference_pose is None or moveit_pose is None:
            return
        self.moveit_fk_position_error_m = math.sqrt(
            (reference_pose.position.x - moveit_pose.position.x) ** 2
            + (reference_pose.position.y - moveit_pose.position.y) ** 2
            + (reference_pose.position.z - moveit_pose.position.z) ** 2
        )
        self.moveit_fk_orientation_error_rad = quaternion_angle(
            (
                reference_pose.orientation.x,
                reference_pose.orientation.y,
                reference_pose.orientation.z,
                reference_pose.orientation.w,
            ),
            (
                moveit_pose.orientation.x,
                moveit_pose.orientation.y,
                moveit_pose.orientation.z,
                moveit_pose.orientation.w,
            ),
        )

    @property
    def maximum_joint_travel_rad(self) -> float:
        travels = [
            self.joint_maximums[name] - minimum
            for name, minimum in self.joint_minimums.items()
            if name in self.joint_maximums
        ]
        return max(travels, default=0.0)

    def _stream_checks(
        self,
        name: str,
        stats: StreamStats,
        *,
        minimum_samples: int,
        minimum_rate_hz: float,
        maximum_gap_sec: float,
        invalid_messages: int,
        required: bool = True,
    ) -> List[dict]:
        return [
            check(
                f'{name}_sample_count',
                stats.count >= minimum_samples,
                required,
                f'{stats.count} received; minimum {minimum_samples}',
            ),
            check(
                f'{name}_rate',
                stats.rate_hz >= minimum_rate_hz,
                required,
                f'{stats.rate_hz:.3f} Hz; minimum {minimum_rate_hz:.3f} Hz',
            ),
            check(
                f'{name}_maximum_gap',
                stats.count >= 2 and stats.max_gap_sec <= maximum_gap_sec,
                required,
                f'{stats.max_gap_sec:.6f} s; maximum {maximum_gap_sec:.6f} s',
            ),
            check(
                f'{name}_message_validity',
                invalid_messages == 0,
                required,
                f'{invalid_messages} invalid messages',
            ),
            check(
                f'{name}_timestamps',
                stats.zero_stamps == 0
                and stats.backwards_stamps == 0
                and stats.stale_samples == 0,
                required,
                (
                    f'zero={stats.zero_stamps}, '
                    f'backwards={stats.backwards_stamps}, '
                    f'stale={stats.stale_samples}, '
                    f'max_age={stats.max_age_sec:.6f} s'
                ),
            ),
        ]

    def build_report(
        self,
        *,
        started_at: str,
        completed_at: str,
        requested_duration_sec: float,
        topics: Dict[str, str],
        report_path: str,
    ) -> dict:
        checks: List[dict] = []
        checks.extend(
            self._stream_checks(
                'arm',
                self.arm,
                minimum_samples=self.config.min_arm_samples,
                minimum_rate_hz=self.config.min_arm_rate_hz,
                maximum_gap_sec=self.config.max_arm_gap_sec,
                invalid_messages=self.arm_invalid_messages,
            )
        )
        checks.extend(
            self._stream_checks(
                'pose',
                self.pose,
                minimum_samples=self.config.min_pose_samples,
                minimum_rate_hz=self.config.min_pose_rate_hz,
                maximum_gap_sec=self.config.max_pose_gap_sec,
                invalid_messages=self.pose_invalid_messages,
            )
        )
        checks.extend(
            self._stream_checks(
                'gripper',
                self.gripper,
                minimum_samples=self.config.min_gripper_samples,
                minimum_rate_hz=self.config.min_gripper_rate_hz,
                maximum_gap_sec=self.config.max_gripper_gap_sec,
                invalid_messages=self.gripper_invalid_messages,
                required=self.config.require_gripper,
            )
        )
        checks.extend(
            self._stream_checks(
                'robot',
                self.robot,
                minimum_samples=self.config.min_robot_samples,
                minimum_rate_hz=self.config.min_robot_rate_hz,
                maximum_gap_sec=self.config.max_robot_gap_sec,
                invalid_messages=0,
                required=self.config.require_robot_state,
            )
        )

        paired_stamps = len(self.arm_stamps.intersection(self.pose_stamps))
        pair_denominator = max(len(self.arm_stamps), len(self.pose_stamps), 1)
        pair_ratio = paired_stamps / pair_denominator
        checks.append(
            check(
                'arm_pose_timestamp_pairing',
                pair_ratio >= self.config.min_stamp_pair_ratio,
                True,
                (
                    f'{paired_stamps} exact pairs; ratio={pair_ratio:.6f}; '
                    f'minimum={self.config.min_stamp_pair_ratio:.6f}'
                ),
            )
        )

        tf_attempts = self.tf_comparisons + self.tf_lookup_failures
        tf_success_ratio = (
            self.tf_comparisons / tf_attempts if tf_attempts else 0.0
        )
        checks.append(
            check(
                'tf_lookup_coverage',
                self.tf_comparisons >= self.config.min_pose_samples
                and tf_success_ratio >= self.config.min_tf_success_ratio,
                True,
                (
                    f'comparisons={self.tf_comparisons}, '
                    f'failures={self.tf_lookup_failures}, '
                    f'ratio={tf_success_ratio:.6f}'
                ),
            )
        )
        checks.append(
            check(
                'ee_pose_tf_consistency',
                self.tf_comparisons > 0
                and self.tf_max_position_error_m
                <= self.config.tf_position_tolerance_m
                and self.tf_max_orientation_error_rad
                <= self.config.tf_orientation_tolerance_rad,
                True,
                (
                    f'max_position={self.tf_max_position_error_m:.9f} m, '
                    'max_orientation='
                    f'{self.tf_max_orientation_error_rad:.9f} rad'
                ),
            )
        )
        checks.append(
            check(
                'robot_error_state',
                self.robot_errors == 0 and self.robot_warnings == 0,
                self.config.require_robot_state,
                (
                    f'error samples={self.robot_errors}, '
                    f'warning samples={self.robot_warnings}'
                ),
            )
        )

        required_diagnostics = [
            ARM_DIAGNOSTIC,
            TF_DIAGNOSTIC,
            INPUT_DIAGNOSTIC,
        ]
        if self.config.require_gripper:
            required_diagnostics.append(GRIPPER_DIAGNOSTIC)
        if self.config.require_robot_state:
            required_diagnostics.append(ROBOT_DIAGNOSTIC)
        missing_diagnostics = [
            name for name in required_diagnostics
            if name not in self.last_diagnostics
        ]
        bad_diagnostics = [
            name for name in required_diagnostics
            if name in self.last_diagnostics
            and self.last_diagnostics[name]['level'] != 0
        ]
        checks.append(
            check(
                'monitor_diagnostics',
                not missing_diagnostics and not bad_diagnostics,
                True,
                (
                    'missing=' + ','.join(missing_diagnostics)
                    + '; non_ok=' + ','.join(bad_diagnostics)
                ),
            )
        )

        moveit_position_ok = (
            self.moveit_fk_position_error_m is not None
            and self.moveit_fk_position_error_m
            <= self.config.moveit_position_tolerance_m
        )
        moveit_orientation_ok = (
            self.moveit_fk_orientation_error_rad is not None
            and self.moveit_fk_orientation_error_rad
            <= self.config.moveit_orientation_tolerance_rad
        )
        checks.append(
            check(
                'moveit_fk_consistency',
                (
                    not self.config.check_moveit_fk
                    or (
                        self.moveit_fk_success
                        and moveit_position_ok
                        and moveit_orientation_ok
                    )
                ),
                self.config.require_moveit_fk,
                self.moveit_fk_message,
            )
        )
        checks.append(
            check(
                'joint_motion_observed',
                self.maximum_joint_travel_rad
                >= self.config.minimum_joint_travel_rad,
                self.config.require_motion,
                (
                    f'maximum travel={self.maximum_joint_travel_rad:.9f} rad; '
                    f'minimum={self.config.minimum_joint_travel_rad:.9f} rad'
                ),
            )
        )
        checks.extend(
            [
                check(
                    'rviz_visual_alignment',
                    False,
                    False,
                    'manual confirmation required during the real-robot run',
                ),
                check(
                    'rosbag_offline_replay',
                    False,
                    False,
                    'run xarm_gate_a_bag_check after the bag is finalized',
                ),
            ]
        )

        passed = all(
            item['passed'] for item in checks if item['required']
        )
        return {
            'schema_version': 1,
            'gate': 'A',
            'passed': passed,
            'started_at': started_at,
            'completed_at': completed_at,
            'requested_duration_sec': requested_duration_sec,
            'report_path': report_path,
            'read_only': {
                'motion_command_publishers': [],
                'motion_action_clients': [],
                'motion_service_clients': [],
                'non_motion_service_clients': (
                    ['/compute_fk'] if self.config.check_moveit_fk else []
                ),
                'note': (
                    'The validator only subscribes to state, publishes its '
                    'report, and optionally calls the kinematics-only FK service.'
                ),
            },
            'topics': topics,
            'checks': checks,
            'metrics': {
                'arm': self.arm.to_dict(),
                'pose': self.pose.to_dict(),
                'gripper': self.gripper.to_dict(),
                'robot': self.robot.to_dict(),
                'diagnostics': self.diagnostics.to_dict(),
                'arm_invalid_messages': self.arm_invalid_messages,
                'pose_invalid_messages': self.pose_invalid_messages,
                'gripper_invalid_messages': self.gripper_invalid_messages,
                'robot_errors': self.robot_errors,
                'robot_warnings': self.robot_warnings,
                'exact_arm_pose_pairs': paired_stamps,
                'arm_pose_pair_ratio': pair_ratio,
                'maximum_joint_travel_rad': self.maximum_joint_travel_rad,
                'tf': {
                    'comparisons': self.tf_comparisons,
                    'lookup_failures': self.tf_lookup_failures,
                    'success_ratio': tf_success_ratio,
                    'max_position_error_m': self.tf_max_position_error_m,
                    'max_orientation_error_rad': (
                        self.tf_max_orientation_error_rad
                    ),
                },
                'moveit_fk': {
                    'attempted': self.moveit_fk_attempted,
                    'available': self.moveit_fk_available,
                    'success': self.moveit_fk_success,
                    'message': self.moveit_fk_message,
                    'position_error_m': self.moveit_fk_position_error_m,
                    'orientation_error_rad': (
                        self.moveit_fk_orientation_error_rad
                    ),
                },
                'last_diagnostics': self.last_diagnostics,
            },
            'manual_follow_up': [
                'Confirm the live joint pose matches RViz.',
                'Inspect and replay the finalized rosbag.',
                'Do not connect policy output or any motion command yet.',
            ],
        }


def check(name: str, passed: bool, required: bool, detail: str) -> dict:
    """Create one stable report check object."""
    return {
        'name': name,
        'passed': bool(passed),
        'required': bool(required),
        'detail': detail,
    }


def diagnostic_level(value) -> int:
    """Normalize ROS uint8 fields across generated-message implementations."""
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, byteorder='little')
    return int(value)


def quaternion_angle(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Return the shortest angular distance between two quaternions."""
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return math.inf
    dot = sum(a * b for a, b in zip(first, second))
    dot = abs(dot / (first_norm * second_norm))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


class GateAValidator(Node):
    """Collect Gate A evidence without owning a robot command interface."""

    def __init__(self) -> None:
        super().__init__('xarm_gate_a_validator')
        self._declare_parameters()
        self._duration_sec = max(
            1.0, float(self.get_parameter('duration_sec').value)
        )
        self._topics = {
            key: str(self.get_parameter(f'{key}_topic').value)
            for key in ('arm', 'pose', 'gripper', 'robot', 'diagnostics')
        }
        config = self._read_config()
        self._evidence = GateAEvidence(config)
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._report_path = self._resolve_report_path()
        self._finished = False
        self._exit_code = 3
        self._last_arm: Optional[JointState] = None
        self._last_pose: Optional[PoseStamped] = None
        self._moveit_future = None
        self._moveit_timeout_timer = None

        self._report_publisher = self.create_publisher(
            String, '/mirage/xarm/gate_a_report', 1
        )
        self.create_subscription(
            JointState,
            self._topics['arm'],
            self._arm_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            self._topics['pose'],
            self._pose_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            self._topics['gripper'],
            self._gripper_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            RobotMsg,
            self._topics['robot'],
            self._robot_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            DiagnosticArray,
            self._topics['diagnostics'],
            self._diagnostic_callback,
            10,
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._moveit_client = self.create_client(
            GetPositionFK,
            str(self.get_parameter('moveit_fk_service').value),
        )
        self._deadline_timer = self.create_timer(
            self._duration_sec, self._begin_finish
        )
        self.get_logger().info(
            f'collecting read-only Gate A evidence for '
            f'{self._duration_sec:.1f}s; report={self._report_path}'
        )

    def _declare_parameters(self) -> None:
        defaults = {
            'duration_sec': 30.0,
            'report_path': '',
            'report_dir': '~/Mirage_xarm_ws/reports',
            'arm_topic': '/mirage/xarm/arm_joint_states',
            'pose_topic': '/mirage/xarm/ee_pose',
            'gripper_topic': '/mirage/xarm/gripper_joint_state',
            'robot_topic': '/xarm/robot_states',
            'diagnostics_topic': '/mirage/xarm/state_diagnostics',
            'base_frame': 'link_base',
            'ee_frame': 'link_eef',
            'arm_joint_names': [
                f'joint{index}' for index in range(1, 8)
            ],
            'min_arm_samples': 20,
            'min_pose_samples': 20,
            'min_gripper_samples': 2,
            'min_robot_samples': 2,
            'min_arm_rate_hz': 8.0,
            'min_pose_rate_hz': 8.0,
            'min_gripper_rate_hz': 1.0,
            'min_robot_rate_hz': 1.0,
            'max_arm_gap_sec': 0.2,
            'max_pose_gap_sec': 0.2,
            'max_gripper_gap_sec': 1.5,
            'max_robot_gap_sec': 1.0,
            'max_message_age_sec': 0.5,
            'min_stamp_pair_ratio': 0.95,
            'min_tf_success_ratio': 0.95,
            'quaternion_norm_tolerance': 0.01,
            'tf_position_tolerance_m': 1e-5,
            'tf_orientation_tolerance_rad': 1e-4,
            'moveit_position_tolerance_m': 1e-4,
            'moveit_orientation_tolerance_rad': 1e-3,
            'require_gripper': True,
            'require_robot_state': False,
            'require_motion': False,
            'minimum_joint_travel_rad': 1e-3,
            'check_moveit_fk': True,
            'require_moveit_fk': False,
            'moveit_fk_service': '/compute_fk',
            'moveit_fk_timeout_sec': 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_config(self) -> GateAConfig:
        values = {}
        for field_name in GateAConfig.__dataclass_fields__:
            value = self.get_parameter(field_name).value
            if field_name == 'arm_joint_names':
                value = tuple(str(name) for name in value)
            values[field_name] = value
        return GateAConfig(**values)

    def _resolve_report_path(self) -> Path:
        configured = str(self.get_parameter('report_path').value).strip()
        if configured:
            return Path(configured).expanduser()
        report_dir = Path(
            str(self.get_parameter('report_dir').value)
        ).expanduser()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return report_dir / f'gate_a_{timestamp}.json'

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _arm_callback(self, message: JointState) -> None:
        self._last_arm = message
        self._evidence.record_arm(message, self._now_ns())

    def _pose_callback(self, message: PoseStamped) -> None:
        self._last_pose = message
        self._evidence.record_pose(message, self._now_ns())
        try:
            transform = self._tf_buffer.lookup_transform(
                self._evidence.config.base_frame,
                self._evidence.config.ee_frame,
                Time.from_msg(message.header.stamp),
                timeout=Duration(),
            )
        except TransformException:
            self._evidence.record_tf_failure()
            return
        self._evidence.record_tf_comparison(
            message.pose, transform.transform
        )

    def _gripper_callback(self, message: JointState) -> None:
        self._evidence.record_gripper(message, self._now_ns())

    def _robot_callback(self, message: RobotMsg) -> None:
        self._evidence.record_robot(message, self._now_ns())

    def _diagnostic_callback(self, message: DiagnosticArray) -> None:
        self._evidence.record_diagnostics(message, self._now_ns())

    def _begin_finish(self) -> None:
        self._deadline_timer.cancel()
        config = self._evidence.config
        if not config.check_moveit_fk:
            self._finalize()
            return
        if self._last_arm is None or self._last_pose is None:
            self._evidence.record_moveit_fk(
                available=self._moveit_client.service_is_ready(),
                success=False,
                message='no paired arm/pose sample available for FK',
            )
            self._finalize()
            return
        if not self._moveit_client.service_is_ready():
            self._evidence.record_moveit_fk(
                available=False,
                success=False,
                message='MoveIt /compute_fk service unavailable',
            )
            self._finalize()
            return

        request = GetPositionFK.Request()
        request.header.frame_id = config.base_frame
        request.header.stamp = self._last_arm.header.stamp
        request.fk_link_names = [config.ee_frame]
        request.robot_state.joint_state = self._last_arm
        self._moveit_future = self._moveit_client.call_async(request)
        self._moveit_future.add_done_callback(self._moveit_fk_done)
        timeout_sec = max(
            0.1,
            float(self.get_parameter('moveit_fk_timeout_sec').value),
        )
        self._moveit_timeout_timer = self.create_timer(
            timeout_sec, self._moveit_fk_timeout
        )

    def _moveit_fk_done(self, future) -> None:
        if self._finished:
            return
        if self._moveit_timeout_timer is not None:
            self._moveit_timeout_timer.cancel()
        try:
            response = future.result()
        except Exception as error:  # ROS future exceptions vary by RMW.
            self._evidence.record_moveit_fk(
                available=True,
                success=False,
                message=f'MoveIt FK call failed: {error}',
            )
            self._finalize()
            return

        success = bool(
            response.error_code.val == response.error_code.SUCCESS
            and response.pose_stamped
        )
        if not success:
            self._evidence.record_moveit_fk(
                available=True,
                success=False,
                message=(
                    f'MoveIt FK returned error {response.error_code.val}'
                ),
            )
        else:
            moveit_pose = response.pose_stamped[0].pose
            self._evidence.record_moveit_fk(
                available=True,
                success=True,
                message='MoveIt FK response received',
                reference_pose=self._last_pose.pose,
                moveit_pose=moveit_pose,
            )
        self._finalize()

    def _moveit_fk_timeout(self) -> None:
        if self._finished:
            return
        self._moveit_timeout_timer.cancel()
        self._evidence.record_moveit_fk(
            available=True,
            success=False,
            message='MoveIt FK request timed out',
        )
        self._finalize()

    def _finalize(self) -> None:
        if self._finished:
            return
        completed_at = datetime.now(timezone.utc).isoformat()
        report = self._evidence.build_report(
            started_at=self._started_at,
            completed_at=completed_at,
            requested_duration_sec=self._duration_sec,
            topics=self._topics,
            report_path=str(self._report_path),
        )
        serialized = json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True
        )
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._report_path.with_suffix(
            self._report_path.suffix + '.tmp'
        )
        temporary_path.write_text(serialized + '\n', encoding='utf-8')
        temporary_path.replace(self._report_path)
        self._report_publisher.publish(String(data=serialized))
        self._exit_code = 0 if report['passed'] else 2
        self._finished = True
        log = self.get_logger().info if report['passed'] else self.get_logger().error
        log(
            f"Gate A {'PASS' if report['passed'] else 'FAIL'}; "
            f'report={self._report_path}'
        )

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def exit_code(self) -> int:
        return self._exit_code


def main(args=None) -> int:
    rclpy.init(args=args)
    node = GateAValidator()
    exit_code = 3
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node.finished and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
        exit_code = node.exit_code
    except (KeyboardInterrupt, ExternalShutdownException):
        exit_code = 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
