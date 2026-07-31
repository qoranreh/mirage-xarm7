"""ROS node that publishes timestamp-aligned, normalized xArm state."""

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque, Iterable, List, Optional

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformException, TransformListener
from xarm_msgs.msg import RobotMsg

from mirage_xarm_state.joint_state import (
    NormalizationResult,
    normalize_joint_state,
    parse_joint_state,
)


@dataclass
class PendingArmSample:
    message: JointState
    enqueued_monotonic: float


class XArmStateMonitor(Node):
    """Normalize xArm state and pair every arm sample with same-time TF."""

    def __init__(self) -> None:
        super().__init__('xarm_state_monitor')

        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('robot_states_topic', '/xarm/robot_states')
        self.declare_parameter('base_frame', 'link_base')
        self.declare_parameter('ee_frame', 'link_eef')
        self.declare_parameter(
            'arm_joint_names', [f'joint{index}' for index in range(1, 8)]
        )
        self.declare_parameter('gripper_joint_name', 'drive_joint')
        self.declare_parameter('tf_wait_timeout_sec', 0.2)
        self.declare_parameter('arm_stale_timeout_sec', 0.1)
        self.declare_parameter('gripper_stale_timeout_sec', 0.3)

        self._joint_states_topic = str(
            self.get_parameter('joint_states_topic').value
        )
        self._robot_states_topic = str(
            self.get_parameter('robot_states_topic').value
        )
        self._base_frame = str(self.get_parameter('base_frame').value)
        self._ee_frame = str(self.get_parameter('ee_frame').value)
        self._arm_joint_names = tuple(
            str(name) for name in self.get_parameter('arm_joint_names').value
        )
        self._gripper_joint_name = str(
            self.get_parameter('gripper_joint_name').value
        )
        self._tf_wait_timeout_sec = max(
            0.0, float(self.get_parameter('tf_wait_timeout_sec').value)
        )
        self._arm_stale_timeout_sec = max(
            0.0, float(self.get_parameter('arm_stale_timeout_sec').value)
        )
        self._gripper_stale_timeout_sec = max(
            0.0, float(self.get_parameter('gripper_stale_timeout_sec').value)
        )

        if len(self._arm_joint_names) != 7:
            raise ValueError(
                'arm_joint_names must contain exactly seven names'
            )
        if len(set(self._arm_joint_names)) != len(self._arm_joint_names):
            raise ValueError('arm_joint_names must not contain duplicates')
        if not self._gripper_joint_name:
            raise ValueError('gripper_joint_name must not be empty')

        self._arm_publisher = self.create_publisher(
            JointState, '/mirage/xarm/arm_joint_states', 10
        )
        self._pose_publisher = self.create_publisher(
            PoseStamped, '/mirage/xarm/ee_pose', 10
        )
        self._gripper_publisher = self.create_publisher(
            JointState, '/mirage/xarm/gripper_joint_state', 10
        )
        self._diagnostic_publisher = self.create_publisher(
            DiagnosticArray, '/mirage/xarm/state_diagnostics', 10
        )

        self._tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._pending_arm: Deque[PendingArmSample] = deque()
        self._last_arm_stamp_ns: Optional[int] = None
        self._last_gripper_stamp_ns: Optional[int] = None
        self._last_robot_stamp_ns: Optional[int] = None
        self._arm_valid = False
        self._gripper_valid = False
        self._last_arm_issue = 'no arm sample received'
        self._last_gripper_issue = 'no gripper sample received'
        self._last_missing_arm: tuple[str, ...] = ()
        self._last_robot: Optional[RobotMsg] = None
        self._joint_input_issue = 'none'

        self._arm_received = 0
        self._arm_published = 0
        self._gripper_received = 0
        self._gripper_published = 0
        self._tf_failures = 0
        self._malformed_messages = 0
        self._out_of_order_messages = 0

        self.create_subscription(
            JointState,
            self._joint_states_topic,
            self._joint_state_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            RobotMsg,
            self._robot_states_topic,
            self._robot_state_callback,
            qos_profile_sensor_data,
        )
        retry_period = min(0.01, max(0.001, self._tf_wait_timeout_sec / 4.0))
        self.create_timer(retry_period, self._process_pending_tf)
        self.create_timer(0.1, self._publish_diagnostics)

        self.get_logger().info(
            f'monitoring {self._joint_states_topic}; TF '
            f'{self._base_frame} -> {self._ee_frame}'
        )

    @staticmethod
    def _stamp_ns(message: JointState | RobotMsg) -> int:
        return Time.from_msg(message.header.stamp).nanoseconds

    def _message_age(self, stamp_ns: Optional[int]) -> Optional[float]:
        if stamp_ns is None:
            return None
        return (self.get_clock().now().nanoseconds - stamp_ns) / 1e9

    @staticmethod
    def _normalization_issue(result: NormalizationResult) -> str:
        parts: List[str] = []
        if result.missing_names:
            parts.append('missing: ' + ', '.join(result.missing_names))
        if result.nonfinite_fields:
            parts.append('non-finite: ' + ', '.join(result.nonfinite_fields))
        parts.extend(result.issues)
        return '; '.join(parts) if parts else 'invalid joint state'

    def _timestamp_is_usable(
        self, stamp_ns: int, previous_ns: Optional[int], source: str
    ) -> bool:
        if stamp_ns == 0:
            self._joint_input_issue = f'{source} timestamp is zero'
            self._malformed_messages += 1
            return False
        age = self._message_age(stamp_ns)
        if age is not None and age < -0.05:
            self._joint_input_issue = (
                f'{source} timestamp is {-age:.3f}s in the future'
            )
            self._malformed_messages += 1
            return False
        if previous_ns is not None and stamp_ns < previous_ns:
            self._joint_input_issue = f'{source} timestamp moved backwards'
            self._out_of_order_messages += 1
            return False
        return True

    def _joint_state_callback(self, message: JointState) -> None:
        names = set(message.name)
        contains_arm = bool(names.intersection(self._arm_joint_names))
        contains_gripper = self._gripper_joint_name in names
        if not contains_arm and not contains_gripper:
            return

        view = parse_joint_state(message)
        if view.issues:
            self._joint_input_issue = '; '.join(view.issues)
            self._malformed_messages += 1

        if contains_gripper:
            self._handle_gripper(message, view)
        if contains_arm:
            self._handle_arm(message, view)

    def _handle_gripper(self, message: JointState, view) -> None:
        self._gripper_received += 1
        stamp_ns = self._stamp_ns(message)
        result = normalize_joint_state(
            message, view, (self._gripper_joint_name,)
        )
        if not result.valid:
            self._gripper_valid = False
            self._last_gripper_issue = self._normalization_issue(result)
            return
        if not self._timestamp_is_usable(
            stamp_ns, self._last_gripper_stamp_ns, 'gripper'
        ):
            self._gripper_valid = False
            self._last_gripper_issue = self._joint_input_issue
            return

        age = self._message_age(stamp_ns)
        self._last_gripper_stamp_ns = stamp_ns
        if age is not None and age > self._gripper_stale_timeout_sec:
            self._gripper_valid = False
            self._last_gripper_issue = f'stale by {age:.3f}s'
            return

        self._gripper_publisher.publish(result.message)
        self._gripper_published += 1
        self._gripper_valid = True
        self._last_gripper_issue = 'ok'

    def _handle_arm(self, message: JointState, view) -> None:
        self._arm_received += 1
        stamp_ns = self._stamp_ns(message)
        result = normalize_joint_state(message, view, self._arm_joint_names)
        self._last_missing_arm = result.missing_names
        if not result.valid:
            self._arm_valid = False
            self._last_arm_issue = self._normalization_issue(result)
            return
        if not self._timestamp_is_usable(
            stamp_ns, self._last_arm_stamp_ns, 'arm'
        ):
            self._arm_valid = False
            self._last_arm_issue = self._joint_input_issue
            return

        age = self._message_age(stamp_ns)
        self._last_arm_stamp_ns = stamp_ns
        if age is not None and age > self._arm_stale_timeout_sec:
            self._arm_valid = False
            self._last_arm_issue = f'stale by {age:.3f}s'
            return

        self._pending_arm.append(
            PendingArmSample(result.message, time.monotonic())
        )
        self._arm_valid = True
        self._last_arm_issue = 'waiting for same-time TF'

    def _process_pending_tf(self) -> None:
        if not self._pending_arm:
            return

        now_monotonic = time.monotonic()
        still_pending: Deque[PendingArmSample] = deque()
        while self._pending_arm:
            pending = self._pending_arm.popleft()
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._base_frame,
                    self._ee_frame,
                    Time.from_msg(pending.message.header.stamp),
                    timeout=Duration(),
                )
            except TransformException as error:
                elapsed = now_monotonic - pending.enqueued_monotonic
                if elapsed < self._tf_wait_timeout_sec:
                    still_pending.append(pending)
                    continue
                self._tf_failures += 1
                self._arm_valid = False
                self._last_arm_issue = (
                    f'same-time TF unavailable after {elapsed:.3f}s: {error}'
                )
                continue

            pose = PoseStamped()
            pose.header.stamp = pending.message.header.stamp
            pose.header.frame_id = self._base_frame
            pose.pose.position.x = transform.transform.translation.x
            pose.pose.position.y = transform.transform.translation.y
            pose.pose.position.z = transform.transform.translation.z
            pose.pose.orientation = transform.transform.rotation
            self._arm_publisher.publish(pending.message)
            self._pose_publisher.publish(pose)
            self._arm_published += 1
            self._arm_valid = True
            self._last_arm_issue = 'ok'

        self._pending_arm = still_pending

    def _robot_state_callback(self, message: RobotMsg) -> None:
        self._last_robot = message
        stamp_ns = self._stamp_ns(message)
        self._last_robot_stamp_ns = (
            stamp_ns if stamp_ns != 0 else self.get_clock().now().nanoseconds
        )

    @staticmethod
    def _values(items: Iterable[tuple[str, object]]) -> List[KeyValue]:
        return [KeyValue(key=key, value=str(value)) for key, value in items]

    def _arm_diagnostic(self) -> DiagnosticStatus:
        age = self._message_age(self._last_arm_stamp_ns)
        stale = age is not None and age > self._arm_stale_timeout_sec
        valid = self._arm_valid and not stale
        if valid:
            level = DiagnosticStatus.OK
            message = self._last_arm_issue
        else:
            level = DiagnosticStatus.WARN
            message = (
                f'stale by {age:.3f}s' if stale and age is not None
                else self._last_arm_issue
            )
        return DiagnosticStatus(
            level=level,
            name='mirage_xarm_state/arm',
            message=message,
            hardware_id='xarm',
            values=self._values(
                (
                    ('valid', valid),
                    ('age_sec', 'unknown' if age is None else f'{age:.6f}'),
                    ('received', self._arm_received),
                    ('published', self._arm_published),
                    ('missing_joints', ','.join(self._last_missing_arm)),
                    ('joint_order', ','.join(self._arm_joint_names)),
                )
            ),
        )

    def _gripper_diagnostic(self) -> DiagnosticStatus:
        age = self._message_age(self._last_gripper_stamp_ns)
        stale = age is not None and age > self._gripper_stale_timeout_sec
        valid = self._gripper_valid and not stale
        message = (
            f'stale by {age:.3f}s' if stale and age is not None
            else self._last_gripper_issue
        )
        return DiagnosticStatus(
            level=DiagnosticStatus.OK if valid else DiagnosticStatus.WARN,
            name='mirage_xarm_state/gripper',
            message=message,
            hardware_id='xarm_gripper',
            values=self._values(
                (
                    ('valid', valid),
                    ('age_sec', 'unknown' if age is None else f'{age:.6f}'),
                    ('received', self._gripper_received),
                    ('published', self._gripper_published),
                    ('joint_name', self._gripper_joint_name),
                )
            ),
        )

    def _tf_diagnostic(self) -> DiagnosticStatus:
        completed = self._arm_published + self._tf_failures
        success_ratio = (
            self._arm_published / completed if completed else math.nan
        )
        level = (
            DiagnosticStatus.WARN
            if self._tf_failures
            else DiagnosticStatus.OK
        )
        return DiagnosticStatus(
            level=level,
            name='mirage_xarm_state/tf',
            message=(
                'TF lookup failures detected' if self._tf_failures else 'ok'
            ),
            hardware_id='xarm',
            values=self._values(
                (
                    ('base_frame', self._base_frame),
                    ('ee_frame', self._ee_frame),
                    ('pending', len(self._pending_arm)),
                    ('failures', self._tf_failures),
                    (
                        'success_ratio',
                        'unknown' if math.isnan(success_ratio)
                        else f'{success_ratio:.6f}',
                    ),
                )
            ),
        )

    def _robot_diagnostic(self) -> DiagnosticStatus:
        if self._last_robot is None:
            return DiagnosticStatus(
                level=DiagnosticStatus.OK,
                name='mirage_xarm_state/robot',
                message='robot state not received (optional in simulation)',
                hardware_id='xarm',
                values=self._values((('received', False),)),
            )

        robot = self._last_robot
        if robot.err:
            level = DiagnosticStatus.ERROR
            message = f'robot error {robot.err}'
        elif robot.warn:
            level = DiagnosticStatus.WARN
            message = f'robot warning {robot.warn}'
        else:
            level = DiagnosticStatus.OK
            message = 'ok'
        age = self._message_age(self._last_robot_stamp_ns)
        return DiagnosticStatus(
            level=level,
            name='mirage_xarm_state/robot',
            message=message,
            hardware_id='xarm',
            values=self._values(
                (
                    ('received', True),
                    ('age_sec', 'unknown' if age is None else f'{age:.6f}'),
                    ('mode', robot.mode),
                    ('state', robot.state),
                    ('error_code', robot.err),
                    ('warning_code', robot.warn),
                )
            ),
        )

    def _input_diagnostic(self) -> DiagnosticStatus:
        has_issue = bool(
            self._malformed_messages or self._out_of_order_messages
        )
        return DiagnosticStatus(
            level=DiagnosticStatus.WARN if has_issue else DiagnosticStatus.OK,
            name='mirage_xarm_state/input',
            message=self._joint_input_issue if has_issue else 'ok',
            hardware_id='xarm',
            values=self._values(
                (
                    ('malformed_messages', self._malformed_messages),
                    ('out_of_order_messages', self._out_of_order_messages),
                )
            ),
        )

    def _publish_diagnostics(self) -> None:
        message = DiagnosticArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.status = [
            self._arm_diagnostic(),
            self._gripper_diagnostic(),
            self._tf_diagnostic(),
            self._robot_diagnostic(),
            self._input_diagnostic(),
        ]
        self._diagnostic_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = XArmStateMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
