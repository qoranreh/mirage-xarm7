"""Adapters from xArm ROS samples and Robosuite observations."""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from mirage_state.model import CanonicalState, NamedArray, make_canonical_state


@dataclass(frozen=True)
class PoseSample:
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame_id: str
    timestamp_ns: int


@dataclass(frozen=True)
class GripperSample:
    position_rad: float
    timestamp_ns: int


@dataclass(frozen=True)
class ObjectSample:
    schema: str
    entries: tuple[NamedArray, ...]
    timestamp_ns: int
    fresh: bool = True
    valid: bool = True
    invalid_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class XArmAdapterConfig:
    source_robot: str = 'xarm7'
    expected_frame_id: str = 'link_base'
    max_pose_age_sec: float = 0.15
    max_gripper_age_sec: float = 0.30
    max_object_age_sec: float = 0.30
    max_component_skew_sec: float = 0.10
    open_angle_rad: float = 0.0
    closed_angle_rad: float = 0.85
    open_width_m: float = 0.085
    closed_width_m: float = 0.0
    angle_tolerance_rad: float = 0.02

    def __post_init__(self) -> None:
        if self.closed_angle_rad <= self.open_angle_rad:
            raise ValueError('closed_angle_rad must exceed open_angle_rad')
        if self.open_width_m <= self.closed_width_m:
            raise ValueError('open_width_m must exceed closed_width_m')
        if min(
            self.max_pose_age_sec,
            self.max_gripper_age_sec,
            self.max_object_age_sec,
            self.max_component_skew_sec,
        ) < 0.0:
            raise ValueError('age and skew thresholds must be non-negative')


class XArm7CanonicalAdapter:
    """Combine asynchronous read-only xArm samples into one state."""

    def __init__(self, config: XArmAdapterConfig | None = None) -> None:
        self.config = config or XArmAdapterConfig()

    def gripper_width_m(self, angle_rad: float) -> tuple[float, tuple[str, ...]]:
        """Convert the xArm virtual gripper joint angle to physical width."""
        angle = float(angle_rad)
        if not math.isfinite(angle):
            return 0.0, ('gripper_angle_nonfinite',)
        config = self.config
        reasons: list[str] = []
        lower = config.open_angle_rad - config.angle_tolerance_rad
        upper = config.closed_angle_rad + config.angle_tolerance_rad
        if angle < lower or angle > upper:
            reasons.append(f'gripper_angle_out_of_range:{angle:.9g}')
        clamped = min(
            config.closed_angle_rad, max(config.open_angle_rad, angle)
        )
        fraction = (clamped - config.open_angle_rad) / (
            config.closed_angle_rad - config.open_angle_rad
        )
        width = config.open_width_m + fraction * (
            config.closed_width_m - config.open_width_m
        )
        return width, tuple(reasons)

    @staticmethod
    def _age_sec(now_ns: int, stamp_ns: int) -> float:
        return (int(now_ns) - int(stamp_ns)) / 1e9

    def adapt(
        self,
        *,
        pose: PoseSample | None,
        gripper: GripperSample | None,
        object_sample: ObjectSample | None,
        now_ns: int,
    ) -> CanonicalState:
        """Create a canonical state, preserving partial data when invalid."""
        reasons: list[str] = []
        position = (0.0, 0.0, 0.0)
        quaternion = (0.0, 0.0, 0.0, 1.0)
        frame_id = self.config.expected_frame_id
        timestamp_ns = int(now_ns)
        component_stamps: list[int] = []
        fresh = True

        if pose is None:
            reasons.append('pose_missing')
            fresh = False
        else:
            position = pose.position_m
            quaternion = pose.quaternion_xyzw
            frame_id = pose.frame_id
            timestamp_ns = pose.timestamp_ns
            component_stamps.append(pose.timestamp_ns)
            if pose.frame_id != self.config.expected_frame_id:
                reasons.append(
                    f'frame_mismatch:{pose.frame_id}!={self.config.expected_frame_id}'
                )
            pose_age = self._age_sec(now_ns, pose.timestamp_ns)
            if pose_age < -self.config.max_component_skew_sec:
                reasons.append(f'pose_timestamp_in_future:{-pose_age:.9g}')
                fresh = False
            elif pose_age > self.config.max_pose_age_sec:
                reasons.append(f'pose_stale:{pose_age:.9g}')
                fresh = False

        width = 0.0
        if gripper is None:
            reasons.append('gripper_missing')
            fresh = False
        else:
            width, width_reasons = self.gripper_width_m(gripper.position_rad)
            reasons.extend(width_reasons)
            component_stamps.append(gripper.timestamp_ns)
            gripper_age = self._age_sec(now_ns, gripper.timestamp_ns)
            if gripper_age < -self.config.max_component_skew_sec:
                reasons.append(f'gripper_timestamp_in_future:{-gripper_age:.9g}')
                fresh = False
            elif gripper_age > self.config.max_gripper_age_sec:
                reasons.append(f'gripper_stale:{gripper_age:.9g}')
                fresh = False

        object_schema = ''
        object_entries: tuple[NamedArray, ...] = ()
        if object_sample is None:
            reasons.append('object_state_missing')
            fresh = False
        else:
            object_schema = object_sample.schema
            object_entries = object_sample.entries
            component_stamps.append(object_sample.timestamp_ns)
            if not object_sample.valid:
                reasons.extend(object_sample.invalid_reasons or ('object_state_invalid',))
            if not object_sample.fresh:
                reasons.append('object_state_stale')
                fresh = False
            object_age = self._age_sec(now_ns, object_sample.timestamp_ns)
            if object_age < -self.config.max_component_skew_sec:
                reasons.append(f'object_timestamp_in_future:{-object_age:.9g}')
                fresh = False
            elif object_age > self.config.max_object_age_sec:
                reasons.append(f'object_stale:{object_age:.9g}')
                fresh = False

        if len(component_stamps) > 1:
            skew = (max(component_stamps) - min(component_stamps)) / 1e9
            if skew > self.config.max_component_skew_sec:
                reasons.append(f'component_timestamp_skew:{skew:.9g}')
                fresh = False

        return make_canonical_state(
            ee_position_m=position,
            ee_quaternion_xyzw=quaternion,
            gripper_width_m=width,
            object_schema=object_schema,
            object_state=object_entries,
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            source_robot=self.config.source_robot,
            fresh=fresh,
            invalid_reasons=reasons,
        )


class PandaRobosuiteAdapter:
    """Convert a Robosuite Panda observation into CanonicalState."""

    def __init__(
        self,
        *,
        env_name: str,
        frame_id: str = 'panda_sim_world',
        source_robot: str = 'panda_robosuite',
        object_keys: Sequence[str] = ('object',),
    ) -> None:
        self.env_name = env_name
        self.frame_id = frame_id
        self.source_robot = source_robot
        self.object_keys = tuple(object_keys)

    @staticmethod
    def _entry(name: str, value: object) -> NamedArray:
        array = np.asarray(value, dtype=np.float64)
        return NamedArray(
            name=name,
            shape=tuple(int(size) for size in array.shape),
            values=tuple(float(item) for item in array.reshape(-1)),
        )

    def adapt(
        self,
        observation: Mapping[str, object],
        *,
        timestamp_ns: int,
    ) -> CanonicalState:
        reasons: list[str] = []

        def required(name: str, length: int) -> np.ndarray:
            if name not in observation:
                reasons.append(f'observation_missing:{name}')
                return np.zeros(length, dtype=np.float64)
            array = np.asarray(observation[name], dtype=np.float64).reshape(-1)
            if array.size != length:
                reasons.append(
                    f'observation_shape:{name}:{tuple(np.asarray(observation[name]).shape)}'
                )
                return np.zeros(length, dtype=np.float64)
            return array

        position = required('robot0_eef_pos', 3)
        quaternion = required('robot0_eef_quat', 4)
        gripper = required('robot0_gripper_qpos', 2)
        width = float(gripper[0] - gripper[1])
        if width < -1e-6 or width > 0.080001:
            reasons.append(f'panda_gripper_width_out_of_range:{width:.9g}')
        width = min(0.08, max(0.0, width))

        entries: list[NamedArray] = []
        for key in self.object_keys:
            if key not in observation:
                reasons.append(f'observation_missing:{key}')
                continue
            try:
                entries.append(self._entry(key, observation[key]))
            except (TypeError, ValueError) as error:
                reasons.append(f'observation_invalid:{key}:{error}')

        return make_canonical_state(
            ee_position_m=position,
            ee_quaternion_xyzw=quaternion,
            gripper_width_m=width,
            object_schema=f'{self.env_name}:robomimic-v1',
            object_state=entries,
            frame_id=self.frame_id,
            timestamp_ns=timestamp_ns,
            source_robot=self.source_robot,
            fresh=True,
            invalid_reasons=reasons,
        )
