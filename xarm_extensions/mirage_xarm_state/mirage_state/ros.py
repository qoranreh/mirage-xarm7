"""Conversions between the pure Python model and Mirage ROS messages."""

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from mirage_state_interfaces.msg import (
    CanonicalState as CanonicalStateMsg,
    NamedFloatArray,
    ObjectState as ObjectStateMsg,
)
from sensor_msgs.msg import JointState

from mirage_state.adapters import GripperSample, ObjectSample, PoseSample
from mirage_state.model import CanonicalState, NamedArray, make_canonical_state


def time_to_ns(stamp: Time) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ns_to_time(timestamp_ns: int) -> Time:
    stamp = Time()
    stamp.sec = int(timestamp_ns) // 1_000_000_000
    stamp.nanosec = int(timestamp_ns) % 1_000_000_000
    return stamp


def pose_sample_from_ros(message: PoseStamped) -> PoseSample:
    return PoseSample(
        position_m=(
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ),
        quaternion_xyzw=(
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ),
        frame_id=message.header.frame_id,
        timestamp_ns=time_to_ns(message.header.stamp),
    )


def gripper_sample_from_ros(
    message: JointState, joint_name: str = 'drive_joint'
) -> GripperSample:
    if joint_name not in message.name:
        raise ValueError(f'gripper joint {joint_name!r} is missing')
    index = list(message.name).index(joint_name)
    if index >= len(message.position):
        raise ValueError(f'gripper joint {joint_name!r} has no position')
    return GripperSample(
        position_rad=float(message.position[index]),
        timestamp_ns=time_to_ns(message.header.stamp),
    )


def object_sample_from_ros(message: ObjectStateMsg) -> ObjectSample:
    reasons = list(message.invalid_reasons)
    entries: list[NamedArray] = []
    for entry in message.entries:
        try:
            entries.append(
                NamedArray(
                    name=entry.name,
                    shape=tuple(entry.shape),
                    values=tuple(entry.values),
                )
            )
        except ValueError as error:
            reasons.append(f'object_entry_invalid:{entry.name}:{error}')
    return ObjectSample(
        schema=message.schema,
        entries=tuple(entries),
        timestamp_ns=time_to_ns(message.header.stamp),
        fresh=bool(message.fresh),
        valid=bool(message.valid) and not reasons,
        invalid_reasons=tuple(reasons),
    )


def _named_array_to_ros(entry: NamedArray) -> NamedFloatArray:
    message = NamedFloatArray()
    message.name = entry.name
    message.shape = list(entry.shape)
    message.values = list(entry.values)
    return message


def canonical_state_to_ros(state: CanonicalState) -> CanonicalStateMsg:
    message = CanonicalStateMsg()
    message.header.stamp = ns_to_time(state.timestamp_ns)
    message.header.frame_id = state.frame_id
    message.ee_position_m.x = state.ee_position_m[0]
    message.ee_position_m.y = state.ee_position_m[1]
    message.ee_position_m.z = state.ee_position_m[2]
    message.ee_quaternion_xyzw.x = state.ee_quaternion_xyzw[0]
    message.ee_quaternion_xyzw.y = state.ee_quaternion_xyzw[1]
    message.ee_quaternion_xyzw.z = state.ee_quaternion_xyzw[2]
    message.ee_quaternion_xyzw.w = state.ee_quaternion_xyzw[3]
    message.gripper_width_m = state.gripper_width_m
    message.object_state.header.stamp = ns_to_time(state.timestamp_ns)
    message.object_state.header.frame_id = state.frame_id
    message.object_state.schema = state.object_schema
    message.object_state.entries = [
        _named_array_to_ros(entry) for entry in state.object_state
    ]
    message.object_state.fresh = state.fresh
    message.object_state.valid = state.valid
    message.object_state.invalid_reasons = list(state.invalid_reasons)
    message.source_robot = state.source_robot
    message.fresh = state.fresh
    message.valid = state.valid
    message.invalid_reasons = list(state.invalid_reasons)
    return message


def canonical_state_from_ros(message: CanonicalStateMsg) -> CanonicalState:
    object_sample = object_sample_from_ros(message.object_state)
    reasons = list(message.invalid_reasons)
    if not message.valid and not reasons:
        reasons.append('canonical_message_invalid')
    if not object_sample.valid:
        reasons.extend(object_sample.invalid_reasons or ('object_state_invalid',))
    return make_canonical_state(
        ee_position_m=(
            message.ee_position_m.x,
            message.ee_position_m.y,
            message.ee_position_m.z,
        ),
        ee_quaternion_xyzw=(
            message.ee_quaternion_xyzw.x,
            message.ee_quaternion_xyzw.y,
            message.ee_quaternion_xyzw.z,
            message.ee_quaternion_xyzw.w,
        ),
        gripper_width_m=message.gripper_width_m,
        object_schema=object_sample.schema,
        object_state=object_sample.entries,
        frame_id=message.header.frame_id,
        timestamp_ns=time_to_ns(message.header.stamp),
        source_robot=message.source_robot,
        fresh=bool(message.fresh) and object_sample.fresh,
        invalid_reasons=reasons,
    )
