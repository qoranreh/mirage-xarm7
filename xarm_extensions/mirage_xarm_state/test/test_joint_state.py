import math

from sensor_msgs.msg import JointState

from mirage_xarm_state.joint_state import (
    normalize_joint_state,
    parse_joint_state,
)


ARM_NAMES = tuple(f'joint{index}' for index in range(1, 8))


def make_message(names, positions, velocities=(), efforts=()):
    message = JointState()
    message.name = list(names)
    message.position = [float(value) for value in positions]
    message.velocity = [float(value) for value in velocities]
    message.effort = [float(value) for value in efforts]
    return message


def test_shuffled_arm_is_normalized_by_name():
    shuffled = (
        'joint5', 'joint2', 'joint7', 'joint1',
        'joint4', 'joint6', 'joint3',
    )
    message = make_message(
        shuffled, [5.0, 2.0, 7.0, 1.0, 4.0, 6.0, 3.0]
    )

    result = normalize_joint_state(
        message, parse_joint_state(message), ARM_NAMES
    )

    assert result.valid
    assert tuple(result.message.name) == ARM_NAMES
    assert tuple(result.message.position) == (
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    )


def test_arm_and_gripper_can_be_normalized_from_separate_messages():
    arm_message = make_message(ARM_NAMES, range(7))
    gripper_message = make_message(('drive_joint',), (0.42,))

    arm = normalize_joint_state(
        arm_message, parse_joint_state(arm_message), ARM_NAMES
    )
    gripper = normalize_joint_state(
        gripper_message,
        parse_joint_state(gripper_message),
        ('drive_joint',),
    )

    assert arm.valid
    assert gripper.valid
    assert tuple(gripper.message.position) == (0.42,)


def test_missing_joint_is_rejected_and_reported():
    message = make_message(ARM_NAMES[:-1], range(6))

    result = normalize_joint_state(
        message, parse_joint_state(message), ARM_NAMES
    )

    assert not result.valid
    assert result.missing_names == ('joint7',)


def test_mismatched_array_length_is_rejected():
    message = make_message(ARM_NAMES, range(6))

    result = normalize_joint_state(
        message, parse_joint_state(message), ARM_NAMES
    )

    assert not result.valid
    assert 'position length 6 does not match name length 7' in result.issues


def test_nonfinite_position_velocity_and_effort_are_rejected():
    parameter_names = {
        'position': 'positions',
        'velocity': 'velocities',
        'effort': 'efforts',
    }
    for field_name, bad_value in (
        ('position', math.nan),
        ('velocity', math.inf),
        ('effort', -math.inf),
    ):
        values = [0.0] * 7
        values[3] = bad_value
        kwargs = {
            'positions': [0.0] * 7,
            'velocities': [0.0] * 7,
            'efforts': [0.0] * 7,
        }
        kwargs[parameter_names[field_name]] = values
        message = make_message(ARM_NAMES, **kwargs)

        result = normalize_joint_state(
            message, parse_joint_state(message), ARM_NAMES
        )

        assert not result.valid
        assert result.nonfinite_fields == (f'{field_name}:joint4',)


def test_absent_optional_arrays_remain_absent():
    message = make_message(ARM_NAMES, range(7))

    result = normalize_joint_state(
        message, parse_joint_state(message), ARM_NAMES
    )

    assert result.valid
    assert not result.message.velocity
    assert not result.message.effort
