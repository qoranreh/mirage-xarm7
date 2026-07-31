from geometry_msgs.msg import TransformStamped
import pytest
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import JointState

from mirage_xarm_state.monitor import XArmStateMonitor


ARM_NAMES = tuple(f'joint{index}' for index in range(1, 8))


class CapturePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


@pytest.fixture
def monitor():
    rclpy.init()
    node = XArmStateMonitor()
    node._arm_publisher = CapturePublisher()
    node._pose_publisher = CapturePublisher()
    node._gripper_publisher = CapturePublisher()
    node._diagnostic_publisher = CapturePublisher()
    yield node
    node.destroy_node()
    rclpy.shutdown()


def stamped_joint_state(node, names, positions, age_sec=0.0):
    message = JointState()
    stamp_ns = node.get_clock().now().nanoseconds - int(age_sec * 1e9)
    message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    message.name = list(names)
    message.position = [float(value) for value in positions]
    return message


def diagnostic_values(status):
    return {item.key: item.value for item in status.values}


def test_separate_caches_exact_tf_and_diagnostics(monitor):
    transform = TransformStamped()
    transform.header.frame_id = 'link_base'
    transform.child_frame_id = 'link_eef'
    transform.transform.translation.x = 0.5
    transform.transform.rotation.w = 1.0
    monitor._tf_buffer.lookup_transform = lambda *args, **kwargs: transform
    gripper = stamped_joint_state(monitor, ('drive_joint',), (0.4,))
    shuffled = (
        'joint5', 'joint2', 'joint7', 'joint1',
        'joint4', 'joint6', 'joint3',
    )
    arm = stamped_joint_state(
        monitor, shuffled, (5.0, 2.0, 7.0, 1.0, 4.0, 6.0, 3.0)
    )

    monitor._joint_state_callback(gripper)
    monitor._joint_state_callback(arm)
    monitor._process_pending_tf()
    monitor._publish_diagnostics()

    assert len(monitor._gripper_publisher.messages) == 1
    assert monitor._gripper_publisher.messages[0].name == ['drive_joint']
    assert len(monitor._arm_publisher.messages) == 1
    assert tuple(monitor._arm_publisher.messages[0].name) == ARM_NAMES
    assert tuple(monitor._arm_publisher.messages[0].position) == (
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
    )
    assert len(monitor._pose_publisher.messages) == 1
    pose = monitor._pose_publisher.messages[0]
    assert pose.header.stamp == arm.header.stamp
    assert pose.header.frame_id == 'link_base'
    assert pose.pose.position.x == 0.5

    statuses = {
        status.name: status
        for status in monitor._diagnostic_publisher.messages[-1].status
    }
    arm_values = diagnostic_values(statuses['mirage_xarm_state/arm'])
    gripper_values = diagnostic_values(
        statuses['mirage_xarm_state/gripper']
    )
    assert arm_values['valid'] == 'True'
    assert arm_values['published'] == '1'
    assert gripper_values['valid'] == 'True'
    assert gripper_values['published'] == '1'


def test_missing_nonfinite_and_stale_samples_are_diagnosed(monitor):
    missing = stamped_joint_state(monitor, ARM_NAMES[:-1], [0.0] * 6)
    monitor._joint_state_callback(missing)
    arm_status = monitor._arm_diagnostic()
    assert diagnostic_values(arm_status)['valid'] == 'False'
    assert diagnostic_values(arm_status)['missing_joints'] == 'joint7'

    nonfinite = stamped_joint_state(monitor, ARM_NAMES, [0.0] * 7)
    nonfinite.position[2] = float('nan')
    monitor._joint_state_callback(nonfinite)
    assert 'non-finite: position:joint3' in monitor._arm_diagnostic().message

    stale = stamped_joint_state(
        monitor, ('drive_joint',), (0.2,), age_sec=1.0
    )
    monitor._joint_state_callback(stale)
    gripper_status = monitor._gripper_diagnostic()
    assert diagnostic_values(gripper_status)['valid'] == 'False'
    assert 'stale by' in gripper_status.message
