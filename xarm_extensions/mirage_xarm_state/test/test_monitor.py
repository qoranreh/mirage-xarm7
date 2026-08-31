from geometry_msgs.msg import TransformStamped
import pytest
import rclpy
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import TransformException

import mirage_xarm_state.monitor as monitor_module
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


def stamped_joint_state(
    node,
    names,
    positions,
    velocities=(),
    efforts=(),
    age_sec=0.0,
):
    message = JointState()
    stamp_ns = node.get_clock().now().nanoseconds - int(age_sec * 1e9)
    message.header.stamp = Time(nanoseconds=stamp_ns).to_msg()
    message.name = list(names)
    message.position = [float(value) for value in positions]
    message.velocity = [float(value) for value in velocities]
    message.effort = [float(value) for value in efforts]
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
        monitor,
        ('drive_joint',),
        (0.2,),
        age_sec=1.0,
    )
    monitor._joint_state_callback(stale)
    gripper_status = monitor._gripper_diagnostic()
    assert diagnostic_values(gripper_status)['valid'] == 'False'
    assert 'stale by' in gripper_status.message


def test_fake_hardware_nan_effort_still_publishes_state(monitor):
    transform = TransformStamped()
    transform.header.frame_id = 'link_base'
    transform.child_frame_id = 'link_eef'
    transform.transform.rotation.w = 1.0
    monitor._tf_buffer.lookup_transform = lambda *args, **kwargs: transform
    names = (
        'drive_joint', 'joint2', 'joint3', 'joint5',
        'joint6', 'joint1', 'joint4', 'joint7',
    )
    message = stamped_joint_state(
        monitor,
        names,
        (0.4, 2.0, 3.0, 5.0, 6.0, 1.0, 4.0, 7.0),
        velocities=(0.0,) * 8,
        efforts=(float('nan'),) * 8,
    )

    monitor._joint_state_callback(message)
    monitor._process_pending_tf()
    monitor._publish_diagnostics()

    arm = monitor._arm_publisher.messages[-1]
    gripper = monitor._gripper_publisher.messages[-1]
    assert tuple(arm.name) == ARM_NAMES
    assert tuple(arm.position) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert tuple(arm.velocity) == (0.0,) * 7
    assert not arm.effort
    assert gripper.name == ['drive_joint']
    assert tuple(gripper.position) == (0.4,)
    assert tuple(gripper.velocity) == (0.0,)
    assert not gripper.effort
    assert len(monitor._pose_publisher.messages) == 1

    statuses = {
        status.name: status
        for status in monitor._diagnostic_publisher.messages[-1].status
    }
    arm_status = statuses['mirage_xarm_state/arm']
    gripper_status = statuses['mirage_xarm_state/gripper']
    assert arm_status.level == arm_status.OK
    assert gripper_status.level == gripper_status.OK
    assert diagnostic_values(arm_status)['omitted_optional_fields'] == 'effort'
    assert (
        diagnostic_values(gripper_status)['omitted_optional_fields']
        == 'effort'
    )


def test_tf_warmup_drops_clear_after_first_success(monitor):
    monitor._tf_wait_timeout_sec = 0.0

    def missing_transform(*args, **kwargs):
        raise TransformException('not buffered yet')

    monitor._tf_buffer.lookup_transform = missing_transform
    monitor._joint_state_callback(
        stamped_joint_state(monitor, ARM_NAMES, (0.0,) * 7)
    )
    monitor._process_pending_tf()

    warming_up = monitor._tf_diagnostic()
    warming_values = diagnostic_values(warming_up)
    assert warming_up.level == warming_up.WARN
    assert warming_values['ready'] == 'False'
    assert warming_values['warmup_drops'] == '1'
    assert warming_values['failures'] == '0'

    transform = TransformStamped()
    transform.transform.rotation.w = 1.0
    monitor._tf_buffer.lookup_transform = lambda *args, **kwargs: transform
    monitor._joint_state_callback(
        stamped_joint_state(monitor, ARM_NAMES, (0.0,) * 7)
    )
    monitor._process_pending_tf()

    ready = monitor._tf_diagnostic()
    ready_values = diagnostic_values(ready)
    assert ready.level == ready.OK
    assert ready_values['ready'] == 'True'
    assert ready_values['warmup_drops'] == '1'
    assert ready_values['failures'] == '0'

    monitor._tf_buffer.lookup_transform = missing_transform
    monitor._joint_state_callback(
        stamped_joint_state(monitor, ARM_NAMES, (0.0,) * 7)
    )
    monitor._process_pending_tf()

    failed = monitor._tf_diagnostic()
    failed_values = diagnostic_values(failed)
    assert failed.level == failed.WARN
    assert failed_values['warmup_drops'] == '1'
    assert failed_values['failures'] == '1'
    assert failed_values['consecutive_failures'] == '1'

    monitor._tf_buffer.lookup_transform = lambda *args, **kwargs: transform
    monitor._joint_state_callback(
        stamped_joint_state(monitor, ARM_NAMES, (0.0,) * 7)
    )
    monitor._process_pending_tf()

    recovered = monitor._tf_diagnostic()
    recovered_values = diagnostic_values(recovered)
    assert recovered.level == recovered.OK
    assert recovered_values['failures'] == '1'
    assert recovered_values['consecutive_failures'] == '0'


def test_main_tolerates_subscription_error_during_shutdown(monkeypatch):
    class FakeNode:
        destroyed = False

        def destroy_node(self):
            self.destroyed = True

    node = FakeNode()
    monkeypatch.setattr(monitor_module, 'XArmStateMonitor', lambda: node)

    def shutdown_during_spin(unused_node):
        monitor_module.rclpy.shutdown()
        raise RuntimeError('subscription stopped during shutdown')

    monkeypatch.setattr(
        monitor_module.rclpy,
        'spin',
        shutdown_during_spin,
    )

    monitor_module.main()

    assert node.destroyed
    assert not monitor_module.rclpy.ok()


def test_main_does_not_hide_runtime_error_while_ros_is_running(monkeypatch):
    class FakeNode:
        destroyed = False

        def destroy_node(self):
            self.destroyed = True

    node = FakeNode()
    monkeypatch.setattr(monitor_module, 'XArmStateMonitor', lambda: node)

    def fail_during_spin(unused_node):
        raise RuntimeError('real runtime failure')

    monkeypatch.setattr(monitor_module.rclpy, 'spin', fail_during_spin)

    with pytest.raises(RuntimeError, match='real runtime failure'):
        monitor_module.main()

    assert node.destroyed
    assert not monitor_module.rclpy.ok()
