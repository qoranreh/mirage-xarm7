from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PoseStamped, Transform
from rclpy.time import Time
from sensor_msgs.msg import JointState
from xarm_msgs.msg import RobotMsg

from mirage_xarm_state.gate_a import (
    ARM_DIAGNOSTIC,
    GRIPPER_DIAGNOSTIC,
    INPUT_DIAGNOSTIC,
    ROBOT_DIAGNOSTIC,
    TF_DIAGNOSTIC,
    GateAConfig,
    GateAEvidence,
)


ARM_NAMES = tuple(f'joint{index}' for index in range(1, 8))


def stamp(message, nanoseconds):
    message.header.stamp = Time(nanoseconds=nanoseconds).to_msg()
    return message


def arm_message(nanoseconds, offset=0.0):
    message = stamp(JointState(), nanoseconds)
    message.name = list(ARM_NAMES)
    message.position = [offset + index * 0.01 for index in range(7)]
    message.velocity = [0.0] * 7
    return message


def pose_message(nanoseconds):
    message = stamp(PoseStamped(), nanoseconds)
    message.header.frame_id = 'link_base'
    message.pose.position.x = 0.4
    message.pose.orientation.w = 1.0
    return message


def gripper_message(nanoseconds):
    message = stamp(JointState(), nanoseconds)
    message.name = ['drive_joint']
    message.position = [0.4]
    return message


def robot_message(nanoseconds):
    return stamp(RobotMsg(), nanoseconds)


def diagnostics_message(nanoseconds):
    message = stamp(DiagnosticArray(), nanoseconds)
    message.status = [
        DiagnosticStatus(level=DiagnosticStatus.OK, name=name, message='ok')
        for name in (
            ARM_DIAGNOSTIC,
            GRIPPER_DIAGNOSTIC,
            TF_DIAGNOSTIC,
            ROBOT_DIAGNOSTIC,
            INPUT_DIAGNOSTIC,
        )
    ]
    return message


def build_passing_evidence():
    config = GateAConfig(
        min_arm_samples=5,
        min_pose_samples=5,
        min_gripper_samples=2,
        min_robot_samples=2,
        min_arm_rate_hz=5.0,
        min_pose_rate_hz=5.0,
        min_gripper_rate_hz=1.0,
        min_robot_rate_hz=1.0,
        max_message_age_sec=1.0,
        require_robot_state=True,
        check_moveit_fk=False,
    )
    evidence = GateAEvidence(config)
    start = 10_000_000_000
    transform = Transform()
    transform.translation.x = 0.4
    transform.rotation.w = 1.0
    for index in range(5):
        stamp_ns = start + index * 100_000_000
        now_ns = stamp_ns + 1_000_000
        evidence.record_arm(
            arm_message(stamp_ns, offset=index * 0.001), now_ns
        )
        pose = pose_message(stamp_ns)
        evidence.record_pose(pose, now_ns)
        evidence.record_tf_comparison(pose.pose, transform)
    for index in range(2):
        stamp_ns = start + index * 500_000_000
        now_ns = stamp_ns + 1_000_000
        evidence.record_gripper(gripper_message(stamp_ns), now_ns)
        evidence.record_robot(robot_message(stamp_ns), now_ns)
    evidence.record_diagnostics(
        diagnostics_message(start + 400_000_000),
        start + 401_000_000,
    )
    return evidence


def report(evidence):
    return evidence.build_report(
        started_at='2026-08-02T00:00:00+00:00',
        completed_at='2026-08-02T00:00:01+00:00',
        requested_duration_sec=1.0,
        topics={},
        report_path='/tmp/gate_a.json',
    )


def test_passing_evidence_reports_gate_a_pass():
    result = report(build_passing_evidence())

    assert result['passed']
    assert result['read_only']['motion_command_publishers'] == []
    assert result['metrics']['exact_arm_pose_pairs'] == 5
    assert result['metrics']['tf']['max_position_error_m'] == 0.0


def test_nonfinite_pose_and_robot_error_fail_required_checks():
    evidence = build_passing_evidence()
    bad_pose = pose_message(10_500_000_000)
    bad_pose.pose.orientation.w = float('nan')
    evidence.record_pose(bad_pose, 10_501_000_000)
    bad_robot = robot_message(10_500_000_000)
    bad_robot.err = 12
    evidence.record_robot(bad_robot, 10_501_000_000)

    result = report(evidence)
    checks = {item['name']: item for item in result['checks']}

    assert not result['passed']
    assert not checks['pose_message_validity']['passed']
    assert not checks['robot_error_state']['passed']


def test_motion_check_is_optional_until_explicitly_required():
    evidence = build_passing_evidence()
    result = report(evidence)
    checks = {item['name']: item for item in result['checks']}

    assert checks['joint_motion_observed']['passed']
    assert not checks['joint_motion_observed']['required']
