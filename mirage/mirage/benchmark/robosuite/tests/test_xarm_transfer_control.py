import math

import numpy as np
import pytest

from mirage.benchmark.robosuite.xarm_transfer.contracts import (
    CartesianPose,
    SourceDesiredPose,
)
from mirage.benchmark.robosuite.xarm_transfer.control import (
    AbsoluteBlockingController,
    PredictionSafetyBounds,
    SafetyLimits,
    SimulationSafetyFilter,
    XArmSimulationGripperMapper,
)
from mirage.benchmark.robosuite.xarm_transfer.simulation_transform import (
    SimulationTaskFrameTransform,
    SimulationToolFrameCorrection,
    SimulationTransform,
    SimulationTransformError,
    TargetDesiredPose,
)


SOURCE_FRAME = "panda_sim_world"
TARGET_FRAME = "xarm7_sim_world"
NOW_NS = 1_000_000_000


def source_pose(position=(0.0, 0.0, 0.8), timestamp=NOW_NS - 1_000_000):
    return CartesianPose(position, (0.0, 0.0, 0.0, 1.0), SOURCE_FRAME, timestamp)


def target_pose(
    position=(0.0, 0.0, 0.8),
    quaternion=(0.0, 0.0, 0.0, 1.0),
    timestamp=NOW_NS - 1_000_000,
):
    return CartesianPose(position, quaternion, TARGET_FRAME, timestamp)


def desired(position=(0.1, 0.0, 0.8), action_timestamp=NOW_NS - 1_000_000):
    return TargetDesiredPose(
        target_pose(position, timestamp=action_timestamp + 50_000_000),
        1.0,
        action_timestamp,
    )


def safety_limits():
    return SafetyLimits(
        target_frame=TARGET_FRAME,
        workspace_min_m=(-0.5, -0.5, 0.4),
        workspace_max_m=(0.5, 0.5, 1.2),
        joint_min_rad=(-2.0,) * 7,
        joint_max_rad=(2.0,) * 7,
    )


def test_task_and_tool_transforms_are_separate_and_round_trip():
    half_sqrt = math.sqrt(0.5)
    transform = SimulationTransform(
        task_frame_transform=SimulationTaskFrameTransform(
            SOURCE_FRAME,
            TARGET_FRAME,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, half_sqrt, half_sqrt),
            "test_task_rotation",
        ),
        tool_frame_correction=SimulationToolFrameCorrection(
            "panda_grip_site",
            "xarm7_grip_site",
            (0.0, 0.0, 0.0),
            (half_sqrt, 0.0, 0.0, half_sqrt),
            "test_tool_rotation",
        ),
    )
    source = source_pose((1.0, 0.0, 0.8))

    mapped = transform.transform_pose(source)

    # The task-frame rotation rotates the world position. The right-multiplied
    # tool rotation changes orientation only; it cannot rotate position motion.
    np.testing.assert_allclose(mapped.position_m, (0.0, 1.0, 0.8), atol=1e-12)
    round_trip = transform.inverse().transform_pose(mapped)
    np.testing.assert_allclose(round_trip.as_array(), source.as_array(), atol=1e-12)
    assert transform.to_dict()["calibrated"] is False
    assert transform.kind == "simulation_only"

    invalid = transform.to_dict()
    invalid["calibrated"] = True
    with pytest.raises(SimulationTransformError, match="calibrated: false"):
        SimulationTransform.from_dict(invalid)


def test_shared_pick_place_world_keeps_position_and_corrects_only_tool_axis():
    transform = SimulationTransform.for_shared_task_world(
        source_frame=SOURCE_FRAME,
        target_frame=TARGET_FRAME,
    )
    source = source_pose((0.17, -0.31, 0.91))

    mapped = transform.transform_pose(source)

    np.testing.assert_allclose(mapped.position_m, source.position_m, atol=1e-12)
    half_sqrt = math.sqrt(0.5)
    np.testing.assert_allclose(
        mapped.quaternion_xyzw,
        (0.0, 0.0, -half_sqrt, half_sqrt),
        atol=1e-12,
    )
    payload = transform.to_dict()
    assert payload["version"] == 2
    assert payload["task_frame_transform"]["rotation_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    np.testing.assert_allclose(
        payload["tool_frame_correction"]["rotation_xyzw"],
        (0.0, 0.0, -half_sqrt, half_sqrt),
        atol=1e-12,
    )
    restored = SimulationTransform.from_dict(payload)
    np.testing.assert_allclose(
        restored.transform_pose(source).as_array(), mapped.as_array()
    )

    with pytest.raises(SimulationTransformError, match="unsafe version-1"):
        SimulationTransform.from_dict({"version": 1})


def test_source_desired_pose_transform_preserves_action_and_gripper_contract():
    transform = SimulationTransform.for_shared_task_world(
        source_frame=SOURCE_FRAME,
        target_frame=TARGET_FRAME,
    )
    source_desired = SourceDesiredPose(source_pose((0.02, 0.0, 0.81)), -1.0, NOW_NS)

    target_desired = transform.transform_desired_pose(source_desired)

    assert target_desired.pose.frame_id == TARGET_FRAME
    assert target_desired.gripper_normalized == -1.0
    assert target_desired.source_action_timestamp_ns == NOW_NS


@pytest.mark.parametrize(
    "target, joints, bounds, now, reason",
    (
        (desired((0.7, 0.0, 0.8)), (0.0,) * 7, (0.0, 0.0), NOW_NS, "workspace_violation"),
        (desired(), (2.0,) + (0.0,) * 6, (0.0, 0.0), NOW_NS, "joint_limit_violation"),
        (desired(action_timestamp=0), (0.0,) * 7, (0.0, 0.0), NOW_NS, "stale_command"),
        (desired(), (0.0,) * 7, (math.inf, 0.0), NOW_NS, "position_prediction_bound"),
        (desired(), (0.0,) * 7, (0.0, math.inf), NOW_NS, "orientation_prediction_bound"),
    ),
)
def test_safety_filter_rejects_before_control(target, joints, bounds, now, reason):
    decision = SimulationSafetyFilter(safety_limits()).evaluate(
        target,
        joint_positions_rad=joints,
        prediction_bounds=PredictionSafetyBounds(*bounds),
        now_ns=now,
    )

    assert not decision.accepted
    assert decision.reason == reason


def test_gripper_mapping_has_explicit_simulation_semantics():
    assert XArmSimulationGripperMapper().map(-1.0) == -1.0
    assert XArmSimulationGripperMapper().map(1.0) == 1.0
    assert XArmSimulationGripperMapper(invert=True).map(-1.0) == 1.0


class FakeAdapter:
    def __init__(
        self,
        *,
        converge=True,
        contact=False,
        contact_residual_m=None,
        contact_residual_orientation_rad=0.0,
        task_success=False,
    ):
        self.pose = target_pose((0.0, 0.0, 0.8))
        self.joints = np.zeros(7)
        self.absolute_mode_calls = 0
        self.actions = []
        self.gripper_commands = []
        self.converge = converge
        self.contact = contact
        self.contact_residual_m = contact_residual_m
        self.contact_residual_orientation_rad = contact_residual_orientation_rad
        self.task_success = task_success

    def set_absolute_mode(self):
        self.absolute_mode_calls += 1

    def read_pose(self):
        return self.pose

    def read_joint_positions(self):
        return self.joints.copy()

    def has_gripper_task_object_contact(self):
        return self.contact

    def task_succeeded(self):
        return self.task_success

    def step_absolute(self, action):
        self.actions.append(action.copy())
        if self.contact_residual_m is not None:
            position = action[:3].copy()
            position[0] -= self.contact_residual_m
            half_angle = self.contact_residual_orientation_rad / 2.0
            quaternion = (0.0, 0.0, math.sin(half_angle), math.cos(half_angle))
            self.pose = target_pose(tuple(position), quaternion)
        elif self.converge:
            position = (
                np.asarray(self.pose.position_m) + action[:3]
            ) / 2.0
            self.pose = target_pose(tuple(position))

    def apply_gripper_command_once(self, normalized_command):
        self.gripper_commands.append(float(normalized_command))


def test_blocking_controller_repeats_absolute_target_and_records_errors():
    adapter = FakeAdapter()
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        position_tolerance_m=0.004,
        orientation_tolerance_rad=0.01,
        max_iterations=20,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert result.succeeded
    assert 1 < result.iterations <= 20
    assert result.position_error_m <= 0.004
    assert result.orientation_error_rad <= 0.01
    assert adapter.absolute_mode_calls == 1
    assert len(adapter.actions) == result.iterations
    for action in adapter.actions:
        np.testing.assert_allclose(action[:3], (0.1, 0.0, 0.8))
        assert action[-1] == 0.0
    assert adapter.gripper_commands == [1.0]
    assert result.gripper_apply_count == 1


def test_blocking_controller_requires_consecutive_settling_iterations():
    adapter = FakeAdapter()
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        position_tolerance_m=0.004,
        orientation_tolerance_rad=0.01,
        settling_iterations=2,
        max_iterations=20,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert result.succeeded
    assert result.iterations == 8
    assert result.position_error_m < 0.004


def test_bounded_outer_loop_removes_absolute_osc_position_bias():
    adapter = FakeAdapter(contact_residual_m=0.02)
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        position_tolerance_m=0.015,
        orientation_tolerance_rad=0.01,
        settling_iterations=2,
        outer_loop_position_gain=0.5,
        max_outer_loop_position_correction_m=0.03,
        max_iterations=20,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
        monotonic=lambda: 0.0,
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert result.succeeded
    assert result.position_error_m <= 0.015
    assert any(not np.isclose(action[0], 0.1) for action in adapter.actions)


def test_outer_loop_orientation_correction_is_angularly_bounded():
    adapter = FakeAdapter()
    five_degrees = math.radians(5.0)
    adapter.pose = CartesianPose(
        (0.0, 0.0, 0.8),
        (0.0, 0.0, math.sin(five_degrees), math.cos(five_degrees)),
        TARGET_FRAME,
        NOW_NS,
    )
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        position_tolerance_m=0.015,
        orientation_tolerance_rad=0.01,
        outer_loop_orientation_gain=1.0,
        max_outer_loop_orientation_correction_rad=five_degrees,
        max_iterations=1,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
        monotonic=lambda: 0.0,
    )

    controller.drive(
        desired((0.0, 0.0, 0.8)),
        prediction_bounds=PredictionSafetyBounds(0.0, 0.0),
    )

    assert np.linalg.norm(adapter.actions[0][3:6]) == pytest.approx(five_degrees)


def test_outer_loop_command_is_safety_checked_before_controller_mode():
    adapter = FakeAdapter()
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        outer_loop_position_gain=0.5,
        max_outer_loop_position_correction_m=0.03,
        wall_clock_ns=lambda: NOW_NS,
    )

    result = controller.drive(
        desired((0.49, 0.0, 0.8)),
        prediction_bounds=PredictionSafetyBounds(0.0, 0.0),
    )

    assert not result.succeeded
    assert result.rejected_reason == "workspace_violation"
    assert adapter.absolute_mode_calls == 0
    assert adapter.actions == []


def test_blocking_controller_does_not_touch_adapter_after_initial_rejection():
    adapter = FakeAdapter()
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        wall_clock_ns=lambda: NOW_NS,
    )

    result = controller.drive(
        desired((0.8, 0.0, 0.8)),
        prediction_bounds=PredictionSafetyBounds(0.0, 0.0),
    )

    assert not result.succeeded
    assert result.rejected_reason == "workspace_violation"
    assert adapter.absolute_mode_calls == 0
    assert adapter.actions == []


def test_contact_tolerance_requires_verified_task_object_contact():
    adapter = FakeAdapter(contact=True, contact_residual_m=0.02)
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        position_tolerance_m=0.015,
        orientation_tolerance_rad=0.01,
        contact_position_tolerance_m=0.03,
        contact_orientation_tolerance_rad=0.02,
        max_iterations=2,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
        monotonic=lambda: 0.0,
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert result.succeeded
    assert result.contact_tolerance_used
    assert result.position_error_m == pytest.approx(0.02)

    adapter.contact = False
    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )
    assert not result.succeeded
    assert result.iteration_limit_reached
    assert not result.contact_tolerance_used


def test_contact_tolerance_can_accept_orientation_only_residual():
    adapter = FakeAdapter(
        contact=True,
        contact_residual_m=0.01,
        contact_residual_orientation_rad=0.015,
    )
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        position_tolerance_m=0.015,
        orientation_tolerance_rad=0.01,
        contact_position_tolerance_m=0.03,
        contact_orientation_tolerance_rad=0.02,
        max_iterations=2,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
        monotonic=lambda: 0.0,
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert result.succeeded
    assert result.contact_tolerance_used
    assert result.position_error_m == pytest.approx(0.01)
    assert result.orientation_error_rad == pytest.approx(0.015)


def test_blocking_controller_checks_task_success_after_alignment_and_gripper():
    adapter = FakeAdapter(converge=True, task_success=True)
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        position_tolerance_m=0.001,
        orientation_tolerance_rad=0.001,
        max_iterations=20,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
        monotonic=lambda: 0.0,
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert result.succeeded
    assert result.task_succeeded
    assert result.iterations > 1
    assert adapter.gripper_commands == [1.0]


def test_blocking_controller_reports_timeout_without_sending_after_deadline():
    adapter = FakeAdapter(converge=False)
    times = iter((0.0, 0.1, 0.2))
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        timeout_sec=0.05,
        wall_clock_ns=lambda: NOW_NS,
        monotonic=lambda: next(times),
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert result.timed_out
    assert result.iterations == 0
    assert adapter.actions == []
    assert adapter.gripper_commands == []


def test_blocking_controller_reports_iteration_limit_as_failure():
    adapter = FakeAdapter(converge=False)
    controller = AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(safety_limits()),
        max_iterations=2,
        timeout_sec=2.0,
        wall_clock_ns=lambda: NOW_NS,
        monotonic=lambda: 0.0,
    )

    result = controller.drive(
        desired(), prediction_bounds=PredictionSafetyBounds(0.0, 0.0)
    )

    assert not result.succeeded
    assert not result.timed_out
    assert result.iteration_limit_reached
    assert result.iterations == 2
    assert adapter.gripper_commands == []
