import numpy as np
import pytest

from mirage.benchmark.robosuite.xarm_transfer.contracts import (
    ContractError,
    CartesianPose,
    OscActionScaler,
    SourceAction,
)
from mirage.benchmark.robosuite.xarm_transfer.dynamics import (
    ForwardDynamicsError,
    ForwardDynamicsRequest,
    PandaOnlineOracle,
)


SOURCE_FRAME = "panda_sim_world"


def pose(position=(0.0, 0.0, 0.8), quaternion=(0.0, 0.0, 0.0, 1.0), timestamp=10):
    return CartesianPose(position, quaternion, SOURCE_FRAME, timestamp)


def test_source_action_layout_and_osc_scaling_are_explicit():
    action = SourceAction(
        (1.0, -1.0, 0.5, -0.5, 0.25, 1.0, -1.0),
        SOURCE_FRAME,
        11,
    )

    command = OscActionScaler().scale(action)

    np.testing.assert_allclose(
        command.translation_delta_m, (0.05, -0.05, 0.025)
    )
    np.testing.assert_allclose(
        command.rotation_delta_axis_angle_rad, (-0.25, 0.125, 0.5)
    )
    assert command.gripper_normalized == -1.0
    np.testing.assert_allclose(
        command.as_array(),
        (0.05, -0.05, 0.025, -0.25, 0.125, 0.5, -1.0),
    )
    assert action.to_dict()["layout"] == [
        "dx",
        "dy",
        "dz",
        "dRx",
        "dRy",
        "dRz",
        "gripper",
    ]


def test_raw_policy_output_is_preserved_separately_from_controller_clipping():
    action = SourceAction(
        (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
        SOURCE_FRAME,
        11,
        (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0002),
    )

    np.testing.assert_allclose(
        action.as_policy_array(),
        (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0002),
    )
    np.testing.assert_allclose(
        action.as_normalized_array(),
        (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    )
    assert action.to_dict()["policy_output_raw"][-1] == -1.0002


def test_normalized_action_and_quaternion_fail_closed():
    with pytest.raises(ContractError, match=r"\[-1, 1\]"):
        SourceAction((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1), SOURCE_FRAME, 1)
    with pytest.raises(ContractError, match="finite"):
        SourceAction((0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0), SOURCE_FRAME, 1)
    with pytest.raises(ContractError, match="norm"):
        pose(quaternion=(0.0, 0.0, 0.0, 0.5))

    canonical = pose(quaternion=(0.0, 0.0, 0.0, -1.0))
    assert canonical.quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)


def test_panda_oracle_executes_exactly_one_policy_step_and_records_achieved_pose():
    state = {
        "pose": pose(timestamp=10),
        "gripper": (0.02, -0.02),
        "calls": [],
    }
    clock_values = iter((100, 145))

    def step_once(action):
        state["calls"].append(action.copy())
        state["pose"] = pose(position=(0.01, -0.02, 0.81), timestamp=12)
        state["gripper"] = (0.01, -0.01)

    oracle = PandaOnlineOracle(
        step_policy_once=step_once,
        read_pose=lambda: state["pose"],
        read_gripper_state=lambda: state["gripper"],
        monotonic_ns=lambda: next(clock_values),
    )
    action = SourceAction(
        (0.2, -0.4, 0.1, 0.0, 0.0, 0.2, 1.0), SOURCE_FRAME, 11
    )

    result = oracle.predict(ForwardDynamicsRequest(state["pose"], action))

    assert len(state["calls"]) == 1
    np.testing.assert_allclose(state["calls"][0], action.normalized)
    assert result.provider == "panda_online_oracle"
    assert result.latency_ns == 45
    assert result.current_pose.position_m == (0.0, 0.0, 0.8)
    assert result.desired_pose.pose.position_m == (0.01, -0.02, 0.81)
    assert result.gripper_state_before == (0.02, -0.02)
    assert result.gripper_state_after == (0.01, -0.01)
    assert result.position_error_bound_m == 0.0
    assert result.orientation_error_bound_rad == 0.0


def test_panda_oracle_rejects_request_not_synchronized_to_simulator():
    oracle = PandaOnlineOracle(
        step_policy_once=lambda _action: pytest.fail("step must not be called"),
        read_pose=lambda: pose(position=(0.1, 0.0, 0.8), timestamp=10),
        read_gripper_state=lambda: (0.02, -0.02),
    )
    request = ForwardDynamicsRequest(
        pose(timestamp=10),
        SourceAction((0.0,) * 7, SOURCE_FRAME, 11),
    )

    with pytest.raises(ForwardDynamicsError, match="not synchronized"):
        oracle.predict(request)
