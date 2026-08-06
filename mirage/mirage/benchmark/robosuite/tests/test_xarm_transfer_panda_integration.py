from pathlib import Path

import numpy as np

from mirage.benchmark.robosuite.xarm_transfer.dynamics import (
    ForwardDynamicsRequest,
)
from mirage.benchmark.robosuite.xarm_transfer.runtime import (
    FrankaPandaSourceRuntime,
)


def test_real_franka_panda_checkpoint_oracle_executes_one_policy_step(monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mirage_mpl")
    checkpoint = (
        Path(__file__).parents[3]
        / "models"
        / "can"
        / "low_dim"
        / "model_epoch_250_PickPlaceCan_success_1.0.pth"
    )
    runtime = FrankaPandaSourceRuntime.from_checkpoint(checkpoint, device="cpu")
    runtime.reset(7)
    current = runtime.read_pose()
    action = runtime.policy_action(timestamp_ns=current.timestamp_ns + 1)

    # Snapshot the existing PickPlaceCan checkpoint output for the same seed and
    # initial observation. This is the raw policy output before controller clip.
    np.testing.assert_allclose(
        action.policy_output_raw,
        (
            0.23363043367862701,
            -0.10862657427787781,
            -0.019257916137576103,
            0.003871426684781909,
            0.15183115005493164,
            -0.03152097761631012,
            -1.0001583099365234,
        ),
        atol=1e-7,
    )

    result = runtime.make_oracle().predict(
        ForwardDynamicsRequest(current, action)
    )

    assert type(runtime.robot.robot_model).__name__ == "Panda"
    assert runtime.policy_step_count == 1
    assert result.provider == "panda_online_oracle"
    assert result.current_pose.frame_id == "panda_sim_world"
    assert result.desired_pose.pose.frame_id == "panda_sim_world"
    assert not np.allclose(
        result.current_pose.as_array(), result.desired_pose.pose.as_array()
    )
