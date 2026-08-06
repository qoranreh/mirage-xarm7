import hashlib
from pathlib import Path

import numpy as np

from mirage.benchmark.robosuite.xarm_transfer.rollout import (
    run_online_oracle_episode,
)
from mirage.benchmark.robosuite.xarm_transfer.robosuite_xarm7 import (
    XARM7_GRIPPER_XML,
    XARM7_ROBOT_XML,
    register_xarm7,
)
from mirage.benchmark.robosuite.xarm_transfer.runtime import (
    TARGET_FRAME_ID,
    FrankaPandaSourceRuntime,
    RobosuiteXArmAbsoluteAdapter,
    create_default_xarm_blocking_controller,
    create_xarm_target_environment,
)


def checkpoint_path():
    return (
        Path(__file__).parents[3]
        / "models"
        / "can"
        / "low_dim"
        / "model_epoch_250_PickPlaceCan_success_1.0.pth"
    )


def git_blob_id(path: Path) -> str:
    payload = path.read_bytes()
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()


def test_pinned_official_xarm7_mjcf_assets_are_unmodified():
    # ARISE-Initiative/robosuite v1.5.2, commit 824ac14cefcfb7ec...
    assert git_blob_id(XARM7_ROBOT_XML) == "b8602f087854bfe43b3056855b383e46da8ca5e3"
    assert git_blob_id(XARM7_GRIPPER_XML) == "a6527d1051f97f5af71a8eac989f621efce93feb"
    robot_meshes = tuple((XARM7_ROBOT_XML.parent / "meshes").glob("*.stl"))
    gripper_meshes = tuple(
        (XARM7_GRIPPER_XML.parent / "meshes" / "xarm7_gripper").glob("*.stl")
    )
    assert len(robot_meshes) == 9
    assert len(gripper_meshes) == 7


def test_xarm7_adapter_creates_only_an_in_process_robosuite_target(monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mirage_mpl")
    register_xarm7()
    environment = create_xarm_target_environment(checkpoint_path())
    try:
        adapter = RobosuiteXArmAbsoluteAdapter(environment)
        observation = adapter.reset(7)
        joint_min, joint_max = adapter.read_joint_limits()

        assert type(adapter.robot.robot_model).__name__ == "XArm7"
        assert adapter.robot.action_dim == 7
        assert adapter.read_pose().frame_id == TARGET_FRAME_ID
        assert observation["robot0_gripper_qpos"].shape == (6,)
        assert joint_min.shape == (7,)
        assert joint_max.shape == (7,)
        assert np.all(joint_min < joint_max)
    finally:
        environment.close()


def test_real_panda_oracle_to_xarm7_blocking_controller_completes_50_steps(
    monkeypatch,
):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mirage_mpl")
    source = FrankaPandaSourceRuntime.from_checkpoint(checkpoint_path(), device="cpu")
    target_environment = create_xarm_target_environment(checkpoint_path())
    target = RobosuiteXArmAbsoluteAdapter(target_environment)
    controller = create_default_xarm_blocking_controller(target)
    try:
        dataset, report = run_online_oracle_episode(
            source_runtime=source,
            target_adapter=target,
            controller=controller,
            seed=7,
            steps=50,
        )

        assert len(dataset) == 50
        assert source.policy_step_count == 50
        assert report.completed
        assert report.completed_steps == 50
        assert report.timeout_count == 0
        assert report.iteration_limit_count == 0
        assert report.rejected_command_count == 0
        assert report.simulation_transform["kind"] == "simulation_only"
        assert report.simulation_transform["calibrated"] is False
        np.testing.assert_allclose(
            report.simulation_transform["task_frame_transform"]["rotation_xyzw"],
            (0.0, 0.0, 0.0, 1.0),
        )
        np.testing.assert_allclose(
            report.simulation_transform["tool_frame_correction"]["rotation_xyzw"],
            (0.0, 0.0, -np.sqrt(0.5), np.sqrt(0.5)),
        )
        assert report.position_error_max_m <= controller.position_tolerance_m
        assert report.orientation_error_max_rad <= controller.orientation_tolerance_rad
        assert report.contact_tolerance_count == 0
        assert all(step.controller_succeeded for step in report.steps)
    finally:
        source.core.close()
        target_environment.close()
