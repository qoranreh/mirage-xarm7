import hashlib
from pathlib import Path

import numpy as np
import pytest

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


def checkpoint_path(task_name="PickPlaceCan"):
    root = Path(__file__).parents[3] / "models"
    paths = {
        "PickPlaceCan": "can/low_dim/model_epoch_250_PickPlaceCan_success_1.0.pth",
        "Lift": "lift/low_dim/lift_ph_low_dim_epoch_1000_succ_100.pth",
        "Square_D0": "square/low_dim/model_epoch_950_Square_D0_success_0.96.pth",
        "Stack_D0": "stack/low_dim/model_epoch_2000.pth",
        "ThreePieceAssembly_D0": (
            "two_piece/low_dim/"
            "model_epoch_1000_ThreePieceAssembly_D0_success_0.68.pth"
        ),
    }
    try:
        return root / paths[task_name]
    except KeyError as error:
        raise ValueError(task_name) from error


def task_object(core, name):
    matches = [obj for obj in core.model.mujoco_objects if obj.name == name]
    assert len(matches) == 1
    return matches[0]


def object_world_pose(core, name):
    obj = task_object(core, name)
    return np.concatenate(
        (
            np.asarray(core.sim.data.get_body_xpos(obj.root_body)),
            np.asarray(core.sim.data.get_body_xquat(obj.root_body)),
        )
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


@pytest.mark.parametrize(
    "task_name",
    ("PickPlaceCan", "Lift", "Square_D0", "Stack_D0", "ThreePieceAssembly_D0"),
)
def test_xarm7_adapter_creates_only_an_in_process_robosuite_target(
    monkeypatch, task_name
):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mirage_mpl")
    register_xarm7()
    environment = create_xarm_target_environment(checkpoint_path(task_name))
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
        assert adapter.task_profile.env_name == task_name
        assert adapter._task_object_geom_ids
        if task_name in ("Lift", "Stack_D0"):
            assert controller_contact_parameters_are_disabled(adapter)
    finally:
        environment.close()


def controller_contact_parameters_are_disabled(adapter):
    controller = create_default_xarm_blocking_controller(adapter)
    return (
        controller.contact_position_tolerance_m is None
        and controller.contact_orientation_tolerance_rad is None
        and controller.contact_max_outer_loop_position_correction_m is None
    )


@pytest.mark.parametrize(
    "task_name",
    ("PickPlaceCan", "Lift", "Square_D0", "Stack_D0", "ThreePieceAssembly_D0"),
)
@pytest.mark.parametrize("seed", (0, 1, 3, 7))
def test_panda_and_xarm_task_object_world_pose_matches_after_reset(
    monkeypatch, task_name, seed
):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mirage_mpl")
    source = FrankaPandaSourceRuntime.from_checkpoint(
        checkpoint_path(task_name), device="cpu"
    )
    target_environment = create_xarm_target_environment(checkpoint_path(task_name))
    target = RobosuiteXArmAbsoluteAdapter(target_environment)
    try:
        source.reset(seed)
        target.reset(seed)
        for object_name in source.task_profile.manipulated_object_names:
            np.testing.assert_allclose(
                object_world_pose(source.core, object_name),
                object_world_pose(target.core, object_name),
                atol=1e-12,
            )
    finally:
        source.core.close()
        target_environment.close()


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
        assert report.gripper_apply_count == 50
        assert all(step.controller_succeeded for step in report.steps)
    finally:
        source.core.close()
        target_environment.close()


@pytest.mark.parametrize(
    "task_name",
    ("Square_D0", "Stack_D0", "ThreePieceAssembly_D0"),
)
def test_new_task_online_oracle_to_xarm7_completes_one_step(
    monkeypatch, task_name
):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mirage_mpl")
    source = FrankaPandaSourceRuntime.from_checkpoint(
        checkpoint_path(task_name), device="cpu"
    )
    target_environment = create_xarm_target_environment(checkpoint_path(task_name))
    target = RobosuiteXArmAbsoluteAdapter(target_environment)
    controller = create_default_xarm_blocking_controller(target)
    try:
        dataset, report = run_online_oracle_episode(
            source_runtime=source,
            target_adapter=target,
            controller=controller,
            seed=0,
            steps=1,
        )

        assert len(dataset) == 1
        assert report.completed
        assert report.completed_steps == 1
        assert report.timeout_count == 0
        assert report.iteration_limit_count == 0
        assert report.rejected_command_count == 0
        assert report.contact_tolerance_count == 0
    finally:
        source.core.close()
        target_environment.close()


@pytest.mark.parametrize(
    "task_name, seed",
    (
        # PickPlaceCan image transfer now evaluates success as an experimental
        # rollout metric; its low-dim seed-1 success is no longer a completion
        # gate after introducing the calibrated one-shot gripper path.
        ("Lift", 0),
        ("Lift", 1),
        ("Lift", 3),
        ("Lift", 7),
        ("Square_D0", 3),
        ("Stack_D0", 0),
        ("ThreePieceAssembly_D0", 0),
    ),
)
def test_full_online_oracle_rollout_reaches_xarm_task_success(
    monkeypatch, task_name, seed
):
    monkeypatch.setenv("MPLCONFIGDIR", "/tmp/mirage_mpl")
    source = FrankaPandaSourceRuntime.from_checkpoint(
        checkpoint_path(task_name), device="cpu"
    )
    target_environment = create_xarm_target_environment(checkpoint_path(task_name))
    target = RobosuiteXArmAbsoluteAdapter(target_environment)
    controller = create_default_xarm_blocking_controller(target)
    horizon = source.task_profile.native_horizon
    try:
        _, report = run_online_oracle_episode(
            source_runtime=source,
            target_adapter=target,
            controller=controller,
            seed=seed,
            steps=horizon,
            terminate_on_success=True,
        )

        assert report.completed
        assert report.task_name == task_name
        assert report.requested_steps == horizon
        assert report.native_rollout_horizon == horizon
        assert report.target_task_succeeded
        assert report.target_success_step is not None
        assert report.target_success_step <= horizon
        assert report.timeout_count == 0
        assert report.iteration_limit_count == 0
        assert report.rejected_command_count == 0
        if task_name in ("Lift", "Stack_D0"):
            assert report.contact_tolerance_count == 0
    finally:
        source.core.close()
        target_environment.close()
