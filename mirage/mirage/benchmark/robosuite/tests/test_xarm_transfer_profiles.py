from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from mirage.benchmark.robosuite.xarm_transfer.contracts import CartesianPose
from mirage.benchmark.robosuite.xarm_transfer.collection import (
    default_artifact_directory,
)
from mirage.benchmark.robosuite.xarm_transfer.profiles import (
    LIFT_CONTROLLER,
    PICK_PLACE_CAN_CONTROLLER,
    ObservationMode,
    RuntimeContractError,
    TASK_PROFILES,
    resolve_checkpoint_contract,
)
from mirage.benchmark.robosuite.xarm_transfer.runtime import (
    FrankaPandaSourceRuntime,
    build_source_environment_metadata,
    build_target_environment_metadata,
    create_xarm_target_environment,
)
from mirage.benchmark.robosuite.xarm_transfer.simulation_transform import (
    SimulationTransform,
)


MODEL_ROOT = Path(__file__).parents[3] / "models"


CHECKPOINTS = {
    ("PickPlaceCan", ObservationMode.LOW_DIM): (
        MODEL_ROOT
        / "can/low_dim/model_epoch_250_PickPlaceCan_success_1.0.pth"
    ),
    ("PickPlaceCan", ObservationMode.IMAGE_NO_PROPRIO): (
        MODEL_ROOT
        / "can/image_no_proprio/model_epoch_500_PickPlaceCan_success_0.8.pth"
    ),
    ("PickPlaceCan", ObservationMode.IMAGE_PROPRIO): (
        MODEL_ROOT
        / "can/image_proprio/model_epoch_160_PickPlaceCan_success_0.94.pth"
    ),
    ("Lift", ObservationMode.LOW_DIM): (
        MODEL_ROOT / "lift/low_dim/lift_ph_low_dim_epoch_1000_succ_100.pth"
    ),
    ("Lift", ObservationMode.IMAGE_NO_PROPRIO): (
        MODEL_ROOT
        / "lift/image_no_proprio/model_epoch_200_Lift_success_1.0.pth"
    ),
    ("Lift", ObservationMode.IMAGE_PROPRIO): (
        MODEL_ROOT
        / "lift/image_proprio/model_epoch_350_Lift_success_0.98.pth"
    ),
    ("Square_D0", ObservationMode.LOW_DIM): (
        MODEL_ROOT / "square/low_dim/model_epoch_950_Square_D0_success_0.96.pth"
    ),
    ("Square_D0", ObservationMode.IMAGE_NO_PROPRIO): (
        MODEL_ROOT
        / "square/image_no_proprio/model_epoch_380_Square_D0_success_0.78.pth"
    ),
    ("Square_D0", ObservationMode.IMAGE_PROPRIO): (
        MODEL_ROOT
        / "square/image_proprio/model_epoch_460_Square_D0_success_0.82.pth"
    ),
    ("Stack_D0", ObservationMode.LOW_DIM): (
        MODEL_ROOT / "stack/low_dim/model_epoch_2000.pth"
    ),
    ("Stack_D0", ObservationMode.IMAGE_NO_PROPRIO): (
        MODEL_ROOT / "stack/image_no_proprio/model_epoch_80_Stack_D0_success_1.0.pth"
    ),
    ("Stack_D0", ObservationMode.IMAGE_PROPRIO): (
        MODEL_ROOT / "stack/image_proprio/model_epoch_600.pth"
    ),
    ("ThreePieceAssembly_D0", ObservationMode.LOW_DIM): (
        MODEL_ROOT
        / "two_piece/low_dim/model_epoch_1000_ThreePieceAssembly_D0_success_0.68.pth"
    ),
    ("ThreePieceAssembly_D0", ObservationMode.IMAGE_NO_PROPRIO): (
        MODEL_ROOT
        / "two_piece/image_no_proprio/model_epoch_160_ThreePieceAssembly_D0_success_1.0.pth"
    ),
    ("ThreePieceAssembly_D0", ObservationMode.IMAGE_PROPRIO): (
        MODEL_ROOT
        / "two_piece/image_proprio/model_epoch_440_ThreePieceAssembly_D0_success_0.72.pth"
    ),
}

EXPECTED_PROFILES = {
    "PickPlaceCan": (14, 400, ("Can",)),
    "Lift": (10, 400, ("cube",)),
    "Square_D0": (14, 400, ("SquareNut",)),
    "Stack_D0": (23, 400, ("cubeA",)),
    "ThreePieceAssembly_D0": (42, 500, ("piece_1", "piece_2")),
}


def checkpoint(task_name, observation_mode):
    return torch.load(CHECKPOINTS[(task_name, observation_mode)], map_location="cpu")


@pytest.mark.parametrize("task_name", tuple(EXPECTED_PROFILES))
@pytest.mark.parametrize("observation_mode", tuple(ObservationMode))
def test_checkpoint_contract_resolves_registered_task_mode_and_native_horizon(
    task_name, observation_mode
):
    contract = resolve_checkpoint_contract(checkpoint(task_name, observation_mode))

    assert contract.task_name == task_name
    assert contract.observation_mode is observation_mode
    assert contract.native_rollout_horizon == EXPECTED_PROFILES[task_name][1]
    assert contract.task_profile is TASK_PROFILES[task_name]


def test_registered_low_dim_shapes_and_manipulated_objects_are_exact():
    assert set(TASK_PROFILES) == set(EXPECTED_PROFILES)
    for task_name, (object_size, horizon, object_names) in EXPECTED_PROFILES.items():
        profile = TASK_PROFILES[task_name]
        assert profile.low_dim_shapes["object"] == (object_size,)
        assert profile.native_horizon == horizon
        assert profile.manipulated_object_names == object_names

    assert TASK_PROFILES["PickPlaceCan"].graspable_object_name == "Can"
    assert TASK_PROFILES["Lift"].graspable_object_name == "cube"
    with pytest.raises(RuntimeContractError, match="multiple manipulated objects"):
        _ = TASK_PROFILES["ThreePieceAssembly_D0"].graspable_object_name
    assert str(
        default_artifact_directory(
            TASK_PROFILES["Lift"], ObservationMode.LOW_DIM
        )
    ).endswith("results/xarm_transfer/lift/low_dim")


def test_checkpoint_contract_rejects_shape_mismatch_and_unknown_combination():
    data = deepcopy(checkpoint("Lift", ObservationMode.LOW_DIM))
    data["shape_metadata"]["all_shapes"]["object"] = [14]
    with pytest.raises(RuntimeContractError, match="object.*shape.*10"):
        resolve_checkpoint_contract(data)

    data = deepcopy(checkpoint("Lift", ObservationMode.IMAGE_PROPRIO))
    data["shape_metadata"]["all_shapes"]["object"] = [10]
    with pytest.raises(RuntimeContractError, match="unknown.*combination"):
        resolve_checkpoint_contract(data)


@pytest.mark.parametrize(
    "task_name",
    ("Square_D0", "Stack_D0", "ThreePieceAssembly_D0"),
)
def test_low_dim_contract_uses_shapes_when_camera_metadata_is_stale(task_name):
    data = checkpoint(task_name, ObservationMode.LOW_DIM)
    assert data["env_metadata"]["env_kwargs"]["use_camera_obs"] is True

    contract = resolve_checkpoint_contract(data)
    source = build_source_environment_metadata(data, contract, render=False)

    assert contract.observation_mode is ObservationMode.LOW_DIM
    assert source["env_kwargs"]["use_camera_obs"] is False
    assert source["env_kwargs"]["has_offscreen_renderer"] is False


@pytest.mark.parametrize(
    "task_name, observation_mode",
    (
        ("PickPlaceCan", ObservationMode.IMAGE_NO_PROPRIO),
        ("Lift", ObservationMode.IMAGE_PROPRIO),
        ("ThreePieceAssembly_D0", ObservationMode.IMAGE_NO_PROPRIO),
    ),
)
def test_image_runtime_fails_before_simulator_creation(task_name, observation_mode):
    checkpoint_path = CHECKPOINTS[(task_name, observation_mode)]
    for factory in (
        create_xarm_target_environment,
        FrankaPandaSourceRuntime.from_checkpoint,
    ):
        with pytest.raises(
            RuntimeContractError, match="image observation runtime not implemented"
        ):
            factory(checkpoint_path)


def test_source_and_target_camera_metadata_are_separate():
    data = checkpoint("Lift", ObservationMode.IMAGE_PROPRIO)
    contract = resolve_checkpoint_contract(data)

    source = build_source_environment_metadata(data, contract, render=False)
    target = build_target_environment_metadata(data, contract, render=False)

    assert source["env_kwargs"]["use_camera_obs"] is True
    assert source["env_kwargs"]["has_offscreen_renderer"] is True
    assert target["env_kwargs"]["use_camera_obs"] is False
    assert target["env_kwargs"]["has_offscreen_renderer"] is False
    assert target["env_kwargs"]["robots"] == ["XArm7"]

    low_dim_data = checkpoint("PickPlaceCan", ObservationMode.LOW_DIM)
    low_dim_contract = resolve_checkpoint_contract(low_dim_data)
    low_dim_source = build_source_environment_metadata(
        low_dim_data, low_dim_contract, render=False
    )
    assert low_dim_source["env_kwargs"]["use_camera_obs"] is False
    assert low_dim_source["env_kwargs"]["has_offscreen_renderer"] is False
    assert low_dim_data["env_metadata"]["env_kwargs"]["robots"] == ["Panda"]


def test_controller_profiles_enable_contact_only_for_tasks_that_require_it():
    assert PICK_PLACE_CAN_CONTROLLER.free_osc_kp == (150.0,) * 6
    assert PICK_PLACE_CAN_CONTROLLER.contact_osc_kp == (
        300.0,
        300.0,
        500.0,
        300.0,
        300.0,
        300.0,
    )
    assert PICK_PLACE_CAN_CONTROLLER.position_tolerance_m == pytest.approx(0.025)
    assert PICK_PLACE_CAN_CONTROLLER.orientation_tolerance_rad == pytest.approx(
        np.deg2rad(7.0)
    )
    assert PICK_PLACE_CAN_CONTROLLER.contact_position_tolerance_m == pytest.approx(
        0.060
    )
    assert PICK_PLACE_CAN_CONTROLLER.contact_orientation_tolerance_rad == pytest.approx(
        np.deg2rad(18.0)
    )
    assert PICK_PLACE_CAN_CONTROLLER.contact_relaxation_enabled

    assert LIFT_CONTROLLER.free_osc_kp == (150.0,) * 6
    assert LIFT_CONTROLLER.position_tolerance_m == pytest.approx(0.025)
    assert LIFT_CONTROLLER.orientation_tolerance_rad == pytest.approx(
        np.deg2rad(7.0)
    )
    assert LIFT_CONTROLLER.contact_osc_kp is None
    assert LIFT_CONTROLLER.contact_position_tolerance_m is None
    assert LIFT_CONTROLLER.contact_orientation_tolerance_rad is None
    assert LIFT_CONTROLLER.contact_max_outer_loop_position_correction_m is None
    assert not LIFT_CONTROLLER.contact_relaxation_enabled

    for task_name in ("Square_D0", "ThreePieceAssembly_D0"):
        assert TASK_PROFILES[task_name].controller_profile is PICK_PLACE_CAN_CONTROLLER
    for task_name in ("Lift", "Stack_D0"):
        assert not TASK_PROFILES[task_name].controller_profile.contact_relaxation_enabled


@pytest.mark.parametrize("task_name", tuple(EXPECTED_PROFILES))
def test_task_transform_provenance_preserves_world_position(task_name):
    profile = TASK_PROFILES[task_name]
    transform = SimulationTransform.for_task_profile(
        profile,
        source_frame="panda_sim_world",
        target_frame="xarm7_sim_world",
    )
    source = CartesianPose(
        (0.13, -0.24, 0.91),
        (0.0, 0.0, 0.0, 1.0),
        "panda_sim_world",
        1,
    )

    target = transform.transform_pose(source)

    np.testing.assert_allclose(target.position_m, source.position_m)
    assert (
        transform.task_frame_transform.provenance
        == profile.task_world_provenance
    )
