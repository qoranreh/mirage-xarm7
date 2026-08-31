import json

import numpy as np
import pytest

from mirage.benchmark.robosuite.xarm_transfer.contracts import (
    CartesianPose,
    OscActionScaler,
    SourceAction,
)
from mirage.benchmark.robosuite.xarm_transfer.dataset import (
    DatasetError,
    TransitionDataset,
    TransitionRecord,
    assert_no_group_leakage,
    split_by_group,
)
from mirage.benchmark.robosuite.xarm_transfer.dynamics import ForwardDynamicsRequest
from mirage.benchmark.robosuite.xarm_transfer.factory import (
    ForwardDynamicsConfig,
    ForwardDynamicsConfigError,
    create_forward_dynamics,
)
from mirage.benchmark.robosuite.xarm_transfer.geometry import (
    integrate_world_rotation_delta,
    rotvec_to_quaternion_xyzw,
)
from mirage.benchmark.robosuite.xarm_transfer.models import (
    ModelError,
    PoseDeltaForwardDynamics,
    compare_linear_models,
)
from mirage.benchmark.robosuite.xarm_transfer.training import train_and_evaluate
from mirage.benchmark.robosuite.xarm_transfer.profiles import (
    CheckpointContract,
    ObservationMode,
    TASK_PROFILES,
)


SOURCE_FRAME = "panda_sim_world"


def synthetic_records(task_name="PickPlaceCan"):
    generator = np.random.default_rng(123)
    scaler = OscActionScaler()
    records = []
    for episode in range(6):
        for step in range(8):
            timestamp = 1_000_000_000 + episode * 1_000_000 + step * 50_000
            position = generator.uniform((-0.2, -0.2, 0.7), (0.2, 0.2, 1.0))
            current_rotvec = generator.normal(0.0, 0.1, size=3)
            current_quaternion = rotvec_to_quaternion_xyzw(current_rotvec)
            normalized = generator.uniform(-0.8, 0.8, size=7)
            action = SourceAction(tuple(normalized), SOURCE_FRAME, timestamp + 1)
            command = scaler.scale(action)
            position_delta = np.array(
                (
                    0.01 * position[0] + 0.03 * normalized[0],
                    -0.02 * position[1] + 0.025 * normalized[1],
                    0.005 * position[2] + 0.02 * normalized[2],
                )
            )
            rotation_delta = 0.12 * normalized[3:6]
            current_pose = CartesianPose(
                tuple(position),
                tuple(current_quaternion),
                SOURCE_FRAME,
                timestamp,
            )
            next_pose = CartesianPose(
                tuple(position + position_delta),
                tuple(
                    integrate_world_rotation_delta(
                        current_quaternion, rotation_delta
                    )
                ),
                SOURCE_FRAME,
                timestamp + 50_000_000,
            )
            records.append(
                TransitionRecord(
                    episode_id=f"episode-{episode}",
                    step_id=step,
                    seed=episode // 2,
                    timestamp_ns=action.timestamp_ns,
                    current_pose=current_pose,
                    normalized_action=action,
                    scaled_action=command,
                    next_pose=next_pose,
                    gripper_state_before=(0.02, -0.02),
                    gripper_state_after=(0.01, -0.01),
                    provider="panda_online_oracle",
                    latency_ns=1000,
                    task_name=task_name,
                    observation_mode=ObservationMode.LOW_DIM,
                )
            )
    return tuple(records)


def test_dataset_jsonl_round_trip_and_quaternion_sign(tmp_path):
    records = synthetic_records()
    path = tmp_path / "transitions.jsonl"

    TransitionDataset(records).write_jsonl(path)
    loaded = TransitionDataset.read_jsonl(path)

    assert len(loaded) == len(records)
    assert loaded.records[0].to_dict() == records[0].to_dict()
    assert loaded.records[0].to_dict()["version"] == 2
    assert loaded.records[0].task_name == "PickPlaceCan"
    assert loaded.records[0].observation_mode is ObservationMode.LOW_DIM
    assert loaded.records[0].current_pose.quaternion_xyzw[3] >= 0.0


def test_version_1_dataset_reads_as_pick_place_can_low_dim(tmp_path):
    payload = synthetic_records()[0].to_dict()
    payload["version"] = 1
    payload.pop("task_name")
    payload.pop("observation_mode")
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    loaded = TransitionDataset.read_jsonl(path)

    assert loaded.records[0].task_name == "PickPlaceCan"
    assert loaded.records[0].observation_mode is ObservationMode.LOW_DIM


def test_lift_dataset_and_model_version_2_round_trip(tmp_path):
    records = synthetic_records("Lift")
    dataset_path = tmp_path / "lift.jsonl"
    TransitionDataset(records).write_jsonl(dataset_path)
    loaded_dataset = TransitionDataset.read_jsonl(dataset_path)
    model = PoseDeltaForwardDynamics(
        model_kind="ridge",
        expected_frame_id=SOURCE_FRAME,
        ridge_alpha=0.1,
        task_name="Lift",
        observation_mode=ObservationMode.LOW_DIM,
    ).fit(loaded_dataset.records)
    model.evaluate(loaded_dataset.records, trust_as_validation=True)
    loaded_model = PoseDeltaForwardDynamics.load(
        model.save(tmp_path / "lift_model.npz")
    )

    assert {record.task_name for record in loaded_dataset.records} == {"Lift"}
    assert loaded_model.task_name == "Lift"
    assert loaded_model.observation_mode is ObservationMode.LOW_DIM


@pytest.mark.parametrize("group_by", ("episode", "seed"))
def test_dataset_split_never_leaks_episode_or_seed(group_by):
    records = synthetic_records()

    train, validation = split_by_group(
        records,
        validation_fraction=0.34,
        group_by=group_by,
        random_seed=7,
    )

    assert train and validation
    assert_no_group_leakage(train, validation, group_by=group_by)


def test_dataset_split_rejects_transition_level_fallback():
    records = synthetic_records()[:8]
    with pytest.raises(DatasetError, match="at least two episode"):
        split_by_group(records, group_by="episode")


def test_linear_and_ridge_compare_with_baselines_and_save_load(tmp_path):
    train, validation = split_by_group(
        synthetic_records(),
        validation_fraction=0.34,
        group_by="episode",
        random_seed=3,
    )

    metrics, best = compare_linear_models(
        train,
        validation,
        expected_frame_id=SOURCE_FRAME,
        ridge_alphas=(0.001, 0.1),
    )

    assert {
        "linear",
        "ridge_alpha_0.001",
        "ridge_alpha_0.1",
        "baseline_identity",
        "baseline_delta_integration",
        "oracle_ground_truth",
    } == set(metrics)
    assert metrics["linear"].position_max_m < 1e-10
    assert metrics["linear"].orientation_max_rad < 1e-7
    assert best.validation_metrics is not None

    artifact = best.save(tmp_path / "forward_dynamics.npz")
    loaded = PoseDeltaForwardDynamics.load(artifact)
    request = ForwardDynamicsRequest(
        validation[0].current_pose, validation[0].normalized_action
    )
    original = best.predict(request)
    restored = loaded.predict(request)
    np.testing.assert_allclose(
        restored.desired_pose.pose.as_array(),
        original.desired_pose.pose.as_array(),
        atol=1e-12,
    )
    assert restored.position_error_bound_m == pytest.approx(
        best.validation_metrics.position_max_m
    )
    assert loaded.task_name == "PickPlaceCan"
    assert loaded.observation_mode is ObservationMode.LOW_DIM


def test_version_1_model_loads_as_pick_place_can_low_dim(tmp_path):
    model = PoseDeltaForwardDynamics(
        model_kind="ridge", expected_frame_id=SOURCE_FRAME, ridge_alpha=0.1
    ).fit(synthetic_records())
    model.evaluate(synthetic_records(), trust_as_validation=True)
    version_2 = model.save(tmp_path / "version_2.npz")
    with np.load(version_2, allow_pickle=False) as artifact:
        metadata = json.loads(str(artifact["metadata"].item()))
        metadata["version"] = 1
        metadata.pop("task_name")
        metadata.pop("observation_mode")
        np.savez_compressed(
            tmp_path / "version_1.npz",
            coef=artifact["coef"],
            intercept=artifact["intercept"],
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        )

    loaded = PoseDeltaForwardDynamics.load(tmp_path / "version_1.npz")

    assert loaded.task_name == "PickPlaceCan"
    assert loaded.observation_mode is ObservationMode.LOW_DIM


def test_ridge_artifact_rejects_checkpoint_task_mismatch(tmp_path):
    train, validation = split_by_group(
        synthetic_records(), validation_fraction=0.34, group_by="episode"
    )
    _, model = compare_linear_models(
        train, validation, expected_frame_id=SOURCE_FRAME, ridge_alphas=(0.1,)
    )
    artifact = model.save(tmp_path / "pick_place_can.npz")
    lift_profile = TASK_PROFILES["Lift"]
    lift_contract = CheckpointContract(
        task_profile=lift_profile,
        observation_mode=ObservationMode.LOW_DIM,
        observation_shapes=tuple(lift_profile.low_dim_observation_shapes),
        native_rollout_horizon=400,
    )

    with pytest.raises(ForwardDynamicsConfigError, match="does not match"):
        create_forward_dynamics(
            ForwardDynamicsConfig("learned_ridge", str(artifact)),
            checkpoint_contract=lift_contract,
        )


def test_learned_model_fails_closed_on_frame_and_non_finite_prediction():
    records = synthetic_records()
    model = PoseDeltaForwardDynamics(
        model_kind="ridge", expected_frame_id=SOURCE_FRAME, ridge_alpha=0.1
    ).fit(records)
    wrong_pose = CartesianPose(
        records[0].current_pose.position_m,
        records[0].current_pose.quaternion_xyzw,
        "wrong_frame",
        records[0].current_pose.timestamp_ns,
    )
    wrong_action = SourceAction(
        records[0].normalized_action.normalized,
        "wrong_frame",
        records[0].normalized_action.timestamp_ns,
    )
    with pytest.raises(ModelError, match="request frame"):
        model.predict(ForwardDynamicsRequest(wrong_pose, wrong_action))

    model.coef_[0, 0] = np.inf
    with pytest.raises(ModelError, match="non-finite"):
        model.predict(
            ForwardDynamicsRequest(
                records[0].current_pose, records[0].normalized_action
            )
        )


def test_training_pipeline_selects_loadable_validated_ridge(tmp_path):
    dataset_path = tmp_path / "transitions.jsonl"
    TransitionDataset(synthetic_records()).write_jsonl(dataset_path)

    outputs = train_and_evaluate(
        dataset_path,
        output_directory=tmp_path / "outputs",
        validation_fraction=0.34,
        group_by="episode",
        split_seed=11,
        ridge_alphas=(0.001, 0.01, 0.1),
        enforce_git_ignored=False,
    )

    assert outputs.selected_model.model_kind == "ridge"
    assert outputs.selected_model.validation_metrics is not None
    assert outputs.model_path.is_file()
    assert outputs.report_path.is_file()
    loaded = create_forward_dynamics(
        ForwardDynamicsConfig("learned_ridge", str(outputs.model_path))
    )
    assert isinstance(loaded, PoseDeltaForwardDynamics)
    assert loaded.model_kind == "ridge"
    assert loaded.validation_metrics == outputs.selected_model.validation_metrics
