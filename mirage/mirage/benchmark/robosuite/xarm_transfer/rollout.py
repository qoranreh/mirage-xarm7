"""Online-oracle and learned-model xArm simulation rollouts."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .contracts import CartesianPose, SourceAction
from .control import (
    AbsoluteBlockingController,
    PredictionSafetyBounds,
)
from .dataset import TransitionDataset, TransitionRecord, record_from_dynamics_result
from .dynamics import (
    ForwardDynamics,
    ForwardDynamicsRequest,
    PandaOnlineOracle,
)
from .geometry import quaternion_angular_error_rad
from .image_runtime import CrossPaintSourceStep, CrossPaintSourceWorker
from .profiles import ObservationMode, task_profile_for_name
from .runtime import FrankaPandaSourceRuntime, RobosuiteXArmAbsoluteAdapter
from .simulation_transform import SimulationTransform


class RolloutError(RuntimeError):
    """Raised when an online or learned rollout violates its contract."""


def _optional_task_success(adapter: object) -> bool:
    checker = getattr(adapter, "task_succeeded", None)
    if not callable(checker):
        return False
    result = checker()
    if not isinstance(result, (bool, np.bool_)):
        raise RolloutError("task success check did not return a bool")
    return bool(result)


@dataclass(frozen=True)
class TransferStepMetrics:
    episode_id: str
    step_id: int
    seed: int
    provider: str
    controller_succeeded: bool
    controller_iterations: int
    controller_timed_out: bool
    controller_iteration_limit_reached: bool
    rejected_reason: str | None
    target_position_error_m: float
    target_orientation_error_rad: float
    source_prediction_position_error_m: float
    source_prediction_orientation_error_rad: float
    dynamics_latency_ms: float
    controller_latency_ms: float
    controller_contact_tolerance_used: bool
    source_task_succeeded: bool
    target_task_succeeded: bool
    visual_input: str | None = None
    perception_latency_ms: float = 0.0
    composition_latency_ms: float = 0.0
    target_robot_pixels: int = 0
    source_robot_pixels: int = 0
    source_visible_pixels: int = 0
    alignment_position_error_m: float = 0.0
    alignment_orientation_error_rad: float = 0.0
    alignment_iterations: int = 0
    gripper_apply_count: int = 0
    protected_object_pixels: int = 0
    object_pixel_loss: int = 0
    target_gripper_aperture_m: float = 0.0
    target_gripper_open_fraction: float = 0.0
    source_gripper_aperture_m: float = 0.0
    source_gripper_open_fraction: float = 0.0


@dataclass(frozen=True)
class TransferRolloutReport:
    mode: str
    task_name: str
    observation_mode: str
    episode_id: str
    seed: int
    native_rollout_horizon: int
    requested_steps: int
    completed_steps: int
    completed: bool
    timeout_count: int
    iteration_limit_count: int
    rejected_command_count: int
    contact_tolerance_count: int
    source_task_succeeded: bool
    target_task_succeeded: bool
    source_success_step: int | None
    target_success_step: int | None
    position_error_mean_m: float
    position_error_max_m: float
    orientation_error_mean_rad: float
    orientation_error_max_rad: float
    source_prediction_position_error_mean_m: float
    source_prediction_position_error_max_m: float
    source_prediction_orientation_error_mean_rad: float
    source_prediction_orientation_error_max_rad: float
    latency_mean_ms: float
    latency_max_ms: float
    simulation_transform: dict[str, object]
    steps: tuple[TransferStepMetrics, ...]
    visual_input: str | None = None
    perception_latency_mean_ms: float = 0.0
    perception_latency_max_ms: float = 0.0
    composition_latency_mean_ms: float = 0.0
    composition_latency_max_ms: float = 0.0
    alignment_position_error_max_m: float = 0.0
    alignment_orientation_error_max_rad: float = 0.0
    gripper_apply_count: int = 0
    protected_object_pixels: int = 0
    object_pixel_loss: int = 0
    gripper_open_fraction_error_max: float = 0.0


def _summarize(
    *,
    mode: str,
    task_name: str,
    observation_mode: ObservationMode,
    episode_id: str,
    seed: int,
    native_rollout_horizon: int,
    requested_steps: int,
    transform: SimulationTransform,
    steps: Sequence[TransferStepMetrics],
) -> TransferRolloutReport:
    source_success_steps = [item.step_id + 1 for item in steps if item.source_task_succeeded]
    target_success_steps = [item.step_id + 1 for item in steps if item.target_task_succeeded]
    source_task_succeeded = bool(source_success_steps)
    target_task_succeeded = bool(target_success_steps)
    position = np.asarray(
        [item.target_position_error_m for item in steps], dtype=np.float64
    )
    orientation = np.asarray(
        [item.target_orientation_error_rad for item in steps], dtype=np.float64
    )
    latency = np.asarray(
        [item.dynamics_latency_ms + item.controller_latency_ms for item in steps],
        dtype=np.float64,
    )
    source_position = np.asarray(
        [item.source_prediction_position_error_m for item in steps],
        dtype=np.float64,
    )
    source_orientation = np.asarray(
        [item.source_prediction_orientation_error_rad for item in steps],
        dtype=np.float64,
    )
    perception = np.asarray(
        [item.perception_latency_ms for item in steps], dtype=np.float64
    )
    composition = np.asarray(
        [item.composition_latency_ms for item in steps], dtype=np.float64
    )
    alignment_position = np.asarray(
        [item.alignment_position_error_m for item in steps], dtype=np.float64
    )
    alignment_orientation = np.asarray(
        [item.alignment_orientation_error_rad for item in steps], dtype=np.float64
    )
    gripper_fraction_error = np.asarray(
        [
            abs(
                item.target_gripper_open_fraction
                - item.source_gripper_open_fraction
            )
            for item in steps
            if item.visual_input is not None
        ],
        dtype=np.float64,
    )
    visual_inputs = {item.visual_input for item in steps if item.visual_input is not None}
    if len(visual_inputs) > 1:
        raise RolloutError("one rollout cannot mix visual input modes")
    return TransferRolloutReport(
        mode=mode,
        task_name=task_name,
        observation_mode=observation_mode.value,
        episode_id=episode_id,
        seed=seed,
        native_rollout_horizon=native_rollout_horizon,
        requested_steps=requested_steps,
        completed_steps=len(steps),
        completed=(
            all(item.controller_succeeded for item in steps)
            and (
                len(steps) == requested_steps
                or target_task_succeeded
            )
        ),
        timeout_count=sum(item.controller_timed_out for item in steps),
        iteration_limit_count=sum(
            item.controller_iteration_limit_reached for item in steps
        ),
        rejected_command_count=sum(item.rejected_reason is not None for item in steps),
        contact_tolerance_count=sum(
            item.controller_contact_tolerance_used for item in steps
        ),
        source_task_succeeded=source_task_succeeded,
        target_task_succeeded=target_task_succeeded,
        source_success_step=(source_success_steps[0] if source_success_steps else None),
        target_success_step=(target_success_steps[0] if target_success_steps else None),
        position_error_mean_m=float(np.mean(position)) if position.size else 0.0,
        position_error_max_m=float(np.max(position)) if position.size else 0.0,
        orientation_error_mean_rad=(
            float(np.mean(orientation)) if orientation.size else 0.0
        ),
        orientation_error_max_rad=(
            float(np.max(orientation)) if orientation.size else 0.0
        ),
        source_prediction_position_error_mean_m=(
            float(np.mean(source_position)) if source_position.size else 0.0
        ),
        source_prediction_position_error_max_m=(
            float(np.max(source_position)) if source_position.size else 0.0
        ),
        source_prediction_orientation_error_mean_rad=(
            float(np.mean(source_orientation)) if source_orientation.size else 0.0
        ),
        source_prediction_orientation_error_max_rad=(
            float(np.max(source_orientation)) if source_orientation.size else 0.0
        ),
        latency_mean_ms=float(np.mean(latency)) if latency.size else 0.0,
        latency_max_ms=float(np.max(latency)) if latency.size else 0.0,
        simulation_transform=transform.to_dict(),
        steps=tuple(steps),
        visual_input=next(iter(visual_inputs)) if visual_inputs else None,
        perception_latency_mean_ms=(
            float(np.mean(perception)) if perception.size else 0.0
        ),
        perception_latency_max_ms=(
            float(np.max(perception)) if perception.size else 0.0
        ),
        composition_latency_mean_ms=(
            float(np.mean(composition)) if composition.size else 0.0
        ),
        composition_latency_max_ms=(
            float(np.max(composition)) if composition.size else 0.0
        ),
        alignment_position_error_max_m=(
            float(np.max(alignment_position)) if alignment_position.size else 0.0
        ),
        alignment_orientation_error_max_rad=(
            float(np.max(alignment_orientation))
            if alignment_orientation.size
            else 0.0
        ),
        gripper_apply_count=sum(item.gripper_apply_count for item in steps),
        protected_object_pixels=sum(
            item.protected_object_pixels for item in steps
        ),
        object_pixel_loss=sum(item.object_pixel_loss for item in steps),
        gripper_open_fraction_error_max=(
            float(np.max(gripper_fraction_error))
            if gripper_fraction_error.size
            else 0.0
        ),
    )


def run_online_oracle_episode(
    *,
    source_runtime: FrankaPandaSourceRuntime,
    target_adapter: RobosuiteXArmAbsoluteAdapter,
    controller: AbsoluteBlockingController,
    seed: int,
    steps: int = 50,
    terminate_on_success: bool = False,
) -> tuple[TransitionDataset, TransferRolloutReport]:
    """Run policy -> one Panda step -> simulation T_TS -> blocking xArm."""

    if type(steps) is not int or steps <= 0:
        raise RolloutError("steps must be a positive integer")
    contract = source_runtime.checkpoint_contract
    if target_adapter.task_profile != contract.task_profile:
        raise RolloutError("source and target task profiles do not match")
    source_runtime.reset(seed)
    target_adapter.reset(seed)
    source_home = source_runtime.read_pose()
    target_home = target_adapter.read_pose()
    transform = SimulationTransform.for_task_profile(
        contract.task_profile,
        source_frame=source_home.frame_id,
        target_frame=target_home.frame_id,
    )
    oracle = source_runtime.make_oracle()
    if not isinstance(oracle, PandaOnlineOracle):
        raise RolloutError("online rollout requires PandaOnlineOracle")
    episode_id = f"{contract.task_profile.slug}-oracle-seed-{seed}"
    transitions = []
    step_metrics = []
    for step_id in range(steps):
        current = source_runtime.read_pose()
        action_timestamp = max(source_runtime.clock_ns(), current.timestamp_ns + 1)
        action = source_runtime.policy_action(timestamp_ns=action_timestamp)
        result = oracle.predict(ForwardDynamicsRequest(current, action))
        transitions.append(
            record_from_dynamics_result(
                result,
                episode_id=episode_id,
                step_id=step_id,
                seed=seed,
                task_name=contract.task_name,
                observation_mode=contract.observation_mode,
            )
        )
        target_desired = transform.transform_desired_pose(result.desired_pose)
        control_result = controller.drive(
            target_desired,
            prediction_bounds=PredictionSafetyBounds(
                result.position_error_bound_m,
                result.orientation_error_bound_rad,
            ),
        )
        source_task_succeeded = source_runtime.task_succeeded()
        target_task_succeeded = target_adapter.task_succeeded()
        step_metrics.append(
            TransferStepMetrics(
                episode_id,
                step_id,
                seed,
                result.provider,
                control_result.succeeded,
                control_result.iterations,
                control_result.timed_out,
                control_result.iteration_limit_reached,
                control_result.rejected_reason,
                control_result.position_error_m,
                control_result.orientation_error_rad,
                0.0,
                0.0,
                result.latency_ns / 1e6,
                control_result.latency_sec * 1e3,
                control_result.contact_tolerance_used,
                source_task_succeeded,
                target_task_succeeded,
                gripper_apply_count=control_result.gripper_apply_count,
            )
        )
        if not control_result.succeeded:
            break
        if terminate_on_success and target_task_succeeded:
            break
    return TransitionDataset(transitions), _summarize(
        mode="online_oracle",
        task_name=contract.task_name,
        observation_mode=contract.observation_mode,
        episode_id=episode_id,
        seed=seed,
        native_rollout_horizon=contract.native_rollout_horizon,
        requested_steps=steps,
        transform=transform,
        steps=step_metrics,
    )


def run_cross_paint_oracle_episode(
    *,
    source_worker: CrossPaintSourceWorker,
    target_adapter: RobosuiteXArmAbsoluteAdapter,
    controller: AbsoluteBlockingController,
    seed: int,
    steps: int = 50,
    terminate_on_success: bool = False,
    visual_input: str = "cross_paint",
    frame_callback: Callable[[int, object, CrossPaintSourceStep], None] | None = None,
) -> tuple[TransitionDataset, TransferRolloutReport]:
    """Run an image policy through process-isolated cross-painting."""

    if type(steps) is not int or steps <= 0:
        raise RolloutError("steps must be a positive integer")
    if visual_input not in {"cross_paint", "naive"}:
        raise RolloutError("visual input must be cross_paint or naive")
    contract = source_worker.checkpoint_contract
    if contract.observation_mode is ObservationMode.LOW_DIM:
        raise RolloutError("cross-paint rollout requires an image checkpoint")
    if target_adapter.task_profile != contract.task_profile:
        raise RolloutError("source and target task profiles do not match")

    source_home = source_worker.reset(seed)
    target_adapter.reset(seed)
    target_home = target_adapter.read_pose()
    transform = SimulationTransform.for_task_profile(
        contract.task_profile,
        source_frame=source_home.frame_id,
        target_frame=target_home.frame_id,
    )
    episode_id = f"{contract.task_profile.slug}-{visual_input}-seed-{seed}"
    transitions = []
    step_metrics = []
    for step_id in range(steps):
        target_pose = target_adapter.read_pose()
        target_frame = target_adapter.render_robot_camera_frame(
            mask_render_scale=getattr(source_worker, "mask_render_scale", 1)
        )
        target_gripper_state = target_adapter.read_gripper_aperture_state()
        source_step = source_worker.step(
            target_frame=target_frame,
            target_pose=target_pose,
            object_joint_states=(
                target_adapter.read_manipulated_object_joint_states()
            ),
            target_gripper_state=target_gripper_state,
            visual_input=visual_input,
        )
        result = source_step.dynamics
        transitions.append(
            record_from_dynamics_result(
                result,
                episode_id=episode_id,
                step_id=step_id,
                seed=seed,
                task_name=contract.task_name,
                observation_mode=contract.observation_mode,
            )
        )
        target_desired = transform.transform_desired_pose(result.desired_pose)
        control_result = controller.drive(
            target_desired,
            prediction_bounds=PredictionSafetyBounds(
                result.position_error_bound_m,
                result.orientation_error_bound_rad,
            ),
        )
        target_task_succeeded = target_adapter.task_succeeded()
        diagnostics = source_step.diagnostics
        step_metrics.append(
            TransferStepMetrics(
                episode_id=episode_id,
                step_id=step_id,
                seed=seed,
                provider=result.provider,
                controller_succeeded=control_result.succeeded,
                controller_iterations=control_result.iterations,
                controller_timed_out=control_result.timed_out,
                controller_iteration_limit_reached=(
                    control_result.iteration_limit_reached
                ),
                rejected_reason=control_result.rejected_reason,
                target_position_error_m=control_result.position_error_m,
                target_orientation_error_rad=control_result.orientation_error_rad,
                source_prediction_position_error_m=0.0,
                source_prediction_orientation_error_rad=0.0,
                dynamics_latency_ms=result.latency_ns / 1e6,
                controller_latency_ms=control_result.latency_sec * 1e3,
                controller_contact_tolerance_used=(
                    control_result.contact_tolerance_used
                ),
                source_task_succeeded=source_step.source_task_succeeded,
                target_task_succeeded=target_task_succeeded,
                visual_input=diagnostics.visual_input,
                perception_latency_ms=diagnostics.perception_latency_ns / 1e6,
                composition_latency_ms=diagnostics.composition_latency_ns / 1e6,
                target_robot_pixels=diagnostics.target_robot_pixels,
                source_robot_pixels=diagnostics.source_robot_pixels,
                source_visible_pixels=diagnostics.source_visible_pixels,
                alignment_position_error_m=(
                    diagnostics.alignment_position_error_m
                ),
                alignment_orientation_error_rad=(
                    diagnostics.alignment_orientation_error_rad
                ),
                alignment_iterations=diagnostics.alignment_iterations,
                gripper_apply_count=control_result.gripper_apply_count,
                protected_object_pixels=(
                    diagnostics.protected_object_pixels
                ),
                object_pixel_loss=diagnostics.object_pixel_loss,
                target_gripper_aperture_m=(
                    diagnostics.target_gripper_aperture_m
                ),
                target_gripper_open_fraction=(
                    diagnostics.target_gripper_open_fraction
                ),
                source_gripper_aperture_m=(
                    diagnostics.source_gripper_aperture_m
                ),
                source_gripper_open_fraction=(
                    diagnostics.source_gripper_open_fraction
                ),
            )
        )
        if frame_callback is not None:
            frame_callback(step_id, target_frame, source_step)
        if not control_result.succeeded:
            break
        if terminate_on_success and target_task_succeeded:
            break
    return TransitionDataset(transitions), _summarize(
        mode=f"online_oracle_{visual_input}",
        task_name=contract.task_name,
        observation_mode=contract.observation_mode,
        episode_id=episode_id,
        seed=seed,
        native_rollout_horizon=contract.native_rollout_horizon,
        requested_steps=steps,
        transform=transform,
        steps=step_metrics,
    )


def run_learned_replay_episode(
    *,
    oracle_transitions: Sequence[TransitionRecord],
    learned_dynamics: ForwardDynamics,
    target_adapter: RobosuiteXArmAbsoluteAdapter,
    controller: AbsoluteBlockingController,
    seed: int,
    steps: int = 50,
    clock_ns=time.time_ns,
) -> TransferRolloutReport:
    """Autoregressively replay oracle actions from the same initial source pose."""

    if type(steps) is not int or steps <= 0:
        raise RolloutError("steps must be a positive integer")
    if len(oracle_transitions) < steps:
        raise RolloutError("oracle replay does not contain the requested horizon")
    selected = tuple(oracle_transitions[:steps])
    if len({record.episode_id for record in selected}) != 1:
        raise RolloutError("learned replay transitions must come from one episode")
    if any(record.seed != seed for record in selected):
        raise RolloutError("learned replay seed does not match transition seed")
    task_modes = {
        (record.task_name, record.observation_mode) for record in selected
    }
    if len(task_modes) != 1:
        raise RolloutError(
            "learned replay must contain one task and observation mode"
        )
    task_name, observation_mode = task_modes.pop()
    task_profile = task_profile_for_name(task_name)
    if getattr(target_adapter, "task_profile", task_profile) != task_profile:
        raise RolloutError("replay and target task profiles do not match")
    model_task = getattr(learned_dynamics, "task_name", task_name)
    model_mode = ObservationMode.parse(
        getattr(learned_dynamics, "observation_mode", observation_mode)
    )
    if model_task != task_name or model_mode is not observation_mode:
        raise RolloutError(
            "learned artifact task/observation mode does not match replay"
        )

    target_adapter.reset(seed)
    first = selected[0]
    initial_time = clock_ns()
    current = CartesianPose(
        first.current_pose.position_m,
        first.current_pose.quaternion_xyzw,
        first.current_pose.frame_id,
        initial_time,
    )
    target_home = target_adapter.read_pose()
    transform = SimulationTransform.for_task_profile(
        task_profile,
        source_frame=current.frame_id,
        target_frame=target_home.frame_id,
    )
    episode_id = f"{task_profile.slug}-learned-replay-seed-{seed}"
    step_metrics = []
    for step_id, oracle_record in enumerate(selected):
        action_timestamp = max(clock_ns(), current.timestamp_ns + 1)
        action = SourceAction(
            oracle_record.normalized_action.normalized,
            current.frame_id,
            action_timestamp,
            oracle_record.normalized_action.policy_output_raw,
        )
        result = learned_dynamics.predict(
            ForwardDynamicsRequest(current_pose=current, action=action)
        )
        oracle_next = oracle_record.next_pose
        source_position_error = float(
            np.linalg.norm(
                np.asarray(result.desired_pose.pose.position_m)
                - np.asarray(oracle_next.position_m)
            )
        )
        source_orientation_error = quaternion_angular_error_rad(
            result.desired_pose.pose.quaternion_xyzw,
            oracle_next.quaternion_xyzw,
        )
        target_desired = transform.transform_desired_pose(result.desired_pose)
        control_result = controller.drive(
            target_desired,
            prediction_bounds=PredictionSafetyBounds(
                result.position_error_bound_m,
                result.orientation_error_bound_rad,
            ),
        )
        target_task_succeeded = _optional_task_success(target_adapter)
        step_metrics.append(
            TransferStepMetrics(
                episode_id,
                step_id,
                seed,
                result.provider,
                control_result.succeeded,
                control_result.iterations,
                control_result.timed_out,
                control_result.iteration_limit_reached,
                control_result.rejected_reason,
                control_result.position_error_m,
                control_result.orientation_error_rad,
                source_position_error,
                source_orientation_error,
                result.latency_ns / 1e6,
                control_result.latency_sec * 1e3,
                control_result.contact_tolerance_used,
                False,
                target_task_succeeded,
                gripper_apply_count=control_result.gripper_apply_count,
            )
        )
        if not control_result.succeeded:
            break
        predicted = result.desired_pose.pose
        current = CartesianPose(
            predicted.position_m,
            predicted.quaternion_xyzw,
            predicted.frame_id,
            clock_ns(),
        )
    return _summarize(
        mode="learned_ridge",
        task_name=task_name,
        observation_mode=observation_mode,
        episode_id=episode_id,
        seed=seed,
        native_rollout_horizon=task_profile.native_horizon,
        requested_steps=steps,
        transform=transform,
        steps=step_metrics,
    )


def write_rollout_reports(
    path: str | Path, reports: Sequence[TransferRolloutReport]
) -> Path:
    """Atomically persist per-seed rollout metrics as JSON."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 2,
        "reports": [
            {
                **asdict(report),
                "steps": [asdict(step) for step in report.steps],
            }
            for report in reports
        ],
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination
