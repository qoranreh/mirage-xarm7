"""Process-isolated source policy and renderer for image transfer rollouts."""

from __future__ import annotations

import multiprocessing
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .contracts import CartesianPose, SourceAction
from .cross_painting import CrossPainter, CrossPaintingInput
from .dynamics import ForwardDynamicsRequest, ForwardDynamicsResult
from .profiles import CheckpointContract, ObservationMode, RuntimeContractError
from .runtime import (
    SOURCE_FRAME_ID,
    FrankaPandaSourceRuntime,
    GripperApertureState,
    ManipulatedObjectState,
    RobotCameraFrame,
    load_checkpoint_contract,
)
from .simulation_transform import SimulationTransform


class ImageRuntimeError(RuntimeError):
    """Raised when the image source worker fails or stops responding."""


@dataclass(frozen=True)
class CrossPaintStepDiagnostics:
    visual_input: str
    perception_latency_ns: int
    composition_latency_ns: int
    target_robot_pixels: int
    source_robot_pixels: int
    source_visible_pixels: int
    alignment_position_error_m: float
    alignment_orientation_error_rad: float
    alignment_iterations: int
    protected_object_pixels: int = 0
    object_pixel_loss: int = 0
    target_gripper_aperture_m: float = 0.0
    target_gripper_open_fraction: float = 0.0
    source_gripper_aperture_m: float = 0.0
    source_gripper_open_fraction: float = 0.0


@dataclass(frozen=True)
class CrossPaintSourceStep:
    current_pose: CartesianPose
    action: SourceAction
    dynamics: ForwardDynamicsResult
    source_task_succeeded: bool
    policy_rgb: np.ndarray
    source_rgb: np.ndarray
    source_robot_mask: np.ndarray
    background_rgb: np.ndarray
    target_inpaint_mask: np.ndarray
    diagnostics: CrossPaintStepDiagnostics


def _camera_projection(intrinsic: np.ndarray) -> np.ndarray:
    projection = np.eye(4, dtype=np.float64)
    projection[:3, :3] = np.asarray(intrinsic, dtype=np.float64)
    return projection


def align_source_camera_frame(
    source: RobotCameraFrame, target: RobotCameraFrame
) -> RobotCameraFrame:
    """Reproject a source RGB-D robot layer into the target camera frame."""

    if source.camera_name != target.camera_name:
        raise ImageRuntimeError("source and target camera names do not match")
    same_intrinsics = np.allclose(
        source.intrinsic_matrix, target.intrinsic_matrix, atol=1e-6, rtol=0.0
    )
    same_extrinsics = np.allclose(
        source.extrinsic_matrix, target.extrinsic_matrix, atol=1e-6, rtol=0.0
    )
    if same_intrinsics and same_extrinsics:
        return source

    mask = np.asarray(source.robot_mask, dtype=bool)
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        raise ImageRuntimeError("source camera robot mask is empty")
    depth = np.asarray(source.depth_m, dtype=np.float64)[rows, columns]
    source_pixels = np.stack(
        (
            columns * depth,
            rows * depth,
            depth,
            np.ones_like(depth),
        ),
        axis=1,
    )
    source_camera_to_world = np.asarray(source.extrinsic_matrix) @ np.linalg.inv(
        _camera_projection(source.intrinsic_matrix)
    )
    target_world_to_camera = _camera_projection(
        target.intrinsic_matrix
    ) @ np.linalg.inv(np.asarray(target.extrinsic_matrix))
    world_points = (source_camera_to_world @ source_pixels.T).T
    target_camera = (target_world_to_camera @ world_points.T).T
    valid_depth = target_camera[:, 2] > 0.0
    projected_x = target_camera[:, 0] / target_camera[:, 2]
    projected_y = target_camera[:, 1] / target_camera[:, 2]

    height, width = target.robot_mask.shape
    output_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    output_depth = np.full((height, width), 1e6, dtype=np.float64)
    output_mask = np.zeros((height, width), dtype=bool)
    output_gripper_mask = np.zeros((height, width), dtype=bool)
    colors = np.asarray(source.rgb)[rows, columns]
    gripper_pixels = np.asarray(source.gripper_mask, dtype=bool)[rows, columns]
    # Splat to the four nearest pixels and use a z-buffer. This avoids holes
    # introduced by the forward projection while preserving occlusion.
    candidates = (
        (np.floor(projected_y), np.floor(projected_x)),
        (np.floor(projected_y), np.ceil(projected_x)),
        (np.ceil(projected_y), np.floor(projected_x)),
        (np.ceil(projected_y), np.ceil(projected_x)),
    )
    for projected_rows, projected_columns in candidates:
        projected_rows = projected_rows.astype(np.int64)
        projected_columns = projected_columns.astype(np.int64)
        valid = (
            valid_depth
            & (projected_rows >= 0)
            & (projected_rows < height)
            & (projected_columns >= 0)
            & (projected_columns < width)
        )
        for index in np.flatnonzero(valid):
            row = projected_rows[index]
            column = projected_columns[index]
            point_depth = target_camera[index, 2]
            if point_depth < output_depth[row, column]:
                output_depth[row, column] = point_depth
                output_rgb[row, column] = colors[index]
                output_mask[row, column] = True
                output_gripper_mask[row, column] = gripper_pixels[index]
    if not np.any(output_mask):
        raise ImageRuntimeError("source robot projects outside the target camera")
    return RobotCameraFrame(
        rgb=output_rgb,
        depth_m=output_depth,
        robot_mask=output_mask,
        gripper_mask=output_gripper_mask,
        object_mask=np.zeros((height, width), dtype=bool),
        camera_name=target.camera_name,
        intrinsic_matrix=np.asarray(target.intrinsic_matrix).copy(),
        extrinsic_matrix=np.asarray(target.extrinsic_matrix).copy(),
    )


def transform_manipulated_object_states(
    states: Mapping[str, ManipulatedObjectState],
    transform: SimulationTransform,
) -> dict[str, ManipulatedObjectState]:
    """Transform object qpos/qvel using only the task/world transform."""

    from scipy.spatial.transform import Rotation

    matrix = np.asarray(transform.task_frame_transform.matrix, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    result: dict[str, ManipulatedObjectState] = {}
    for joint, state in states.items():
        if not isinstance(state, ManipulatedObjectState):
            raise ImageRuntimeError("object state must contain qpos and qvel")
        qpos = state.qpos.copy()
        qvel = state.qvel.copy()
        if qpos.shape == (7,) and qvel.shape == (6,):
            qpos[:3] = rotation @ qpos[:3] + translation
            source_xyzw = qpos[[4, 5, 6, 3]]
            transformed_xyzw = (
                Rotation.from_matrix(rotation) * Rotation.from_quat(source_xyzw)
            ).as_quat()
            qpos[3:] = transformed_xyzw[[3, 0, 1, 2]]
            qvel[:3] = rotation @ qvel[:3]
            qvel[3:] = rotation @ qvel[3:]
        elif qpos.shape == (4,) and qvel.shape == (3,):
            source_xyzw = qpos[[1, 2, 3, 0]]
            transformed_xyzw = (
                Rotation.from_matrix(rotation) * Rotation.from_quat(source_xyzw)
            ).as_quat()
            qpos[:] = transformed_xyzw[[3, 0, 1, 2]]
            qvel[:] = rotation @ qvel
        elif qpos.shape != (1,) or qvel.shape != (1,):
            raise ImageRuntimeError(
                f"unsupported object joint state dimensions for {joint!r}"
            )
        result[joint] = ManipulatedObjectState(qpos, qvel)
    return result


def _raw_rgb_to_policy_image(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb)
    if value.shape != (84, 84, 3) or value.dtype != np.uint8:
        raise ImageRuntimeError("naive target image must be uint8 with shape (84, 84, 3)")
    return np.ascontiguousarray(value.transpose(2, 0, 1), dtype=np.float32) / 255.0


def _worker_main(
    connection,
    checkpoint_path: str,
    device: str,
    mask_render_scale: int,
) -> None:
    runtime = None
    try:
        runtime = FrankaPandaSourceRuntime.from_checkpoint(
            checkpoint_path, device=device
        )
        contract = runtime.checkpoint_contract
        if contract.observation_mode is ObservationMode.LOW_DIM:
            raise ImageRuntimeError("image source worker rejects low-dimensional checkpoints")
        connection.send(("ready", contract))
        painter = CrossPainter()
        transform = None
        oracle = runtime.make_oracle()
        while True:
            message = connection.recv()
            command = message[0]
            try:
                if command == "close":
                    connection.send(("closed", None))
                    break
                if command == "reset":
                    seed = message[1]
                    runtime.reset(seed)
                    oracle = runtime.make_oracle()
                    transform = None
                    connection.send(("ok", runtime.read_pose()))
                    continue
                if command != "step":
                    raise ImageRuntimeError(f"unknown image worker command {command!r}")
                if runtime.observation is None:
                    raise ImageRuntimeError("image source worker must be reset first")
                (
                    _,
                    target_frame,
                    target_pose,
                    object_joint_states,
                    target_gripper_state,
                    visual_input,
                ) = message
                if visual_input not in {"cross_paint", "naive"}:
                    raise ImageRuntimeError("visual input must be cross_paint or naive")
                if transform is None:
                    transform = SimulationTransform.for_task_profile(
                        contract.task_profile,
                        source_frame=SOURCE_FRAME_ID,
                        target_frame=target_pose.frame_id,
                    )
                started_ns = time.monotonic_ns()
                desired_source = transform.inverse().transform_pose(target_pose)
                source_object_states = transform_manipulated_object_states(
                    object_joint_states, transform.inverse()
                )
                if not isinstance(target_gripper_state, GripperApertureState):
                    raise ImageRuntimeError("target gripper aperture state is invalid")
                runtime.set_manipulated_object_joint_states(source_object_states)
                runtime.synchronize_gripper_open_fraction(
                    target_gripper_state.open_fraction
                )
                position_error, orientation_error, align_iterations = (
                    runtime.align_to_pose(desired_source)
                )
                # Absolute OSC advances physics and may disturb a contacting
                # object. Reapply both qpos and qvel immediately before the
                # policy observation and render.
                runtime.set_manipulated_object_joint_states(source_object_states)
                runtime._refresh_observation()
                source_frame = align_source_camera_frame(
                    runtime.render_robot_camera_frame(
                        mask_render_scale=mask_render_scale
                    ),
                    target_frame,
                )
                source_gripper_state = runtime.read_gripper_aperture_state()
                composition_started_ns = time.monotonic_ns()
                if visual_input == "cross_paint":
                    painted = painter.compose(
                        CrossPaintingInput(
                            target_rgb=target_frame.rgb,
                            target_depth_m=target_frame.depth_m,
                            target_robot_mask=target_frame.robot_mask,
                            target_gripper_mask=target_frame.gripper_mask,
                            target_object_mask=target_frame.object_mask,
                            source_rgb=source_frame.rgb,
                            source_depth_m=source_frame.depth_m,
                            source_robot_mask=source_frame.robot_mask,
                        )
                    )
                    policy_image = painted.policy_chw_float32()
                    policy_rgb = painted.policy_rgb
                    background_rgb = painted.background_rgb
                    target_inpaint_mask = painted.target_inpaint_mask
                    target_pixels = painted.target_robot_pixels
                    source_pixels = painted.source_robot_pixels
                    visible_pixels = painted.source_visible_pixels
                    protected_object_pixels = painted.protected_object_pixels
                    object_pixel_loss = painted.object_pixel_loss
                else:
                    policy_image = _raw_rgb_to_policy_image(target_frame.rgb)
                    policy_rgb = target_frame.rgb.copy()
                    background_rgb = target_frame.rgb.copy()
                    target_inpaint_mask = target_frame.robot_mask.copy()
                    target_pixels = int(np.count_nonzero(target_frame.robot_mask))
                    source_pixels = int(np.count_nonzero(source_frame.robot_mask))
                    visible_pixels = 0
                    protected_object_pixels = 0
                    object_pixel_loss = 0
                composition_finished_ns = time.monotonic_ns()
                current = runtime.read_pose()
                action_timestamp = max(runtime.clock_ns(), current.timestamp_ns + 1)
                action = runtime.policy_action(
                    timestamp_ns=action_timestamp,
                    image_override=policy_image,
                )
                dynamics = oracle.predict(ForwardDynamicsRequest(current, action))
                finished_ns = time.monotonic_ns()
                diagnostics = CrossPaintStepDiagnostics(
                    visual_input=visual_input,
                    perception_latency_ns=max(0, finished_ns - started_ns),
                    composition_latency_ns=max(
                        0, composition_finished_ns - composition_started_ns
                    ),
                    target_robot_pixels=target_pixels,
                    source_robot_pixels=source_pixels,
                    source_visible_pixels=visible_pixels,
                    alignment_position_error_m=position_error,
                    alignment_orientation_error_rad=orientation_error,
                    alignment_iterations=align_iterations,
                    protected_object_pixels=protected_object_pixels,
                    object_pixel_loss=object_pixel_loss,
                    target_gripper_aperture_m=(
                        target_gripper_state.aperture_m
                    ),
                    target_gripper_open_fraction=(
                        target_gripper_state.open_fraction
                    ),
                    source_gripper_aperture_m=(
                        source_gripper_state.aperture_m
                    ),
                    source_gripper_open_fraction=(
                        source_gripper_state.open_fraction
                    ),
                )
                connection.send(
                    (
                        "ok",
                        CrossPaintSourceStep(
                            current_pose=current,
                            action=action,
                            dynamics=dynamics,
                            source_task_succeeded=runtime.task_succeeded(),
                            policy_rgb=np.ascontiguousarray(policy_rgb),
                            source_rgb=np.ascontiguousarray(source_frame.rgb),
                            source_robot_mask=np.ascontiguousarray(
                                source_frame.robot_mask
                            ),
                            background_rgb=np.ascontiguousarray(background_rgb),
                            target_inpaint_mask=np.ascontiguousarray(
                                target_inpaint_mask
                            ),
                            diagnostics=diagnostics,
                        ),
                    )
                )
            except Exception as error:
                connection.send(
                    (
                        "error",
                        type(error).__name__,
                        str(error),
                        traceback.format_exc(),
                    )
                )
    except Exception as error:
        try:
            connection.send(
                ("error", type(error).__name__, str(error), traceback.format_exc())
            )
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if runtime is not None:
            try:
                runtime.core.close()
            except Exception:
                pass
        connection.close()


class CrossPaintSourceWorker:
    """Spawn-safe proxy that owns the Panda policy and its EGL context."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        mask_render_scale: int = 1,
        startup_timeout_sec: float = 120.0,
        step_timeout_sec: float = 10.0,
    ) -> None:
        source = Path(checkpoint_path).expanduser().resolve()
        _, contract = load_checkpoint_contract(source)
        contract.require_runtime_supported()
        if contract.observation_mode is ObservationMode.LOW_DIM:
            raise ImageRuntimeError("cross-paint worker requires an image checkpoint")
        self.checkpoint_path = source
        self.checkpoint_contract: CheckpointContract = contract
        self.task_profile = contract.task_profile
        self.device = str(device)
        if mask_render_scale not in (1, 2):
            raise ImageRuntimeError("mask render scale must be 1 or 2")
        self.mask_render_scale = mask_render_scale
        self.startup_timeout_sec = float(startup_timeout_sec)
        self.step_timeout_sec = float(step_timeout_sec)
        if self.startup_timeout_sec <= 0.0 or self.step_timeout_sec <= 0.0:
            raise ImageRuntimeError("worker timeouts must be positive")
        self._context = multiprocessing.get_context("spawn")
        self._connection = None
        self._process = None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def start(self) -> None:
        if self._process is not None:
            raise ImageRuntimeError("image source worker already started")
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main,
            args=(
                child,
                str(self.checkpoint_path),
                self.device,
                self.mask_render_scale,
            ),
            daemon=True,
        )
        process.start()
        child.close()
        self._connection = parent
        self._process = process
        status, payload = self._receive(self.startup_timeout_sec)
        if status != "ready" or payload != self.checkpoint_contract:
            self.close()
            raise ImageRuntimeError("image source worker returned an invalid contract")

    def _receive(self, timeout_sec: float):
        if self._connection is None or self._process is None:
            raise ImageRuntimeError("image source worker is not started")
        if not self._connection.poll(timeout_sec):
            self.close()
            raise ImageRuntimeError("image source worker timed out")
        try:
            message = self._connection.recv()
        except EOFError as error:
            self.close()
            raise ImageRuntimeError("image source worker exited unexpectedly") from error
        if message[0] == "error":
            _, error_type, detail, worker_traceback = message
            self.close()
            raise ImageRuntimeError(
                f"image source worker failed with {error_type}: {detail}\n{worker_traceback}"
            )
        return message

    def reset(self, seed: int) -> CartesianPose:
        if type(seed) is not int or seed < 0:
            raise ImageRuntimeError("image source seed must be non-negative")
        self._connection.send(("reset", seed))
        status, pose = self._receive(self.step_timeout_sec)
        if status != "ok" or not isinstance(pose, CartesianPose):
            raise ImageRuntimeError("image source reset returned an invalid pose")
        return pose

    def step(
        self,
        *,
        target_frame: RobotCameraFrame,
        target_pose: CartesianPose,
        object_joint_states: Mapping[str, ManipulatedObjectState],
        target_gripper_state: GripperApertureState,
        visual_input: str = "cross_paint",
    ) -> CrossPaintSourceStep:
        self._connection.send(
            (
                "step",
                target_frame,
                target_pose,
                dict(object_joint_states),
                target_gripper_state,
                visual_input,
            )
        )
        status, result = self._receive(self.step_timeout_sec)
        if status != "ok" or not isinstance(result, CrossPaintSourceStep):
            raise ImageRuntimeError("image source step returned an invalid result")
        return result

    def close(self) -> None:
        connection = self._connection
        process = self._process
        self._connection = None
        self._process = None
        if connection is not None:
            try:
                if process is not None and process.is_alive():
                    connection.send(("close",))
                    if connection.poll(2.0):
                        connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                connection.close()
        if process is not None:
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)

    def __enter__(self) -> "CrossPaintSourceWorker":
        self.start()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()
