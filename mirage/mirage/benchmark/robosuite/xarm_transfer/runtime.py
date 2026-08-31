"""Robosuite runtime adapters for the Franka Panda source and xArm target.

This module only talks to in-process Robosuite environments. It contains no
ROS publisher, service client, action client, robot IP, or hardware SDK path.
"""

from __future__ import annotations

import copy
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import CartesianPose, SourceAction
from .dynamics import PandaOnlineOracle
from .geometry import quaternion_angular_error_rad
from .profiles import (
    CheckpointContract,
    ObservationMode,
    RuntimeContractError,
    TaskProfile,
    resolve_checkpoint_contract,
    task_profile_for_name,
)


SOURCE_ROBOSUITE_ROBOT_ID = "Panda"
SOURCE_FRAME_ID = "panda_sim_world"
TARGET_ROBOSUITE_ROBOT_ID = "XArm7"
TARGET_FRAME_ID = "xarm7_sim_world"
DEFAULT_XARM_WORKSPACE_MIN_M = (-0.65, -0.60, 0.40)
DEFAULT_XARM_WORKSPACE_MAX_M = (0.35, 0.60, 1.35)
# PandaGripper advances 0.01 on every 2 ms physics tick. At the checkpoint's
# 20 Hz policy rate that is 25 ticks, or 0.25 normalized actuator units per
# policy step. The xArm adapter applies this calibrated increment explicitly
# once rather than letting XArm7Gripper.format_action accumulate per tick.
XARM_GRIPPER_POLICY_INCREMENT = 0.25
XARM_GRIPPER_DRIVER_RANGE_RAD = (0.0, 0.85)


@dataclass(frozen=True)
class RobotCameraFrame:
    """One camera-aligned robot layer in OpenCV image convention."""

    rgb: np.ndarray
    depth_m: np.ndarray
    robot_mask: np.ndarray
    gripper_mask: np.ndarray
    object_mask: np.ndarray
    camera_name: str
    intrinsic_matrix: np.ndarray
    extrinsic_matrix: np.ndarray


@dataclass(frozen=True)
class ManipulatedObjectState:
    """MuJoCo joint position and velocity for one manipulated object joint."""

    qpos: np.ndarray
    qvel: np.ndarray

    def __post_init__(self) -> None:
        qpos = np.asarray(self.qpos, dtype=np.float64).reshape(-1)
        qvel = np.asarray(self.qvel, dtype=np.float64).reshape(-1)
        if (
            qpos.ndim != 1
            or qpos.size == 0
            or qvel.ndim != 1
            or qvel.size == 0
            or not np.all(np.isfinite(qpos))
            or not np.all(np.isfinite(qvel))
        ):
            raise RuntimeContractError(
                "manipulated object qpos/qvel must be non-empty finite vectors"
            )
        object.__setattr__(self, "qpos", np.ascontiguousarray(qpos.copy()))
        object.__setattr__(self, "qvel", np.ascontiguousarray(qvel.copy()))


@dataclass(frozen=True)
class GripperApertureState:
    """Physical pad aperture plus an embodiment-independent open fraction."""

    aperture_m: float
    open_fraction: float

    def __post_init__(self) -> None:
        aperture = float(self.aperture_m)
        fraction = float(self.open_fraction)
        if (
            not np.isfinite(aperture)
            or aperture < 0.0
            or not np.isfinite(fraction)
            or fraction < 0.0
            or fraction > 1.0
        ):
            raise RuntimeContractError("gripper aperture state is invalid")
        object.__setattr__(self, "aperture_m", aperture)
        object.__setattr__(self, "open_fraction", fraction)


def _register_task_environment(profile: TaskProfile) -> None:
    """Import the package that registers a profile's external Robosuite task."""

    module_name = profile.environment_registration_module
    if module_name is None:
        return
    try:
        importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeContractError(
            f"task {profile.env_name!r} requires environment package {module_name!r}"
        ) from error


def _disable_policy_cameras(kwargs: dict[str, Any]) -> None:
    kwargs["has_offscreen_renderer"] = False
    kwargs["use_camera_obs"] = False
    kwargs["camera_depths"] = False
    for key in ("camera_segmentations", "render_gpu_device_id"):
        kwargs.pop(key, None)


def _enable_image_policy_camera(kwargs: dict[str, Any], camera_name: str) -> None:
    kwargs["has_offscreen_renderer"] = True
    kwargs["use_camera_obs"] = True
    kwargs["camera_depths"] = False
    kwargs["camera_names"] = [camera_name]
    kwargs["camera_heights"] = 84
    kwargs["camera_widths"] = 84
    for key in ("camera_segmentations", "render_gpu_device_id"):
        kwargs.pop(key, None)


def _enable_compositor_camera(kwargs: dict[str, Any], camera_name: str) -> None:
    kwargs["has_offscreen_renderer"] = True
    kwargs["use_camera_obs"] = False
    kwargs["camera_depths"] = False
    kwargs["camera_names"] = [camera_name]
    kwargs["camera_heights"] = 84
    kwargs["camera_widths"] = 84
    for key in ("camera_segmentations", "render_gpu_device_id"):
        kwargs.pop(key, None)


def build_source_environment_metadata(
    checkpoint_data: Mapping[str, Any],
    contract: CheckpointContract,
    *,
    render: bool,
) -> dict[str, Any]:
    """Build source metadata while preserving future image policy cameras."""

    metadata = copy.deepcopy(checkpoint_data["env_metadata"])
    kwargs = metadata["env_kwargs"]
    kwargs["robots"] = [SOURCE_ROBOSUITE_ROBOT_ID]
    kwargs["has_renderer"] = bool(render)
    if contract.observation_mode is ObservationMode.LOW_DIM:
        _disable_policy_cameras(kwargs)
    else:
        _enable_image_policy_camera(kwargs, contract.camera_name)
    return metadata


def build_target_environment_metadata(
    checkpoint_data: Mapping[str, Any],
    contract: CheckpointContract,
    *,
    render: bool,
) -> dict[str, Any]:
    """Build the target with a camera only when cross-painting needs one."""

    metadata = copy.deepcopy(checkpoint_data["env_metadata"])
    metadata["env_name"] = contract.task_name
    kwargs = metadata["env_kwargs"]
    kwargs["robots"] = [TARGET_ROBOSUITE_ROBOT_ID]
    kwargs["has_renderer"] = bool(render)
    if contract.observation_mode is ObservationMode.LOW_DIM:
        _disable_policy_cameras(kwargs)
    else:
        _enable_compositor_camera(kwargs, contract.camera_name)
    return metadata


def create_xarm_target_environment(
    checkpoint_path: str | Path,
    *,
    render: bool = False,
) -> Any:
    """Create the simulation-only xArm7 target from Panda task metadata.

    This deliberately constructs an in-process Robosuite environment directly;
    there is no ROS, UFactory SDK, IP address, service, or action connection.
    """

    checkpoint_data, contract = load_checkpoint_contract(checkpoint_path)
    contract.require_runtime_supported()
    metadata = build_target_environment_metadata(
        checkpoint_data, contract, render=render
    )
    _register_task_environment(contract.task_profile)
    import robosuite
    from .robosuite_xarm7 import register_xarm7

    register_xarm7()
    environment = robosuite.make(metadata["env_name"], **metadata["env_kwargs"])
    setattr(environment, "_mirage_checkpoint_contract", contract)
    return environment


def load_checkpoint_contract(
    checkpoint_path: str | Path,
) -> tuple[Mapping[str, Any], CheckpointContract]:
    """Load checkpoint metadata and resolve its static transfer contract."""

    from robomimic.utils import file_utils as FileUtils

    source = Path(checkpoint_path).expanduser()
    if not source.is_file():
        raise RuntimeContractError(f"checkpoint does not exist: {source}")
    checkpoint_data = FileUtils.load_dict_from_checkpoint(str(source))
    return checkpoint_data, resolve_checkpoint_contract(checkpoint_data)


def validate_pick_place_can_checkpoint_contract(
    env_metadata: Mapping[str, Any],
    shape_metadata: Mapping[str, Any],
) -> None:
    """Compatibility wrapper for the former PickPlaceCan low-dim validator."""

    checkpoint_data = {
        "env_metadata": env_metadata,
        "shape_metadata": shape_metadata,
        "config": {"experiment": {"rollout": {"horizon": 400}}},
    }
    contract = resolve_checkpoint_contract(checkpoint_data)
    if (
        contract.task_name != "PickPlaceCan"
        or contract.observation_mode is not ObservationMode.LOW_DIM
    ):
        raise RuntimeContractError(
            "source checkpoint must be PickPlaceCan low_dim"
        )


def _unwrap_robosuite_environment(environment: Any) -> Any:
    core = environment
    visited: set[int] = set()
    while hasattr(core, "env"):
        if id(core) in visited:
            raise RuntimeContractError("environment wrapper cycle detected")
        visited.add(id(core))
        core = core.env
    if not hasattr(core, "robots") or not hasattr(core, "sim"):
        raise RuntimeContractError("environment does not expose a Robosuite core")
    return core


def _single_robot(core: Any) -> Any:
    robots = getattr(core, "robots", None)
    if not isinstance(robots, (list, tuple)) or len(robots) != 1:
        raise RuntimeContractError("state transfer requires exactly one robot")
    return robots[0]


def _robot_model_name(robot: Any) -> str:
    robot_model = getattr(robot, "robot_model", None)
    if robot_model is None:
        raise RuntimeContractError("Robosuite robot has no robot_model")
    return type(robot_model).__name__


def _make_render_context_current(core: Any) -> None:
    context = getattr(core.sim, "_render_context_offscreen", None)
    if context is None:
        return
    gl_context = getattr(context, "gl_ctx", None)
    if gl_context is None:
        gl_context = getattr(context, "opengl_context", None)
    make_current = getattr(gl_context, "make_current", None)
    if callable(make_current):
        make_current()


def _robot_geom_ids(core: Any, robot: Any) -> frozenset[int]:
    """Resolve robot and gripper geoms without Robosuite's static robot list."""

    body_names = []
    for model in (getattr(robot, "robot_model", None), getattr(robot, "gripper", None)):
        names = getattr(model, "bodies", ())
        if isinstance(names, (list, tuple)):
            body_names.extend(names)
    try:
        body_ids = {
            int(core.sim.model.body_name2id(name)) for name in body_names
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeContractError("robot camera body IDs cannot be resolved") from error
    geom_body_ids = np.asarray(core.sim.model.geom_bodyid, dtype=np.int64)
    geom_ids = frozenset(
        int(index)
        for index in np.flatnonzero(np.isin(geom_body_ids, tuple(body_ids)))
    )
    if not geom_ids:
        raise RuntimeContractError("robot camera geometry is empty")
    return geom_ids


def _body_geom_ids(core: Any, body_names: Sequence[str], label: str) -> frozenset[int]:
    try:
        body_ids = {
            int(core.sim.model.body_name2id(name)) for name in body_names
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeContractError(f"{label} body IDs cannot be resolved") from error
    geom_body_ids = np.asarray(core.sim.model.geom_bodyid, dtype=np.int64)
    return frozenset(
        int(index)
        for index in np.flatnonzero(np.isin(geom_body_ids, tuple(body_ids)))
    )


def _gripper_geom_ids(core: Any, robot: Any) -> frozenset[int]:
    gripper = getattr(robot, "gripper", None)
    bodies = getattr(gripper, "bodies", ())
    if not isinstance(bodies, (list, tuple)) or not bodies:
        raise RuntimeContractError("robot gripper exposes no bodies")
    geom_ids = _body_geom_ids(core, bodies, "gripper camera")
    if not geom_ids:
        raise RuntimeContractError("gripper camera geometry is empty")
    return geom_ids


def _object_geom_ids(core: Any, task_objects: Sequence[Any]) -> frozenset[int]:
    body_names = [
        name for model in task_objects for name in getattr(model, "bodies", ())
    ]
    if not body_names:
        raise RuntimeContractError("task object exposes no segmentation bodies")
    geom_ids = _body_geom_ids(core, body_names, "task object segmentation")
    if not geom_ids:
        raise RuntimeContractError("task object segmentation geometry is empty")
    return geom_ids


def _downsample_mask(mask: np.ndarray, scale: int, size: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if scale == 1:
        if value.shape != (size, size):
            raise RuntimeContractError("camera segmentation mask shape is invalid")
        return np.ascontiguousarray(value)
    expected = size * scale
    if value.shape != (expected, expected):
        raise RuntimeContractError("supersampled camera mask shape is invalid")
    # Conservative max pooling keeps thin robot edges and every protected
    # object subpixel when returning to the 84 x 84 policy grid.
    return np.ascontiguousarray(
        value.reshape(size, scale, size, scale).any(axis=(1, 3))
    )


def _gripper_pad_aperture_m(core: Any, gripper: Any) -> float:
    important = getattr(gripper, "important_geoms", None)
    if not isinstance(important, Mapping):
        raise RuntimeContractError("gripper exposes no important geom mapping")
    left_names = important.get("left_fingerpad") or important.get("left_finger")
    right_names = important.get("right_fingerpad") or important.get("right_finger")
    if not left_names or not right_names:
        raise RuntimeContractError("gripper finger-pad geoms are missing")
    try:
        left_ids = [core.sim.model.geom_name2id(name) for name in left_names]
        right_ids = [core.sim.model.geom_name2id(name) for name in right_names]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeContractError("gripper finger-pad geom IDs cannot be resolved") from error
    left = np.mean(np.asarray(core.sim.data.geom_xpos)[left_ids], axis=0)
    right = np.mean(np.asarray(core.sim.data.geom_xpos)[right_ids], axis=0)
    aperture = float(np.linalg.norm(left - right))
    if not np.isfinite(aperture) or aperture < 0.0:
        raise RuntimeContractError("gripper physical aperture is invalid")
    return aperture


def _metric_depth(core: Any, normalized_depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(normalized_depth, dtype=np.float64)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    if not np.all(np.isfinite(depth)) or np.any(depth < 0.0) or np.any(depth > 1.0):
        raise RuntimeContractError("camera depth is not normalized to [0, 1]")
    extent = float(core.sim.model.stat.extent)
    far = float(core.sim.model.vis.map.zfar) * extent
    near = float(core.sim.model.vis.map.znear) * extent
    if not (0.0 < near < far):
        raise RuntimeContractError("camera near/far planes are invalid")
    return np.ascontiguousarray(near / (1.0 - depth * (1.0 - near / far)))


def _camera_geometry(core: Any, camera_name: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    from robosuite.utils import camera_utils

    intrinsic = camera_utils.get_camera_intrinsic_matrix(
        core.sim, camera_name=camera_name, camera_height=size, camera_width=size
    )
    extrinsic = camera_utils.get_camera_extrinsic_matrix(
        core.sim, camera_name=camera_name
    )
    return (
        np.asarray(intrinsic, dtype=np.float64),
        np.asarray(extrinsic, dtype=np.float64),
    )


def _render_robot_camera_frame(
    core: Any,
    robot: Any,
    task_objects: Sequence[Any],
    *,
    camera_name: str,
    size: int = 84,
    mask_render_scale: int = 1,
) -> RobotCameraFrame:
    """Render RGB/depth and derive a robot mask from element geom IDs."""

    if not isinstance(camera_name, str) or not camera_name:
        raise RuntimeContractError("camera name must not be empty")
    if mask_render_scale not in (1, 2):
        raise RuntimeContractError("mask render scale must be 1 or 2")
    _make_render_context_current(core)
    rendered = core.sim.render(
        camera_name=camera_name, width=size, height=size, depth=True
    )
    if not isinstance(rendered, tuple) or len(rendered) != 2:
        raise RuntimeContractError("camera RGB/depth render returned an invalid value")
    rgb, normalized_depth = rendered
    segmentation = core.sim.render(
        camera_name=camera_name,
        width=size * mask_render_scale,
        height=size * mask_render_scale,
        depth=False,
        segmentation=True,
    )
    rgb = np.ascontiguousarray(np.asarray(rgb)[::-1])
    normalized_depth = np.ascontiguousarray(np.asarray(normalized_depth)[::-1])
    segmentation = np.asarray(segmentation)
    if rgb.shape != (size, size, 3) or segmentation.shape != (
        size * mask_render_scale,
        size * mask_render_scale,
        2,
    ):
        raise RuntimeContractError("camera render shape does not match checkpoint")
    geom_ids = np.ascontiguousarray(segmentation[::-1, :, 1])
    robot_mask = _downsample_mask(
        np.isin(geom_ids, tuple(_robot_geom_ids(core, robot))),
        mask_render_scale,
        size,
    )
    gripper_mask = _downsample_mask(
        np.isin(geom_ids, tuple(_gripper_geom_ids(core, robot))),
        mask_render_scale,
        size,
    )
    object_mask = _downsample_mask(
        np.isin(geom_ids, tuple(_object_geom_ids(core, task_objects))),
        mask_render_scale,
        size,
    )
    if not np.any(robot_mask):
        raise RuntimeContractError("camera robot mask is empty")
    if np.any(gripper_mask & ~robot_mask):
        # Independent conservative pooling can only produce a subset because
        # both masks share the same pooling grid. Keep the invariant explicit.
        raise RuntimeContractError("camera gripper mask is outside robot mask")
    intrinsic, extrinsic = _camera_geometry(core, camera_name, size)
    return RobotCameraFrame(
        rgb=rgb,
        depth_m=_metric_depth(core, normalized_depth),
        robot_mask=np.ascontiguousarray(robot_mask),
        gripper_mask=np.ascontiguousarray(gripper_mask),
        object_mask=np.ascontiguousarray(object_mask),
        camera_name=camera_name,
        intrinsic_matrix=intrinsic,
        extrinsic_matrix=extrinsic,
    )


class FrankaPandaSourceRuntime:
    """Deterministic policy/pose adapter for one Franka Panda environment."""

    def __init__(
        self,
        *,
        environment: Any,
        policy: Any,
        checkpoint_data: Mapping[str, Any],
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        contract = resolve_checkpoint_contract(checkpoint_data)
        contract.require_runtime_supported()
        self.environment = environment
        self.policy = policy
        self.checkpoint_data = checkpoint_data
        self.checkpoint_contract = contract
        self.task_profile = contract.task_profile
        self.clock_ns = clock_ns
        self.core = _unwrap_robosuite_environment(environment)
        self.robot = None
        self._eef_site_name = ""
        self._task_objects: tuple[Any, ...] = ()
        self._last_gripper_command = -1.0
        self._refresh_robot_reference()
        self.observation: Mapping[str, np.ndarray] | None = None
        self.policy_step_count = 0
        self.last_step_output: tuple[object, object, object, object] | None = None

    def _refresh_robot_reference(self) -> None:
        """Refresh references after Robosuite performs a hard model reset."""

        self.core = _unwrap_robosuite_environment(self.environment)
        self.robot = _single_robot(self.core)
        if _robot_model_name(self.robot) != SOURCE_ROBOSUITE_ROBOT_ID:
            raise RuntimeContractError(
                "source simulator robot model is not the Franka Panda"
            )
        controller = getattr(self.robot, "controller", None)
        if controller is None or not hasattr(controller, "eef_name"):
            raise RuntimeContractError("source robot has no Cartesian controller")
        self._eef_site_name = controller.eef_name
        self._task_objects = self._resolve_task_objects()

    def _resolve_task_objects(self) -> tuple[Any, ...]:
        objects = getattr(getattr(self.core, "model", None), "mujoco_objects", None)
        if not isinstance(objects, (list, tuple)):
            raise RuntimeContractError("source task exposes no mujoco_objects sequence")
        selected = []
        for name in self.task_profile.manipulated_object_names:
            matches = [item for item in objects if getattr(item, "name", None) == name]
            if len(matches) != 1:
                raise RuntimeContractError(
                    f"source task must contain exactly one object named {name!r}; "
                    f"found {len(matches)}"
                )
            selected.append(matches[0])
        return tuple(selected)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        render: bool = False,
    ) -> "FrankaPandaSourceRuntime":
        """Load the policy and its matching Franka Panda environment.

        The legacy ``FileUtils.env_from_checkpoint`` path forcibly enables
        extra camera observations in this repository. This runtime builds from
        deep-copied metadata and enables exactly the checkpoint camera for
        image policies while keeping low-dimensional policies camera-free.
        """

        from robomimic.utils import file_utils as FileUtils

        source = Path(checkpoint_path).expanduser()
        if not source.is_file():
            raise RuntimeContractError(f"checkpoint does not exist: {source}")
        checkpoint_data = FileUtils.load_dict_from_checkpoint(str(source))
        contract = resolve_checkpoint_contract(checkpoint_data)
        contract.require_runtime_supported()
        _register_task_environment(contract.task_profile)
        from robomimic.utils import env_utils as EnvUtils

        policy, policy_checkpoint_data = FileUtils.policy_from_checkpoint(
            ckpt_path=str(source), device=device, verbose=False
        )
        policy_contract = resolve_checkpoint_contract(policy_checkpoint_data)
        if policy_contract != contract:
            raise RuntimeContractError(
                "checkpoint metadata changed while loading the source policy"
            )
        env_metadata = build_source_environment_metadata(
            checkpoint_data, contract, render=render
        )
        uses_images = contract.observation_mode is not ObservationMode.LOW_DIM
        environment = EnvUtils.create_env_from_metadata(
            env_meta=env_metadata,
            render=render,
            render_offscreen=uses_images,
            use_image_obs=uses_images,
            use_depth_obs=False,
        )
        config, _ = FileUtils.config_from_checkpoint(
            algo_name=checkpoint_data["algo_name"],
            ckpt_dict=checkpoint_data,
            verbose=False,
        )
        environment = EnvUtils.wrap_env_from_config(environment, config=config)
        return cls(
            environment=environment,
            policy=policy,
            checkpoint_data=checkpoint_data,
        )

    def reset(self, seed: int) -> Mapping[str, np.ndarray]:
        if type(seed) is not int or seed < 0:
            raise RuntimeContractError("seed must be a non-negative integer")
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)
        observation = self.environment.reset()
        # Match the legacy benchmark's deterministic Robosuite reset behavior.
        state = self.environment.get_state()
        observation = self.environment.reset_to(state)
        self._refresh_robot_reference()
        if not isinstance(observation, Mapping):
            raise RuntimeContractError("source reset did not return an observation")
        self.observation = copy.deepcopy(observation)
        self.policy.start_episode()
        self.policy_step_count = 0
        self.last_step_output = None
        self._last_gripper_command = -1.0
        return copy.deepcopy(self.observation)

    def read_pose(self) -> CartesianPose:
        site_id = self.core.sim.model.site_name2id(self._eef_site_name)
        position = np.asarray(self.core.sim.data.site_xpos[site_id], dtype=np.float64)
        rotation_matrix = np.asarray(
            self.core.sim.data.site_xmat[site_id], dtype=np.float64
        ).reshape(3, 3)
        from robosuite.utils import transform_utils as transform

        quaternion = np.asarray(transform.mat2quat(rotation_matrix), dtype=np.float64)
        return CartesianPose(
            tuple(position), tuple(quaternion), SOURCE_FRAME_ID, self.clock_ns()
        )

    def read_gripper_state(self) -> tuple[float, ...]:
        if self.observation is None or self.robot is None:
            raise RuntimeContractError("source runtime must be reset first")
        indexes = getattr(self.robot, "_ref_gripper_joint_pos_indexes", None)
        if not isinstance(indexes, (list, tuple, np.ndarray)):
            raise RuntimeContractError("Franka Panda gripper qpos indexes are missing")
        value = np.asarray(self.core.sim.data.qpos[indexes], dtype=np.float64)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise RuntimeContractError("Franka Panda simulator gripper qpos is invalid")
        return tuple(float(item) for item in value)

    def read_gripper_aperture_state(self) -> GripperApertureState:
        qpos = np.asarray(self.read_gripper_state(), dtype=np.float64)
        ranges = []
        for joint_name in self.robot.gripper.joints:
            joint_id = self.core.sim.model.joint_name2id(joint_name)
            ranges.append(self.core.sim.model.jnt_range[joint_id])
        joint_ranges = np.asarray(ranges, dtype=np.float64)
        if joint_ranges.shape != (2, 2) or not np.all(np.isfinite(joint_ranges)):
            raise RuntimeContractError("Panda gripper joint ranges are invalid")
        closed = joint_ranges[
            np.arange(2), np.argmin(np.abs(joint_ranges), axis=1)
        ]
        opened = joint_ranges[
            np.arange(2), np.argmax(np.abs(joint_ranges), axis=1)
        ]
        denominator = opened - closed
        fractions = np.divide(
            qpos - closed,
            denominator,
            out=np.zeros_like(qpos),
            where=np.abs(denominator) > 1e-12,
        )
        return GripperApertureState(
            _gripper_pad_aperture_m(self.core, self.robot.gripper),
            float(np.clip(np.mean(fractions), 0.0, 1.0)),
        )

    def synchronize_gripper_open_fraction(self, open_fraction: float) -> None:
        """Set only the virtual Panda fingers from the measured xArm aperture."""

        fraction = float(open_fraction)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise RuntimeContractError("target gripper open fraction must be in [0, 1]")
        qpos_indexes = np.asarray(
            self.robot._ref_gripper_joint_pos_indexes, dtype=np.int64
        )
        qvel_indexes = np.asarray(
            self.robot._ref_gripper_joint_vel_indexes, dtype=np.int64
        )
        ranges = np.asarray(
            [
                self.core.sim.model.jnt_range[
                    self.core.sim.model.joint_name2id(joint_name)
                ]
                for joint_name in self.robot.gripper.joints
            ],
            dtype=np.float64,
        )
        if (
            qpos_indexes.shape != (2,)
            or qvel_indexes.shape != (2,)
            or ranges.shape != (2, 2)
            or not np.all(np.isfinite(ranges))
        ):
            raise RuntimeContractError("Panda gripper calibration references are invalid")
        closed = ranges[np.arange(2), np.argmin(np.abs(ranges), axis=1)]
        opened = ranges[np.arange(2), np.argmax(np.abs(ranges), axis=1)]
        self.core.sim.data.qpos[qpos_indexes] = closed + fraction * (opened - closed)
        self.core.sim.data.qvel[qvel_indexes] = 0.0
        normalized_target = np.asarray(
            [2.0 * fraction - 1.0, 1.0 - 2.0 * fraction],
            dtype=np.float64,
        )
        self.robot.gripper.current_action = normalized_target.copy()
        actuator_indexes = np.asarray(
            self.robot._ref_joint_gripper_actuator_indexes, dtype=np.int64
        )
        control_range = np.asarray(
            self.core.sim.model.actuator_ctrlrange[actuator_indexes],
            dtype=np.float64,
        )
        if actuator_indexes.shape != (2,) or control_range.shape != (2, 2):
            raise RuntimeContractError("Panda gripper actuator calibration is invalid")
        bias = 0.5 * (control_range[:, 1] + control_range[:, 0])
        weight = 0.5 * (control_range[:, 1] - control_range[:, 0])
        self.core.sim.data.ctrl[actuator_indexes] = bias + weight * normalized_target
        self.core.sim.forward()

    def policy_action(
        self,
        *,
        timestamp_ns: int | None = None,
        image_override: np.ndarray | None = None,
    ) -> SourceAction:
        if self.observation is None:
            raise RuntimeContractError("source runtime must be reset first")
        observation = copy.deepcopy(dict(self.observation))
        image_key = self.checkpoint_contract.image_observation_key
        if image_override is not None:
            if image_key is None:
                raise RuntimeContractError(
                    "low-dimensional policy does not accept an image override"
                )
            image = np.asarray(image_override, dtype=np.float32)
            if image.shape != self.checkpoint_contract.shapes[image_key]:
                raise RuntimeContractError(
                    "policy image override shape does not match checkpoint"
                )
            if not np.all(np.isfinite(image)) or np.any(image < 0.0) or np.any(image > 1.0):
                raise RuntimeContractError(
                    "policy image override must be finite and normalized to [0, 1]"
                )
            observation[image_key] = np.ascontiguousarray(image)
        for key, value in observation.items():
            if np.isscalar(value):
                observation[key] = np.asarray([value])
        action = np.asarray(self.policy(ob=observation), dtype=np.float64)
        return SourceAction(
            tuple(np.clip(action, -1.0, 1.0)),
            SOURCE_FRAME_ID,
            self.clock_ns() if timestamp_ns is None else timestamp_ns,
            tuple(action),
        )

    def render_robot_camera_frame(
        self, *, mask_render_scale: int = 1
    ) -> RobotCameraFrame:
        camera_name = self.checkpoint_contract.camera_name
        if camera_name is None:
            raise RuntimeContractError(
                "low-dimensional source has no cross-paint camera"
            )
        return _render_robot_camera_frame(
            self.core,
            self.robot,
            self._task_objects,
            camera_name=camera_name,
            mask_render_scale=mask_render_scale,
        )

    def step_policy_once(self, normalized_action: np.ndarray) -> object:
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise RuntimeContractError("source step action must be finite with shape (7,)")
        self.robot.controller.use_delta = True
        step_output = self.environment.step(action.copy())
        if not isinstance(step_output, tuple) or len(step_output) != 4:
            raise RuntimeContractError("source environment step returned an invalid tuple")
        next_observation, reward, done, info = step_output
        if not isinstance(next_observation, Mapping):
            raise RuntimeContractError("source environment step returned no observation")
        self.observation = copy.deepcopy(next_observation)
        self.policy_step_count += 1
        self._last_gripper_command = float(np.clip(action[-1], -1.0, 1.0))
        self.last_step_output = (next_observation, reward, done, info)
        return step_output

    def set_manipulated_object_joint_states(
        self, states: Mapping[str, ManipulatedObjectState]
    ) -> None:
        expected = {
            joint
            for model in self._task_objects
            for joint in getattr(model, "joints", ())
        }
        if set(states) != expected:
            raise RuntimeContractError(
                "source object joint-state keys do not match task profile"
            )
        for joint, value in states.items():
            if not isinstance(value, ManipulatedObjectState):
                raise RuntimeContractError(
                    "source object state must contain qpos and qvel"
                )
            try:
                self.core.sim.data.set_joint_qpos(joint, value.qpos.copy())
                self.core.sim.data.set_joint_qvel(joint, value.qvel.copy())
            except (TypeError, ValueError) as error:
                raise RuntimeContractError(
                    f"source object joint state shape is invalid for {joint!r}"
                ) from error
        self.core.sim.forward()

    def _refresh_observation(self) -> None:
        getter = getattr(self.environment, "get_observation", None)
        if callable(getter):
            observation = getter()
        else:
            observation = self.core._get_observations(force_update=True)
        if not isinstance(observation, Mapping):
            raise RuntimeContractError("source observation refresh failed")
        self.observation = copy.deepcopy(observation)

    def align_to_pose(
        self,
        desired: CartesianPose,
        *,
        position_tolerance_m: float = 0.003,
        orientation_tolerance_rad: float = float(np.deg2rad(2.0)),
        max_iterations: int = 100,
    ) -> tuple[float, float, int]:
        """Align the source embodiment to the transformed target state."""

        if desired.frame_id != SOURCE_FRAME_ID:
            raise RuntimeContractError("source alignment pose has the wrong frame")
        from robosuite.utils import transform_utils as transform

        self.robot.controller.use_delta = False
        iterations = 0
        try:
            while True:
                current = self.read_pose()
                position_error = float(
                    np.linalg.norm(
                        np.asarray(current.position_m)
                        - np.asarray(desired.position_m)
                    )
                )
                orientation_error = quaternion_angular_error_rad(
                    current.quaternion_xyzw, desired.quaternion_xyzw
                )
                if (
                    position_error <= position_tolerance_m
                    and orientation_error <= orientation_tolerance_rad
                ):
                    break
                if iterations >= max_iterations:
                    raise RuntimeContractError(
                        "Panda could not align to the transformed xArm pose"
                    )
                action = np.zeros(7, dtype=np.float64)
                action[:3] = desired.position_m
                action[3:6] = transform.quat2axisangle(
                    np.asarray(desired.quaternion_xyzw, dtype=np.float64)
                )
                # Match Mirage's original absolute-OSC source alignment: the
                # arm tracks the target pose while the gripper command is held
                # at zero. Reusing the previous sign would integrate another
                # open/close increment on every alignment physics step.
                action[-1] = 0.0
                output = self.environment.step(action)
                if not isinstance(output, tuple) or len(output) != 4:
                    raise RuntimeContractError("source alignment step failed")
                self.observation = copy.deepcopy(output[0])
                iterations += 1
        finally:
            self.robot.controller.use_delta = True
        self._refresh_observation()
        return position_error, orientation_error, iterations

    def make_oracle(self) -> PandaOnlineOracle:
        return PandaOnlineOracle(
            step_policy_once=self.step_policy_once,
            read_pose=self.read_pose,
            read_gripper_state=self.read_gripper_state,
        )

    def task_succeeded(self) -> bool:
        checker = getattr(self.core, "_check_success", None)
        if not callable(checker):
            raise RuntimeContractError("source environment has no success check")
        result = checker()
        if not isinstance(result, (bool, np.bool_)):
            raise RuntimeContractError("source success check did not return a bool")
        return bool(result)


class RobosuiteXArmAbsoluteAdapter:
    """Expose an in-process xArm7 Robosuite environment to blocking control."""

    def __init__(
        self,
        environment: Any,
        *,
        task_profile: TaskProfile | None = None,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self.environment = environment
        self.clock_ns = clock_ns
        self.core = _unwrap_robosuite_environment(environment)
        embedded_contract = getattr(
            environment, "_mirage_checkpoint_contract", None
        )
        if embedded_contract is not None and not isinstance(
            embedded_contract, CheckpointContract
        ):
            raise RuntimeContractError(
                "target environment has an invalid checkpoint contract"
            )
        inferred_profile = (
            embedded_contract.task_profile
            if embedded_contract is not None
            else task_profile_for_name(type(self.core).__name__)
        )
        if task_profile is not None and task_profile != inferred_profile:
            raise RuntimeContractError(
                "target adapter task profile does not match its environment"
            )
        self.checkpoint_contract = embedded_contract
        self.task_profile = task_profile or inferred_profile
        self.robot = None
        self._eef_site_name = ""
        self._gripper_body_ids: frozenset[int] = frozenset()
        self._gripper_driver_qpos_indexes: tuple[int, ...] = ()
        self._gripper_driver_ranges: tuple[tuple[float, float], ...] = ()
        self._task_object_geom_ids: frozenset[int] = frozenset()
        self._task_objects: tuple[Any, ...] = ()
        self._refresh_robot_reference()
        self.observation: Mapping[str, np.ndarray] | None = None
        self.step_count = 0

    def _refresh_robot_reference(self) -> None:
        """Refresh references after Robosuite performs a hard model reset."""

        self.core = _unwrap_robosuite_environment(self.environment)
        self.robot = _single_robot(self.core)
        if _robot_model_name(self.robot) != TARGET_ROBOSUITE_ROBOT_ID:
            raise RuntimeContractError("target simulator robot model is not XArm7")
        controller = getattr(self.robot, "controller", None)
        if controller is None or not hasattr(controller, "eef_name"):
            raise RuntimeContractError("target robot has no Cartesian controller")
        self._eef_site_name = controller.eef_name
        self._refresh_contact_references()
        self._refresh_gripper_calibration_references()

    def _refresh_contact_references(self) -> None:
        """Resolve all profile-selected manipulated objects in the MJCF model."""

        task_model = getattr(self.core, "model", None)
        objects = getattr(task_model, "mujoco_objects", None)
        if not isinstance(objects, (list, tuple)):
            raise RuntimeContractError(
                "target task model exposes no mujoco_objects sequence"
            )
        selected_objects = []
        for expected_name in self.task_profile.manipulated_object_names:
            matches = [
                model
                for model in objects
                if getattr(model, "name", None) == expected_name
            ]
            if len(matches) != 1:
                raise RuntimeContractError(
                    f"target task must contain exactly one object named {expected_name!r}; "
                    f"found {len(matches)}"
                )
            contact_geoms = getattr(matches[0], "contact_geoms", None)
            if not isinstance(contact_geoms, (list, tuple)) or not contact_geoms:
                raise RuntimeContractError(
                    f"target task object {expected_name!r} has no contact geoms"
                )
            selected_objects.append(matches[0])
        try:
            object_geom_ids = frozenset(
                int(self.core.sim.model.geom_name2id(name))
                for model in selected_objects
                for name in model.contact_geoms
            )
            gripper_body_ids = frozenset(
                int(self.core.sim.model.body_name2id(name))
                for name in self.robot.gripper.bodies
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeContractError(
                "target task contact geometry cannot be resolved"
            ) from error
        if not object_geom_ids or not gripper_body_ids:
            raise RuntimeContractError("target task contact geometry is empty")
        self._task_objects = tuple(selected_objects)
        self._task_object_geom_ids = object_geom_ids
        self._gripper_body_ids = gripper_body_ids

    def _refresh_gripper_calibration_references(self) -> None:
        driver_names = [
            name
            for name in self.robot.gripper.joints
            if name.endswith("left_driver_joint")
            or name.endswith("right_driver_joint")
        ]
        if len(driver_names) != 2:
            raise RuntimeContractError(
                "xArm gripper calibration requires two driver joints"
            )
        indexes = []
        ranges = []
        for name in driver_names:
            address = self.core.sim.model.get_joint_qpos_addr(name)
            joint_id = self.core.sim.model.joint_name2id(name)
            value = np.asarray(self.core.sim.model.jnt_range[joint_id], dtype=np.float64)
            if not isinstance(address, (int, np.integer)):
                raise RuntimeContractError("xArm driver joint calibration is invalid")
            if (
                value.shape != (2,)
                or not np.all(np.isfinite(value))
                or value[0] >= value[1]
            ):
                # Robosuite 1.4 drops the inherited default range while
                # merging this gripper asset, even though the approved local
                # MJCF declares 0..0.85 for every xArm gripper joint.
                value = np.asarray(XARM_GRIPPER_DRIVER_RANGE_RAD)
            indexes.append(int(address))
            ranges.append((float(value[0]), float(value[1])))
        self._gripper_driver_qpos_indexes = tuple(indexes)
        self._gripper_driver_ranges = tuple(ranges)

    def reset(self, seed: int | None = None) -> Mapping[str, np.ndarray]:
        if seed is not None:
            if type(seed) is not int or seed < 0:
                raise RuntimeContractError("target seed must be non-negative")
            np.random.seed(seed)
        observation = self.environment.reset()
        self._refresh_robot_reference()
        if not isinstance(observation, Mapping):
            raise RuntimeContractError("target reset did not return an observation")
        self.observation = copy.deepcopy(observation)
        self.step_count = 0
        self._synchronize_gripper_actuator_to_state()
        return copy.deepcopy(self.observation)

    def read_gripper_aperture_state(self) -> GripperApertureState:
        indexes = np.asarray(self._gripper_driver_qpos_indexes, dtype=np.int64)
        ranges = np.asarray(self._gripper_driver_ranges, dtype=np.float64)
        qpos = np.abs(np.asarray(self.core.sim.data.qpos[indexes], dtype=np.float64))
        # The official xArm asset uses mirrored driver axes. Absolute joint
        # displacement is therefore the stable closure coordinate for both.
        closure = np.divide(
            qpos - np.abs(ranges[:, 0]),
            np.abs(ranges[:, 1]) - np.abs(ranges[:, 0]),
        )
        if not np.all(np.isfinite(closure)):
            raise RuntimeContractError("xArm gripper closure calibration is invalid")
        return GripperApertureState(
            _gripper_pad_aperture_m(self.core, self.robot.gripper),
            float(1.0 - np.clip(np.mean(closure), 0.0, 1.0)),
        )

    def _write_gripper_actuator_target(self) -> None:
        indexes = np.asarray(
            self.robot._ref_joint_gripper_actuator_indexes, dtype=np.int64
        )
        current = np.asarray(self.robot.gripper.current_action, dtype=np.float64)
        if indexes.shape != current.shape or current.shape != (1,):
            raise RuntimeContractError("xArm gripper actuator references are invalid")
        control_range = np.asarray(
            self.core.sim.model.actuator_ctrlrange[indexes], dtype=np.float64
        )
        bias = 0.5 * (control_range[:, 1] + control_range[:, 0])
        weight = 0.5 * (control_range[:, 1] - control_range[:, 0])
        self.core.sim.data.ctrl[indexes] = bias + weight * current

    def _synchronize_gripper_actuator_to_state(self) -> None:
        state = self.read_gripper_aperture_state()
        # Robosuite/xArm semantics are -1=open and +1=closed.
        self.robot.gripper.current_action = np.asarray(
            [1.0 - 2.0 * state.open_fraction], dtype=np.float64
        )
        self._write_gripper_actuator_target()

    def apply_gripper_command_once(self, normalized_command: float) -> None:
        """Advance the calibrated xArm actuator target exactly once."""

        value = float(normalized_command)
        if not np.isfinite(value) or abs(value) > 1.0 + 1e-8:
            raise RuntimeContractError("xArm gripper command must be in [-1, 1]")
        current = np.asarray(self.robot.gripper.current_action, dtype=np.float64)
        if current.shape != (1,) or not np.all(np.isfinite(current)):
            raise RuntimeContractError("xArm gripper current action is invalid")
        self.robot.gripper.current_action = np.clip(
            current + XARM_GRIPPER_POLICY_INCREMENT * np.sign(value),
            -1.0,
            1.0,
        )
        self._write_gripper_actuator_target()

    def set_absolute_mode(self) -> None:
        self.robot.controller.use_delta = False

    def _configure_simulation_osc_gains(self) -> None:
        """Use higher simulated stiffness only while carrying a task object."""

        profile = self.task_profile.controller_profile
        kp = profile.free_osc_kp
        if (
            profile.contact_osc_kp is not None
            and self.has_gripper_task_object_contact()
        ):
            kp = profile.contact_osc_kp
        kp = np.asarray(kp, dtype=np.float64)
        self.robot.controller.kp = kp.copy()
        self.robot.controller.kd = 2.0 * np.sqrt(kp)

    def read_pose(self) -> CartesianPose:
        site_id = self.core.sim.model.site_name2id(self._eef_site_name)
        position = np.asarray(self.core.sim.data.site_xpos[site_id], dtype=np.float64)
        rotation_matrix = np.asarray(
            self.core.sim.data.site_xmat[site_id], dtype=np.float64
        ).reshape(3, 3)
        from robosuite.utils import transform_utils as transform

        quaternion = transform.mat2quat(rotation_matrix)
        return CartesianPose(
            tuple(position), tuple(quaternion), TARGET_FRAME_ID, self.clock_ns()
        )

    def read_joint_positions(self) -> np.ndarray:
        positions = np.asarray(self.robot._joint_positions, dtype=np.float64)
        if positions.shape != (7,) or not np.all(np.isfinite(positions)):
            raise RuntimeContractError("xArm7 joint state must be finite with shape (7,)")
        return positions.copy()

    def read_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Return MJCF joint ranges in the xArm7 model's declared order."""

        ranges = []
        for joint_name in self.robot.robot_model.joints:
            joint_id = self.core.sim.model.joint_name2id(joint_name)
            ranges.append(self.core.sim.model.jnt_range[joint_id])
        value = np.asarray(ranges, dtype=np.float64)
        if (
            value.shape != (7, 2)
            or not np.all(np.isfinite(value))
            or np.any(value[:, 0] >= value[:, 1])
        ):
            raise RuntimeContractError("xArm7 model joint limits are invalid")
        return value[:, 0].copy(), value[:, 1].copy()

    def render_robot_camera_frame(
        self, *, mask_render_scale: int = 1
    ) -> RobotCameraFrame:
        contract = self.checkpoint_contract
        camera_name = None if contract is None else contract.camera_name
        if camera_name is None:
            raise RuntimeContractError(
                "low-dimensional target has no cross-paint camera"
            )
        return _render_robot_camera_frame(
            self.core,
            self.robot,
            self._task_objects,
            camera_name=camera_name,
            mask_render_scale=mask_render_scale,
        )

    def read_manipulated_object_joint_states(
        self,
    ) -> dict[str, ManipulatedObjectState]:
        states: dict[str, ManipulatedObjectState] = {}
        for model in self._task_objects:
            joints = getattr(model, "joints", None)
            if not isinstance(joints, (list, tuple)) or not joints:
                raise RuntimeContractError(
                    f"target task object {getattr(model, 'name', '')!r} has no joints"
                )
            for joint in joints:
                qpos = np.asarray(
                    self.core.sim.data.get_joint_qpos(joint), dtype=np.float64
                )
                qvel = np.asarray(
                    self.core.sim.data.get_joint_qvel(joint), dtype=np.float64
                )
                states[joint] = ManipulatedObjectState(qpos, qvel)
        if not states:
            raise RuntimeContractError("target object joint-state snapshot is empty")
        return states

    def has_gripper_task_object_contact(self) -> bool:
        """Detect xArm gripper-body contact with any simulated task object.

        The official xArm gripper contains an unnamed collision mesh at its
        base, so checking only named finger-pad geoms misses a real contact
        that can prevent Cartesian convergence. Body IDs keep this check
        simulation-local and include that mesh without guessing geom names.
        """

        for index in range(self.core.sim.data.ncon):
            contact = self.core.sim.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self._task_object_geom_ids:
                other_geom = geom2
            elif geom2 in self._task_object_geom_ids:
                other_geom = geom1
            else:
                continue
            other_body = int(self.core.sim.model.geom_bodyid[other_geom])
            if other_body in self._gripper_body_ids:
                return True
        return False

    def step_absolute(self, action: np.ndarray) -> object:
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (7,) or not np.all(np.isfinite(value)):
            raise RuntimeContractError("xArm7 absolute action must be finite with shape (7,)")
        if self.robot.controller.use_delta is not False:
            raise RuntimeContractError("xArm7 controller is not in absolute mode")
        self._configure_simulation_osc_gains()
        output = self.environment.step(value.copy())
        if not isinstance(output, tuple) or len(output) != 4:
            raise RuntimeContractError("target environment step returned an invalid tuple")
        observation = output[0]
        if not isinstance(observation, Mapping):
            raise RuntimeContractError("target environment step returned no observation")
        self.observation = copy.deepcopy(observation)
        self.step_count += 1
        return output

    def task_succeeded(self) -> bool:
        checker = getattr(self.core, "_check_success", None)
        if not callable(checker):
            raise RuntimeContractError("target environment has no success check")
        result = checker()
        if not isinstance(result, (bool, np.bool_)):
            raise RuntimeContractError("target success check did not return a bool")
        return bool(result)


def create_default_xarm_blocking_controller(
    adapter: RobosuiteXArmAbsoluteAdapter,
):
    """Build the documented fail-closed controller from MJCF joint limits."""

    from .control import (
        AbsoluteBlockingController,
        SafetyLimits,
        SimulationSafetyFilter,
    )

    joint_min, joint_max = adapter.read_joint_limits()
    limits = SafetyLimits(
        target_frame=TARGET_FRAME_ID,
        workspace_min_m=DEFAULT_XARM_WORKSPACE_MIN_M,
        workspace_max_m=DEFAULT_XARM_WORKSPACE_MAX_M,
        joint_min_rad=tuple(joint_min),
        joint_max_rad=tuple(joint_max),
        max_command_age_sec=0.15,
        max_future_skew_sec=0.02,
        max_prediction_position_error_m=0.03,
        max_prediction_orientation_error_rad=np.deg2rad(10.0),
        joint_limit_margin_rad=0.01,
    )
    profile = adapter.task_profile.controller_profile
    return AbsoluteBlockingController(
        adapter=adapter,
        safety_filter=SimulationSafetyFilter(limits),
        position_tolerance_m=profile.position_tolerance_m,
        orientation_tolerance_rad=profile.orientation_tolerance_rad,
        contact_position_tolerance_m=profile.contact_position_tolerance_m,
        contact_orientation_tolerance_rad=profile.contact_orientation_tolerance_rad,
        settling_iterations=profile.settling_iterations,
        outer_loop_position_gain=profile.outer_loop_position_gain,
        max_outer_loop_position_correction_m=(
            profile.max_outer_loop_position_correction_m
        ),
        contact_max_outer_loop_position_correction_m=(
            profile.contact_max_outer_loop_position_correction_m
        ),
        outer_loop_orientation_gain=profile.outer_loop_orientation_gain,
        max_outer_loop_orientation_correction_rad=(
            profile.max_outer_loop_orientation_correction_rad
        ),
        max_iterations=profile.max_iterations,
        timeout_sec=profile.timeout_sec,
    )
