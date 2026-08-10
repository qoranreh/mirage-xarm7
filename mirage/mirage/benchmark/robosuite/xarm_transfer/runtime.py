"""Robosuite runtime adapters for the Franka Panda source and xArm target.

This module only talks to in-process Robosuite environments. It contains no
ROS publisher, service client, action client, robot IP, or hardware SDK path.
"""

from __future__ import annotations

import copy
import importlib
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import CartesianPose, SourceAction
from .dynamics import PandaOnlineOracle
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
    return metadata


def build_target_environment_metadata(
    checkpoint_data: Mapping[str, Any],
    contract: CheckpointContract,
    *,
    render: bool,
) -> dict[str, Any]:
    """Build a camera-free target independent of source policy modality."""

    metadata = copy.deepcopy(checkpoint_data["env_metadata"])
    metadata["env_name"] = contract.task_name
    kwargs = metadata["env_kwargs"]
    kwargs["robots"] = [TARGET_ROBOSUITE_ROBOT_ID]
    kwargs["has_renderer"] = bool(render)
    _disable_policy_cameras(kwargs)
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

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        render: bool = False,
    ) -> "FrankaPandaSourceRuntime":
        """Load the policy and a low-dimensional Franka Panda environment.

        The legacy ``FileUtils.env_from_checkpoint`` path forcibly enables
        camera observations in this repository.  This simulation-only state
        transfer intentionally builds from a deep-copied metadata document
        with all offscreen, depth, and segmentation paths disabled.
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
        environment = EnvUtils.create_env_from_metadata(
            env_meta=env_metadata,
            render=render,
            render_offscreen=False,
            use_image_obs=False,
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

    def policy_action(self, *, timestamp_ns: int | None = None) -> SourceAction:
        if self.observation is None:
            raise RuntimeContractError("source runtime must be reset first")
        observation = copy.deepcopy(dict(self.observation))
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
        self.last_step_output = (next_observation, reward, done, info)
        return step_output

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
        self._task_object_geom_ids: frozenset[int] = frozenset()
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
        self._task_object_geom_ids = object_geom_ids
        self._gripper_body_ids = gripper_body_ids

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
        return copy.deepcopy(self.observation)

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
