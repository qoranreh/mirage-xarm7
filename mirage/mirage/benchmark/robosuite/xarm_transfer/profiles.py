"""Static task, observation, and blocking-controller transfer profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


SOURCE_ROBOT_ID = "Panda"
POLICY_RATE_HZ = 20
ACTION_DIM = 7
EXPECTED_OSC_OUTPUT_MAX = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)
PROPRIO_OBSERVATION_SHAPES = MappingProxyType(
    {
        "robot0_eef_pos": (3,),
        "robot0_eef_quat": (4,),
        "robot0_gripper_qpos": (2,),
    }
)


class RuntimeContractError(RuntimeError):
    """Raised when checkpoint metadata cannot safely drive transfer runtime."""


class ObservationMode(str, Enum):
    """Supported checkpoint observation layouts."""

    LOW_DIM = "low_dim"
    IMAGE_NO_PROPRIO = "image_no_proprio"
    IMAGE_PROPRIO = "image_proprio"

    @classmethod
    def parse(cls, value: object) -> "ObservationMode":
        try:
            return cls(value)
        except (TypeError, ValueError) as error:
            raise RuntimeContractError(
                f"unsupported observation mode {value!r}"
            ) from error


def _finite_tuple(
    value: Sequence[float], length: int, name: str
) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise RuntimeContractError(f"{name} must be numeric") from error
    if len(result) != length or not all(np.isfinite(result)):
        raise RuntimeContractError(
            f"{name} must contain {length} finite values"
        )
    return result


@dataclass(frozen=True)
class BlockingControllerProfile:
    """Task-specific, statically reviewed simulation controller parameters."""

    free_osc_kp: tuple[float, float, float, float, float, float]
    position_tolerance_m: float
    orientation_tolerance_rad: float
    settling_iterations: int
    outer_loop_position_gain: float
    max_outer_loop_position_correction_m: float
    outer_loop_orientation_gain: float
    max_outer_loop_orientation_correction_rad: float
    max_iterations: int
    timeout_sec: float
    contact_osc_kp: tuple[float, float, float, float, float, float] | None = None
    contact_position_tolerance_m: float | None = None
    contact_orientation_tolerance_rad: float | None = None
    contact_max_outer_loop_position_correction_m: float | None = None

    def __post_init__(self) -> None:
        free_kp = _finite_tuple(self.free_osc_kp, 6, "free_osc_kp")
        if any(item <= 0.0 for item in free_kp):
            raise RuntimeContractError("free_osc_kp must be positive")
        object.__setattr__(self, "free_osc_kp", free_kp)
        if self.contact_osc_kp is not None:
            contact_kp = _finite_tuple(self.contact_osc_kp, 6, "contact_osc_kp")
            if any(item <= 0.0 for item in contact_kp):
                raise RuntimeContractError("contact_osc_kp must be positive")
            object.__setattr__(self, "contact_osc_kp", contact_kp)

        numeric_nonnegative = (
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "outer_loop_position_gain",
            "max_outer_loop_position_correction_m",
            "outer_loop_orientation_gain",
            "max_outer_loop_orientation_correction_rad",
        )
        for name in numeric_nonnegative:
            number = float(getattr(self, name))
            if not np.isfinite(number) or number < 0.0:
                raise RuntimeContractError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, number)
        if type(self.settling_iterations) is not int or self.settling_iterations < 0:
            raise RuntimeContractError("settling_iterations must be non-negative")
        if type(self.max_iterations) is not int or self.max_iterations <= 0:
            raise RuntimeContractError("max_iterations must be positive")
        timeout = float(self.timeout_sec)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise RuntimeContractError("timeout_sec must be finite and positive")
        object.__setattr__(self, "timeout_sec", timeout)

        contact_values = (
            self.contact_osc_kp,
            self.contact_position_tolerance_m,
            self.contact_orientation_tolerance_rad,
            self.contact_max_outer_loop_position_correction_m,
        )
        if any(item is not None for item in contact_values) and any(
            item is None for item in contact_values
        ):
            raise RuntimeContractError(
                "contact gain, tolerances, and correction must be enabled together"
            )
        if self.contact_position_tolerance_m is not None:
            contact_position = float(self.contact_position_tolerance_m)
            contact_orientation = float(self.contact_orientation_tolerance_rad)
            contact_correction = float(
                self.contact_max_outer_loop_position_correction_m
            )
            if (
                not np.isfinite(contact_position)
                or contact_position < self.position_tolerance_m
                or not np.isfinite(contact_orientation)
                or contact_orientation < self.orientation_tolerance_rad
                or not np.isfinite(contact_correction)
                or contact_correction < self.max_outer_loop_position_correction_m
            ):
                raise RuntimeContractError("contact controller parameters are invalid")
            object.__setattr__(
                self, "contact_position_tolerance_m", contact_position
            )
            object.__setattr__(
                self, "contact_orientation_tolerance_rad", contact_orientation
            )
            object.__setattr__(
                self,
                "contact_max_outer_loop_position_correction_m",
                contact_correction,
            )

    @property
    def contact_relaxation_enabled(self) -> bool:
        return self.contact_osc_kp is not None


@dataclass(frozen=True)
class TaskProfile:
    """All task-specific facts needed by simulation transfer."""

    env_name: str
    slug: str
    native_horizon: int
    low_dim_observation_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    manipulated_object_names: tuple[str, ...]
    task_world_provenance: str
    controller_profile: BlockingControllerProfile
    environment_registration_module: str | None = None

    def __post_init__(self) -> None:
        for name in ("env_name", "slug", "task_world_provenance"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeContractError(f"{name} must not be empty")
            object.__setattr__(self, name, value.strip())
        try:
            object_names = tuple(
                name.strip() for name in self.manipulated_object_names
            )
        except (AttributeError, TypeError) as error:
            raise RuntimeContractError(
                "manipulated_object_names must contain strings"
            ) from error
        if (
            not object_names
            or any(not name for name in object_names)
            or len(set(object_names)) != len(object_names)
        ):
            raise RuntimeContractError(
                "manipulated_object_names must be non-empty and unique"
            )
        object.__setattr__(self, "manipulated_object_names", object_names)
        registration_module = self.environment_registration_module
        if registration_module is not None:
            if (
                not isinstance(registration_module, str)
                or not registration_module.strip()
            ):
                raise RuntimeContractError(
                    "environment_registration_module must be a non-empty string"
                )
            object.__setattr__(
                self,
                "environment_registration_module",
                registration_module.strip(),
            )
        if type(self.native_horizon) is not int or self.native_horizon <= 0:
            raise RuntimeContractError("native_horizon must be positive")
        shapes: list[tuple[str, tuple[int, ...]]] = []
        for key, shape in self.low_dim_observation_shapes:
            dimensions = tuple(int(item) for item in shape)
            if not key or not dimensions or any(item <= 0 for item in dimensions):
                raise RuntimeContractError("low-dimensional observation shape is invalid")
            shapes.append((str(key), dimensions))
        if len({key for key, _ in shapes}) != len(shapes):
            raise RuntimeContractError("low-dimensional observation keys must be unique")
        object.__setattr__(self, "low_dim_observation_shapes", tuple(shapes))

    @property
    def low_dim_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self.low_dim_observation_shapes)

    @property
    def graspable_object_name(self) -> str:
        """Return the legacy single-object name for single-object tasks."""

        if len(self.manipulated_object_names) != 1:
            raise RuntimeContractError(
                f"task {self.env_name!r} has multiple manipulated objects"
            )
        return self.manipulated_object_names[0]


CONTACT_MANIPULATION_CONTROLLER = BlockingControllerProfile(
    free_osc_kp=(150.0,) * 6,
    position_tolerance_m=0.025,
    orientation_tolerance_rad=float(np.deg2rad(7.0)),
    settling_iterations=4,
    outer_loop_position_gain=1.0,
    max_outer_loop_position_correction_m=0.030,
    outer_loop_orientation_gain=0.5,
    max_outer_loop_orientation_correction_rad=float(np.deg2rad(5.0)),
    max_iterations=300,
    timeout_sec=2.0,
    contact_osc_kp=(300.0, 300.0, 500.0, 300.0, 300.0, 300.0),
    contact_position_tolerance_m=0.060,
    contact_orientation_tolerance_rad=float(np.deg2rad(18.0)),
    contact_max_outer_loop_position_correction_m=0.100,
)

# Compatibility name retained for existing callers and documentation.
PICK_PLACE_CAN_CONTROLLER = CONTACT_MANIPULATION_CONTROLLER

DEFAULT_BLOCKING_CONTROLLER = BlockingControllerProfile(
    free_osc_kp=(150.0,) * 6,
    position_tolerance_m=0.025,
    orientation_tolerance_rad=float(np.deg2rad(7.0)),
    settling_iterations=4,
    outer_loop_position_gain=1.0,
    max_outer_loop_position_correction_m=0.030,
    outer_loop_orientation_gain=0.5,
    max_outer_loop_orientation_correction_rad=float(np.deg2rad(5.0)),
    max_iterations=300,
    timeout_sec=2.0,
)

# Compatibility name retained for existing callers and serialized reports.
LIFT_CONTROLLER = DEFAULT_BLOCKING_CONTROLLER


def _low_dim_shapes(object_size: int) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (
        ("object", (object_size,)),
        ("robot0_eef_pos", (3,)),
        ("robot0_eef_quat", (4,)),
        ("robot0_gripper_qpos", (2,)),
    )


def _task_profile(
    *,
    env_name: str,
    slug: str,
    object_size: int,
    manipulated_object_names: tuple[str, ...],
    native_horizon: int = 400,
    controller_profile: BlockingControllerProfile = DEFAULT_BLOCKING_CONTROLLER,
    environment_registration_module: str | None = None,
) -> TaskProfile:
    """Build a registry entry from the transfer invariants shared by all tasks."""

    return TaskProfile(
        env_name=env_name,
        slug=slug,
        native_horizon=native_horizon,
        low_dim_observation_shapes=_low_dim_shapes(object_size),
        manipulated_object_names=manipulated_object_names,
        task_world_provenance=f"shared_{slug.replace('-', '_')}_task_world_simulation_only",
        controller_profile=controller_profile,
        environment_registration_module=environment_registration_module,
    )


TASK_PROFILES = MappingProxyType(
    {
        "PickPlaceCan": _task_profile(
            env_name="PickPlaceCan",
            slug="pick-place-can",
            object_size=14,
            manipulated_object_names=("Can",),
            controller_profile=CONTACT_MANIPULATION_CONTROLLER,
        ),
        "Lift": _task_profile(
            env_name="Lift",
            slug="lift",
            object_size=10,
            manipulated_object_names=("cube",),
        ),
        "Square_D0": _task_profile(
            env_name="Square_D0",
            slug="square-d0",
            object_size=14,
            manipulated_object_names=("SquareNut",),
            controller_profile=CONTACT_MANIPULATION_CONTROLLER,
            environment_registration_module="mimicgen_envs",
        ),
        "Stack_D0": _task_profile(
            env_name="Stack_D0",
            slug="stack-d0",
            object_size=23,
            manipulated_object_names=("cubeA",),
            environment_registration_module="mimicgen_envs",
        ),
        "ThreePieceAssembly_D0": _task_profile(
            env_name="ThreePieceAssembly_D0",
            slug="three-piece-assembly-d0",
            object_size=42,
            manipulated_object_names=("piece_1", "piece_2"),
            native_horizon=500,
            controller_profile=CONTACT_MANIPULATION_CONTROLLER,
            environment_registration_module="mimicgen_envs",
        ),
    }
)


@dataclass(frozen=True)
class CheckpointContract:
    """Resolved checkpoint facts validated before simulator construction."""

    task_profile: TaskProfile
    observation_mode: ObservationMode
    observation_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    native_rollout_horizon: int

    def __post_init__(self) -> None:
        if not isinstance(self.task_profile, TaskProfile):
            raise RuntimeContractError("checkpoint task_profile is invalid")
        mode = ObservationMode.parse(self.observation_mode)
        object.__setattr__(self, "observation_mode", mode)
        shapes = _normalized_shapes(dict(self.observation_shapes))
        classified_mode = _classify_observation_mode(shapes, self.task_profile)
        if classified_mode is not mode:
            raise RuntimeContractError(
                "checkpoint observation mode does not match observation shapes"
            )
        object.__setattr__(self, "observation_shapes", tuple(sorted(shapes.items())))
        if self.native_rollout_horizon != self.task_profile.native_horizon:
            raise RuntimeContractError(
                "checkpoint native horizon does not match task profile"
            )

    @property
    def task_name(self) -> str:
        return self.task_profile.env_name

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self.observation_shapes)

    @property
    def image_observation_key(self) -> str | None:
        image_keys = tuple(
            key for key, _ in self.observation_shapes if key.endswith("_image")
        )
        if not image_keys:
            return None
        if len(image_keys) != 1:
            raise RuntimeContractError(
                "image runtime requires exactly one camera observation"
            )
        return image_keys[0]

    @property
    def camera_name(self) -> str | None:
        key = self.image_observation_key
        return None if key is None else key[: -len("_image")]

    def require_runtime_supported(self) -> None:
        if self.observation_mode is ObservationMode.LOW_DIM:
            return
        key = self.image_observation_key
        if key != "agentview_image" or self.shapes[key] != (3, 84, 84):
            raise RuntimeContractError(
                "image runtime requires one agentview_image with shape [3, 84, 84]"
            )

    def require_artifact_contract(
        self, *, task_name: str, observation_mode: ObservationMode | str
    ) -> None:
        mode = ObservationMode.parse(observation_mode)
        if task_name != self.task_name or mode is not self.observation_mode:
            raise RuntimeContractError(
                "artifact task/observation mode does not match checkpoint contract"
            )


def task_profile_for_name(env_name: object) -> TaskProfile:
    if not isinstance(env_name, str) or env_name not in TASK_PROFILES:
        raise RuntimeContractError(f"unregistered source task {env_name!r}")
    return TASK_PROFILES[env_name]


def _normalized_shapes(value: object) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise RuntimeContractError("source checkpoint has no observation shapes")
    result: dict[str, tuple[int, ...]] = {}
    for key, shape in value.items():
        if not isinstance(key, str) or not key:
            raise RuntimeContractError("source observation key is invalid")
        try:
            dimensions = tuple(int(item) for item in shape)
        except (TypeError, ValueError) as error:
            raise RuntimeContractError(
                f"source observation {key!r} has an invalid shape"
            ) from error
        if not dimensions or any(item <= 0 for item in dimensions):
            raise RuntimeContractError(
                f"source observation {key!r} has an invalid shape"
            )
        result[key] = dimensions
    if not result:
        raise RuntimeContractError("source checkpoint has no observation shapes")
    return result


def _classify_observation_mode(
    shapes: Mapping[str, tuple[int, ...]], profile: TaskProfile
) -> ObservationMode:
    expected_low_dim = profile.low_dim_shapes
    if set(shapes) == set(expected_low_dim):
        for key, expected in expected_low_dim.items():
            if shapes[key] != expected:
                raise RuntimeContractError(
                    f"source observation {key!r} must have shape {list(expected)}"
                )
        return ObservationMode.LOW_DIM

    image_keys = {key for key in shapes if key.endswith("_image")}
    if image_keys:
        for key in image_keys:
            shape = shapes[key]
            if len(shape) != 3 or shape[0] not in (1, 3, 4):
                raise RuntimeContractError(
                    f"source image observation {key!r} has an invalid shape"
                )
        keys = set(shapes)
        if keys == image_keys:
            return ObservationMode.IMAGE_NO_PROPRIO
        proprio = set(PROPRIO_OBSERVATION_SHAPES)
        if keys == image_keys | proprio:
            for key, expected in PROPRIO_OBSERVATION_SHAPES.items():
                if shapes[key] != expected:
                    raise RuntimeContractError(
                        f"source observation {key!r} must have shape {list(expected)}"
                    )
            return ObservationMode.IMAGE_PROPRIO
    raise RuntimeContractError("unknown source observation combination")


def _native_horizon(checkpoint_data: Mapping[str, Any], profile: TaskProfile) -> int:
    config = checkpoint_data.get("config")
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError as error:
            raise RuntimeContractError("source checkpoint config is not valid JSON") from error
    if not isinstance(config, Mapping):
        raise RuntimeContractError("source checkpoint has no config mapping")
    try:
        horizon = config["experiment"]["rollout"]["horizon"]
    except (KeyError, TypeError) as error:
        raise RuntimeContractError(
            "source checkpoint has no native rollout horizon"
        ) from error
    if type(horizon) is not int or horizon != profile.native_horizon:
        raise RuntimeContractError(
            f"source checkpoint native horizon must be {profile.native_horizon}"
        )
    return horizon


def resolve_checkpoint_contract(
    checkpoint_data: Mapping[str, Any],
) -> CheckpointContract:
    """Resolve and validate task and observation metadata before runtime use."""

    if not isinstance(checkpoint_data, Mapping):
        raise RuntimeContractError("checkpoint data must be a mapping")
    env_metadata = checkpoint_data.get("env_metadata")
    shape_metadata = checkpoint_data.get("shape_metadata")
    if not isinstance(env_metadata, Mapping):
        raise RuntimeContractError("source checkpoint has no env_metadata mapping")
    if not isinstance(shape_metadata, Mapping):
        raise RuntimeContractError("source checkpoint has no shape_metadata mapping")
    profile = task_profile_for_name(env_metadata.get("env_name"))
    kwargs = env_metadata.get("env_kwargs")
    if not isinstance(kwargs, Mapping):
        raise RuntimeContractError("source checkpoint has no env_kwargs mapping")
    if kwargs.get("robots") != [SOURCE_ROBOT_ID]:
        raise RuntimeContractError(
            "source checkpoint must contain exactly one Franka Panda robot"
        )
    if kwargs.get("control_freq") != POLICY_RATE_HZ:
        raise RuntimeContractError("source checkpoint control_freq must be 20 Hz")
    controller = kwargs.get("controller_configs")
    if not isinstance(controller, Mapping):
        raise RuntimeContractError("source checkpoint has no controller config")
    if controller.get("type") != "OSC_POSE":
        raise RuntimeContractError("source controller must be OSC_POSE")
    if controller.get("control_delta") is not True:
        raise RuntimeContractError("source controller must set control_delta=true")
    if controller.get("input_min") != -1 or controller.get("input_max") != 1:
        raise RuntimeContractError("source controller input range must be [-1, 1]")
    try:
        output_max = np.asarray(controller.get("output_max"), dtype=np.float64)
        output_min = np.asarray(controller.get("output_min"), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeContractError("source OSC output scales must be numeric") from error
    expected_output_max = np.asarray(EXPECTED_OSC_OUTPUT_MAX)
    if not np.array_equal(output_max, expected_output_max) or not np.array_equal(
        output_min, -expected_output_max
    ):
        raise RuntimeContractError(
            "source OSC output scales must be ±[0.05m, 0.5rad] per axis"
        )
    if shape_metadata.get("ac_dim") != ACTION_DIM:
        raise RuntimeContractError("source policy action dimension must be 7")
    shapes = _normalized_shapes(shape_metadata.get("all_shapes"))
    mode = _classify_observation_mode(shapes, profile)
    horizon = _native_horizon(checkpoint_data, profile)
    return CheckpointContract(
        task_profile=profile,
        observation_mode=mode,
        observation_shapes=tuple(sorted(shapes.items())),
        native_rollout_horizon=horizon,
    )
