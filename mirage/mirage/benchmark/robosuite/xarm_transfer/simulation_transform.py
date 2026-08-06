"""Simulation-only task-frame transforms and tool-frame corrections.

The source and target task worlds are related by a left-multiplied transform,
while differences between the two end-effector conventions are represented by
a separate right-multiplied correction::

    T_targetWorld_targetTool =
        T_targetWorld_sourceWorld
        @ T_sourceWorld_sourceTool
        @ T_sourceTool_targetTool

Keeping the two terms separate prevents a robot's home tool orientation from
rotating task-space motion directions. These types cannot represent a real
calibration, and reject any configuration containing ``calibrated: true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contracts import CartesianPose, ContractError, SourceDesiredPose
from .geometry import (
    canonicalize_quaternion_xyzw,
    finite_vector,
    matrix_pose,
    pose_matrix,
)


SIMULATION_TRANSFORM_VERSION = 2
SIMULATION_TRANSFORM_KIND = "simulation_only"
IDENTITY_TRANSLATION_M = (0.0, 0.0, 0.0)
IDENTITY_QUATERNION_XYZW = (0.0, 0.0, 0.0, 1.0)
PANDA_TO_XARM7_TOOL_QUATERNION_XYZW = (
    0.0,
    0.0,
    -float(np.sqrt(0.5)),
    float(np.sqrt(0.5)),
)


class SimulationTransformError(ValueError):
    """Raised when a simulation alignment is invalid or mislabeled."""


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SimulationTransformError(f"{name} must not be empty")
    return value.strip()


def _rigid_components(
    translation_m: object,
    rotation_xyzw: object,
    *,
    prefix: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    try:
        translation = tuple(
            float(item)
            for item in finite_vector(translation_m, 3, f"{prefix}.translation_m")
        )
        quaternion = tuple(
            float(item) for item in canonicalize_quaternion_xyzw(rotation_xyzw)
        )
    except ValueError as error:
        raise SimulationTransformError(str(error)) from error
    return translation, quaternion


@dataclass(frozen=True)
class TargetDesiredPose:
    """xArm simulation pose produced from a source desired pose."""

    pose: CartesianPose
    gripper_normalized: float
    source_action_timestamp_ns: int
    transform_kind: str = SIMULATION_TRANSFORM_KIND

    def __post_init__(self) -> None:
        if self.transform_kind != SIMULATION_TRANSFORM_KIND:
            raise SimulationTransformError(
                "target desired pose must be marked simulation_only"
            )
        value = float(self.gripper_normalized)
        if not np.isfinite(value) or abs(value) > 1.0 + 1e-8:
            raise SimulationTransformError(
                "target gripper command must be finite in [-1, 1]"
            )
        if type(self.source_action_timestamp_ns) is not int or self.source_action_timestamp_ns < 0:
            raise SimulationTransformError(
                "source_action_timestamp_ns must be non-negative"
            )
        object.__setattr__(self, "gripper_normalized", value)


@dataclass(frozen=True)
class SimulationTaskFrameTransform:
    """Left-multiplied transform between two simulator task/world frames."""

    source_frame: str
    target_frame: str
    translation_m: tuple[float, float, float] = IDENTITY_TRANSLATION_M
    rotation_xyzw: tuple[float, float, float, float] = IDENTITY_QUATERNION_XYZW
    provenance: str = "shared_simulation_task_world"

    def __post_init__(self) -> None:
        source_frame = _non_empty_string(self.source_frame, "source_frame")
        target_frame = _non_empty_string(self.target_frame, "target_frame")
        provenance = _non_empty_string(self.provenance, "task provenance")
        if source_frame == target_frame:
            raise SimulationTransformError(
                "source and target simulation task frames must differ"
            )
        translation, quaternion = _rigid_components(
            self.translation_m, self.rotation_xyzw, prefix="task_frame_transform"
        )
        object.__setattr__(self, "source_frame", source_frame)
        object.__setattr__(self, "target_frame", target_frame)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "rotation_xyzw", quaternion)
        object.__setattr__(self, "provenance", provenance)

    @property
    def matrix(self) -> np.ndarray:
        return pose_matrix(self.translation_m, self.rotation_xyzw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "translation_m": list(self.translation_m),
            "rotation_xyzw": list(self.rotation_xyzw),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SimulationToolFrameCorrection:
    """Right-multiplied correction from source to target tool convention."""

    source_tool_frame: str
    target_tool_frame: str
    translation_m: tuple[float, float, float] = IDENTITY_TRANSLATION_M
    rotation_xyzw: tuple[float, float, float, float] = IDENTITY_QUATERNION_XYZW
    provenance: str = "standardized_robosuite_grip_site"

    def __post_init__(self) -> None:
        source_tool_frame = _non_empty_string(
            self.source_tool_frame, "source_tool_frame"
        )
        target_tool_frame = _non_empty_string(
            self.target_tool_frame, "target_tool_frame"
        )
        provenance = _non_empty_string(self.provenance, "tool provenance")
        translation, quaternion = _rigid_components(
            self.translation_m, self.rotation_xyzw, prefix="tool_frame_correction"
        )
        object.__setattr__(self, "source_tool_frame", source_tool_frame)
        object.__setattr__(self, "target_tool_frame", target_tool_frame)
        object.__setattr__(self, "translation_m", translation)
        object.__setattr__(self, "rotation_xyzw", quaternion)
        object.__setattr__(self, "provenance", provenance)

    @property
    def matrix(self) -> np.ndarray:
        return pose_matrix(self.translation_m, self.rotation_xyzw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tool_frame": self.source_tool_frame,
            "target_tool_frame": self.target_tool_frame,
            "translation_m": list(self.translation_m),
            "rotation_xyzw": list(self.rotation_xyzw),
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class SimulationTransform:
    """Composition of independent task/world and tool-frame transforms."""

    task_frame_transform: SimulationTaskFrameTransform
    tool_frame_correction: SimulationToolFrameCorrection
    kind: str = SIMULATION_TRANSFORM_KIND
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.task_frame_transform, SimulationTaskFrameTransform):
            raise SimulationTransformError(
                "task_frame_transform must be a SimulationTaskFrameTransform"
            )
        if not isinstance(self.tool_frame_correction, SimulationToolFrameCorrection):
            raise SimulationTransformError(
                "tool_frame_correction must be a SimulationToolFrameCorrection"
            )
        if self.kind != SIMULATION_TRANSFORM_KIND:
            raise SimulationTransformError("transform kind must be simulation_only")
        if self.calibrated is not False:
            raise SimulationTransformError(
                "a simulation transform can never be marked calibrated"
            )

    @property
    def source_frame(self) -> str:
        return self.task_frame_transform.source_frame

    @property
    def target_frame(self) -> str:
        return self.task_frame_transform.target_frame

    @classmethod
    def for_shared_task_world(
        cls,
        *,
        source_frame: str,
        target_frame: str,
        source_tool_frame: str = "panda_grip_site",
        target_tool_frame: str = "xarm7_grip_site",
    ) -> "SimulationTransform":
        """Build the PickPlaceCan mapping without inspecting robot home poses.

        Robosuite places both PickPlaceCan scenes in the same task/world
        coordinates, so the task transform is identity. The Panda and xArm7
        gripper axes differ by a nominal -90 degree local-Z rotation, which is
        represented only by the right-multiplied tool correction. It therefore
        cannot rotate world-frame position deltas.
        """

        return cls(
            task_frame_transform=SimulationTaskFrameTransform(
                source_frame=source_frame,
                target_frame=target_frame,
                provenance="shared_pick_place_can_task_world",
            ),
            tool_frame_correction=SimulationToolFrameCorrection(
                source_tool_frame=source_tool_frame,
                target_tool_frame=target_tool_frame,
                rotation_xyzw=PANDA_TO_XARM7_TOOL_QUATERNION_XYZW,
                provenance="nominal_panda_to_xarm7_grip_axis",
            ),
        )

    def transform_pose(self, source_pose: CartesianPose) -> CartesianPose:
        if source_pose.frame_id != self.source_frame:
            raise SimulationTransformError(
                f"source pose frame {source_pose.frame_id!r} does not match "
                f"{self.source_frame!r}"
            )
        source_matrix = pose_matrix(
            source_pose.position_m, source_pose.quaternion_xyzw
        )
        target_matrix = (
            self.task_frame_transform.matrix
            @ source_matrix
            @ self.tool_frame_correction.matrix
        )
        position, quaternion = matrix_pose(target_matrix)
        try:
            return CartesianPose(
                tuple(position),
                tuple(quaternion),
                self.target_frame,
                source_pose.timestamp_ns,
            )
        except ContractError as error:
            raise SimulationTransformError(str(error)) from error

    def transform_desired_pose(
        self, desired: SourceDesiredPose
    ) -> TargetDesiredPose:
        return TargetDesiredPose(
            pose=self.transform_pose(desired.pose),
            gripper_normalized=desired.gripper_normalized,
            source_action_timestamp_ns=desired.action_timestamp_ns,
        )

    def inverse(self) -> "SimulationTransform":
        task_translation, task_quaternion = matrix_pose(
            np.linalg.inv(self.task_frame_transform.matrix)
        )
        tool_translation, tool_quaternion = matrix_pose(
            np.linalg.inv(self.tool_frame_correction.matrix)
        )
        return SimulationTransform(
            task_frame_transform=SimulationTaskFrameTransform(
                source_frame=self.target_frame,
                target_frame=self.source_frame,
                translation_m=tuple(task_translation),
                rotation_xyzw=tuple(task_quaternion),
                provenance=f"inverse({self.task_frame_transform.provenance})",
            ),
            tool_frame_correction=SimulationToolFrameCorrection(
                source_tool_frame=self.tool_frame_correction.target_tool_frame,
                target_tool_frame=self.tool_frame_correction.source_tool_frame,
                translation_m=tuple(tool_translation),
                rotation_xyzw=tuple(tool_quaternion),
                provenance=f"inverse({self.tool_frame_correction.provenance})",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SIMULATION_TRANSFORM_VERSION,
            "kind": self.kind,
            "calibrated": False,
            "task_frame_transform": self.task_frame_transform.to_dict(),
            "tool_frame_correction": self.tool_frame_correction.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationTransform":
        if value.get("version") == 1:
            raise SimulationTransformError(
                "unsafe version-1 home-alignment transform is not supported"
            )
        expected = {
            "version",
            "kind",
            "calibrated",
            "task_frame_transform",
            "tool_frame_correction",
        }
        if set(value) != expected:
            raise SimulationTransformError(
                "simulation transform keys do not match version-2 schema"
            )
        if value["version"] != SIMULATION_TRANSFORM_VERSION:
            raise SimulationTransformError("unsupported simulation transform version")
        if value["calibrated"] is not False:
            raise SimulationTransformError(
                "simulation transform configuration must set calibrated: false"
            )
        task = value["task_frame_transform"]
        tool = value["tool_frame_correction"]
        if not isinstance(task, Mapping) or set(task) != {
            "source_frame",
            "target_frame",
            "translation_m",
            "rotation_xyzw",
            "provenance",
        }:
            raise SimulationTransformError("invalid task_frame_transform schema")
        if not isinstance(tool, Mapping) or set(tool) != {
            "source_tool_frame",
            "target_tool_frame",
            "translation_m",
            "rotation_xyzw",
            "provenance",
        }:
            raise SimulationTransformError("invalid tool_frame_correction schema")
        return cls(
            task_frame_transform=SimulationTaskFrameTransform(**dict(task)),
            tool_frame_correction=SimulationToolFrameCorrection(**dict(tool)),
            kind=value["kind"],
            calibrated=value["calibrated"],
        )
