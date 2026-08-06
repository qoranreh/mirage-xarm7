"""Explicit action, pose, and forward-dynamics contracts.

The PickPlaceCan policy action layout is fixed here as
``[dx, dy, dz, dRx, dRy, dRz, gripper]``. Translation is measured in metres,
rotation-vector components in radians, timestamps in nanoseconds, and
quaternions are always XYZW.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .geometry import GeometryError, canonicalize_quaternion_xyzw, finite_vector


ACTION_DIM = 7
POSITION_SLICE = slice(0, 3)
ORIENTATION_SLICE = slice(3, 6)
GRIPPER_INDEX = 6
DEFAULT_POSITION_SCALE_M = (0.05, 0.05, 0.05)
DEFAULT_ORIENTATION_SCALE_RAD = (0.5, 0.5, 0.5)
POLICY_RATE_HZ = 20.0
QUATERNION_ORDER = "xyzw"


class ContractError(ValueError):
    """Raised when state-transfer data violates its declared contract."""


def _timestamp_ns(value: int, name: str = "timestamp_ns") -> int:
    if type(value) is not int or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _frame_id(value: str, name: str = "frame_id") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must not be empty")
    return value.strip()


def _float_tuple(
    value: Sequence[float] | np.ndarray,
    length: int,
    name: str,
) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in finite_vector(value, length, name))
    except GeometryError as error:
        raise ContractError(str(error)) from error


@dataclass(frozen=True)
class CartesianPose:
    """Timestamped Cartesian pose in SI units and an explicit frame."""

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame_id: str
    timestamp_ns: int

    def __post_init__(self) -> None:
        position = _float_tuple(self.position_m, 3, "position_m")
        try:
            quaternion = tuple(
                float(item)
                for item in canonicalize_quaternion_xyzw(self.quaternion_xyzw)
            )
        except GeometryError as error:
            raise ContractError(str(error)) from error
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_xyzw", quaternion)
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        object.__setattr__(self, "timestamp_ns", _timestamp_ns(self.timestamp_ns))

    @classmethod
    def from_vector(
        cls,
        pose: Sequence[float] | np.ndarray,
        *,
        frame_id: str,
        timestamp_ns: int,
    ) -> "CartesianPose":
        value = _float_tuple(pose, 7, "pose")
        return cls(value[:3], value[3:], frame_id, timestamp_ns)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.position_m + self.quaternion_xyzw, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_m": list(self.position_m),
            "quaternion_xyzw": list(self.quaternion_xyzw),
            "quaternion_order": QUATERNION_ORDER,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CartesianPose":
        expected = {
            "position_m",
            "quaternion_xyzw",
            "quaternion_order",
            "frame_id",
            "timestamp_ns",
        }
        if set(value) != expected:
            raise ContractError("CartesianPose keys do not match version-1 contract")
        if value["quaternion_order"] != QUATERNION_ORDER:
            raise ContractError("only xyzw quaternion order is supported")
        return cls(
            value["position_m"],
            value["quaternion_xyzw"],
            value["frame_id"],
            value["timestamp_ns"],
        )


@dataclass(frozen=True)
class SourceAction:
    """Raw policy output and effective normalized controller input.

    Robosuite clips OSC dimensions to ``[-1, 1]`` internally. Keeping both
    vectors preserves exact policy-action parity while making the physical
    command derived from the clipped vector explicit.
    """

    normalized: tuple[float, float, float, float, float, float, float]
    frame_id: str
    timestamp_ns: int
    policy_output_raw: (
        tuple[float, float, float, float, float, float, float] | None
    ) = None

    def __post_init__(self) -> None:
        normalized = _float_tuple(self.normalized, ACTION_DIM, "normalized action")
        if any(abs(item) > 1.0 + 1e-8 for item in normalized):
            raise ContractError("normalized action must remain within [-1, 1]")
        raw = (
            normalized
            if self.policy_output_raw is None
            else _float_tuple(
                self.policy_output_raw, ACTION_DIM, "raw policy output"
            )
        )
        expected_normalized = tuple(float(item) for item in np.clip(raw, -1.0, 1.0))
        if not np.allclose(normalized, expected_normalized, atol=1e-8, rtol=0.0):
            raise ContractError(
                "normalized action must equal clip(raw policy output, -1, 1)"
            )
        object.__setattr__(self, "normalized", normalized)
        object.__setattr__(self, "policy_output_raw", raw)
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        object.__setattr__(self, "timestamp_ns", _timestamp_ns(self.timestamp_ns))

    @property
    def gripper_normalized(self) -> float:
        return self.normalized[GRIPPER_INDEX]

    def as_normalized_array(self) -> np.ndarray:
        return np.asarray(self.normalized, dtype=np.float64)

    def as_policy_array(self) -> np.ndarray:
        return np.asarray(self.policy_output_raw, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized": list(self.normalized),
            "policy_output_raw": list(self.policy_output_raw),
            "layout": ["dx", "dy", "dz", "dRx", "dRy", "dRz", "gripper"],
            "translation_unit": "normalized",
            "orientation_unit": "normalized",
            "gripper_unit": "normalized[-1,1]",
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceAction":
        required = {
            "normalized",
            "policy_output_raw",
            "frame_id",
            "timestamp_ns",
        }
        if not required.issubset(value):
            raise ContractError("SourceAction is missing required fields")
        return cls(
            value["normalized"],
            value["frame_id"],
            value["timestamp_ns"],
            value["policy_output_raw"],
        )


@dataclass(frozen=True)
class ScaledOscCommand:
    """Physical OSC delta command corresponding to one normalized action."""

    translation_delta_m: tuple[float, float, float]
    rotation_delta_axis_angle_rad: tuple[float, float, float]
    gripper_normalized: float
    frame_id: str
    timestamp_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "translation_delta_m",
            _float_tuple(self.translation_delta_m, 3, "translation_delta_m"),
        )
        object.__setattr__(
            self,
            "rotation_delta_axis_angle_rad",
            _float_tuple(
                self.rotation_delta_axis_angle_rad,
                3,
                "rotation_delta_axis_angle_rad",
            ),
        )
        gripper = float(self.gripper_normalized)
        if not np.isfinite(gripper) or abs(gripper) > 1.0 + 1e-8:
            raise ContractError("gripper_normalized must be finite in [-1, 1]")
        object.__setattr__(self, "gripper_normalized", gripper)
        object.__setattr__(self, "frame_id", _frame_id(self.frame_id))
        object.__setattr__(self, "timestamp_ns", _timestamp_ns(self.timestamp_ns))

    def as_array(self) -> np.ndarray:
        return np.asarray(
            self.translation_delta_m
            + self.rotation_delta_axis_angle_rad
            + (self.gripper_normalized,),
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation_delta_m": list(self.translation_delta_m),
            "rotation_delta_axis_angle_rad": list(
                self.rotation_delta_axis_angle_rad
            ),
            "gripper_normalized": self.gripper_normalized,
            "frame_id": self.frame_id,
            "timestamp_ns": self.timestamp_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScaledOscCommand":
        return cls(
            value["translation_delta_m"],
            value["rotation_delta_axis_angle_rad"],
            value["gripper_normalized"],
            value["frame_id"],
            value["timestamp_ns"],
        )


@dataclass(frozen=True)
class OscActionScaler:
    """Axis-specific normalized-action scaling used by OSC_POSE."""

    position_scale_m: tuple[float, float, float] = DEFAULT_POSITION_SCALE_M
    orientation_scale_rad: tuple[float, float, float] = (
        DEFAULT_ORIENTATION_SCALE_RAD
    )
    control_delta: bool = True
    policy_rate_hz: float = POLICY_RATE_HZ

    def __post_init__(self) -> None:
        position = _float_tuple(self.position_scale_m, 3, "position_scale_m")
        orientation = _float_tuple(
            self.orientation_scale_rad, 3, "orientation_scale_rad"
        )
        if any(item <= 0.0 for item in position + orientation):
            raise ContractError("OSC scales must be positive")
        if self.control_delta is not True:
            raise ContractError("source action contract requires control_delta=true")
        rate = float(self.policy_rate_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ContractError("policy_rate_hz must be finite and positive")
        object.__setattr__(self, "position_scale_m", position)
        object.__setattr__(self, "orientation_scale_rad", orientation)
        object.__setattr__(self, "policy_rate_hz", rate)

    def scale(self, action: SourceAction) -> ScaledOscCommand:
        normalized = action.as_normalized_array()
        translation = normalized[POSITION_SLICE] * np.asarray(self.position_scale_m)
        orientation = normalized[ORIENTATION_SLICE] * np.asarray(
            self.orientation_scale_rad
        )
        return ScaledOscCommand(
            tuple(translation),
            tuple(orientation),
            action.gripper_normalized,
            action.frame_id,
            action.timestamp_ns,
        )


@dataclass(frozen=True)
class SourceDesiredPose:
    """Source next EE pose plus the associated simulation gripper command."""

    pose: CartesianPose
    gripper_normalized: float
    action_timestamp_ns: int

    def __post_init__(self) -> None:
        gripper = float(self.gripper_normalized)
        if not np.isfinite(gripper) or abs(gripper) > 1.0 + 1e-8:
            raise ContractError("gripper_normalized must be finite in [-1, 1]")
        object.__setattr__(self, "gripper_normalized", gripper)
        object.__setattr__(
            self,
            "action_timestamp_ns",
            _timestamp_ns(self.action_timestamp_ns, "action_timestamp_ns"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose": self.pose.to_dict(),
            "gripper_normalized": self.gripper_normalized,
            "action_timestamp_ns": self.action_timestamp_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceDesiredPose":
        return cls(
            CartesianPose.from_dict(value["pose"]),
            value["gripper_normalized"],
            value["action_timestamp_ns"],
        )
