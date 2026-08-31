"""Quaternion and Cartesian-pose helpers for simulation state transfer.

All quaternions use ``[x, y, z, w]`` ordering.  The helpers canonicalize the
otherwise ambiguous quaternion sign so serialized datasets are deterministic.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


class GeometryError(ValueError):
    """Raised when a pose or rotation cannot be represented safely."""


def finite_vector(
    value: Sequence[float] | np.ndarray,
    length: int,
    name: str,
) -> np.ndarray:
    """Return a finite float64 vector with an exact shape."""

    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise GeometryError(f"{name} must be numeric") from error
    if vector.shape != (length,):
        raise GeometryError(f"{name} must have shape ({length},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise GeometryError(f"{name} must contain only finite values")
    return vector


def canonicalize_quaternion_xyzw(
    quaternion_xyzw: Sequence[float] | np.ndarray,
    *,
    norm_tolerance: float = 1e-3,
) -> np.ndarray:
    """Normalize a quaternion and choose a deterministic sign."""

    quaternion = finite_vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise GeometryError("quaternion_xyzw norm must be non-zero")
    if abs(norm - 1.0) > norm_tolerance:
        raise GeometryError(
            f"quaternion_xyzw norm {norm:.9g} exceeds tolerance {norm_tolerance:.9g}"
        )
    quaternion = quaternion / norm

    # q and -q describe the same rotation. Prefer positive w. At an exact half
    # turn, prefer the sign whose first non-zero xyz component is positive.
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    elif abs(quaternion[3]) <= 1e-15:
        for component in quaternion[:3]:
            if abs(component) > 1e-15:
                if component < 0.0:
                    quaternion = -quaternion
                break
    quaternion[np.abs(quaternion) < 1e-15] = 0.0
    return quaternion


def quaternion_angular_error_rad(
    first_xyzw: Sequence[float] | np.ndarray,
    second_xyzw: Sequence[float] | np.ndarray,
) -> float:
    """Return the shortest sign-invariant angular distance in radians."""

    first = canonicalize_quaternion_xyzw(first_xyzw)
    second = canonicalize_quaternion_xyzw(second_xyzw)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def quaternion_to_rotvec(
    quaternion_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert a canonical XYZW quaternion to a rotation vector."""

    quaternion = canonicalize_quaternion_xyzw(quaternion_xyzw)
    rotvec = Rotation.from_quat(quaternion).as_rotvec()
    if not np.all(np.isfinite(rotvec)):
        raise GeometryError("quaternion conversion produced a non-finite rotation")
    return np.asarray(rotvec, dtype=np.float64)


def rotvec_to_quaternion_xyzw(
    rotation_vector_rad: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Convert a finite rotation vector to a canonical XYZW quaternion."""

    rotation_vector = finite_vector(rotation_vector_rad, 3, "rotation_vector_rad")
    return canonicalize_quaternion_xyzw(
        Rotation.from_rotvec(rotation_vector).as_quat()
    )


def relative_rotation_vector(
    current_xyzw: Sequence[float] | np.ndarray,
    next_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return world-frame delta ``R_next * inverse(R_current)``."""

    current = Rotation.from_quat(canonicalize_quaternion_xyzw(current_xyzw))
    next_rotation = Rotation.from_quat(canonicalize_quaternion_xyzw(next_xyzw))
    delta = next_rotation * current.inv()
    result = delta.as_rotvec()
    if not np.all(np.isfinite(result)):
        raise GeometryError("relative rotation produced non-finite values")
    return np.asarray(result, dtype=np.float64)


def integrate_world_rotation_delta(
    current_xyzw: Sequence[float] | np.ndarray,
    delta_rotation_vector_rad: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Apply a world-frame rotation delta to an orientation."""

    current = Rotation.from_quat(canonicalize_quaternion_xyzw(current_xyzw))
    delta = Rotation.from_rotvec(
        finite_vector(delta_rotation_vector_rad, 3, "delta_rotation_vector_rad")
    )
    return canonicalize_quaternion_xyzw((delta * current).as_quat())


def pose_matrix(
    position_m: Sequence[float] | np.ndarray,
    quaternion_xyzw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Build a homogeneous transform from a Cartesian pose."""

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        canonicalize_quaternion_xyzw(quaternion_xyzw)
    ).as_matrix()
    matrix[:3, 3] = finite_vector(position_m, 3, "position_m")
    return matrix


def matrix_pose(matrix: Sequence[Sequence[float]] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract position and canonical XYZW orientation from a rigid matrix."""

    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise GeometryError("pose matrix must be finite with shape (4, 4)")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9):
        raise GeometryError("pose matrix has an invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise GeometryError("pose matrix rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise GeometryError("pose matrix rotation determinant must be +1")
    return (
        value[:3, 3].copy(),
        canonicalize_quaternion_xyzw(Rotation.from_matrix(rotation).as_quat()),
    )
