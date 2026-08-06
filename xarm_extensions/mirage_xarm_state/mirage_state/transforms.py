"""Validated rigid transforms between Mirage source and target frames."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml


class TransformError(ValueError):
    """Raised when a transform or pose cannot be used safely."""


class TransformConfigError(TransformError):
    """Raised when a source/target transform configuration is invalid."""


def _finite_vector(
    name: str, values: Sequence[float], expected_length: int
) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise TransformError(f'{name} must be numeric') from error
    if len(result) != expected_length:
        raise TransformError(
            f'{name} must contain {expected_length} values, got {len(result)}'
        )
    if not all(math.isfinite(value) for value in result):
        raise TransformError(f'{name} must contain only finite values')
    return result


def _normalized_quaternion(
    quaternion_xyzw: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> tuple[float, float, float, float]:
    values = _finite_vector('quaternion_xyzw', quaternion_xyzw, 4)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12 or abs(norm - 1.0) > tolerance:
        raise TransformError(
            f'quaternion_xyzw norm {norm:.9g} is outside tolerance '
            f'{tolerance:.9g}'
        )
    normalized = tuple(value / norm for value in values)
    vector_sign = next(
        (value for value in normalized[:3] if abs(value) > 1e-15), 0.0
    )
    if normalized[3] < 0.0 or (
        abs(normalized[3]) <= 1e-15 and vector_sign < 0.0
    ):
        normalized = tuple(-value for value in normalized)
    return normalized  # type: ignore[return-value]


def _quaternion_multiply(
    left_xyzw: Sequence[float], right_xyzw: Sequence[float]
) -> tuple[float, float, float, float]:
    """Return the Hamilton product left * right in xyzw ordering."""
    lx, ly, lz, lw = left_xyzw
    rx, ry, rz, rw = right_xyzw
    result = (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
    return _normalized_quaternion(result)


def _rotation_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def quaternion_angular_distance(
    first_xyzw: Sequence[float], second_xyzw: Sequence[float]
) -> float:
    """Return the shortest angular distance between two orientations."""
    first = np.asarray(_normalized_quaternion(first_xyzw), dtype=np.float64)
    second = np.asarray(_normalized_quaternion(second_xyzw), dtype=np.float64)
    dot = float(np.clip(abs(np.dot(first, second)), 0.0, 1.0))
    return 2.0 * math.acos(dot)


@dataclass(frozen=True)
class CartesianPose:
    """Cartesian pose in metres, xyzw convention, and an explicit frame."""

    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    frame_id: str

    def __post_init__(self) -> None:
        position = _finite_vector('position_m', self.position_m, 3)
        quaternion = _normalized_quaternion(self.quaternion_xyzw)
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise TransformError('frame_id must not be empty')
        object.__setattr__(self, 'position_m', position)
        object.__setattr__(self, 'quaternion_xyzw', quaternion)
        object.__setattr__(self, 'frame_id', self.frame_id.strip())


@dataclass(frozen=True)
class RigidTransform:
    """
    Map coordinates from ``source_frame`` into ``target_frame``.

    ``T_TS`` follows ``p_T = R_TS * p_S + t_TS`` and applies orientation as
    ``q_TE = q_TS * q_SE`` using the Hamilton product in xyzw order.
    """

    source_frame: str
    target_frame: str
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.source_frame, str) or not self.source_frame.strip():
            raise TransformError('source_frame must not be empty')
        if not isinstance(self.target_frame, str) or not self.target_frame.strip():
            raise TransformError('target_frame must not be empty')
        source_frame = self.source_frame.strip()
        target_frame = self.target_frame.strip()
        if source_frame == target_frame:
            raise TransformError('source_frame and target_frame must differ')
        object.__setattr__(self, 'source_frame', source_frame)
        object.__setattr__(self, 'target_frame', target_frame)
        object.__setattr__(
            self,
            'translation_m',
            _finite_vector('translation_m', self.translation_m, 3),
        )
        object.__setattr__(
            self,
            'rotation_xyzw',
            _normalized_quaternion(self.rotation_xyzw),
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Return a copy of the 3-by-3 source-to-target rotation matrix."""
        return _rotation_matrix(self.rotation_xyzw)

    def transform_vector(
        self, vector_source: Sequence[float]
    ) -> tuple[float, float, float]:
        """Rotate a free vector without applying translation."""
        vector = np.asarray(
            _finite_vector('vector_source', vector_source, 3),
            dtype=np.float64,
        )
        result = self.rotation_matrix @ vector
        return tuple(float(value) for value in result)  # type: ignore[return-value]

    def transform_position(
        self, position_source_m: Sequence[float]
    ) -> tuple[float, float, float]:
        """Transform a source-frame position into the target frame."""
        rotated = np.asarray(
            self.transform_vector(position_source_m), dtype=np.float64
        )
        result = rotated + np.asarray(self.translation_m, dtype=np.float64)
        return tuple(float(value) for value in result)  # type: ignore[return-value]

    def transform_orientation(
        self, quaternion_source_xyzw: Sequence[float]
    ) -> tuple[float, float, float, float]:
        """Transform a source-frame EE orientation into the target frame."""
        source = _normalized_quaternion(quaternion_source_xyzw)
        return _quaternion_multiply(self.rotation_xyzw, source)

    def transform_pose(self, pose_source: CartesianPose) -> CartesianPose:
        """Transform a pose, rejecting an unexpected source frame."""
        if pose_source.frame_id != self.source_frame:
            raise TransformError(
                f'pose frame {pose_source.frame_id!r} does not match '
                f'source frame {self.source_frame!r}'
            )
        return CartesianPose(
            position_m=self.transform_position(pose_source.position_m),
            quaternion_xyzw=self.transform_orientation(
                pose_source.quaternion_xyzw
            ),
            frame_id=self.target_frame,
        )

    def inverse(self) -> 'RigidTransform':
        """Return ``T_ST``, the exact inverse mapping."""
        x, y, z, w = self.rotation_xyzw
        inverse_rotation = (-x, -y, -z, w)
        inverse_matrix = _rotation_matrix(inverse_rotation)
        inverse_translation = inverse_matrix @ (
            -np.asarray(self.translation_m, dtype=np.float64)
        )
        return RigidTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            translation_m=tuple(float(value) for value in inverse_translation),
            rotation_xyzw=inverse_rotation,
        )


_CONFIG_KEYS = {
    'version',
    'calibrated',
    'source_frame',
    'target_frame',
    'translation_m',
    'rotation_xyzw',
}


def load_transform_config(
    path: str | Path, *, require_calibrated: bool = True
) -> RigidTransform:
    """Load a strict version-1 transform YAML file."""
    config_path = Path(path).expanduser()
    try:
        with config_path.open('r', encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise TransformConfigError(
            f'cannot read transform config {config_path}: {error}'
        ) from error
    if not isinstance(raw, Mapping):
        raise TransformConfigError('transform config must be a mapping')
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    missing = sorted(_CONFIG_KEYS - set(raw))
    if unknown:
        raise TransformConfigError(
            'unknown transform config keys: ' + ', '.join(unknown)
        )
    if missing:
        raise TransformConfigError(
            'missing transform config keys: ' + ', '.join(missing)
        )
    if raw['version'] != 1:
        raise TransformConfigError('transform config version must be 1')
    if type(raw['calibrated']) is not bool:
        raise TransformConfigError('calibrated must be a boolean')
    if require_calibrated and not raw['calibrated']:
        raise TransformConfigError(
            'transform is not calibrated; measured T_TS values are required'
        )
    try:
        return RigidTransform(
            source_frame=raw['source_frame'],
            target_frame=raw['target_frame'],
            translation_m=raw['translation_m'],
            rotation_xyzw=raw['rotation_xyzw'],
        )
    except (KeyError, TypeError, TransformError) as error:
        raise TransformConfigError(f'invalid transform config: {error}') from error
