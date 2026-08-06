"""Immutable canonical state values with strict unit and shape validation."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


def _finite_tuple(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class NamedArray:
    """A named numeric array whose shape is preserved across transports."""

    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'shape', tuple(int(size) for size in self.shape))
        object.__setattr__(self, 'values', _finite_tuple(self.values))
        if not self.name:
            raise ValueError('named array name must not be empty')
        if any(size < 0 for size in self.shape):
            raise ValueError(f'{self.name}: shape entries must be non-negative')
        expected = math.prod(self.shape) if self.shape else 1
        if expected != len(self.values):
            raise ValueError(
                f'{self.name}: shape {self.shape} requires {expected} values, '
                f'got {len(self.values)}'
            )
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError(f'{self.name}: values must be finite')

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Sequence[float],
        shape: Sequence[int] | None = None,
    ) -> 'NamedArray':
        """Create an entry, defaulting one-dimensional data to its length."""
        flat = _finite_tuple(values)
        resolved_shape = tuple(shape) if shape is not None else (len(flat),)
        return cls(name=name, shape=resolved_shape, values=flat)


@dataclass(frozen=True)
class CanonicalState:
    """Robot-independent state in SI units and an explicit reference frame."""

    ee_position_m: tuple[float, float, float]
    ee_quaternion_xyzw: tuple[float, float, float, float]
    gripper_width_m: float
    object_schema: str
    object_state: tuple[NamedArray, ...]
    frame_id: str
    timestamp_ns: int
    source_robot: str
    fresh: bool
    valid: bool
    invalid_reasons: tuple[str, ...] = ()

    def object_entry(self, name: str) -> NamedArray | None:
        """Return a named object entry without relying on array order."""
        return next((entry for entry in self.object_state if entry.name == name), None)


def _normalized_quaternion(
    quaternion: Sequence[float],
    reasons: list[str],
    norm_tolerance: float,
) -> tuple[float, float, float, float]:
    values = _finite_tuple(quaternion)
    if len(values) != 4:
        reasons.append(f'ee_quaternion_xyzw_length:{len(values)}')
        return (0.0, 0.0, 0.0, 1.0)
    if not all(math.isfinite(value) for value in values):
        reasons.append('ee_quaternion_xyzw_nonfinite')
        return (0.0, 0.0, 0.0, 1.0)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12 or abs(norm - 1.0) > norm_tolerance:
        reasons.append(f'ee_quaternion_xyzw_norm:{norm:.9g}')
        return (0.0, 0.0, 0.0, 1.0)
    normalized = tuple(value / norm for value in values)
    if normalized[3] < 0.0:
        normalized = tuple(-value for value in normalized)
    return normalized  # type: ignore[return-value]


def make_canonical_state(
    *,
    ee_position_m: Sequence[float],
    ee_quaternion_xyzw: Sequence[float],
    gripper_width_m: float,
    object_schema: str,
    object_state: Iterable[NamedArray],
    frame_id: str,
    timestamp_ns: int,
    source_robot: str,
    fresh: bool,
    invalid_reasons: Iterable[str] = (),
    quaternion_norm_tolerance: float = 1e-3,
) -> CanonicalState:
    """Validate raw fields and return a deterministic canonical state."""
    reasons = list(dict.fromkeys(str(reason) for reason in invalid_reasons))
    position = _finite_tuple(ee_position_m)
    if len(position) != 3:
        reasons.append(f'ee_position_m_length:{len(position)}')
        position = (0.0, 0.0, 0.0)
    elif not all(math.isfinite(value) for value in position):
        reasons.append('ee_position_m_nonfinite')
        position = (0.0, 0.0, 0.0)

    quaternion = _normalized_quaternion(
        ee_quaternion_xyzw, reasons, quaternion_norm_tolerance
    )
    width = float(gripper_width_m)
    if not math.isfinite(width):
        reasons.append('gripper_width_m_nonfinite')
        width = 0.0
    elif width < 0.0:
        reasons.append(f'gripper_width_m_negative:{width:.9g}')
        width = 0.0

    entries = tuple(object_state)
    names = [entry.name for entry in entries]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        reasons.append('object_state_duplicate_names:' + ','.join(duplicates))
    if entries and not object_schema:
        reasons.append('object_schema_missing')
    if not frame_id:
        reasons.append('frame_id_missing')
    if int(timestamp_ns) <= 0:
        reasons.append('timestamp_missing')
    if not source_robot:
        reasons.append('source_robot_missing')
    if not fresh:
        reasons.append('state_stale')

    final_reasons = tuple(dict.fromkeys(reasons))
    return CanonicalState(
        ee_position_m=position,  # type: ignore[arg-type]
        ee_quaternion_xyzw=quaternion,
        gripper_width_m=width,
        object_schema=object_schema,
        object_state=entries,
        frame_id=frame_id,
        timestamp_ns=int(timestamp_ns),
        source_robot=source_robot,
        fresh=bool(fresh),
        valid=not final_reasons,
        invalid_reasons=final_reasons,
    )
