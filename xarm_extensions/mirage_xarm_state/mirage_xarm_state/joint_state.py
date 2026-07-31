"""Validation and name-based normalization for JointState messages."""

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from sensor_msgs.msg import JointState


@dataclass(frozen=True)
class JointStateView:
    """A JointState represented by name instead of array offset."""

    position: Dict[str, float]
    velocity: Dict[str, float]
    effort: Dict[str, float]
    issues: Tuple[str, ...]


@dataclass(frozen=True)
class NormalizationResult:
    """The normalized message and any reason that made it invalid."""

    message: JointState | None
    missing_names: Tuple[str, ...]
    nonfinite_fields: Tuple[str, ...]
    issues: Tuple[str, ...]

    @property
    def valid(self) -> bool:
        return self.message is not None


def _field_map(
    names: Sequence[str], values: Sequence[float], field_name: str
) -> Tuple[Dict[str, float], List[str]]:
    if not values:
        return {}, []
    if len(values) != len(names):
        return {}, [
            f'{field_name} length {len(values)} does not match name length '
            f'{len(names)}'
        ]
    return dict(zip(names, values)), []


def parse_joint_state(message: JointState) -> JointStateView:
    """Parse a JointState while enforcing ROS array-length invariants."""
    issues: List[str] = []
    names = list(message.name)
    if not names:
        issues.append('name array is empty')

    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        issues.append('duplicate joint names: ' + ', '.join(duplicate_names))

    position, field_issues = _field_map(names, message.position, 'position')
    issues.extend(field_issues)
    velocity, field_issues = _field_map(names, message.velocity, 'velocity')
    issues.extend(field_issues)
    effort, field_issues = _field_map(names, message.effort, 'effort')
    issues.extend(field_issues)

    return JointStateView(position, velocity, effort, tuple(issues))


def normalize_joint_state(
    source: JointState,
    view: JointStateView,
    ordered_names: Iterable[str],
    *,
    require_position: bool = True,
) -> NormalizationResult:
    """Build a deterministic JointState ordered by ``ordered_names``."""
    names = tuple(ordered_names)
    missing = tuple(name for name in names if name not in view.position)
    if not require_position:
        missing = ()

    nonfinite: List[str] = []
    for field_name, values in (
        ('position', view.position),
        ('velocity', view.velocity),
        ('effort', view.effort),
    ):
        for name in names:
            if name in values and not math.isfinite(values[name]):
                nonfinite.append(f'{field_name}:{name}')

    fatal_issues = bool(view.issues) or bool(missing) or bool(nonfinite)
    if fatal_issues:
        return NormalizationResult(
            None, missing, tuple(nonfinite), view.issues
        )

    normalized = JointState()
    normalized.header = source.header
    normalized.name = list(names)
    if require_position:
        normalized.position = [view.position[name] for name in names]
    elif all(name in view.position for name in names):
        normalized.position = [view.position[name] for name in names]

    if all(name in view.velocity for name in names):
        normalized.velocity = [view.velocity[name] for name in names]
    if all(name in view.effort for name in names):
        normalized.effort = [view.effort[name] for name in names]

    return NormalizationResult(normalized, (), (), ())
