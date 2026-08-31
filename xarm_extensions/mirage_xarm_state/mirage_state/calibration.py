"""Pure-data tools for measuring and estimating the physical ``T_TS``."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
import yaml

from mirage_state.transforms import (
    CartesianPose,
    RigidTransform,
    TransformError,
    quaternion_angular_distance,
)


class CalibrationError(TransformError):
    """Raised when calibration evidence is incomplete or unsafe to use."""


def _vector(
    value: object, length: int, name: str
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CalibrationError(f'{name} must contain {length} values')
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f'{name} must be numeric') from error
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise CalibrationError(f'{name} must contain {length} finite values')
    return result


def _nonnegative(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CalibrationError(f'{name} must be numeric') from error
    if not math.isfinite(result) or result < 0.0:
        raise CalibrationError(f'{name} must be finite and non-negative')
    return result


def _strict_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise CalibrationError(f'{context} has unknown keys: ' + ', '.join(unknown))
    if missing:
        raise CalibrationError(f'{context} is missing keys: ' + ', '.join(missing))


@dataclass(frozen=True)
class CaptureEvidence:
    """One stable target-side sample burst for a landmark."""

    position_m: tuple[float, float, float]
    position_std_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    orientation_spread_rad: float
    sample_count: int
    captured_at: str
    max_sample_age_sec: float

    def __post_init__(self) -> None:
        pose = CartesianPose(
            position_m=self.position_m,
            quaternion_xyzw=self.orientation_xyzw,
            frame_id='capture',
        )
        object.__setattr__(self, 'position_m', pose.position_m)
        object.__setattr__(self, 'orientation_xyzw', pose.quaternion_xyzw)
        std = _vector(self.position_std_m, 3, 'position_std_m')
        if any(item < 0.0 for item in std):
            raise CalibrationError('position_std_m must be non-negative')
        object.__setattr__(self, 'position_std_m', std)
        object.__setattr__(
            self,
            'orientation_spread_rad',
            _nonnegative(self.orientation_spread_rad, 'orientation_spread_rad'),
        )
        if type(self.sample_count) is not int or self.sample_count < 2:
            raise CalibrationError('sample_count must be an integer of at least 2')
        if not isinstance(self.captured_at, str) or not self.captured_at.strip():
            raise CalibrationError('captured_at must not be empty')
        object.__setattr__(
            self,
            'max_sample_age_sec',
            _nonnegative(self.max_sample_age_sec, 'max_sample_age_sec'),
        )


@dataclass(frozen=True)
class LandmarkCorrespondence:
    """Source landmark and repeated target measurements of the same point."""

    name: str
    role: str
    source_position_m: tuple[float, float, float]
    target_captures: tuple[CaptureEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise CalibrationError('landmark name must not be empty')
        if self.role not in {'fit', 'holdout'}:
            raise CalibrationError('landmark role must be fit or holdout')
        object.__setattr__(self, 'name', self.name.strip())
        object.__setattr__(
            self,
            'source_position_m',
            _vector(self.source_position_m, 3, 'source_position_m'),
        )
        object.__setattr__(self, 'target_captures', tuple(self.target_captures))


@dataclass(frozen=True)
class CalibrationThresholds:
    """Capture gates and physical acceptance limits."""

    smoke_position_m: float = 0.005
    smoke_orientation_rad: float = math.radians(1.0)
    strict_position_m: float = 0.002
    strict_orientation_rad: float = math.radians(0.5)
    capture_position_std_m: float = 0.0005
    capture_orientation_spread_rad: float = math.radians(0.25)

    def __post_init__(self) -> None:
        for name in (
            'smoke_position_m',
            'smoke_orientation_rad',
            'strict_position_m',
            'strict_orientation_rad',
            'capture_position_std_m',
            'capture_orientation_spread_rad',
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))

    def physical(self, mode: str) -> tuple[float, float]:
        if mode == 'smoke':
            return self.smoke_position_m, self.smoke_orientation_rad
        if mode == 'strict':
            return self.strict_position_m, self.strict_orientation_rad
        raise CalibrationError('mode must be smoke or strict')


@dataclass(frozen=True)
class CalibrationDataset:
    """Versioned source/target landmark measurement contract."""

    task: str
    profile: str
    source_frame: str
    target_frame: str
    minimum_captures_per_landmark: int
    thresholds: CalibrationThresholds
    landmarks: tuple[LandmarkCorrespondence, ...]

    def __post_init__(self) -> None:
        for name in ('task', 'profile', 'source_frame', 'target_frame'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CalibrationError(f'{name} must not be empty')
            object.__setattr__(self, name, value.strip())
        if self.source_frame == self.target_frame:
            raise CalibrationError('source_frame and target_frame must differ')
        if (
            type(self.minimum_captures_per_landmark) is not int
            or self.minimum_captures_per_landmark < 1
        ):
            raise CalibrationError(
                'minimum_captures_per_landmark must be a positive integer'
            )
        landmarks = tuple(self.landmarks)
        if len(landmarks) < 5:
            raise CalibrationError('at least four fit landmarks and one holdout are required')
        names = [landmark.name for landmark in landmarks]
        if len(names) != len(set(names)):
            raise CalibrationError('landmark names must be unique')
        if sum(item.role == 'fit' for item in landmarks) < 4:
            raise CalibrationError('at least four fit landmarks are required')
        if not any(item.role == 'holdout' for item in landmarks):
            raise CalibrationError('at least one holdout landmark is required')
        source = np.asarray([item.source_position_m for item in landmarks])
        _reject_near_duplicates(source, 'source landmarks')
        object.__setattr__(self, 'landmarks', landmarks)


@dataclass(frozen=True)
class CalibrationResult:
    """Estimated transform and independent physical validation metrics."""

    transform: RigidTransform
    target_positions_m: Mapping[str, tuple[float, float, float]]
    fit_rms_error_m: float
    fit_max_error_m: float
    holdout_max_error_m: float
    orientation_max_error_rad: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    passed: bool


@dataclass(frozen=True)
class ToolOffset:
    """Measured rigid offset from ``link_eef`` to a pointer tip."""

    parent_frame: str
    tool_frame: str
    translation_m: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        pose = CartesianPose(
            position_m=self.translation_m,
            quaternion_xyzw=self.rotation_xyzw,
            frame_id=self.parent_frame,
        )
        if not isinstance(self.tool_frame, str) or not self.tool_frame.strip():
            raise CalibrationError('tool_frame must not be empty')
        object.__setattr__(self, 'parent_frame', pose.frame_id)
        object.__setattr__(self, 'translation_m', pose.position_m)
        object.__setattr__(self, 'rotation_xyzw', pose.quaternion_xyzw)
        object.__setattr__(self, 'tool_frame', self.tool_frame.strip())


_DATASET_KEYS = {
    'version',
    'task',
    'profile',
    'source_frame',
    'target_frame',
    'minimum_captures_per_landmark',
    'thresholds',
    'landmarks',
}
_THRESHOLD_FIELDS = (
    'smoke_position_m',
    'smoke_orientation_rad',
    'strict_position_m',
    'strict_orientation_rad',
    'capture_position_std_m',
    'capture_orientation_spread_rad',
)
_THRESHOLD_KEYS = set(_THRESHOLD_FIELDS)
_LANDMARK_KEYS = {'name', 'role', 'source_position_m', 'target_captures'}
_CAPTURE_KEYS = {
    'position_m',
    'position_std_m',
    'orientation_xyzw',
    'orientation_spread_rad',
    'sample_count',
    'captured_at',
    'max_sample_age_sec',
}


def _load_yaml(path: str | Path, description: str) -> Mapping[str, object]:
    file_path = Path(path).expanduser()
    try:
        with file_path.open('r', encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise CalibrationError(f'cannot read {description} {file_path}: {error}') from error
    if not isinstance(raw, Mapping):
        raise CalibrationError(f'{description} must be a mapping')
    return raw


def load_calibration_dataset(path: str | Path) -> CalibrationDataset:
    """Load a strict version-1 correspondence YAML document."""
    raw = _load_yaml(path, 'calibration dataset')
    _strict_keys(raw, _DATASET_KEYS, 'calibration dataset')
    if raw['version'] != 1:
        raise CalibrationError('calibration dataset version must be 1')
    threshold_raw = raw['thresholds']
    if not isinstance(threshold_raw, Mapping):
        raise CalibrationError('thresholds must be a mapping')
    _strict_keys(threshold_raw, _THRESHOLD_KEYS, 'thresholds')
    landmarks_raw = raw['landmarks']
    if not isinstance(landmarks_raw, list):
        raise CalibrationError('landmarks must be a list')
    landmarks = []
    for index, landmark_raw in enumerate(landmarks_raw):
        context = f'landmarks[{index}]'
        if not isinstance(landmark_raw, Mapping):
            raise CalibrationError(f'{context} must be a mapping')
        _strict_keys(landmark_raw, _LANDMARK_KEYS, context)
        captures_raw = landmark_raw['target_captures']
        if not isinstance(captures_raw, list):
            raise CalibrationError(f'{context}.target_captures must be a list')
        captures = []
        for capture_index, capture_raw in enumerate(captures_raw):
            capture_context = f'{context}.target_captures[{capture_index}]'
            if not isinstance(capture_raw, Mapping):
                raise CalibrationError(f'{capture_context} must be a mapping')
            _strict_keys(capture_raw, _CAPTURE_KEYS, capture_context)
            captures.append(CaptureEvidence(**capture_raw))
        landmarks.append(
            LandmarkCorrespondence(
                name=landmark_raw['name'],
                role=landmark_raw['role'],
                source_position_m=landmark_raw['source_position_m'],
                target_captures=tuple(captures),
            )
        )
    return CalibrationDataset(
        task=raw['task'],
        profile=raw['profile'],
        source_frame=raw['source_frame'],
        target_frame=raw['target_frame'],
        minimum_captures_per_landmark=raw['minimum_captures_per_landmark'],
        thresholds=CalibrationThresholds(**threshold_raw),
        landmarks=tuple(landmarks),
    )


def calibration_dataset_dict(dataset: CalibrationDataset) -> dict[str, object]:
    """Convert a dataset into stable YAML-compatible primitives."""
    thresholds = dataset.thresholds
    return {
        'version': 1,
        'task': dataset.task,
        'profile': dataset.profile,
        'source_frame': dataset.source_frame,
        'target_frame': dataset.target_frame,
        'minimum_captures_per_landmark': dataset.minimum_captures_per_landmark,
        'thresholds': {
            name: getattr(thresholds, name) for name in _THRESHOLD_FIELDS
        },
        'landmarks': [
            {
                'name': item.name,
                'role': item.role,
                'source_position_m': list(item.source_position_m),
                'target_captures': [
                    {
                        'position_m': list(capture.position_m),
                        'position_std_m': list(capture.position_std_m),
                        'orientation_xyzw': list(capture.orientation_xyzw),
                        'orientation_spread_rad': capture.orientation_spread_rad,
                        'sample_count': capture.sample_count,
                        'captured_at': capture.captured_at,
                        'max_sample_age_sec': capture.max_sample_age_sec,
                    }
                    for capture in item.target_captures
                ],
            }
            for item in dataset.landmarks
        ],
    }


def _atomic_yaml(path: str | Path, value: Mapping[str, object]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f'.{output.name}.', suffix='.tmp'
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            yaml.safe_dump(value, stream, sort_keys=False)
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def save_calibration_dataset(
    path: str | Path, dataset: CalibrationDataset, *, overwrite: bool = False
) -> None:
    """Atomically save a dataset, refusing accidental replacement by default."""
    output = Path(path).expanduser()
    if output.exists() and not overwrite:
        raise CalibrationError(f'output already exists: {output}')
    try:
        _atomic_yaml(output, calibration_dataset_dict(dataset))
    except OSError as error:
        raise CalibrationError(f'cannot write calibration dataset: {error}') from error


def append_capture(
    path: str | Path, landmark_name: str, capture: CaptureEvidence
) -> CalibrationDataset:
    """Append one immutable capture attempt and atomically replace the draft."""
    dataset = load_calibration_dataset(path)
    updated = []
    found = False
    for landmark in dataset.landmarks:
        if landmark.name == landmark_name:
            found = True
            landmark = LandmarkCorrespondence(
                name=landmark.name,
                role=landmark.role,
                source_position_m=landmark.source_position_m,
                target_captures=landmark.target_captures + (capture,),
            )
        updated.append(landmark)
    if not found:
        raise CalibrationError(f'unknown landmark: {landmark_name}')
    result = CalibrationDataset(
        task=dataset.task,
        profile=dataset.profile,
        source_frame=dataset.source_frame,
        target_frame=dataset.target_frame,
        minimum_captures_per_landmark=dataset.minimum_captures_per_landmark,
        thresholds=dataset.thresholds,
        landmarks=tuple(updated),
    )
    save_calibration_dataset(path, result, overwrite=True)
    return result


def load_tool_offset(path: str | Path) -> ToolOffset:
    """Load a measured, explicitly enabled pointer offset."""
    raw = _load_yaml(path, 'tool offset')
    keys = {
        'version',
        'calibrated',
        'parent_frame',
        'tool_frame',
        'translation_m',
        'rotation_xyzw',
    }
    _strict_keys(raw, keys, 'tool offset')
    if raw['version'] != 1:
        raise CalibrationError('tool offset version must be 1')
    if raw['calibrated'] is not True:
        raise CalibrationError('tool offset must be measured and calibrated')
    return ToolOffset(
        parent_frame=raw['parent_frame'],
        tool_frame=raw['tool_frame'],
        translation_m=raw['translation_m'],
        rotation_xyzw=raw['rotation_xyzw'],
    )


def tool_tip_pose(ee_pose: CartesianPose, tool: ToolOffset) -> CartesianPose:
    """Apply ``link_eef`` to pointer-tip offset to a measured EE pose."""
    if tool.parent_frame != 'link_eef':
        raise CalibrationError("tool offset parent_frame must be 'link_eef'")
    rotation = RigidTransform(
        source_frame='link_eef',
        target_frame=ee_pose.frame_id,
        translation_m=ee_pose.position_m,
        rotation_xyzw=ee_pose.quaternion_xyzw,
    )
    tip_orientation = rotation.transform_orientation(tool.rotation_xyzw)
    return CartesianPose(
        position_m=rotation.transform_position(tool.translation_m),
        quaternion_xyzw=tip_orientation,
        frame_id=ee_pose.frame_id,
    )


def mean_quaternion_xyzw(
    quaternions: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """Return the sign-invariant Markley mean of xyzw quaternions."""
    if not quaternions:
        raise CalibrationError('at least one quaternion is required')
    normalized = [
        CartesianPose((0.0, 0.0, 0.0), tuple(item), 'mean').quaternion_xyzw
        for item in quaternions
    ]
    accumulator = np.zeros((4, 4), dtype=np.float64)
    for quaternion in normalized:
        vector = np.asarray(quaternion, dtype=np.float64)
        accumulator += np.outer(vector, vector)
    _, eigenvectors = np.linalg.eigh(accumulator)
    result = eigenvectors[:, -1]
    return CartesianPose(
        (0.0, 0.0, 0.0), tuple(float(item) for item in result), 'mean'
    ).quaternion_xyzw


def capture_evidence(
    poses: Sequence[CartesianPose],
    *,
    captured_at: str | None = None,
    max_sample_age_sec: float,
) -> CaptureEvidence:
    """Reduce a stable sample burst using a robust position median."""
    if len(poses) < 2:
        raise CalibrationError('at least two poses are required for a capture')
    frame_ids = {pose.frame_id for pose in poses}
    if len(frame_ids) != 1:
        raise CalibrationError('all capture poses must use the same frame')
    positions = np.asarray([pose.position_m for pose in poses], dtype=np.float64)
    center = np.median(positions, axis=0)
    std = np.std(positions, axis=0)
    orientation = mean_quaternion_xyzw(
        [pose.quaternion_xyzw for pose in poses]
    )
    spread = max(
        quaternion_angular_distance(pose.quaternion_xyzw, orientation)
        for pose in poses
    )
    timestamp = captured_at or datetime.now(timezone.utc).isoformat()
    return CaptureEvidence(
        position_m=tuple(float(item) for item in center),
        position_std_m=tuple(float(item) for item in std),
        orientation_xyzw=orientation,
        orientation_spread_rad=spread,
        sample_count=len(poses),
        captured_at=timestamp,
        max_sample_age_sec=max_sample_age_sec,
    )


def _reject_near_duplicates(
    points: np.ndarray, description: str, minimum_separation_m: float = 0.02
) -> None:
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            separation = float(np.linalg.norm(points[first] - points[second]))
            if separation < minimum_separation_m:
                raise CalibrationError(
                    f'{description} contain points closer than '
                    f'{minimum_separation_m:.3f} m'
                )


def _matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3-by-3 rotation matrix to canonical xyzw."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = (
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        )
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quaternion = (
            0.25 * scale,
            (matrix[0, 1] + matrix[1, 0]) / scale,
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
        )
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quaternion = (
            (matrix[0, 1] + matrix[1, 0]) / scale,
            0.25 * scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
        )
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quaternion = (
            (matrix[0, 2] + matrix[2, 0]) / scale,
            (matrix[1, 2] + matrix[2, 1]) / scale,
            0.25 * scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    return CartesianPose(
        (0.0, 0.0, 0.0), tuple(float(item) for item in quaternion), 'rotation'
    ).quaternion_xyzw


def _aggregate_target(
    dataset: CalibrationDataset, landmark: LandmarkCorrespondence
) -> tuple[float, float, float]:
    if len(landmark.target_captures) < dataset.minimum_captures_per_landmark:
        raise CalibrationError(
            f'{landmark.name} has {len(landmark.target_captures)} captures; '
            f'{dataset.minimum_captures_per_landmark} required'
        )
    for index, capture in enumerate(landmark.target_captures):
        if max(capture.position_std_m) > dataset.thresholds.capture_position_std_m:
            raise CalibrationError(
                f'{landmark.name} capture {index + 1} position std exceeds gate'
            )
        if (
            capture.orientation_spread_rad
            > dataset.thresholds.capture_orientation_spread_rad
        ):
            raise CalibrationError(
                f'{landmark.name} capture {index + 1} orientation spread exceeds gate'
            )
    positions = np.asarray(
        [capture.position_m for capture in landmark.target_captures],
        dtype=np.float64,
    )
    result = np.median(positions, axis=0)
    return tuple(float(item) for item in result)


def estimate_rigid_transform(
    dataset: CalibrationDataset, *, mode: str = 'smoke'
) -> CalibrationResult:
    """Estimate ``p_T = R_TS p_S + t_TS`` using Kabsch/SVD."""
    position_tolerance, orientation_tolerance = dataset.thresholds.physical(mode)
    target_positions = {
        item.name: _aggregate_target(dataset, item) for item in dataset.landmarks
    }
    _reject_near_duplicates(
        np.asarray(list(target_positions.values()), dtype=np.float64),
        'target landmarks',
    )
    fit = [item for item in dataset.landmarks if item.role == 'fit']
    source = np.asarray([item.source_position_m for item in fit], dtype=np.float64)
    target = np.asarray([target_positions[item.name] for item in fit])
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    if np.linalg.matrix_rank(source_centered, tol=1e-8) < 3:
        raise CalibrationError('source fit landmarks are rank deficient')
    if np.linalg.matrix_rank(target_centered, tol=1e-8) < 3:
        raise CalibrationError('target fit landmarks are rank deficient')
    covariance = source_centered.T @ target_centered
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        raise CalibrationError('fit requires a reflection, not a rigid rotation')
    translation = target_center - rotation @ source_center
    transform = RigidTransform(
        source_frame=dataset.source_frame,
        target_frame=dataset.target_frame,
        translation_m=tuple(float(item) for item in translation),
        rotation_xyzw=_matrix_to_quaternion(rotation),
    )

    fit_errors = []
    for item in fit:
        predicted = np.asarray(transform.transform_position(item.source_position_m))
        observed = np.asarray(target_positions[item.name])
        fit_errors.append(float(np.linalg.norm(predicted - observed)))
    holdout_errors = []
    for item in dataset.landmarks:
        if item.role != 'holdout':
            continue
        predicted = np.asarray(transform.transform_position(item.source_position_m))
        observed = np.asarray(target_positions[item.name])
        holdout_errors.append(float(np.linalg.norm(predicted - observed)))

    angular_errors = []
    for first in range(len(fit)):
        for second in range(first + 1, len(fit)):
            source_vector = source[second] - source[first]
            observed_vector = target[second] - target[first]
            predicted_vector = rotation @ source_vector
            cosine = float(
                np.clip(
                    np.dot(predicted_vector, observed_vector)
                    / (
                        np.linalg.norm(predicted_vector)
                        * np.linalg.norm(observed_vector)
                    ),
                    -1.0,
                    1.0,
                )
            )
            angular_errors.append(math.acos(cosine))
    fit_rms = math.sqrt(float(np.mean(np.square(fit_errors))))
    fit_max = max(fit_errors)
    holdout_max = max(holdout_errors)
    orientation_max = max(angular_errors)
    passed = (
        fit_max <= position_tolerance
        and holdout_max <= position_tolerance
        and orientation_max <= orientation_tolerance
    )
    return CalibrationResult(
        transform=transform,
        target_positions_m=target_positions,
        fit_rms_error_m=fit_rms,
        fit_max_error_m=fit_max,
        holdout_max_error_m=holdout_max,
        orientation_max_error_rad=orientation_max,
        position_tolerance_m=position_tolerance,
        orientation_tolerance_rad=orientation_tolerance,
        passed=passed,
    )
