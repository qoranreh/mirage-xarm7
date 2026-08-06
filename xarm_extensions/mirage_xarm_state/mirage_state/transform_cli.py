"""Validate a calibrated source-to-target transform against known fixtures."""

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np
import yaml

from mirage_state.transforms import (
    CartesianPose,
    TransformError,
    load_transform_config,
    quaternion_angular_distance,
)


class FixtureError(TransformError):
    """Raised when a transform validation fixture is malformed."""


_FIXTURE_V1_KEYS = {
    'version',
    'position_tolerance_m',
    'orientation_tolerance_rad',
    'vector_tolerance',
    'pose_cases',
    'axis_cases',
}
_FIXTURE_V2_KEYS = {
    'version',
    'physical_position_tolerance_m',
    'physical_orientation_tolerance_rad',
    'round_trip_position_tolerance_m',
    'round_trip_orientation_tolerance_rad',
    'vector_angular_tolerance_rad',
    'pose_cases',
    'axis_cases',
}
_POSE_KEYS = {'frame_id', 'position_m', 'quaternion_xyzw'}
_POSE_CASE_KEYS = {'name', 'source', 'expected_target'}
_AXIS_CASE_KEYS = {'name', 'source_vector', 'expected_target_vector'}


def _strict_keys(
    value: Mapping[str, object], expected: set[str], context: str
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise FixtureError(f'{context} has unknown keys: ' + ', '.join(unknown))
    if missing:
        raise FixtureError(f'{context} is missing keys: ' + ', '.join(missing))


def _finite_nonnegative(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FixtureError(f'{name} must be numeric') from error
    if not math.isfinite(result) or result < 0.0:
        raise FixtureError(f'{name} must be finite and non-negative')
    return result


def _pose(value: object, context: str) -> CartesianPose:
    if not isinstance(value, Mapping):
        raise FixtureError(f'{context} must be a mapping')
    _strict_keys(value, _POSE_KEYS, context)
    try:
        return CartesianPose(
            frame_id=value['frame_id'],
            position_m=value['position_m'],
            quaternion_xyzw=value['quaternion_xyzw'],
        )
    except (KeyError, TypeError, TransformError) as error:
        raise FixtureError(f'{context} is invalid: {error}') from error


def _vector(value: object, context: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FixtureError(f'{context} must contain three numeric values')
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise FixtureError(f'{context} must contain numeric values') from error
    if len(vector) != 3 or not all(math.isfinite(item) for item in vector):
        raise FixtureError(f'{context} must contain three finite values')
    return vector  # type: ignore[return-value]


def _load_fixtures(path: str | Path) -> dict[str, object]:
    fixture_path = Path(path).expanduser()
    try:
        with fixture_path.open('r', encoding='utf-8') as stream:
            raw = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as error:
        raise FixtureError(f'cannot read fixture {fixture_path}: {error}') from error
    if not isinstance(raw, Mapping):
        raise FixtureError('validation fixture must be a mapping')
    version = raw.get('version')
    if version == 1:
        _strict_keys(raw, _FIXTURE_V1_KEYS, 'validation fixture')
    elif version == 2:
        _strict_keys(raw, _FIXTURE_V2_KEYS, 'validation fixture')
    else:
        raise FixtureError('validation fixture version must be 1 or 2')
    pose_cases = raw['pose_cases']
    axis_cases = raw['axis_cases']
    if not isinstance(pose_cases, list) or not pose_cases:
        raise FixtureError('pose_cases must be a non-empty list')
    if not isinstance(axis_cases, list) or not axis_cases:
        raise FixtureError('axis_cases must be a non-empty list')
    if version == 1:
        position = _finite_nonnegative(
            raw['position_tolerance_m'], 'position_tolerance_m'
        )
        orientation = _finite_nonnegative(
            raw['orientation_tolerance_rad'], 'orientation_tolerance_rad'
        )
        return {
            'version': version,
            'physical_position_tolerance_m': position,
            'physical_orientation_tolerance_rad': orientation,
            'round_trip_position_tolerance_m': position,
            'round_trip_orientation_tolerance_rad': orientation,
            'vector_tolerance': _finite_nonnegative(
                raw['vector_tolerance'], 'vector_tolerance'
            ),
            'axis_metric': 'euclidean',
            'pose_cases': pose_cases,
            'axis_cases': axis_cases,
        }
    return {
        'version': version,
        'physical_position_tolerance_m': _finite_nonnegative(
            raw['physical_position_tolerance_m'],
            'physical_position_tolerance_m',
        ),
        'physical_orientation_tolerance_rad': _finite_nonnegative(
            raw['physical_orientation_tolerance_rad'],
            'physical_orientation_tolerance_rad',
        ),
        'round_trip_position_tolerance_m': _finite_nonnegative(
            raw['round_trip_position_tolerance_m'],
            'round_trip_position_tolerance_m',
        ),
        'round_trip_orientation_tolerance_rad': _finite_nonnegative(
            raw['round_trip_orientation_tolerance_rad'],
            'round_trip_orientation_tolerance_rad',
        ),
        'vector_tolerance': _finite_nonnegative(
            raw['vector_angular_tolerance_rad'],
            'vector_angular_tolerance_rad',
        ),
        'axis_metric': 'angular',
        'pose_cases': pose_cases,
        'axis_cases': axis_cases,
    }


def validate_transform(config_path: str | Path, fixture_path: str | Path) -> dict:
    """Validate one calibrated transform and return a JSON-compatible report."""
    transform = load_transform_config(config_path, require_calibrated=True)
    inverse = transform.inverse()
    fixtures = _load_fixtures(fixture_path)
    physical_position_tolerance = fixtures[
        'physical_position_tolerance_m'
    ]
    physical_orientation_tolerance = fixtures[
        'physical_orientation_tolerance_rad'
    ]
    round_trip_position_tolerance = fixtures[
        'round_trip_position_tolerance_m'
    ]
    round_trip_orientation_tolerance = fixtures[
        'round_trip_orientation_tolerance_rad'
    ]
    vector_tolerance = fixtures['vector_tolerance']

    pose_results = []
    names: set[str] = set()
    for index, raw_case in enumerate(fixtures['pose_cases']):
        context = f'pose_cases[{index}]'
        if not isinstance(raw_case, Mapping):
            raise FixtureError(f'{context} must be a mapping')
        _strict_keys(raw_case, _POSE_CASE_KEYS, context)
        name = str(raw_case['name']).strip()
        if not name or name in names:
            raise FixtureError(f'{context} name must be non-empty and unique')
        names.add(name)
        source = _pose(raw_case['source'], f'{context}.source')
        expected = _pose(
            raw_case['expected_target'], f'{context}.expected_target'
        )
        if expected.frame_id != transform.target_frame:
            raise FixtureError(
                f'{context} expected frame {expected.frame_id!r} does not '
                f'match {transform.target_frame!r}'
            )
        actual = transform.transform_pose(source)
        round_trip = inverse.transform_pose(actual)
        position_error = float(
            np.linalg.norm(
                np.asarray(actual.position_m) - np.asarray(expected.position_m)
            )
        )
        orientation_error = quaternion_angular_distance(
            actual.quaternion_xyzw, expected.quaternion_xyzw
        )
        round_trip_position_error = float(
            np.linalg.norm(
                np.asarray(round_trip.position_m)
                - np.asarray(source.position_m)
            )
        )
        round_trip_orientation_error = quaternion_angular_distance(
            round_trip.quaternion_xyzw, source.quaternion_xyzw
        )
        passed = (
            position_error <= physical_position_tolerance
            and orientation_error <= physical_orientation_tolerance
            and round_trip_position_error <= round_trip_position_tolerance
            and round_trip_orientation_error
            <= round_trip_orientation_tolerance
        )
        pose_results.append(
            {
                'name': name,
                'position_error_m': position_error,
                'orientation_error_rad': orientation_error,
                'round_trip_position_error_m': round_trip_position_error,
                'round_trip_orientation_error_rad': (
                    round_trip_orientation_error
                ),
                'passed': passed,
            }
        )

    axis_results = []
    for index, raw_case in enumerate(fixtures['axis_cases']):
        context = f'axis_cases[{index}]'
        if not isinstance(raw_case, Mapping):
            raise FixtureError(f'{context} must be a mapping')
        _strict_keys(raw_case, _AXIS_CASE_KEYS, context)
        name = str(raw_case['name']).strip()
        if not name or name in names:
            raise FixtureError(f'{context} name must be non-empty and unique')
        names.add(name)
        source = _vector(raw_case['source_vector'], f'{context}.source_vector')
        expected = _vector(
            raw_case['expected_target_vector'],
            f'{context}.expected_target_vector',
        )
        actual = transform.transform_vector(source)
        if fixtures['axis_metric'] == 'angular':
            actual_norm = float(np.linalg.norm(actual))
            expected_norm = float(np.linalg.norm(expected))
            if actual_norm <= 1e-12 or expected_norm <= 1e-12:
                raise FixtureError(f'{context} vectors must be non-zero')
            cosine = float(
                np.clip(
                    np.dot(actual, expected) / (actual_norm * expected_norm),
                    -1.0,
                    1.0,
                )
            )
            error = math.acos(cosine)
            error_key = 'angular_error_rad'
        else:
            error = float(
                np.linalg.norm(np.asarray(actual) - np.asarray(expected))
            )
            error_key = 'vector_error'
        axis_results.append(
            {'name': name, error_key: error, 'passed': error <= vector_tolerance}
        )

    passed = all(result['passed'] for result in pose_results + axis_results)
    return {
        'passed': passed,
        'calibrated': True,
        'source_frame': transform.source_frame,
        'target_frame': transform.target_frame,
        'translation_m': list(transform.translation_m),
        'rotation_xyzw': list(transform.rotation_xyzw),
        'tolerances': {
            'physical_position_m': physical_position_tolerance,
            'physical_orientation_rad': physical_orientation_tolerance,
            'round_trip_position_m': round_trip_position_tolerance,
            'round_trip_orientation_rad': round_trip_orientation_tolerance,
            (
                'vector_angular_rad'
                if fixtures['axis_metric'] == 'angular'
                else 'vector'
            ): vector_tolerance,
        },
        'pose_cases': pose_results,
        'axis_cases': axis_results,
    }


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Validate calibrated T_TS and T_ST pose transforms. Uncalibrated '
            'configurations are rejected.'
        )
    )
    parser.add_argument('--config', required=True, help='T_TS YAML configuration')
    parser.add_argument(
        '--fixtures', required=True, help='Known pose and axis validation YAML'
    )
    parser.add_argument('--report', help='Optional JSON report output path')
    return parser.parse_args(args)


def run(args=None) -> int:
    """Run the validation command and return its process status."""
    parsed = _parse_args(args)
    try:
        report = validate_transform(parsed.config, parsed.fixtures)
    except TransformError as error:
        report = {'passed': False, 'error': str(error)}
        status = 2
    else:
        status = 0 if report['passed'] else 1
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if parsed.report:
        report_path = Path(parsed.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + '\n', encoding='utf-8')
    return status


def main(args=None) -> None:
    """Console entry point."""
    raise SystemExit(run(args))


if __name__ == '__main__':
    main(sys.argv[1:])
