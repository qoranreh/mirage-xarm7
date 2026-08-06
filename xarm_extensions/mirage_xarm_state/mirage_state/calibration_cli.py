"""Estimate and physically validate a measured source-to-target transform."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import yaml

from mirage_state.calibration import (
    CalibrationError,
    CalibrationResult,
    estimate_rigid_transform,
    load_calibration_dataset,
)
from mirage_state.transform_cli import validate_transform
from mirage_state.transforms import TransformError


def _transform_document(result: CalibrationResult, calibrated: bool) -> dict:
    transform = result.transform
    return {
        'version': 1,
        'calibrated': calibrated,
        'source_frame': transform.source_frame,
        'target_frame': transform.target_frame,
        'translation_m': list(transform.translation_m),
        'rotation_xyzw': list(transform.rotation_xyzw),
    }


def _fixture_document(dataset, result: CalibrationResult) -> dict:
    transform = result.transform
    pose_cases = []
    for landmark in dataset.landmarks:
        pose_cases.append(
            {
                'name': landmark.name,
                'source': {
                    'frame_id': dataset.source_frame,
                    'position_m': list(landmark.source_position_m),
                    'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0],
                },
                'expected_target': {
                    'frame_id': dataset.target_frame,
                    'position_m': list(
                        result.target_positions_m[landmark.name]
                    ),
                    'quaternion_xyzw': list(transform.rotation_xyzw),
                },
            }
        )

    by_name = {item.name: item for item in dataset.landmarks}
    origin = by_name.get('origin')
    if origin is None:
        raise CalibrationError("profile must include an 'origin' landmark")
    axis_cases = []
    for name, source_name in (
        ('source_x', 'x_100mm'),
        ('source_y', 'y_100mm'),
        ('source_z', 'z_80mm'),
    ):
        endpoint = by_name.get(source_name)
        if endpoint is None:
            raise CalibrationError(f'profile must include {source_name!r}')
        source_vector = [
            endpoint.source_position_m[index]
            - origin.source_position_m[index]
            for index in range(3)
        ]
        target_vector = [
            result.target_positions_m[endpoint.name][index]
            - result.target_positions_m[origin.name][index]
            for index in range(3)
        ]
        axis_cases.append(
            {
                'name': name,
                'source_vector': source_vector,
                'expected_target_vector': target_vector,
            }
        )
    return {
        'version': 2,
        'physical_position_tolerance_m': result.position_tolerance_m,
        'physical_orientation_tolerance_rad': result.orientation_tolerance_rad,
        'round_trip_position_tolerance_m': 1.0e-9,
        'round_trip_orientation_tolerance_rad': 1.0e-8,
        'vector_angular_tolerance_rad': result.orientation_tolerance_rad,
        'pose_cases': pose_cases,
        'axis_cases': axis_cases,
    }


def _atomic_text(path: str | Path, content: str) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent, prefix=f'.{output.name}.', suffix='.tmp'
    )
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _yaml(value: dict) -> str:
    return yaml.safe_dump(value, sort_keys=False)


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Estimate T_TS from repeated landmark measurements. Output files '
            'are created only at new paths; calibrated is true only when all '
            'physical gates pass.'
        )
    )
    parser.add_argument('--correspondences', required=True)
    parser.add_argument('--output', required=True, help='New transform YAML')
    parser.add_argument('--fixtures', required=True, help='New v2 fixture YAML')
    parser.add_argument('--report', required=True, help='New JSON report')
    parser.add_argument('--mode', choices=['smoke', 'strict'], default='smoke')
    return parser.parse_args(args)


def _ensure_new_outputs(paths) -> None:
    existing = [str(Path(path).expanduser()) for path in paths if Path(path).expanduser().exists()]
    if existing:
        raise CalibrationError('refusing to overwrite: ' + ', '.join(existing))


def run(args=None) -> int:
    parsed = _parse_args(args)
    try:
        _ensure_new_outputs((parsed.output, parsed.fixtures, parsed.report))
        dataset = load_calibration_dataset(parsed.correspondences)
        result = estimate_rigid_transform(dataset, mode=parsed.mode)
        fixture = _fixture_document(dataset, result)
    except CalibrationError as error:
        print(json.dumps({'passed': False, 'error': str(error)}, indent=2))
        return 2

    validation = None
    calibrated = result.passed
    if calibrated:
        try:
            with tempfile.TemporaryDirectory(prefix='mirage-calibration-') as temp:
                config_path = Path(temp) / 'transform.yaml'
                fixture_path = Path(temp) / 'fixtures.yaml'
                config_path.write_text(
                    _yaml(_transform_document(result, True)), encoding='utf-8'
                )
                fixture_path.write_text(_yaml(fixture), encoding='utf-8')
                validation = validate_transform(config_path, fixture_path)
            calibrated = bool(validation['passed'])
        except TransformError as error:
            validation = {'passed': False, 'error': str(error)}
            calibrated = False

    report = {
        'passed': calibrated,
        'mode': parsed.mode,
        'calibrated': calibrated,
        'source_frame': result.transform.source_frame,
        'target_frame': result.transform.target_frame,
        'translation_m': list(result.transform.translation_m),
        'rotation_xyzw': list(result.transform.rotation_xyzw),
        'metrics': {
            'fit_rms_error_m': result.fit_rms_error_m,
            'fit_max_error_m': result.fit_max_error_m,
            'holdout_max_error_m': result.holdout_max_error_m,
            'orientation_max_error_rad': result.orientation_max_error_rad,
        },
        'tolerances': {
            'physical_position_m': result.position_tolerance_m,
            'physical_orientation_rad': result.orientation_tolerance_rad,
            'round_trip_position_m': 1.0e-9,
            'round_trip_orientation_rad': 1.0e-8,
        },
        'validation': validation,
    }
    try:
        _atomic_text(
            parsed.output,
            _yaml(_transform_document(result, calibrated)),
        )
        _atomic_text(parsed.fixtures, _yaml(fixture))
        _atomic_text(
            parsed.report,
            json.dumps(report, indent=2, sort_keys=True) + '\n',
        )
    except OSError as error:
        print(json.dumps({'passed': False, 'error': str(error)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if calibrated else 1


def main(args=None) -> None:
    raise SystemExit(run(args))


if __name__ == '__main__':
    main(sys.argv[1:])
