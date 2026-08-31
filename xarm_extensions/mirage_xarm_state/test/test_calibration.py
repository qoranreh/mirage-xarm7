import math

import numpy as np
import pytest

from mirage_state.calibration import (
    CalibrationDataset,
    CalibrationError,
    CalibrationThresholds,
    CaptureEvidence,
    LandmarkCorrespondence,
    estimate_rigid_transform,
    save_calibration_dataset,
)
from mirage_state.calibration_cli import run
from mirage_state.transform_cli import validate_transform
from mirage_state.transforms import RigidTransform, quaternion_angular_distance


POINTS = (
    ('origin', 'fit', (0.1, -0.25, 0.8)),
    ('x_100mm', 'fit', (0.2, -0.25, 0.8)),
    ('y_100mm', 'fit', (0.1, -0.15, 0.8)),
    ('z_80mm', 'fit', (0.1, -0.25, 0.88)),
    ('xy_100mm_holdout', 'holdout', (0.2, -0.15, 0.8)),
)


def dataset_from_target(transform, *, holdout_offset=(0.0, 0.0, 0.0)):
    generator = np.random.default_rng(11)
    landmarks = []
    for name, role, source in POINTS:
        target = np.asarray(transform.transform_position(source))
        if role == 'holdout':
            target += np.asarray(holdout_offset)
        captures = []
        for _ in range(3):
            measured = target + generator.normal(0.0, 0.00005, size=3)
            captures.append(
                CaptureEvidence(
                    position_m=tuple(measured),
                    position_std_m=(0.0001, 0.0001, 0.0001),
                    orientation_xyzw=transform.rotation_xyzw,
                    orientation_spread_rad=0.001,
                    sample_count=50,
                    captured_at='2026-08-03T00:00:00+00:00',
                    max_sample_age_sec=0.02,
                )
            )
        landmarks.append(
            LandmarkCorrespondence(name, role, source, tuple(captures))
        )
    return CalibrationDataset(
        task='PickPlaceCan',
        profile='pick_place_can_bin1',
        source_frame='panda_sim_world',
        target_frame='link_base',
        minimum_captures_per_landmark=3,
        thresholds=CalibrationThresholds(),
        landmarks=tuple(landmarks),
    )


def test_kabsch_recovers_noisy_rigid_transform_and_passes_holdout():
    expected = RigidTransform(
        source_frame='panda_sim_world',
        target_frame='link_base',
        translation_m=(0.42, -0.13, 0.07),
        rotation_xyzw=(0.1, -0.2, 0.05, math.sqrt(0.9475)),
    )

    result = estimate_rigid_transform(dataset_from_target(expected))

    assert result.passed
    np.testing.assert_allclose(
        result.transform.translation_m, expected.translation_m, atol=0.001
    )
    assert (
        quaternion_angular_distance(
            result.transform.rotation_xyzw, expected.rotation_xyzw
        )
        < math.radians(0.5)
    )
    assert result.holdout_max_error_m < 0.001


def test_independent_holdout_failure_does_not_enable_calibration():
    expected = RigidTransform(
        'panda_sim_world',
        'link_base',
        (0.2, 0.3, -0.1),
        (0.0, 0.0, 0.0, 1.0),
    )

    result = estimate_rigid_transform(
        dataset_from_target(expected, holdout_offset=(0.02, 0.0, 0.0))
    )

    assert not result.passed
    assert result.holdout_max_error_m > result.position_tolerance_m


def test_estimator_rejects_reflection():
    landmarks = []
    for name, role, source in POINTS:
        reflected = (-source[0], source[1], source[2])
        capture = CaptureEvidence(
            reflected,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            0.0,
            50,
            'now',
            0.0,
        )
        landmarks.append(
            LandmarkCorrespondence(name, role, source, (capture,) * 3)
        )
    dataset = CalibrationDataset(
        'PickPlaceCan',
        'pick_place_can_bin1',
        'panda_sim_world',
        'link_base',
        3,
        CalibrationThresholds(),
        tuple(landmarks),
    )

    with pytest.raises(CalibrationError, match='reflection'):
        estimate_rigid_transform(dataset)


def test_capture_repeatability_gate_is_enforced():
    expected = RigidTransform(
        'panda_sim_world',
        'link_base',
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    dataset = dataset_from_target(expected)
    bad = dataset.landmarks[0].target_captures[0]
    bad = CaptureEvidence(
        bad.position_m,
        (0.001, 0.0, 0.0),
        bad.orientation_xyzw,
        bad.orientation_spread_rad,
        bad.sample_count,
        bad.captured_at,
        bad.max_sample_age_sec,
    )
    first = dataset.landmarks[0]
    landmarks = (
        LandmarkCorrespondence(
            first.name,
            first.role,
            first.source_position_m,
            (bad,) + first.target_captures[1:],
        ),
    ) + dataset.landmarks[1:]
    dataset = CalibrationDataset(
        dataset.task,
        dataset.profile,
        dataset.source_frame,
        dataset.target_frame,
        dataset.minimum_captures_per_landmark,
        dataset.thresholds,
        landmarks,
    )

    with pytest.raises(CalibrationError, match='position std'):
        estimate_rigid_transform(dataset)


def test_calibration_cli_writes_enabled_config_and_v2_fixture(tmp_path):
    expected = RigidTransform(
        'panda_sim_world',
        'link_base',
        (0.2, -0.1, 0.3),
        (0.0, 0.0, math.sin(0.2), math.cos(0.2)),
    )
    correspondences = tmp_path / 'correspondences.yaml'
    config = tmp_path / 'transform.yaml'
    fixtures = tmp_path / 'fixtures.yaml'
    report = tmp_path / 'report.json'
    save_calibration_dataset(
        correspondences, dataset_from_target(expected)
    )

    status = run(
        [
            '--correspondences',
            str(correspondences),
            '--output',
            str(config),
            '--fixtures',
            str(fixtures),
            '--report',
            str(report),
        ]
    )

    assert status == 0
    assert 'calibrated: true' in config.read_text(encoding='utf-8')
    assert 'version: 2' in fixtures.read_text(encoding='utf-8')
    assert validate_transform(config, fixtures)['passed']


def test_calibration_cli_fail_closes_on_bad_holdout(tmp_path):
    identity = RigidTransform(
        'panda_sim_world',
        'link_base',
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    correspondences = tmp_path / 'correspondences.yaml'
    config = tmp_path / 'transform.yaml'
    save_calibration_dataset(
        correspondences,
        dataset_from_target(identity, holdout_offset=(0.02, 0.0, 0.0)),
    )

    status = run(
        [
            '--correspondences',
            str(correspondences),
            '--output',
            str(config),
            '--fixtures',
            str(tmp_path / 'fixtures.yaml'),
            '--report',
            str(tmp_path / 'report.json'),
        ]
    )

    assert status == 1
    assert 'calibrated: false' in config.read_text(encoding='utf-8')
