import json
import math

import numpy as np
import pytest

from mirage_state import (
    CartesianPose,
    RigidTransform,
    TransformConfigError,
    TransformError,
    load_transform_config,
    quaternion_angular_distance,
)
from mirage_state.transform_cli import run, validate_transform


SQRT_HALF = math.sqrt(0.5)


def z_quarter_turn():
    return RigidTransform(
        source_frame='panda_sim_world',
        target_frame='link_base',
        translation_m=(1.0, 2.0, 3.0),
        rotation_xyzw=(0.0, 0.0, SQRT_HALF, SQRT_HALF),
    )


def test_t_ts_maps_source_axes_and_position_into_target_frame():
    transform = z_quarter_turn()

    np.testing.assert_allclose(transform.transform_vector((1.0, 0.0, 0.0)), (0, 1, 0), atol=1e-12)
    np.testing.assert_allclose(transform.transform_vector((0.0, 1.0, 0.0)), (-1, 0, 0), atol=1e-12)
    np.testing.assert_allclose(transform.transform_vector((0.0, 0.0, 1.0)), (0, 0, 1), atol=1e-12)
    np.testing.assert_allclose(
        transform.transform_position((0.5, 0.0, 0.0)),
        (1.0, 2.5, 3.0),
        atol=1e-12,
    )


def test_orientation_uses_q_ts_times_q_se_hamilton_order():
    transform = z_quarter_turn()
    source_x_quarter_turn = (SQRT_HALF, 0.0, 0.0, SQRT_HALF)

    actual = transform.transform_orientation(source_x_quarter_turn)

    np.testing.assert_allclose(actual, (0.5, 0.5, 0.5, 0.5), atol=1e-12)


def test_t_ts_and_t_st_round_trip_random_poses():
    transform = RigidTransform(
        source_frame='source',
        target_frame='target',
        translation_m=(0.37, -0.12, 0.81),
        rotation_xyzw=(0.2, -0.3, 0.1, math.sqrt(0.86)),
    )
    inverse = transform.inverse()
    generator = np.random.default_rng(23)

    for _ in range(50):
        raw_quaternion = generator.normal(size=4)
        raw_quaternion /= np.linalg.norm(raw_quaternion)
        pose = CartesianPose(
            position_m=tuple(generator.normal(size=3)),
            quaternion_xyzw=tuple(raw_quaternion),
            frame_id='source',
        )
        round_trip = inverse.transform_pose(transform.transform_pose(pose))
        np.testing.assert_allclose(
            round_trip.position_m, pose.position_m, atol=1e-9
        )
        assert quaternion_angular_distance(
            round_trip.quaternion_xyzw, pose.quaternion_xyzw
        ) <= 1e-8


def test_quaternion_signs_describe_the_same_orientation():
    quaternion = (0.2, -0.3, 0.1, math.sqrt(0.86))
    assert quaternion_angular_distance(
        quaternion, tuple(-value for value in quaternion)
    ) == pytest.approx(0.0)


def test_half_turn_quaternion_sign_is_deterministic():
    pose = CartesianPose(
        position_m=(0.0, 0.0, 0.0),
        quaternion_xyzw=(-1.0, 0.0, 0.0, 0.0),
        frame_id='source',
    )

    assert pose.quaternion_xyzw == (1.0, -0.0, -0.0, -0.0)


def test_transform_pose_rejects_wrong_frame():
    pose = CartesianPose(
        position_m=(0.0, 0.0, 0.0),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        frame_id='wrong_frame',
    )

    with pytest.raises(TransformError, match='does not match'):
        z_quarter_turn().transform_pose(pose)


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('translation_m', (math.nan, 0.0, 0.0), 'finite'),
        ('rotation_xyzw', (0.0, 0.0, 0.0, 2.0), 'norm'),
    ],
)
def test_transform_rejects_invalid_numeric_values(field, value, message):
    arguments = {
        'source_frame': 'source',
        'target_frame': 'target',
        'translation_m': (0.0, 0.0, 0.0),
        'rotation_xyzw': (0.0, 0.0, 0.0, 1.0),
    }
    arguments[field] = value
    with pytest.raises(TransformError, match=message):
        RigidTransform(**arguments)


def write_config(path, *, calibrated, extra=''):
    path.write_text(
        '\n'.join(
            [
                'version: 1',
                f'calibrated: {str(calibrated).lower()}',
                'source_frame: panda_sim_world',
                'target_frame: link_base',
                'translation_m: [1.0, 2.0, 3.0]',
                'rotation_xyzw: [0.0, 0.0, 0.7071067811865476, '
                '0.7071067811865476]',
                extra,
            ]
        ),
        encoding='utf-8',
    )


def write_fixtures(path, *, expected_position='[1.0, 3.0, 3.0]'):
    path.write_text(
        f'''version: 1
position_tolerance_m: 1.0e-9
orientation_tolerance_rad: 1.0e-8
vector_tolerance: 1.0e-9
pose_cases:
  - name: known_pose
    source:
      frame_id: panda_sim_world
      position_m: [1.0, 0.0, 0.0]
      quaternion_xyzw: [0.0, 0.0, 0.0, 1.0]
    expected_target:
      frame_id: link_base
      position_m: {expected_position}
      quaternion_xyzw: [0.0, 0.0, 0.7071067811865476, 0.7071067811865476]
axis_cases:
  - name: source_x
    source_vector: [1.0, 0.0, 0.0]
    expected_target_vector: [0.0, 1.0, 0.0]
  - name: source_y
    source_vector: [0.0, 1.0, 0.0]
    expected_target_vector: [-1.0, 0.0, 0.0]
  - name: source_z
    source_vector: [0.0, 0.0, 1.0]
    expected_target_vector: [0.0, 0.0, 1.0]
''',
        encoding='utf-8',
    )


def test_config_is_fail_closed_until_measured(tmp_path):
    config = tmp_path / 'transform.yaml'
    write_config(config, calibrated=False)

    with pytest.raises(TransformConfigError, match='not calibrated'):
        load_transform_config(config)

    transform = load_transform_config(config, require_calibrated=False)
    assert transform.source_frame == 'panda_sim_world'


def test_config_rejects_camera_extrinsic_or_other_unknown_fields(tmp_path):
    config = tmp_path / 'transform.yaml'
    write_config(config, calibrated=True, extra='camera_translation_m: [0, 0, 0]')

    with pytest.raises(TransformConfigError, match='unknown.*camera'):
        load_transform_config(config)


def test_validation_cli_passes_known_pose_axes_and_round_trip(tmp_path, capsys):
    config = tmp_path / 'transform.yaml'
    fixtures = tmp_path / 'fixtures.yaml'
    report_path = tmp_path / 'reports' / 'transform.json'
    write_config(config, calibrated=True)
    write_fixtures(fixtures)

    status = run(
        [
            '--config',
            str(config),
            '--fixtures',
            str(fixtures),
            '--report',
            str(report_path),
        ]
    )

    assert status == 0
    report = json.loads(capsys.readouterr().out)
    assert report['passed']
    assert all(case['passed'] for case in report['pose_cases'])
    assert all(case['passed'] for case in report['axis_cases'])
    assert json.loads(report_path.read_text(encoding='utf-8')) == report


def test_validation_returns_failure_for_wrong_expected_pose(tmp_path):
    config = tmp_path / 'transform.yaml'
    fixtures = tmp_path / 'fixtures.yaml'
    write_config(config, calibrated=True)
    write_fixtures(fixtures, expected_position='[9.0, 9.0, 9.0]')

    report = validate_transform(config, fixtures)

    assert not report['passed']
    assert not report['pose_cases'][0]['passed']


def test_validation_cli_rejects_uncalibrated_config(tmp_path, capsys):
    config = tmp_path / 'transform.yaml'
    fixtures = tmp_path / 'fixtures.yaml'
    write_config(config, calibrated=False)
    write_fixtures(fixtures)

    status = run(['--config', str(config), '--fixtures', str(fixtures)])

    assert status == 2
    report = json.loads(capsys.readouterr().out)
    assert not report['passed']
    assert 'not calibrated' in report['error']
