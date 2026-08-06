import math

import numpy as np
import pytest

from mirage_state import (
    CanonicalPolicyObservationProjector,
    CanonicalStateError,
    GripperSample,
    NamedArray,
    ObjectSample,
    PandaRobosuiteAdapter,
    PoseSample,
    XArm7CanonicalAdapter,
    XArmAdapterConfig,
    make_canonical_state,
)


NOW_NS = 10_000_000_000


def object_sample(stamp_ns=NOW_NS):
    return ObjectSample(
        schema='PickPlaceCan:robomimic-v1',
        entries=(NamedArray.from_values('object', [0.0] * 14),),
        timestamp_ns=stamp_ns,
    )


def test_named_array_rejects_shape_and_nonfinite_values():
    with pytest.raises(ValueError, match='requires 4 values'):
        NamedArray('bad', (2, 2), (1.0, 2.0))
    with pytest.raises(ValueError, match='finite'):
        NamedArray.from_values('bad', [math.nan])


def test_quaternion_is_normalized_and_sign_is_canonical():
    state = make_canonical_state(
        ee_position_m=(1.0, 2.0, 3.0),
        ee_quaternion_xyzw=(0.0, 0.0, 0.0, -1.0001),
        gripper_width_m=0.04,
        object_schema='fixture:v1',
        object_state=(NamedArray.from_values('object', [1.0]),),
        frame_id='world',
        timestamp_ns=NOW_NS,
        source_robot='fixture',
        fresh=True,
    )

    assert state.valid
    assert state.ee_quaternion_xyzw == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ('angle', 'expected_width'),
    [(0.0, 0.085), (0.425, 0.0425), (0.85, 0.0)],
)
def test_xarm_gripper_calibration(angle, expected_width):
    width, reasons = XArm7CanonicalAdapter().gripper_width_m(angle)
    assert width == pytest.approx(expected_width)
    assert not reasons


def test_xarm_calibration_is_configurable_and_clamped():
    adapter = XArm7CanonicalAdapter(
        XArmAdapterConfig(
            open_angle_rad=0.1,
            closed_angle_rad=0.9,
            open_width_m=0.09,
            closed_width_m=0.01,
        )
    )
    width, reasons = adapter.gripper_width_m(0.0)
    assert width == pytest.approx(0.09)
    assert reasons == ('gripper_angle_out_of_range:0',)


def test_xarm_adapter_builds_valid_si_state():
    adapter = XArm7CanonicalAdapter()
    state = adapter.adapt(
        pose=PoseSample(
            (0.5, -0.2, 0.4),
            (0.0, 0.0, 0.0, 1.0),
            'link_base',
            NOW_NS,
        ),
        gripper=GripperSample(0.425, NOW_NS),
        object_sample=object_sample(),
        now_ns=NOW_NS,
    )

    assert state.valid
    assert state.fresh
    assert state.gripper_width_m == pytest.approx(0.0425)
    assert state.object_entry('object').shape == (14,)


def test_xarm_adapter_keeps_partial_state_but_blocks_missing_object():
    state = XArm7CanonicalAdapter().adapt(
        pose=PoseSample(
            (0.5, 0.0, 0.4),
            (0.0, 0.0, 0.0, 1.0),
            'link_base',
            NOW_NS,
        ),
        gripper=GripperSample(0.0, NOW_NS),
        object_sample=None,
        now_ns=NOW_NS,
    )

    assert not state.valid
    assert not state.fresh
    assert state.ee_position_m == (0.5, 0.0, 0.4)
    assert 'object_state_missing' in state.invalid_reasons


def test_xarm_adapter_rejects_stale_and_skewed_components():
    state = XArm7CanonicalAdapter().adapt(
        pose=PoseSample(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            'link_base',
            NOW_NS,
        ),
        gripper=GripperSample(0.0, NOW_NS - 500_000_000),
        object_sample=object_sample(NOW_NS),
        now_ns=NOW_NS,
    )

    assert not state.valid
    assert 'gripper_stale:0.5' in state.invalid_reasons
    assert 'component_timestamp_skew:0.5' in state.invalid_reasons


def test_panda_adapter_and_policy_projector_round_trip():
    observation = {
        'robot0_eef_pos': np.array([0.1, 0.2, 0.3]),
        'robot0_eef_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        'robot0_gripper_qpos': np.array([0.025, -0.025]),
        'object': np.arange(14, dtype=np.float64),
        'Can_pos': np.array([0.4, 0.5, 0.6]),
    }
    state = PandaRobosuiteAdapter(
        env_name='PickPlaceCan', object_keys=('object', 'Can_pos')
    ).adapt(observation, timestamp_ns=NOW_NS)
    projector = CanonicalPolicyObservationProjector(
        expected_shapes={
            'object': (14,),
            'robot0_eef_pos': (3,),
            'robot0_eef_quat': (4,),
            'robot0_gripper_qpos': (2,),
        },
        expected_frame_id='panda_sim_world',
        expected_object_schema='PickPlaceCan:robomimic-v1',
    )

    projected = projector.project(state, now_ns=NOW_NS)

    for key in projector.expected_shapes:
        np.testing.assert_allclose(projected[key], observation[key])
    assert state.object_entry('Can_pos').shape == (3,)


def test_policy_projector_blocks_untransformed_xarm_frame():
    state = XArm7CanonicalAdapter().adapt(
        pose=PoseSample(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            'link_base',
            NOW_NS,
        ),
        gripper=GripperSample(0.0, NOW_NS),
        object_sample=object_sample(),
        now_ns=NOW_NS,
    )
    projector = CanonicalPolicyObservationProjector(
        expected_shapes={'robot0_eef_pos': (3,)},
        expected_frame_id='panda_sim_world',
        expected_object_schema='PickPlaceCan:robomimic-v1',
    )

    with pytest.raises(CanonicalStateError, match='not transformed'):
        projector.project(state, now_ns=NOW_NS)


def test_policy_projector_rechecks_age_at_inference_time():
    state = PandaRobosuiteAdapter(env_name='PickPlaceCan').adapt(
        {
            'robot0_eef_pos': [0.0, 0.0, 0.0],
            'robot0_eef_quat': [0.0, 0.0, 0.0, 1.0],
            'robot0_gripper_qpos': [0.0, 0.0],
            'object': [0.0] * 14,
        },
        timestamp_ns=NOW_NS,
    )
    projector = CanonicalPolicyObservationProjector(
        expected_shapes={'object': (14,)},
        expected_frame_id='panda_sim_world',
        expected_object_schema='PickPlaceCan:robomimic-v1',
        max_state_age_sec=0.1,
    )

    with pytest.raises(CanonicalStateError, match='age'):
        projector.project(state, now_ns=NOW_NS + 200_000_000)


def test_policy_projector_blocks_invalid_state_before_inference():
    state = XArm7CanonicalAdapter().adapt(
        pose=PoseSample(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
            'link_base',
            NOW_NS,
        ),
        gripper=GripperSample(0.0, NOW_NS),
        object_sample=None,
        now_ns=NOW_NS,
    )
    projector = CanonicalPolicyObservationProjector(
        expected_shapes={'robot0_eef_pos': (3,)},
        expected_frame_id='link_base',
        expected_object_schema='PickPlaceCan:robomimic-v1',
    )

    with pytest.raises(CanonicalStateError, match='object_state_missing'):
        projector.project(state, now_ns=NOW_NS)


def test_policy_projector_blocks_object_schema_mismatch():
    state = PandaRobosuiteAdapter(env_name='PickPlaceCan').adapt(
        {
            'robot0_eef_pos': [0.0, 0.0, 0.0],
            'robot0_eef_quat': [0.0, 0.0, 0.0, 1.0],
            'robot0_gripper_qpos': [0.0, 0.0],
            'object': [0.0] * 14,
        },
        timestamp_ns=NOW_NS,
    )
    projector = CanonicalPolicyObservationProjector(
        expected_shapes={'object': (14,)},
        expected_frame_id='panda_sim_world',
        expected_object_schema='Lift:robomimic-v1',
    )

    with pytest.raises(CanonicalStateError, match='object schema'):
        projector.project(state, now_ns=NOW_NS)
