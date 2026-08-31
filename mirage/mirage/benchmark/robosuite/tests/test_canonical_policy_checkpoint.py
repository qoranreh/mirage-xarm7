from pathlib import Path

import numpy as np
import robomimic.utils.file_utils as FileUtils
import torch

from mirage_state import PandaRobosuiteAdapter
from mirage_state.policy import CanonicalPolicyObservationProjector


def test_pick_place_checkpoint_action_is_preserved():
    checkpoint = (
        Path(__file__).parents[3]
        / 'models'
        / 'can'
        / 'low_dim'
        / 'model_epoch_250_PickPlaceCan_success_1.0.pth'
    )
    policy, checkpoint_data = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device='cpu', verbose=False
    )
    original = {
        'robot0_eef_pos': np.array([0.0, 0.0, 0.9], dtype=np.float32),
        'robot0_eef_quat': np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        'robot0_gripper_qpos': np.array(
            [0.02, -0.02], dtype=np.float32
        ),
        'object': np.zeros(14, dtype=np.float32),
    }
    state = PandaRobosuiteAdapter(env_name='PickPlaceCan').adapt(
        original, timestamp_ns=1_000_000_000
    )
    projector = CanonicalPolicyObservationProjector.from_shape_metadata(
        checkpoint_data['shape_metadata'],
        expected_frame_id='panda_sim_world',
        expected_object_schema='PickPlaceCan:robomimic-v1',
    )
    projected = projector.project(state, now_ns=1_000_000_000)

    np.random.seed(7)
    torch.manual_seed(7)
    policy.start_episode()
    original_action = policy(ob=original)
    np.random.seed(7)
    torch.manual_seed(7)
    policy.start_episode()
    projected_action = policy(ob=projected)

    np.testing.assert_allclose(projected_action, original_action, atol=1e-7)


def test_lift_checkpoint_action_is_preserved():
    checkpoint = (
        Path(__file__).parents[3]
        / 'models'
        / 'lift'
        / 'low_dim'
        / 'lift_ph_low_dim_epoch_1000_succ_100.pth'
    )
    policy, checkpoint_data = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device='cpu', verbose=False
    )
    original = {
        'robot0_eef_pos': np.array([0.0, 0.0, 0.9], dtype=np.float32),
        'robot0_eef_quat': np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        'robot0_gripper_qpos': np.array(
            [0.02, -0.02], dtype=np.float32
        ),
        'object': np.zeros(10, dtype=np.float32),
    }
    state = PandaRobosuiteAdapter(env_name='Lift').adapt(
        original, timestamp_ns=1_000_000_000
    )
    projector = CanonicalPolicyObservationProjector.from_shape_metadata(
        checkpoint_data['shape_metadata'],
        expected_frame_id='panda_sim_world',
        expected_object_schema='Lift:robomimic-v1',
    )
    projected = projector.project(state, now_ns=1_000_000_000)

    np.random.seed(7)
    torch.manual_seed(7)
    policy.start_episode()
    original_action = policy(ob=original)
    np.random.seed(7)
    torch.manual_seed(7)
    policy.start_episode()
    projected_action = policy(ob=projected)

    np.testing.assert_allclose(projected_action, original_action, atol=1e-7)
