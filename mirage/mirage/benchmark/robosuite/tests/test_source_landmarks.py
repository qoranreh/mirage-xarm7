from pathlib import Path

import numpy as np

from mirage.benchmark.robosuite.source_landmarks import (
    pick_place_can_bin1_dataset,
    source_dataset_from_checkpoint,
)


def test_profile_uses_bin1_origin_four_fit_points_and_holdout():
    dataset = pick_place_can_bin1_dataset((0.1, -0.25, 0.8))

    assert dataset.source_frame == 'panda_sim_world'
    assert dataset.target_frame == 'link_base'
    assert [item.role for item in dataset.landmarks].count('fit') == 4
    assert [item.role for item in dataset.landmarks].count('holdout') == 1
    np.testing.assert_allclose(
        dataset.landmarks[-1].source_position_m, (0.2, -0.15, 0.8)
    )


def test_real_checkpoint_environment_produces_expected_source_landmarks():
    checkpoint = (
        Path(__file__).parents[3]
        / 'models'
        / 'can'
        / 'low_dim'
        / 'model_epoch_250_PickPlaceCan_success_1.0.pth'
    )

    dataset = source_dataset_from_checkpoint(
        checkpoint, 'pick_place_can_bin1'
    )

    assert dataset.task == 'PickPlaceCan'
    np.testing.assert_allclose(
        dataset.landmarks[0].source_position_m, (0.1, -0.25, 0.8)
    )
