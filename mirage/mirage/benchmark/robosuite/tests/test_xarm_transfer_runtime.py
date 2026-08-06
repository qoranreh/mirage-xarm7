from copy import deepcopy
from pathlib import Path

import pytest
import torch

from mirage.benchmark.robosuite.xarm_transfer.runtime import (
    RuntimeContractError,
    validate_pick_place_can_checkpoint_contract,
)


def checkpoint_data():
    checkpoint = (
        Path(__file__).parents[3]
        / "models"
        / "can"
        / "low_dim"
        / "model_epoch_250_PickPlaceCan_success_1.0.pth"
    )
    return torch.load(checkpoint, map_location="cpu")


def test_checkpoint_is_exact_franka_panda_osc_pose_contract():
    data = checkpoint_data()

    validate_pick_place_can_checkpoint_contract(
        data["env_metadata"], data["shape_metadata"]
    )


@pytest.mark.parametrize(
    "field, value, message",
    (
        ("robots", ["Sawyer"], "Franka Panda"),
        ("control_freq", 10, "20 Hz"),
    ),
)
def test_checkpoint_contract_rejects_wrong_robot_or_rate(field, value, message):
    data = checkpoint_data()
    env_metadata = deepcopy(data["env_metadata"])
    env_metadata["env_kwargs"][field] = value

    with pytest.raises(RuntimeContractError, match=message):
        validate_pick_place_can_checkpoint_contract(
            env_metadata, data["shape_metadata"]
        )


def test_checkpoint_contract_rejects_wrong_controller_scaling():
    data = checkpoint_data()
    env_metadata = deepcopy(data["env_metadata"])
    env_metadata["env_kwargs"]["controller_configs"]["output_max"][0] = 0.1

    with pytest.raises(RuntimeContractError, match="output scales"):
        validate_pick_place_can_checkpoint_contract(
            env_metadata, data["shape_metadata"]
        )
