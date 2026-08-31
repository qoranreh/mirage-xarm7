from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mirage.benchmark.robosuite.xarm_transfer.runtime import (
    SOURCE_FRAME_ID,
    FrankaPandaSourceRuntime,
    RobosuiteXArmAbsoluteAdapter,
    RuntimeContractError,
    validate_pick_place_can_checkpoint_contract,
)
from mirage.benchmark.robosuite.xarm_transfer.contracts import CartesianPose
from mirage.benchmark.robosuite.xarm_transfer.profiles import TASK_PROFILES


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


def test_source_absolute_osc_alignment_holds_gripper_at_zero_and_restores_delta():
    runtime = object.__new__(FrankaPandaSourceRuntime)
    runtime.robot = SimpleNamespace(
        controller=SimpleNamespace(use_delta=True)
    )
    desired = CartesianPose(
        (0.01, 0.0, 0.8),
        (0.0, 0.0, 0.0, 1.0),
        SOURCE_FRAME_ID,
        2,
    )
    pose = CartesianPose(
        (0.0, 0.0, 0.8),
        (0.0, 0.0, 0.0, 1.0),
        SOURCE_FRAME_ID,
        1,
    )
    actions = []

    def step(action):
        nonlocal pose
        actions.append(np.asarray(action).copy())
        pose = desired
        return {}, 0.0, False, {}

    runtime.environment = SimpleNamespace(step=step)
    runtime.read_pose = lambda: pose
    runtime._refresh_observation = lambda: None
    runtime.observation = {}

    position_error, orientation_error, iterations = (
        FrankaPandaSourceRuntime.align_to_pose(runtime, desired)
    )

    assert iterations == 1
    assert position_error == pytest.approx(0.0)
    assert orientation_error == pytest.approx(0.0)
    assert actions[0][-1] == 0.0
    assert runtime.robot.controller.use_delta is True


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


class FakeContactModel:
    def __init__(self, objects):
        self.mujoco_objects = objects


class FakeCompiledModel:
    geom_bodyid = np.asarray((0, 20, 30, 40, 50), dtype=np.int64)

    @staticmethod
    def geom_name2id(name):
        return {
            "Can_g0": 1,
            "cube_g0": 1,
            "Milk_g0": 2,
            "distractor_g0": 2,
            "piece_1_g0": 1,
            "piece_2_g0": 4,
        }[name]

    @staticmethod
    def body_name2id(name):
        return {"gripper": 40}[name]


def fake_adapter(objects, contacts=(), task_name="PickPlaceCan"):
    adapter = object.__new__(RobosuiteXArmAbsoluteAdapter)
    adapter.task_profile = TASK_PROFILES[task_name]
    adapter.robot = SimpleNamespace(gripper=SimpleNamespace(bodies=["gripper"]))
    adapter.core = SimpleNamespace(
        model=FakeContactModel(objects),
        sim=SimpleNamespace(
            model=FakeCompiledModel(),
            data=SimpleNamespace(
                ncon=len(contacts),
                contact=[SimpleNamespace(geom1=a, geom2=b) for a, b in contacts],
            ),
        ),
    )
    adapter._gripper_body_ids = frozenset()
    adapter._task_object_geom_ids = frozenset()
    return adapter


@pytest.mark.parametrize(
    "objects, message",
    (
        ([], "found 0"),
        (
            [
                SimpleNamespace(name="Can", contact_geoms=["Can_g0"]),
                SimpleNamespace(name="Can", contact_geoms=["Can_g0"]),
            ],
            "found 2",
        ),
        ([SimpleNamespace(name="Can", contact_geoms=[])], "no contact geoms"),
    ),
)
def test_task_object_resolution_rejects_missing_duplicate_or_empty(objects, message):
    adapter = fake_adapter(objects)

    with pytest.raises(RuntimeContractError, match=message):
        adapter._refresh_contact_references()


@pytest.mark.parametrize(
    "task_name, expected_name, expected_geom",
    (
        ("PickPlaceCan", "Can", "Can_g0"),
        ("Lift", "cube", "cube_g0"),
    ),
)
def test_contact_probe_uses_only_profile_object_geoms(
    task_name, expected_name, expected_geom
):
    objects = [
        SimpleNamespace(name="distractor", contact_geoms=["distractor_g0"]),
        SimpleNamespace(name=expected_name, contact_geoms=[expected_geom]),
    ]
    distractor_contact = fake_adapter(
        objects, contacts=((2, 3),), task_name=task_name
    )
    distractor_contact._refresh_contact_references()
    expected_contact = fake_adapter(
        objects, contacts=((1, 3),), task_name=task_name
    )
    expected_contact._refresh_contact_references()

    assert not distractor_contact.has_gripper_task_object_contact()
    assert expected_contact.has_gripper_task_object_contact()


def test_contact_probe_combines_all_profile_manipulated_objects():
    objects = [
        SimpleNamespace(name="base", contact_geoms=["distractor_g0"]),
        SimpleNamespace(name="piece_1", contact_geoms=["piece_1_g0"]),
        SimpleNamespace(name="piece_2", contact_geoms=["piece_2_g0"]),
    ]
    first_piece_contact = fake_adapter(
        objects,
        contacts=((1, 3),),
        task_name="ThreePieceAssembly_D0",
    )
    second_piece_contact = fake_adapter(
        objects,
        contacts=((4, 3),),
        task_name="ThreePieceAssembly_D0",
    )
    base_contact = fake_adapter(
        objects,
        contacts=((2, 3),),
        task_name="ThreePieceAssembly_D0",
    )
    for adapter in (first_piece_contact, second_piece_contact, base_contact):
        adapter._refresh_contact_references()

    assert first_piece_contact.has_gripper_task_object_contact()
    assert second_piece_contact.has_gripper_task_object_contact()
    assert not base_contact.has_gripper_task_object_contact()


@pytest.mark.parametrize(
    "task_name, expected_kp",
    (
        ("PickPlaceCan", (300.0, 300.0, 500.0, 300.0, 300.0, 300.0)),
        ("Lift", (150.0,) * 6),
    ),
)
def test_osc_gain_switch_is_controlled_by_task_profile(task_name, expected_kp):
    adapter = object.__new__(RobosuiteXArmAbsoluteAdapter)
    adapter.task_profile = TASK_PROFILES[task_name]
    adapter.robot = SimpleNamespace(controller=SimpleNamespace())
    adapter.has_gripper_task_object_contact = lambda: True

    adapter._configure_simulation_osc_gains()

    np.testing.assert_allclose(adapter.robot.controller.kp, expected_kp)
    np.testing.assert_allclose(
        adapter.robot.controller.kd, 2.0 * np.sqrt(expected_kp)
    )
