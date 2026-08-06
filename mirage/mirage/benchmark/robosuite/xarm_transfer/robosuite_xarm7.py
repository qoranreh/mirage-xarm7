"""Robosuite v1.4 adapter for the official xArm7 MJCF assets.

The robot and gripper definitions are simulation-only. Importing this module
does not import ROS, the UFactory SDK, or any real-robot transport.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


ASSET_ROOT = Path(__file__).resolve().parent / "assets"
XARM7_ROBOT_XML = ASSET_ROOT / "robots" / "xarm7" / "robot.xml"
XARM7_GRIPPER_XML = ASSET_ROOT / "grippers" / "xarm7_gripper.xml"


def _require_asset(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"required xArm7 simulation asset is missing: {path}")
    return str(path)


def _robosuite_types():
    """Import Robosuite lazily so contract-only code has no MuJoCo dependency."""

    from robosuite.models.grippers.gripper_model import GripperModel
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel

    return GripperModel, ManipulatorModel


GripperModel, ManipulatorModel = _robosuite_types()


class XArm7GripperBase(GripperModel):
    """Six-joint UFactory gripper driven by its single tendon actuator."""

    def __init__(self, idn=0):
        super().__init__(_require_asset(XARM7_GRIPPER_XML), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return np.array([0.02, 0.0, 0.0, -0.02, 0.0, 0.0])

    @property
    def _important_geoms(self):
        return {
            "left_finger": ["left_finger_pad_1", "left_finger_pad_2"],
            "right_finger": ["right_finger_pad_1", "right_finger_pad_2"],
            "left_fingerpad": ["left_finger_pad_1", "left_finger_pad_2"],
            "right_fingerpad": ["right_finger_pad_1", "right_finger_pad_2"],
        }


class XArm7Gripper(XArm7GripperBase):
    """One-dimensional gripper with Robosuite's -1=open, +1=closed contract."""

    def format_action(self, action):
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (self.dof,) or not np.all(np.isfinite(value)):
            raise ValueError("xArm7 gripper action must be finite with shape (1,)")
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(value), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1


class XArm7(ManipulatorModel):
    """UFactory xArm7 model adapted to the Robosuite v1.4 model API."""

    def __init__(self, idn=0):
        super().__init__(_require_asset(XARM7_ROBOT_XML), idn=idn)
        self.set_joint_attribute(
            attrib="damping",
            values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01)),
        )

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "XArm7Gripper"

    @property
    def default_controller_config(self):
        # The transfer runtime supplies the checkpoint's explicit OSC_POSE
        # configuration. This v1.4 fallback only satisfies the model contract.
        return "default_panda"

    @property
    def init_qpos(self):
        return np.array([0.0, 0.0, 0.0, 0.39 * np.pi, 0.0, 0.39 * np.pi, 0.0])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0.0),
            "empty": (-0.6, 0.0, 0.0),
            "table": lambda table_length: (-0.16 - table_length / 2.0, 0.0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0.0, 0.0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"


def register_xarm7() -> None:
    """Register the adapter with Robosuite v1.4's two manual registries."""

    from robosuite.models import grippers
    from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
    from robosuite.robots import ROBOT_CLASS_MAPPING, SingleArm

    registered_model = REGISTERED_ROBOTS.get("XArm7")
    if registered_model is not XArm7:
        raise RuntimeError("Robosuite XArm7 model registry contains a conflicting class")
    registered_gripper = grippers.GRIPPER_MAPPING.get("XArm7Gripper")
    if registered_gripper not in (None, XArm7Gripper):
        raise RuntimeError("Robosuite xArm7 gripper registry contains a conflicting class")
    registered_robot = ROBOT_CLASS_MAPPING.get("XArm7")
    if registered_robot not in (None, SingleArm):
        raise RuntimeError("Robosuite XArm7 runtime registry contains a conflicting class")
    grippers.GRIPPER_MAPPING["XArm7Gripper"] = XArm7Gripper
    ROBOT_CLASS_MAPPING["XArm7"] = SingleArm
