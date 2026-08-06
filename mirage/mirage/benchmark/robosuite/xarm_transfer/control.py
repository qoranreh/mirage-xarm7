"""Fail-closed xArm simulation safety and absolute blocking control."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import CartesianPose
from .geometry import quaternion_angular_error_rad, quaternion_to_rotvec
from .simulation_transform import (
    SIMULATION_TRANSFORM_KIND,
    TargetDesiredPose,
)


class ControlError(RuntimeError):
    """Raised for invalid controller construction or adapter behavior."""


class AbsoluteCartesianAdapter(Protocol):
    """Minimal Robosuite-facing adapter used by the blocking controller."""

    def set_absolute_mode(self) -> None:
        """Configure the underlying OSC controller for absolute pose input."""

    def read_pose(self) -> CartesianPose:
        """Return the achieved target end-effector pose."""

    def read_joint_positions(self) -> Sequence[float]:
        """Return current target arm joint positions in model order."""

    def step_absolute(self, action: np.ndarray) -> object:
        """Advance the simulator using [xyz, absolute rotvec, gripper]."""

    def has_gripper_task_object_contact(self) -> bool:
        """Return whether the simulated gripper touches a task object."""


@dataclass(frozen=True)
class PredictionSafetyBounds:
    """Validation error bounds attached to a dynamics result."""

    position_error_bound_m: float
    orientation_error_bound_rad: float

    def __post_init__(self) -> None:
        for name in ("position_error_bound_m", "orientation_error_bound_rad"):
            value = float(getattr(self, name))
            if np.isnan(value) or value < 0.0:
                raise ControlError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SafetyLimits:
    """Explicit simulation workspace, freshness, model, and joint gates."""

    target_frame: str
    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    joint_min_rad: tuple[float, ...]
    joint_max_rad: tuple[float, ...]
    max_command_age_sec: float = 0.15
    max_future_skew_sec: float = 0.02
    max_prediction_position_error_m: float = 0.03
    max_prediction_orientation_error_rad: float = np.deg2rad(10.0)
    joint_limit_margin_rad: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.target_frame, str) or not self.target_frame.strip():
            raise ControlError("target_frame must not be empty")
        object.__setattr__(self, "target_frame", self.target_frame.strip())
        for name, length in (("workspace_min_m", 3), ("workspace_max_m", 3)):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (length,) or not np.all(np.isfinite(value)):
                raise ControlError(f"{name} must contain three finite values")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        if np.any(
            np.asarray(self.workspace_min_m) >= np.asarray(self.workspace_max_m)
        ):
            raise ControlError("workspace min must be strictly below max")
        joint_min = np.asarray(self.joint_min_rad, dtype=np.float64)
        joint_max = np.asarray(self.joint_max_rad, dtype=np.float64)
        if (
            joint_min.ndim != 1
            or joint_min.size == 0
            or joint_max.shape != joint_min.shape
            or not np.all(np.isfinite(joint_min))
            or not np.all(np.isfinite(joint_max))
            or np.any(joint_min >= joint_max)
        ):
            raise ControlError("joint limits must be finite ordered vectors")
        object.__setattr__(self, "joint_min_rad", tuple(float(x) for x in joint_min))
        object.__setattr__(self, "joint_max_rad", tuple(float(x) for x in joint_max))
        for name in (
            "max_command_age_sec",
            "max_future_skew_sec",
            "max_prediction_position_error_m",
            "max_prediction_orientation_error_rad",
            "joint_limit_margin_rad",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ControlError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if 2.0 * self.joint_limit_margin_rad >= float(
            np.min(joint_max - joint_min)
        ):
            raise ControlError("joint limit margin removes the valid joint range")


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.accepted and self.reason is not None:
            raise ControlError("accepted safety decision cannot have a reason")
        if not self.accepted and not self.reason:
            raise ControlError("rejected safety decision must have a reason")


class SimulationSafetyFilter:
    """Reject unsafe target commands before any controller call."""

    def __init__(self, limits: SafetyLimits) -> None:
        self.limits = limits

    def evaluate(
        self,
        desired: TargetDesiredPose,
        *,
        joint_positions_rad: Sequence[float],
        prediction_bounds: PredictionSafetyBounds,
        now_ns: int,
    ) -> SafetyDecision:
        if desired.transform_kind != SIMULATION_TRANSFORM_KIND:
            return SafetyDecision(False, "not_simulation_only")
        if desired.pose.frame_id != self.limits.target_frame:
            return SafetyDecision(False, "frame_mismatch")
        if type(now_ns) is not int or now_ns < 0:
            return SafetyDecision(False, "invalid_clock")
        age_sec = (now_ns - desired.source_action_timestamp_ns) / 1e9
        if age_sec > self.limits.max_command_age_sec:
            return SafetyDecision(False, "stale_command")
        if age_sec < -self.limits.max_future_skew_sec:
            return SafetyDecision(False, "future_command")

        position = np.asarray(desired.pose.position_m, dtype=np.float64)
        if not np.all(np.isfinite(position)):
            return SafetyDecision(False, "non_finite_pose")
        if np.any(position < np.asarray(self.limits.workspace_min_m)) or np.any(
            position > np.asarray(self.limits.workspace_max_m)
        ):
            return SafetyDecision(False, "workspace_violation")

        joints = np.asarray(joint_positions_rad, dtype=np.float64)
        joint_min = np.asarray(self.limits.joint_min_rad)
        joint_max = np.asarray(self.limits.joint_max_rad)
        if joints.shape != joint_min.shape:
            return SafetyDecision(False, "joint_shape_mismatch")
        if not np.all(np.isfinite(joints)):
            return SafetyDecision(False, "non_finite_joint_state")
        margin = self.limits.joint_limit_margin_rad
        if np.any(joints < joint_min + margin) or np.any(joints > joint_max - margin):
            return SafetyDecision(False, "joint_limit_violation")

        if not np.isfinite(prediction_bounds.position_error_bound_m) or (
            prediction_bounds.position_error_bound_m
            > self.limits.max_prediction_position_error_m
        ):
            return SafetyDecision(False, "position_prediction_bound")
        if not np.isfinite(prediction_bounds.orientation_error_bound_rad) or (
            prediction_bounds.orientation_error_bound_rad
            > self.limits.max_prediction_orientation_error_rad
        ):
            return SafetyDecision(False, "orientation_prediction_bound")
        return SafetyDecision(True)


@dataclass(frozen=True)
class XArmSimulationGripperMapper:
    """Map Franka Panda policy semantics to the xArm simulation actuator."""

    invert: bool = False

    def map(self, source_normalized: float) -> float:
        value = float(source_normalized)
        if not np.isfinite(value) or abs(value) > 1.0 + 1e-8:
            raise ControlError("source gripper value must be finite in [-1, 1]")
        # Both Robosuite conventions are -1=open, +1=closed. Inversion remains
        # explicit for alternative simulation assets and is never implicit.
        return float(-value if self.invert else value)


@dataclass(frozen=True)
class BlockingControlResult:
    """Per-policy-step target tracking diagnostics."""

    succeeded: bool
    iterations: int
    timed_out: bool
    iteration_limit_reached: bool
    rejected_reason: str | None
    position_error_m: float
    orientation_error_rad: float
    latency_sec: float
    contact_tolerance_used: bool = False

    def __post_init__(self) -> None:
        if self.succeeded and (
            self.timed_out
            or self.iteration_limit_reached
            or self.rejected_reason is not None
        ):
            raise ControlError("successful control result cannot be rejected or timed out")
        if self.timed_out and self.iteration_limit_reached:
            raise ControlError("controller cannot hit both time and iteration limits")
        if type(self.iterations) is not int or self.iterations < 0:
            raise ControlError("iterations must be a non-negative integer")
        if type(self.contact_tolerance_used) is not bool:
            raise ControlError("contact_tolerance_used must be a bool")
        if self.contact_tolerance_used and not self.succeeded:
            raise ControlError("contact tolerance can only be used for success")
        for name in ("position_error_m", "orientation_error_rad", "latency_sec"):
            value = float(getattr(self, name))
            if np.isnan(value) or value < 0.0:
                raise ControlError(f"{name} must be non-negative")


class AbsoluteBlockingController:
    """Repeat absolute OSC control until separate pose tolerances are met."""

    def __init__(
        self,
        *,
        adapter: AbsoluteCartesianAdapter,
        safety_filter: SimulationSafetyFilter,
        position_tolerance_m: float = 0.015,
        orientation_tolerance_rad: float = np.deg2rad(5.0),
        contact_position_tolerance_m: float | None = None,
        settling_iterations: int = 0,
        outer_loop_position_gain: float = 0.0,
        max_outer_loop_position_correction_m: float = 0.0,
        outer_loop_orientation_gain: float = 0.0,
        max_outer_loop_orientation_correction_rad: float = 0.0,
        max_iterations: int = 300,
        timeout_sec: float = 2.0,
        gripper_mapper: XArmSimulationGripperMapper | None = None,
        wall_clock_ns: Callable[[], int] = time.time_ns,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.adapter = adapter
        self.safety_filter = safety_filter
        self.position_tolerance_m = float(position_tolerance_m)
        self.orientation_tolerance_rad = float(orientation_tolerance_rad)
        self.contact_position_tolerance_m = (
            None
            if contact_position_tolerance_m is None
            else float(contact_position_tolerance_m)
        )
        self.settling_iterations = settling_iterations
        self.outer_loop_position_gain = float(outer_loop_position_gain)
        self.max_outer_loop_position_correction_m = float(
            max_outer_loop_position_correction_m
        )
        self.outer_loop_orientation_gain = float(outer_loop_orientation_gain)
        self.max_outer_loop_orientation_correction_rad = float(
            max_outer_loop_orientation_correction_rad
        )
        self.max_iterations = max_iterations
        self.timeout_sec = float(timeout_sec)
        self.gripper_mapper = gripper_mapper or XArmSimulationGripperMapper()
        self.wall_clock_ns = wall_clock_ns
        self.monotonic = monotonic
        if (
            not np.isfinite(self.position_tolerance_m)
            or self.position_tolerance_m < 0.0
            or not np.isfinite(self.orientation_tolerance_rad)
            or self.orientation_tolerance_rad < 0.0
            or (
                self.contact_position_tolerance_m is not None
                and (
                    not np.isfinite(self.contact_position_tolerance_m)
                    or self.contact_position_tolerance_m < self.position_tolerance_m
                )
            )
            or type(self.settling_iterations) is not int
            or self.settling_iterations < 0
            or not np.isfinite(self.outer_loop_position_gain)
            or self.outer_loop_position_gain < 0.0
            or not np.isfinite(self.max_outer_loop_position_correction_m)
            or self.max_outer_loop_position_correction_m < 0.0
            or (
                self.outer_loop_position_gain > 0.0
                and self.max_outer_loop_position_correction_m <= 0.0
            )
            or not np.isfinite(self.outer_loop_orientation_gain)
            or self.outer_loop_orientation_gain < 0.0
            or not np.isfinite(self.max_outer_loop_orientation_correction_rad)
            or self.max_outer_loop_orientation_correction_rad < 0.0
            or (
                self.outer_loop_orientation_gain > 0.0
                and self.max_outer_loop_orientation_correction_rad <= 0.0
            )
            or type(max_iterations) is not int
            or max_iterations <= 0
            or not np.isfinite(self.timeout_sec)
            or self.timeout_sec <= 0.0
        ):
            raise ControlError("blocking controller limits are invalid")

    def _contact_tolerance_is_met(
        self,
        *,
        position_error_m: float,
        orientation_error_rad: float,
    ) -> bool:
        """Apply the looser position bound only during verified object contact."""

        if (
            self.contact_position_tolerance_m is None
            or position_error_m > self.contact_position_tolerance_m
            or orientation_error_rad > self.orientation_tolerance_rad
        ):
            return False
        probe = getattr(self.adapter, "has_gripper_task_object_contact", None)
        if not callable(probe):
            return False
        in_contact = probe()
        if type(in_contact) is not bool:
            raise ControlError("target adapter contact probe must return a bool")
        return in_contact

    def _compensated_desired(
        self,
        desired: TargetDesiredPose,
        current: CartesianPose,
    ) -> TargetDesiredPose:
        """Apply bounded Cartesian outer-loop compensation in world axes."""

        if (
            self.outer_loop_position_gain == 0.0
            and self.outer_loop_orientation_gain == 0.0
        ):
            return desired
        command_position = np.asarray(desired.pose.position_m, dtype=np.float64)
        if self.outer_loop_position_gain > 0.0:
            position_error = command_position - np.asarray(current.position_m)
            position_correction = self.outer_loop_position_gain * position_error
            correction_norm = float(np.linalg.norm(position_correction))
            if correction_norm > self.max_outer_loop_position_correction_m:
                position_correction *= (
                    self.max_outer_loop_position_correction_m / correction_norm
                )
            command_position = command_position + position_correction
        if not np.all(np.isfinite(command_position)):
            raise ControlError("outer-loop position command is non-finite")

        command_quaternion = np.asarray(
            desired.pose.quaternion_xyzw, dtype=np.float64
        )
        if self.outer_loop_orientation_gain > 0.0:
            desired_rotation = Rotation.from_quat(command_quaternion)
            current_rotation = Rotation.from_quat(current.quaternion_xyzw)
            orientation_correction = (
                desired_rotation * current_rotation.inv()
            ).as_rotvec() * self.outer_loop_orientation_gain
            correction_angle = float(np.linalg.norm(orientation_correction))
            if correction_angle > self.max_outer_loop_orientation_correction_rad:
                orientation_correction *= (
                    self.max_outer_loop_orientation_correction_rad / correction_angle
                )
            command_quaternion = (
                Rotation.from_rotvec(orientation_correction) * desired_rotation
            ).as_quat()
        if not np.all(np.isfinite(command_quaternion)):
            raise ControlError("outer-loop orientation command is non-finite")
        return TargetDesiredPose(
            pose=CartesianPose(
                tuple(command_position),
                tuple(command_quaternion),
                desired.pose.frame_id,
                desired.pose.timestamp_ns,
            ),
            gripper_normalized=desired.gripper_normalized,
            source_action_timestamp_ns=desired.source_action_timestamp_ns,
        )

    @staticmethod
    def _pose_errors(current: CartesianPose, goal: CartesianPose) -> tuple[float, float]:
        if current.frame_id != goal.frame_id:
            raise ControlError("target adapter returned an unexpected pose frame")
        position_error = float(
            np.linalg.norm(
                np.asarray(current.position_m) - np.asarray(goal.position_m)
            )
        )
        orientation_error = quaternion_angular_error_rad(
            current.quaternion_xyzw, goal.quaternion_xyzw
        )
        if not np.isfinite(position_error) or not np.isfinite(orientation_error):
            raise ControlError("target adapter returned a non-finite pose error")
        return position_error, orientation_error

    def drive(
        self,
        desired: TargetDesiredPose,
        *,
        prediction_bounds: PredictionSafetyBounds,
    ) -> BlockingControlResult:
        started = self.monotonic()
        current = self.adapter.read_pose()
        position_error, orientation_error = self._pose_errors(current, desired.pose)
        iterations = 0
        strict_tolerance_streak = 0
        acceptance_time_ns = self.wall_clock_ns()

        # Evaluate all gates before changing controller mode or stepping physics.
        decision = self.safety_filter.evaluate(
            desired,
            joint_positions_rad=self.adapter.read_joint_positions(),
            prediction_bounds=prediction_bounds,
            now_ns=acceptance_time_ns,
        )
        if not decision.accepted:
            return BlockingControlResult(
                False,
                0,
                False,
                False,
                decision.reason,
                position_error,
                orientation_error,
                max(0.0, self.monotonic() - started),
            )

        target_gripper = np.asarray(
            [self.gripper_mapper.map(desired.gripper_normalized)],
            dtype=np.float64,
        )

        absolute_mode_set = False
        while iterations < self.max_iterations:
            elapsed = self.monotonic() - started
            if elapsed >= self.timeout_sec:
                return BlockingControlResult(
                    False,
                    iterations,
                    True,
                    False,
                    None,
                    position_error,
                    orientation_error,
                    max(0.0, elapsed),
                )

            # Freshness is an admission check: once a goal is accepted, its
            # age must not expire merely because blocking control is still
            # converging. Spatial, joint, and model gates are re-evaluated.
            command_desired = self._compensated_desired(desired, current)
            decision = self.safety_filter.evaluate(
                command_desired,
                joint_positions_rad=self.adapter.read_joint_positions(),
                prediction_bounds=prediction_bounds,
                now_ns=acceptance_time_ns,
            )
            if not decision.accepted:
                return BlockingControlResult(
                    False,
                    iterations,
                    False,
                    False,
                    decision.reason,
                    position_error,
                    orientation_error,
                    max(0.0, self.monotonic() - started),
                )

            # Always execute at least once so a gripper-only command is applied.
            if not absolute_mode_set:
                self.adapter.set_absolute_mode()
                absolute_mode_set = True
            target_action = np.concatenate(
                (
                    np.asarray(command_desired.pose.position_m, dtype=np.float64),
                    quaternion_to_rotvec(command_desired.pose.quaternion_xyzw),
                    target_gripper,
                )
            )
            if target_action.shape != (7,) or not np.all(np.isfinite(target_action)):
                raise ControlError("absolute target action is invalid")
            self.adapter.step_absolute(target_action.copy())
            iterations += 1
            current = self.adapter.read_pose()
            position_error, orientation_error = self._pose_errors(
                current, desired.pose
            )
            if (
                position_error <= self.position_tolerance_m
                and orientation_error <= self.orientation_tolerance_rad
            ):
                strict_tolerance_streak += 1
                if strict_tolerance_streak > self.settling_iterations:
                    return BlockingControlResult(
                        True,
                        iterations,
                        False,
                        False,
                        None,
                        position_error,
                        orientation_error,
                        max(0.0, self.monotonic() - started),
                    )
            else:
                strict_tolerance_streak = 0
            if (
                position_error > self.position_tolerance_m
                and self._contact_tolerance_is_met(
                    position_error_m=position_error,
                    orientation_error_rad=orientation_error,
                )
            ):
                return BlockingControlResult(
                    True,
                    iterations,
                    False,
                    False,
                    None,
                    position_error,
                    orientation_error,
                    max(0.0, self.monotonic() - started),
                    True,
                )

        return BlockingControlResult(
            False,
            iterations,
            False,
            True,
            None,
            position_error,
            orientation_error,
            max(0.0, self.monotonic() - started),
        )
