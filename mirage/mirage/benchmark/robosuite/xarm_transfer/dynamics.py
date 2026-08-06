"""Exchangeable online and learned forward-dynamics interfaces."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .contracts import (
    CartesianPose,
    OscActionScaler,
    ScaledOscCommand,
    SourceAction,
    SourceDesiredPose,
)
from .geometry import quaternion_angular_error_rad


class ForwardDynamicsError(RuntimeError):
    """Raised when a forward-dynamics implementation fails closed."""


@dataclass(frozen=True)
class ForwardDynamicsRequest:
    """One current source pose and normalized source action."""

    current_pose: CartesianPose
    action: SourceAction

    def __post_init__(self) -> None:
        if self.current_pose.frame_id != self.action.frame_id:
            raise ForwardDynamicsError(
                "current pose and source action frame_id must match"
            )
        if self.action.timestamp_ns < self.current_pose.timestamp_ns:
            raise ForwardDynamicsError("source action predates the current pose")


@dataclass(frozen=True)
class ForwardDynamicsResult:
    """Next-pose result and auditable one-step diagnostics."""

    current_pose: CartesianPose
    action: SourceAction
    scaled_action: ScaledOscCommand
    desired_pose: SourceDesiredPose
    gripper_state_before: tuple[float, ...]
    gripper_state_after: tuple[float, ...]
    provider: str
    latency_ns: int
    position_error_bound_m: float
    orientation_error_bound_rad: float

    def __post_init__(self) -> None:
        if self.current_pose.frame_id != self.action.frame_id:
            raise ForwardDynamicsError("result current/action frame mismatch")
        if self.desired_pose.pose.frame_id != self.current_pose.frame_id:
            raise ForwardDynamicsError("result changed source frame")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ForwardDynamicsError("provider must not be empty")
        if type(self.latency_ns) is not int or self.latency_ns < 0:
            raise ForwardDynamicsError("latency_ns must be non-negative")
        for name in ("position_error_bound_m", "orientation_error_bound_rad"):
            value = float(getattr(self, name))
            if np.isnan(value) or value < 0.0:
                raise ForwardDynamicsError(f"{name} must be non-negative")
        for name in ("gripper_state_before", "gripper_state_after"):
            try:
                value = tuple(float(item) for item in getattr(self, name))
            except (TypeError, ValueError) as error:
                raise ForwardDynamicsError(f"{name} must be numeric") from error
            if not all(np.isfinite(value)):
                raise ForwardDynamicsError(f"{name} must be finite")
            object.__setattr__(self, name, value)


class ForwardDynamics(ABC):
    """Common API for online Franka Panda and learned next-pose providers."""

    @abstractmethod
    def predict(self, request: ForwardDynamicsRequest) -> ForwardDynamicsResult:
        """Return exactly one source policy-step prediction or observation."""


class PandaOnlineOracle(ForwardDynamics):
    """Ground-truth oracle backed by one Robosuite Franka Panda policy step.

    The adapter deliberately accepts narrow callbacks, which keeps legacy
    benchmark classes out of this contract and makes the one-step invariant
    independently testable. Robosuite registers this Franka embodiment under
    the implementation identifier ``Panda``.
    """

    def __init__(
        self,
        *,
        step_policy_once: Callable[[np.ndarray], object],
        read_pose: Callable[[], CartesianPose],
        read_gripper_state: Callable[[], Sequence[float]],
        scaler: OscActionScaler | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        position_sync_tolerance_m: float = 1e-9,
        orientation_sync_tolerance_rad: float = 1e-6,
    ) -> None:
        self._step_policy_once = step_policy_once
        self._read_pose = read_pose
        self._read_gripper_state = read_gripper_state
        self._scaler = scaler or OscActionScaler()
        self._monotonic_ns = monotonic_ns
        self._position_sync_tolerance_m = float(position_sync_tolerance_m)
        self._orientation_sync_tolerance_rad = float(
            orientation_sync_tolerance_rad
        )
        if (
            not np.isfinite(self._position_sync_tolerance_m)
            or self._position_sync_tolerance_m < 0.0
            or not np.isfinite(self._orientation_sync_tolerance_rad)
            or self._orientation_sync_tolerance_rad < 0.0
        ):
            raise ForwardDynamicsError("oracle sync tolerances must be non-negative")

    def _validated_gripper(self) -> tuple[float, ...]:
        try:
            state = tuple(float(item) for item in self._read_gripper_state())
        except (TypeError, ValueError) as error:
            raise ForwardDynamicsError("Panda gripper state must be numeric") from error
        if not state or not all(np.isfinite(state)):
            raise ForwardDynamicsError("Panda gripper state must be non-empty and finite")
        return state

    def predict(self, request: ForwardDynamicsRequest) -> ForwardDynamicsResult:
        actual_current = self._read_pose()
        if actual_current.frame_id != request.current_pose.frame_id:
            raise ForwardDynamicsError("Panda oracle pose frame does not match request")
        position_error = float(
            np.linalg.norm(
                np.asarray(actual_current.position_m)
                - np.asarray(request.current_pose.position_m)
            )
        )
        orientation_error = quaternion_angular_error_rad(
            actual_current.quaternion_xyzw,
            request.current_pose.quaternion_xyzw,
        )
        if (
            position_error > self._position_sync_tolerance_m
            or orientation_error > self._orientation_sync_tolerance_rad
        ):
            raise ForwardDynamicsError(
                "Panda oracle request is not synchronized with simulator state"
            )

        scaled = self._scaler.scale(request.action)
        gripper_before = self._validated_gripper()
        started_ns = self._monotonic_ns()
        # This is the only policy-step call in the oracle.
        self._step_policy_once(request.action.as_policy_array().copy())
        achieved = self._read_pose()
        finished_ns = self._monotonic_ns()
        if achieved.frame_id != actual_current.frame_id:
            raise ForwardDynamicsError("Panda oracle changed pose frame after step")
        if achieved.timestamp_ns < actual_current.timestamp_ns:
            raise ForwardDynamicsError("Panda oracle returned a regressing timestamp")
        gripper_after = self._validated_gripper()
        desired = SourceDesiredPose(
            achieved,
            request.action.gripper_normalized,
            request.action.timestamp_ns,
        )
        return ForwardDynamicsResult(
            # Preserve the pose timestamp captured before policy inference.
            # ``actual_current`` is a synchronization read taken afterward.
            current_pose=request.current_pose,
            action=request.action,
            scaled_action=scaled,
            desired_pose=desired,
            gripper_state_before=gripper_before,
            gripper_state_after=gripper_after,
            provider="panda_online_oracle",
            latency_ns=max(0, finished_ns - started_ns),
            position_error_bound_m=0.0,
            orientation_error_bound_rad=0.0,
        )
