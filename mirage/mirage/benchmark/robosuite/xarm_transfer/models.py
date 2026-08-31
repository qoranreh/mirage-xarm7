"""Linear and Ridge source forward-dynamics models and baselines."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from .contracts import CartesianPose, OscActionScaler, SourceDesiredPose
from .dataset import TransitionRecord
from .dynamics import (
    ForwardDynamics,
    ForwardDynamicsError,
    ForwardDynamicsRequest,
    ForwardDynamicsResult,
)
from .geometry import (
    integrate_world_rotation_delta,
    quaternion_angular_error_rad,
    quaternion_to_rotvec,
    relative_rotation_vector,
)
from .profiles import (
    CheckpointContract,
    ObservationMode,
    RuntimeContractError,
    task_profile_for_name,
)


MODEL_VERSION = 2
LEGACY_TASK_NAME = "PickPlaceCan"
LEGACY_OBSERVATION_MODE = ObservationMode.LOW_DIM
FEATURE_DIM = 13
OUTPUT_DIM = 6


class ModelError(ForwardDynamicsError):
    """Raised when a learned dynamics artifact is invalid or unsafe."""


@dataclass(frozen=True)
class PoseErrorMetrics:
    """Separate held-out position and orientation errors."""

    sample_count: int
    position_mean_m: float
    position_rmse_m: float
    position_max_m: float
    orientation_mean_rad: float
    orientation_rmse_rad: float
    orientation_max_rad: float
    orientation_mean_deg: float
    orientation_rmse_deg: float
    orientation_max_deg: float


def pose_features(record_or_request: object) -> np.ndarray:
    """Encode current pose + normalized action without quaternion sign ambiguity."""

    if isinstance(record_or_request, TransitionRecord):
        pose = record_or_request.current_pose
        action = record_or_request.normalized_action
    elif isinstance(record_or_request, ForwardDynamicsRequest):
        pose = record_or_request.current_pose
        action = record_or_request.action
    else:
        raise ModelError("features require TransitionRecord or ForwardDynamicsRequest")
    feature = np.concatenate(
        (
            np.asarray(pose.position_m, dtype=np.float64),
            quaternion_to_rotvec(pose.quaternion_xyzw),
            action.as_normalized_array(),
        )
    )
    if feature.shape != (FEATURE_DIM,) or not np.all(np.isfinite(feature)):
        raise ModelError("feature encoding produced invalid values")
    return feature


def pose_delta_target(record: TransitionRecord) -> np.ndarray:
    """Encode position and world-frame orientation deltas."""

    delta = np.concatenate(
        (
            np.asarray(record.next_pose.position_m)
            - np.asarray(record.current_pose.position_m),
            relative_rotation_vector(
                record.current_pose.quaternion_xyzw,
                record.next_pose.quaternion_xyzw,
            ),
        )
    )
    if delta.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(delta)):
        raise ModelError("target encoding produced invalid values")
    return delta


class PoseDeltaForwardDynamics(ForwardDynamics):
    """Serializable linear map from pose/action features to stable pose deltas."""

    def __init__(
        self,
        *,
        model_kind: str,
        expected_frame_id: str,
        ridge_alpha: float = 1.0,
        policy_period_ns: int = 50_000_000,
        scaler: OscActionScaler | None = None,
        task_name: str = LEGACY_TASK_NAME,
        observation_mode: ObservationMode | str = LEGACY_OBSERVATION_MODE,
    ) -> None:
        if model_kind not in {"linear", "ridge"}:
            raise ModelError("model_kind must be 'linear' or 'ridge'")
        if not isinstance(expected_frame_id, str) or not expected_frame_id.strip():
            raise ModelError("expected_frame_id must not be empty")
        alpha = float(ridge_alpha)
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ModelError("ridge_alpha must be finite and non-negative")
        if type(policy_period_ns) is not int or policy_period_ns <= 0:
            raise ModelError("policy_period_ns must be a positive integer")
        try:
            profile = task_profile_for_name(task_name)
            resolved_mode = ObservationMode.parse(observation_mode)
        except RuntimeContractError as error:
            raise ModelError(str(error)) from error
        self.model_kind = model_kind
        self.expected_frame_id = expected_frame_id.strip()
        self.ridge_alpha = alpha
        self.policy_period_ns = policy_period_ns
        self.scaler = scaler or OscActionScaler()
        self.task_name = profile.env_name
        self.observation_mode = resolved_mode
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.validation_metrics: PoseErrorMetrics | None = None

    @property
    def fitted(self) -> bool:
        return self.coef_ is not None and self.intercept_ is not None

    def _validate_records(
        self, records: Sequence[TransitionRecord], description: str
    ) -> None:
        if not records:
            raise ModelError(f"{description} records must not be empty")
        for record in records:
            if record.current_pose.frame_id != self.expected_frame_id:
                raise ModelError(
                    f"{description} frame {record.current_pose.frame_id!r} does not "
                    f"match {self.expected_frame_id!r}"
                )
            if (
                record.task_name != self.task_name
                or record.observation_mode is not self.observation_mode
            ):
                raise ModelError(
                    f"{description} task/observation mode does not match model"
                )

    def require_checkpoint_contract(self, contract: CheckpointContract) -> None:
        try:
            contract.require_artifact_contract(
                task_name=self.task_name,
                observation_mode=self.observation_mode,
            )
        except RuntimeContractError as error:
            raise ModelError(str(error)) from error

    def fit(self, records: Sequence[TransitionRecord]) -> "PoseDeltaForwardDynamics":
        self._validate_records(records, "training")
        if len(records) < 2:
            raise ModelError("at least two training transitions are required")
        features = np.stack([pose_features(record) for record in records])
        targets = np.stack([pose_delta_target(record) for record in records])
        estimator = (
            LinearRegression()
            if self.model_kind == "linear"
            else Ridge(alpha=self.ridge_alpha)
        )
        estimator.fit(features, targets)
        coefficient = np.asarray(estimator.coef_, dtype=np.float64)
        intercept = np.asarray(estimator.intercept_, dtype=np.float64)
        if coefficient.shape != (OUTPUT_DIM, FEATURE_DIM):
            raise ModelError(f"estimator coefficient shape is {coefficient.shape}")
        if intercept.shape != (OUTPUT_DIM,):
            raise ModelError(f"estimator intercept shape is {intercept.shape}")
        if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(intercept)):
            raise ModelError("estimator produced non-finite parameters")
        self.coef_ = coefficient
        self.intercept_ = intercept
        self.validation_metrics = None
        return self

    def _prediction_delta(self, request: ForwardDynamicsRequest) -> np.ndarray:
        if not self.fitted:
            raise ModelError("forward-dynamics model is not fitted")
        if request.current_pose.frame_id != self.expected_frame_id:
            raise ModelError(
                f"request frame {request.current_pose.frame_id!r} does not match "
                f"model frame {self.expected_frame_id!r}"
            )
        feature = pose_features(request)
        delta = self.coef_ @ feature + self.intercept_
        if delta.shape != (OUTPUT_DIM,) or not np.all(np.isfinite(delta)):
            raise ModelError("model prediction is non-finite or has the wrong shape")
        return delta

    def predict(self, request: ForwardDynamicsRequest) -> ForwardDynamicsResult:
        started_ns = time.monotonic_ns()
        delta = self._prediction_delta(request)
        position = np.asarray(request.current_pose.position_m) + delta[:3]
        quaternion = integrate_world_rotation_delta(
            request.current_pose.quaternion_xyzw, delta[3:]
        )
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise ModelError("pose integration produced a non-finite prediction")
        next_pose = CartesianPose(
            tuple(position),
            tuple(quaternion),
            self.expected_frame_id,
            request.action.timestamp_ns + self.policy_period_ns,
        )
        metrics = self.validation_metrics
        position_bound = math.inf if metrics is None else metrics.position_max_m
        orientation_bound = (
            math.inf if metrics is None else metrics.orientation_max_rad
        )
        return ForwardDynamicsResult(
            current_pose=request.current_pose,
            action=request.action,
            scaled_action=self.scaler.scale(request.action),
            desired_pose=SourceDesiredPose(
                next_pose,
                request.action.gripper_normalized,
                request.action.timestamp_ns,
            ),
            gripper_state_before=(),
            gripper_state_after=(),
            provider=f"learned_{self.model_kind}",
            latency_ns=max(0, time.monotonic_ns() - started_ns),
            position_error_bound_m=position_bound,
            orientation_error_bound_rad=orientation_bound,
        )

    def evaluate(
        self,
        records: Sequence[TransitionRecord],
        *,
        trust_as_validation: bool = False,
    ) -> PoseErrorMetrics:
        self._validate_records(records, "evaluation")
        position_errors = []
        orientation_errors = []
        for record in records:
            request = ForwardDynamicsRequest(
                record.current_pose, record.normalized_action
            )
            prediction = self.predict(request).desired_pose.pose
            position_errors.append(
                float(
                    np.linalg.norm(
                        np.asarray(prediction.position_m)
                        - np.asarray(record.next_pose.position_m)
                    )
                )
            )
            orientation_errors.append(
                quaternion_angular_error_rad(
                    prediction.quaternion_xyzw,
                    record.next_pose.quaternion_xyzw,
                )
            )
        position = np.asarray(position_errors, dtype=np.float64)
        orientation = np.asarray(orientation_errors, dtype=np.float64)
        metrics = PoseErrorMetrics(
            sample_count=len(records),
            position_mean_m=float(np.mean(position)),
            position_rmse_m=float(np.sqrt(np.mean(position**2))),
            position_max_m=float(np.max(position)),
            orientation_mean_rad=float(np.mean(orientation)),
            orientation_rmse_rad=float(np.sqrt(np.mean(orientation**2))),
            orientation_max_rad=float(np.max(orientation)),
            orientation_mean_deg=float(np.degrees(np.mean(orientation))),
            orientation_rmse_deg=float(
                np.degrees(np.sqrt(np.mean(orientation**2)))
            ),
            orientation_max_deg=float(np.degrees(np.max(orientation))),
        )
        if trust_as_validation:
            self.validation_metrics = metrics
        return metrics

    def save(self, path: str | Path) -> Path:
        if not self.fitted:
            raise ModelError("cannot save an unfitted model")
        destination = Path(path).expanduser()
        if destination.suffix != ".npz":
            raise ModelError("model artifact path must end in .npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": MODEL_VERSION,
            "model_kind": self.model_kind,
            "expected_frame_id": self.expected_frame_id,
            "ridge_alpha": self.ridge_alpha,
            "policy_period_ns": self.policy_period_ns,
            "feature_dim": FEATURE_DIM,
            "output_dim": OUTPUT_DIM,
            "position_scale_m": list(self.scaler.position_scale_m),
            "orientation_scale_rad": list(self.scaler.orientation_scale_rad),
            "control_delta": self.scaler.control_delta,
            "policy_rate_hz": self.scaler.policy_rate_hz,
            "validation_metrics": (
                None
                if self.validation_metrics is None
                else asdict(self.validation_metrics)
            ),
            "task_name": self.task_name,
            "observation_mode": self.observation_mode.value,
        }
        np.savez_compressed(
            destination,
            coef=self.coef_,
            intercept=self.intercept_,
            metadata=np.asarray(
                json.dumps(metadata, sort_keys=True, allow_nan=False)
            ),
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "PoseDeltaForwardDynamics":
        source = Path(path).expanduser()
        try:
            with np.load(source, allow_pickle=False) as artifact:
                if set(artifact.files) != {"coef", "intercept", "metadata"}:
                    raise ModelError("model artifact keys do not match schema")
                coefficient = np.asarray(artifact["coef"], dtype=np.float64)
                intercept = np.asarray(artifact["intercept"], dtype=np.float64)
                metadata_value = artifact["metadata"]
                if metadata_value.shape != ():
                    raise ModelError("model metadata must be a scalar JSON string")
                metadata = json.loads(str(metadata_value.item()))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ModelError(f"cannot load model artifact {source}: {error}") from error
        expected_v1 = {
            "version",
            "model_kind",
            "expected_frame_id",
            "ridge_alpha",
            "policy_period_ns",
            "feature_dim",
            "output_dim",
            "position_scale_m",
            "orientation_scale_rad",
            "control_delta",
            "policy_rate_hz",
            "validation_metrics",
        }
        expected_v2 = expected_v1 | {"task_name", "observation_mode"}
        version = metadata.get("version")
        expected_keys = expected_v1 if version == 1 else expected_v2
        if set(metadata) != expected_keys or version not in (1, MODEL_VERSION):
            raise ModelError("model metadata does not match a supported schema")
        if metadata["feature_dim"] != FEATURE_DIM or metadata["output_dim"] != OUTPUT_DIM:
            raise ModelError("model artifact dimensions are incompatible")
        if coefficient.shape != (OUTPUT_DIM, FEATURE_DIM):
            raise ModelError("model coefficient shape is incompatible")
        if intercept.shape != (OUTPUT_DIM,):
            raise ModelError("model intercept shape is incompatible")
        if not np.all(np.isfinite(coefficient)) or not np.all(np.isfinite(intercept)):
            raise ModelError("model parameters contain non-finite values")
        scaler = OscActionScaler(
            tuple(metadata["position_scale_m"]),
            tuple(metadata["orientation_scale_rad"]),
            metadata["control_delta"],
            metadata["policy_rate_hz"],
        )
        model = cls(
            model_kind=metadata["model_kind"],
            expected_frame_id=metadata["expected_frame_id"],
            ridge_alpha=metadata["ridge_alpha"],
            policy_period_ns=metadata["policy_period_ns"],
            scaler=scaler,
            task_name=(
                LEGACY_TASK_NAME if version == 1 else metadata["task_name"]
            ),
            observation_mode=(
                LEGACY_OBSERVATION_MODE
                if version == 1
                else metadata["observation_mode"]
            ),
        )
        model.coef_ = coefficient
        model.intercept_ = intercept
        if metadata["validation_metrics"] is not None:
            model.validation_metrics = PoseErrorMetrics(
                **metadata["validation_metrics"]
            )
        return model


class AnalyticBaseline(ForwardDynamics):
    """Identity or ideal delta-integration baseline under the same API."""

    def __init__(
        self,
        *,
        mode: str,
        expected_frame_id: str,
        scaler: OscActionScaler | None = None,
        policy_period_ns: int = 50_000_000,
    ) -> None:
        if mode not in {"identity", "delta_integration"}:
            raise ModelError("baseline mode must be identity or delta_integration")
        self.mode = mode
        self.expected_frame_id = expected_frame_id
        self.scaler = scaler or OscActionScaler()
        self.policy_period_ns = policy_period_ns

    def predict(self, request: ForwardDynamicsRequest) -> ForwardDynamicsResult:
        if request.current_pose.frame_id != self.expected_frame_id:
            raise ModelError("baseline request frame mismatch")
        scaled = self.scaler.scale(request.action)
        if self.mode == "identity":
            position = np.asarray(request.current_pose.position_m)
            quaternion = np.asarray(request.current_pose.quaternion_xyzw)
        else:
            position = np.asarray(request.current_pose.position_m) + np.asarray(
                scaled.translation_delta_m
            )
            quaternion = integrate_world_rotation_delta(
                request.current_pose.quaternion_xyzw,
                scaled.rotation_delta_axis_angle_rad,
            )
        pose = CartesianPose(
            tuple(position),
            tuple(quaternion),
            self.expected_frame_id,
            request.action.timestamp_ns + self.policy_period_ns,
        )
        return ForwardDynamicsResult(
            request.current_pose,
            request.action,
            scaled,
            SourceDesiredPose(
                pose,
                request.action.gripper_normalized,
                request.action.timestamp_ns,
            ),
            (),
            (),
            f"baseline_{self.mode}",
            0,
            math.inf,
            math.inf,
        )


def evaluate_provider(
    provider: ForwardDynamics,
    records: Sequence[TransitionRecord],
) -> PoseErrorMetrics:
    """Evaluate any provider against recorded held-out transitions."""

    if not records:
        raise ModelError("evaluation records must not be empty")
    position_errors = []
    orientation_errors = []
    for record in records:
        result = provider.predict(
            ForwardDynamicsRequest(record.current_pose, record.normalized_action)
        )
        predicted = result.desired_pose.pose
        if predicted.frame_id != record.next_pose.frame_id:
            raise ModelError("provider changed frame during evaluation")
        position_errors.append(
            float(
                np.linalg.norm(
                    np.asarray(predicted.position_m)
                    - np.asarray(record.next_pose.position_m)
                )
            )
        )
        orientation_errors.append(
            quaternion_angular_error_rad(
                predicted.quaternion_xyzw, record.next_pose.quaternion_xyzw
            )
        )
    position = np.asarray(position_errors)
    orientation = np.asarray(orientation_errors)
    return PoseErrorMetrics(
        len(records),
        float(np.mean(position)),
        float(np.sqrt(np.mean(position**2))),
        float(np.max(position)),
        float(np.mean(orientation)),
        float(np.sqrt(np.mean(orientation**2))),
        float(np.max(orientation)),
        float(np.degrees(np.mean(orientation))),
        float(np.degrees(np.sqrt(np.mean(orientation**2)))),
        float(np.degrees(np.max(orientation))),
    )


def compare_linear_models(
    train: Sequence[TransitionRecord],
    validation: Sequence[TransitionRecord],
    *,
    expected_frame_id: str,
    ridge_alphas: Iterable[float] = (0.01, 0.1, 1.0, 10.0),
) -> tuple[dict[str, PoseErrorMetrics], PoseDeltaForwardDynamics]:
    """Fit Linear/Ridge candidates and return metrics plus the best Ridge.

    Linear regression remains an evaluated comparator, while the deployable
    learned provider is deliberately Ridge as required by the transfer path.
    """

    if not train or not validation:
        raise ModelError("training and validation records must not be empty")
    task_name = train[0].task_name
    observation_mode = train[0].observation_mode

    candidates: list[tuple[str, PoseDeltaForwardDynamics]] = [
        (
            "linear",
            PoseDeltaForwardDynamics(
                model_kind="linear",
                expected_frame_id=expected_frame_id,
                task_name=task_name,
                observation_mode=observation_mode,
            ),
        )
    ]
    for alpha_value in ridge_alphas:
        alpha = float(alpha_value)
        candidates.append(
            (
                f"ridge_alpha_{alpha:g}",
                PoseDeltaForwardDynamics(
                    model_kind="ridge",
                    expected_frame_id=expected_frame_id,
                    ridge_alpha=alpha,
                    task_name=task_name,
                    observation_mode=observation_mode,
                ),
            )
        )
    metrics: dict[str, PoseErrorMetrics] = {}
    best: PoseDeltaForwardDynamics | None = None
    best_score = math.inf
    for name, model in candidates:
        model.fit(train)
        result = model.evaluate(validation, trust_as_validation=True)
        metrics[name] = result
        score = result.position_rmse_m + result.orientation_rmse_rad
        if model.model_kind == "ridge" and score < best_score:
            best = model
            best_score = score
    for mode in ("identity", "delta_integration"):
        metrics[f"baseline_{mode}"] = evaluate_provider(
            AnalyticBaseline(mode=mode, expected_frame_id=expected_frame_id),
            validation,
        )
    metrics["oracle_ground_truth"] = PoseErrorMetrics(
        sample_count=len(validation),
        position_mean_m=0.0,
        position_rmse_m=0.0,
        position_max_m=0.0,
        orientation_mean_rad=0.0,
        orientation_rmse_rad=0.0,
        orientation_max_rad=0.0,
        orientation_mean_deg=0.0,
        orientation_rmse_deg=0.0,
        orientation_max_deg=0.0,
    )
    if best is None:
        raise ModelError("no Ridge forward-dynamics candidates were evaluated")
    return metrics, best
