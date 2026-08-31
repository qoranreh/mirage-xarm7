"""Configuration-driven forward-dynamics provider selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .dynamics import ForwardDynamics, PandaOnlineOracle
from .models import ModelError, PoseDeltaForwardDynamics
from .profiles import CheckpointContract


class ForwardDynamicsConfigError(ValueError):
    """Raised when oracle/model selection is incomplete or unsafe."""


@dataclass(frozen=True)
class ForwardDynamicsConfig:
    mode: str
    model_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"online_oracle", "learned_ridge"}:
            raise ForwardDynamicsConfigError(
                "mode must be online_oracle or learned_ridge"
            )
        if self.mode == "online_oracle" and self.model_path is not None:
            raise ForwardDynamicsConfigError(
                "online_oracle configuration must not set model_path"
            )
        if self.mode == "learned_ridge" and (
            not isinstance(self.model_path, str) or not self.model_path.strip()
        ):
            raise ForwardDynamicsConfigError(
                "learned_ridge configuration requires model_path"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ForwardDynamicsConfig":
        if set(value) - {"mode", "model_path"}:
            raise ForwardDynamicsConfigError(
                "forward_dynamics configuration has unknown keys"
            )
        if "mode" not in value:
            raise ForwardDynamicsConfigError("forward_dynamics mode is required")
        return cls(value["mode"], value.get("model_path"))


def create_forward_dynamics(
    config: ForwardDynamicsConfig,
    *,
    online_oracle: PandaOnlineOracle | None = None,
    checkpoint_contract: CheckpointContract | None = None,
) -> ForwardDynamics:
    """Create one provider under the common interface."""

    if config.mode == "online_oracle":
        if online_oracle is None:
            raise ForwardDynamicsConfigError(
                "online_oracle mode requires a PandaOnlineOracle"
            )
        return online_oracle

    try:
        model = PoseDeltaForwardDynamics.load(Path(config.model_path).expanduser())
    except ModelError as error:
        raise ForwardDynamicsConfigError(str(error)) from error
    if model.model_kind != "ridge":
        raise ForwardDynamicsConfigError(
            "learned_ridge mode refuses a non-Ridge artifact"
        )
    if model.validation_metrics is None:
        raise ForwardDynamicsConfigError(
            "learned Ridge artifact has no held-out validation bounds"
        )
    if checkpoint_contract is not None:
        try:
            model.require_checkpoint_contract(checkpoint_contract)
        except ModelError as error:
            raise ForwardDynamicsConfigError(str(error)) from error
    return model
