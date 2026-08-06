"""Leakage-safe Ridge training and held-out comparison reporting."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .collection import default_artifact_directory
from .dataset import (
    TransitionDataset,
    assert_no_group_leakage,
    split_by_group,
)
from .models import (
    PoseDeltaForwardDynamics,
    PoseErrorMetrics,
    compare_linear_models,
)


class TrainingError(RuntimeError):
    """Raised when dataset training or artifact policy fails."""


@dataclass(frozen=True)
class TrainingOutputs:
    model_path: Path
    report_path: Path
    selected_model: PoseDeltaForwardDynamics
    metrics: dict[str, PoseErrorMetrics]


def require_git_ignored(path: str | Path, *, repository_root: str | Path) -> None:
    """Refuse generated artifacts that Git does not ignore."""

    target = Path(path).expanduser().resolve()
    root = Path(repository_root).expanduser().resolve()
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise TrainingError("artifact path must remain inside the repository") from error
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", str(relative)],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise TrainingError(f"generated artifact path is not Git-ignored: {target}")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def train_and_evaluate(
    dataset_path: str | Path,
    *,
    output_directory: str | Path | None = None,
    validation_fraction: float = 0.25,
    group_by: str = "episode",
    split_seed: int = 0,
    ridge_alphas: Iterable[float] = (0.001, 0.01, 0.1, 1.0, 10.0),
    enforce_git_ignored: bool = True,
    repository_root: str | Path | None = None,
) -> TrainingOutputs:
    """Train Linear/Ridge, select Ridge, save it, and report all baselines."""

    dataset = TransitionDataset.read_jsonl(dataset_path)
    train, validation = split_by_group(
        dataset.records,
        validation_fraction=validation_fraction,
        group_by=group_by,
        random_seed=split_seed,
    )
    assert_no_group_leakage(train, validation, group_by=group_by)
    frame_ids = {record.current_pose.frame_id for record in dataset.records}
    if len(frame_ids) != 1:
        raise TrainingError("dataset must contain exactly one source frame")
    expected_frame_id = frame_ids.pop()
    metrics, selected_model = compare_linear_models(
        train,
        validation,
        expected_frame_id=expected_frame_id,
        ridge_alphas=ridge_alphas,
    )
    if selected_model.model_kind != "ridge":
        raise TrainingError("training did not select a Ridge model")

    output = (
        default_artifact_directory()
        if output_directory is None
        else Path(output_directory).expanduser()
    )
    model_path = output / "ridge_forward_dynamics.npz"
    report_path = output / "forward_dynamics_evaluation.json"
    if enforce_git_ignored:
        if repository_root is None:
            repository_root = Path(__file__).resolve().parents[5]
        require_git_ignored(model_path, repository_root=repository_root)
        require_git_ignored(report_path, repository_root=repository_root)
    selected_model.save(model_path)

    train_groups = sorted(
        {
            record.episode_id if group_by == "episode" else record.seed
            for record in train
        },
        key=str,
    )
    validation_groups = sorted(
        {
            record.episode_id if group_by == "episode" else record.seed
            for record in validation
        },
        key=str,
    )
    report = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "source_frame": expected_frame_id,
        "transition_count": len(dataset),
        "train_transition_count": len(train),
        "validation_transition_count": len(validation),
        "split": {
            "group_by": group_by,
            "split_seed": split_seed,
            "validation_fraction": validation_fraction,
            "train_groups": train_groups,
            "validation_groups": validation_groups,
        },
        "selected": {
            "model_kind": selected_model.model_kind,
            "ridge_alpha": selected_model.ridge_alpha,
            "artifact_path": str(model_path.resolve()),
        },
        "metrics": {name: asdict(value) for name, value in metrics.items()},
    }
    _atomic_json(report_path, report)
    return TrainingOutputs(model_path, report_path, selected_model, metrics)
