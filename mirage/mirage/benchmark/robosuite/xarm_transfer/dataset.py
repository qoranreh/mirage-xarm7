"""Versioned transition records and leakage-safe dataset splitting."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import CartesianPose, ScaledOscCommand, SourceAction


DATASET_VERSION = 1


class DatasetError(ValueError):
    """Raised when a transition dataset violates its contract."""


def _finite_tuple(value: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise DatasetError(f"{name} must be numeric") from error
    if not all(np.isfinite(result)):
        raise DatasetError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class TransitionRecord:
    """One source simulator policy transition."""

    episode_id: str
    step_id: int
    seed: int
    timestamp_ns: int
    current_pose: CartesianPose
    normalized_action: SourceAction
    scaled_action: ScaledOscCommand
    next_pose: CartesianPose
    gripper_state_before: tuple[float, ...]
    gripper_state_after: tuple[float, ...]
    provider: str
    latency_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise DatasetError("episode_id must not be empty")
        object.__setattr__(self, "episode_id", self.episode_id.strip())
        for name in ("step_id", "seed", "timestamp_ns", "latency_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise DatasetError(f"{name} must be a non-negative integer")
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise DatasetError("provider must not be empty")
        object.__setattr__(self, "provider", self.provider.strip())

        frame = self.current_pose.frame_id
        if any(
            item != frame
            for item in (
                self.normalized_action.frame_id,
                self.scaled_action.frame_id,
                self.next_pose.frame_id,
            )
        ):
            raise DatasetError("transition pose and action frames must match")
        if self.timestamp_ns != self.normalized_action.timestamp_ns:
            raise DatasetError("transition timestamp must equal action timestamp")
        if self.scaled_action.timestamp_ns != self.normalized_action.timestamp_ns:
            raise DatasetError("scaled and normalized action timestamps must match")
        if self.next_pose.timestamp_ns < self.current_pose.timestamp_ns:
            raise DatasetError("next pose timestamp predates current pose")

        object.__setattr__(
            self,
            "gripper_state_before",
            _finite_tuple(self.gripper_state_before, "gripper_state_before"),
        )
        object.__setattr__(
            self,
            "gripper_state_after",
            _finite_tuple(self.gripper_state_after, "gripper_state_after"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": DATASET_VERSION,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "seed": self.seed,
            "timestamp_ns": self.timestamp_ns,
            "current_pose": self.current_pose.to_dict(),
            "normalized_action": self.normalized_action.to_dict(),
            "scaled_action": self.scaled_action.to_dict(),
            "next_pose": self.next_pose.to_dict(),
            "gripper_state_before": list(self.gripper_state_before),
            "gripper_state_after": list(self.gripper_state_after),
            "provider": self.provider,
            "latency_ns": self.latency_ns,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TransitionRecord":
        expected = {
            "version",
            "episode_id",
            "step_id",
            "seed",
            "timestamp_ns",
            "current_pose",
            "normalized_action",
            "scaled_action",
            "next_pose",
            "gripper_state_before",
            "gripper_state_after",
            "provider",
            "latency_ns",
        }
        if set(value) != expected:
            unknown = sorted(set(value) - expected)
            missing = sorted(expected - set(value))
            raise DatasetError(
                f"transition keys mismatch; unknown={unknown}, missing={missing}"
            )
        if value["version"] != DATASET_VERSION:
            raise DatasetError(
                f"unsupported transition version {value['version']!r}"
            )
        return cls(
            episode_id=value["episode_id"],
            step_id=value["step_id"],
            seed=value["seed"],
            timestamp_ns=value["timestamp_ns"],
            current_pose=CartesianPose.from_dict(value["current_pose"]),
            normalized_action=SourceAction.from_dict(value["normalized_action"]),
            scaled_action=ScaledOscCommand.from_dict(value["scaled_action"]),
            next_pose=CartesianPose.from_dict(value["next_pose"]),
            gripper_state_before=value["gripper_state_before"],
            gripper_state_after=value["gripper_state_after"],
            provider=value["provider"],
            latency_ns=value["latency_ns"],
        )


def record_from_dynamics_result(
    result: object,
    *,
    episode_id: str,
    step_id: int,
    seed: int,
) -> TransitionRecord:
    """Convert a ForwardDynamicsResult without introducing a circular import."""

    required = (
        "current_pose",
        "action",
        "scaled_action",
        "desired_pose",
        "gripper_state_before",
        "gripper_state_after",
        "provider",
        "latency_ns",
    )
    if any(not hasattr(result, name) for name in required):
        raise DatasetError("result does not implement ForwardDynamicsResult")
    return TransitionRecord(
        episode_id=episode_id,
        step_id=step_id,
        seed=seed,
        timestamp_ns=result.action.timestamp_ns,
        current_pose=result.current_pose,
        normalized_action=result.action,
        scaled_action=result.scaled_action,
        next_pose=result.desired_pose.pose,
        gripper_state_before=result.gripper_state_before,
        gripper_state_after=result.gripper_state_after,
        provider=result.provider,
        latency_ns=result.latency_ns,
    )


class TransitionDataset:
    """Ordered collection with deterministic JSONL serialization."""

    def __init__(self, records: Iterable[TransitionRecord] = ()) -> None:
        self._records = tuple(records)
        seen: set[tuple[str, int]] = set()
        last_step_by_episode: dict[str, int] = {}
        for record in self._records:
            key = (record.episode_id, record.step_id)
            if key in seen:
                raise DatasetError(f"duplicate transition key {key!r}")
            seen.add(key)
            previous = last_step_by_episode.get(record.episode_id)
            if previous is not None and record.step_id <= previous:
                raise DatasetError("steps within each episode must be increasing")
            last_step_by_episode[record.episode_id] = record.step_id

    @property
    def records(self) -> tuple[TransitionRecord, ...]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def write_jsonl(self, path: str | Path) -> Path:
        """Atomically write records; callers must select an ignored artifact path."""

        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                for record in self._records:
                    stream.write(
                        json.dumps(
                            record.to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                    )
                    stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return destination

    @classmethod
    def read_jsonl(cls, path: str | Path) -> "TransitionDataset":
        records = []
        source = Path(path).expanduser()
        try:
            with source.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                        records.append(TransitionRecord.from_dict(value))
                    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
                        raise DatasetError(
                            f"invalid transition at {source}:{line_number}: {error}"
                        ) from error
        except OSError as error:
            raise DatasetError(f"cannot read transition dataset {source}: {error}") from error
        return cls(records)


def split_by_group(
    records: Sequence[TransitionRecord],
    *,
    validation_fraction: float = 0.2,
    group_by: str = "episode",
    random_seed: int = 0,
) -> tuple[tuple[TransitionRecord, ...], tuple[TransitionRecord, ...]]:
    """Split whole episodes or whole seeds, never individual transitions."""

    fraction = float(validation_fraction)
    if not np.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise DatasetError("validation_fraction must be strictly between 0 and 1")
    if group_by not in {"episode", "seed"}:
        raise DatasetError("group_by must be 'episode' or 'seed'")
    if type(random_seed) is not int or random_seed < 0:
        raise DatasetError("random_seed must be a non-negative integer")
    if not records:
        raise DatasetError("cannot split an empty dataset")

    key = (
        (lambda item: item.episode_id)
        if group_by == "episode"
        else (lambda item: item.seed)
    )
    groups = sorted({key(record) for record in records}, key=str)
    if len(groups) < 2:
        raise DatasetError(f"at least two {group_by} groups are required")
    shuffled = list(np.random.default_rng(random_seed).permutation(groups))
    validation_count = max(1, int(round(len(groups) * fraction)))
    validation_count = min(validation_count, len(groups) - 1)
    validation_groups = set(shuffled[:validation_count])
    train = tuple(record for record in records if key(record) not in validation_groups)
    validation = tuple(
        record for record in records if key(record) in validation_groups
    )
    if not train or not validation:
        raise DatasetError("group split unexpectedly produced an empty partition")
    return train, validation


def assert_no_group_leakage(
    train: Sequence[TransitionRecord],
    validation: Sequence[TransitionRecord],
    *,
    group_by: str = "episode",
) -> None:
    """Fail if a grouping key appears in both partitions."""

    if group_by == "episode":
        train_groups = {item.episode_id for item in train}
        validation_groups = {item.episode_id for item in validation}
    elif group_by == "seed":
        train_groups = {item.seed for item in train}
        validation_groups = {item.seed for item in validation}
    else:
        raise DatasetError("group_by must be 'episode' or 'seed'")
    overlap = train_groups & validation_groups
    if overlap:
        raise DatasetError(f"{group_by} leakage detected: {sorted(overlap, key=str)}")
