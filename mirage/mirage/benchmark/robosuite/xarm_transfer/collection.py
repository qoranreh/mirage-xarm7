"""Transition collection from the real Robosuite Franka Panda oracle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .dataset import (
    TransitionDataset,
    TransitionRecord,
    record_from_dynamics_result,
)
from .dynamics import ForwardDynamicsRequest
from .runtime import FrankaPandaSourceRuntime
from .profiles import ObservationMode, TaskProfile, TASK_PROFILES


class CollectionError(RuntimeError):
    """Raised when an oracle dataset rollout violates its horizon contract."""


@dataclass(frozen=True)
class CollectionSummary:
    episode_count: int
    transition_count: int
    seeds: tuple[int, ...]
    steps_per_episode: int


def collect_panda_oracle_transitions(
    runtime: FrankaPandaSourceRuntime,
    *,
    seeds: Iterable[int],
    steps_per_episode: int,
) -> tuple[TransitionDataset, CollectionSummary]:
    """Collect full fixed-horizon episodes with exactly one step per action."""

    seed_values = tuple(int(seed) for seed in seeds)
    if not seed_values or any(seed < 0 for seed in seed_values):
        raise CollectionError("seeds must contain non-negative integers")
    if len(set(seed_values)) != len(seed_values):
        raise CollectionError("seeds must be unique")
    if type(steps_per_episode) is not int or steps_per_episode <= 0:
        raise CollectionError("steps_per_episode must be a positive integer")

    records: list[TransitionRecord] = []
    for seed in seed_values:
        runtime.reset(seed)
        oracle = runtime.make_oracle()
        contract = runtime.checkpoint_contract
        episode_id = f"{contract.task_profile.slug}-panda-seed-{seed}"
        for step_id in range(steps_per_episode):
            current_pose = runtime.read_pose()
            action_timestamp = max(runtime.clock_ns(), current_pose.timestamp_ns + 1)
            action = runtime.policy_action(timestamp_ns=action_timestamp)
            result = oracle.predict(ForwardDynamicsRequest(current_pose, action))
            records.append(
                record_from_dynamics_result(
                    result,
                    episode_id=episode_id,
                    step_id=step_id,
                    seed=seed,
                    task_name=contract.task_name,
                    observation_mode=contract.observation_mode,
                )
            )
        if runtime.policy_step_count != steps_per_episode:
            raise CollectionError(
                f"seed {seed} executed {runtime.policy_step_count} policy steps, "
                f"expected {steps_per_episode}"
            )
    dataset = TransitionDataset(records)
    return dataset, CollectionSummary(
        episode_count=len(seed_values),
        transition_count=len(records),
        seeds=seed_values,
        steps_per_episode=steps_per_episode,
    )


def default_artifact_directory(
    task_profile: TaskProfile | None = None,
    observation_mode: ObservationMode | str = ObservationMode.LOW_DIM,
) -> Path:
    profile = task_profile or TASK_PROFILES["PickPlaceCan"]
    mode = ObservationMode.parse(observation_mode)
    return (
        Path(__file__).resolve().parents[1]
        / "results"
        / "xarm_transfer"
        / profile.slug
        / mode.value
    )
