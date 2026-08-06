"""Command-line collection and training for xArm simulation transfer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .collection import collect_panda_oracle_transitions, default_artifact_directory
from .dataset import TransitionDataset
from .factory import (
    ForwardDynamicsConfig,
    create_forward_dynamics,
)
from .rollout import (
    run_learned_replay_episode,
    run_online_oracle_episode,
    write_rollout_reports,
)
from .runtime import (
    FrankaPandaSourceRuntime,
    RobosuiteXArmAbsoluteAdapter,
    create_default_xarm_blocking_controller,
    create_xarm_target_environment,
)
from .training import require_git_ignored, train_and_evaluate


def _seeds(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Franka Panda forward-dynamics dataset and Ridge tools"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="collect online Panda transitions")
    collect.add_argument("--checkpoint", required=True)
    collect.add_argument("--seeds", type=_seeds, default=(0, 1, 2, 3))
    collect.add_argument("--steps", type=int, default=50)
    collect.add_argument("--device", default="cpu")
    collect.add_argument(
        "--output",
        type=Path,
        default=default_artifact_directory() / "panda_transitions.jsonl",
    )

    train = subparsers.add_parser("train", help="train and evaluate Linear/Ridge")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--output-directory", type=Path, default=default_artifact_directory())
    train.add_argument("--group-by", choices=("episode", "seed"), default="episode")
    train.add_argument("--validation-fraction", type=float, default=0.25)
    train.add_argument("--split-seed", type=int, default=0)
    train.add_argument(
        "--ridge-alphas", type=float, nargs="+", default=(0.001, 0.01, 0.1, 1.0, 10.0)
    )

    rollout = subparsers.add_parser(
        "rollout", help="run real Panda-oracle and/or learned xArm transfer"
    )
    rollout.add_argument("--checkpoint", required=True)
    rollout.add_argument(
        "--mode", choices=("oracle", "learned", "both"), default="both"
    )
    rollout.add_argument("--model", type=Path)
    rollout.add_argument("--dataset", type=Path)
    rollout.add_argument("--seeds", type=_seeds, default=(0, 1, 2))
    rollout.add_argument("--steps", type=int, default=50)
    rollout.add_argument(
        "--terminate-on-success",
        action="store_true",
        help="stop an oracle episode when either simulation task succeeds",
    )
    rollout.add_argument("--device", default="cpu")
    rollout.add_argument(
        "--output",
        type=Path,
        default=default_artifact_directory() / "xarm_rollout_report.json",
    )
    rollout.add_argument(
        "--oracle-output",
        type=Path,
        default=default_artifact_directory() / "rollout_oracle_transitions.jsonl",
    )
    return parser


def _seed_episode(dataset: TransitionDataset, seed: int):
    records = tuple(record for record in dataset.records if record.seed == seed)
    episode_ids = {record.episode_id for record in records}
    if len(episode_ids) != 1:
        raise ValueError(
            f"dataset must contain exactly one episode for seed {seed}, "
            f"found {len(episode_ids)}"
        )
    return records


def _rollout(parsed: argparse.Namespace) -> int:
    if parsed.steps <= 0:
        raise ValueError("rollout steps must be positive")
    if parsed.mode in {"learned", "both"} and parsed.model is None:
        raise ValueError("--model is required for learned or both mode")
    if parsed.mode == "learned" and parsed.dataset is None:
        raise ValueError("--dataset is required for learned-only mode")

    repository_root = Path(__file__).resolve().parents[5]
    require_git_ignored(parsed.output, repository_root=repository_root)
    if parsed.mode in {"oracle", "both"}:
        require_git_ignored(parsed.oracle_output, repository_root=repository_root)

    source_runtime = None
    target_environment = create_xarm_target_environment(parsed.checkpoint)
    target_adapter = RobosuiteXArmAbsoluteAdapter(target_environment)
    controller = create_default_xarm_blocking_controller(target_adapter)
    if parsed.mode in {"oracle", "both"}:
        source_runtime = FrankaPandaSourceRuntime.from_checkpoint(
            parsed.checkpoint, device=parsed.device
        )
    learned = None
    replay_dataset = None
    if parsed.mode in {"learned", "both"}:
        learned = create_forward_dynamics(
            ForwardDynamicsConfig("learned_ridge", str(parsed.model))
        )
    if parsed.mode == "learned":
        replay_dataset = TransitionDataset.read_jsonl(parsed.dataset)

    reports = []
    oracle_records = []
    try:
        for seed in parsed.seeds:
            seed_records = None
            if parsed.mode in {"oracle", "both"}:
                dataset, report = run_online_oracle_episode(
                    source_runtime=source_runtime,
                    target_adapter=target_adapter,
                    controller=controller,
                    seed=seed,
                    steps=parsed.steps,
                    terminate_on_success=parsed.terminate_on_success,
                )
                reports.append(report)
                oracle_records.extend(dataset.records)
                seed_records = dataset.records
            if parsed.mode in {"learned", "both"}:
                if parsed.mode == "learned":
                    seed_records = _seed_episode(replay_dataset, seed)
                reports.append(
                    run_learned_replay_episode(
                        oracle_transitions=seed_records,
                        learned_dynamics=learned,
                        target_adapter=target_adapter,
                        controller=controller,
                        seed=seed,
                        steps=(
                            min(parsed.steps, len(seed_records))
                            if parsed.mode == "both" and parsed.terminate_on_success
                            else parsed.steps
                        ),
                    )
                )
    finally:
        target_environment.close()
        if source_runtime is not None:
            source_runtime.core.close()

    if oracle_records:
        TransitionDataset(oracle_records).write_jsonl(parsed.oracle_output)
    output = write_rollout_reports(parsed.output, reports)
    summaries = []
    for report in reports:
        summary = asdict(report)
        summary.pop("steps")
        summaries.append(summary)
    print(json.dumps({"output": str(output), "reports": summaries}, sort_keys=True))
    return 0


def main(args: list[str] | None = None) -> int:
    parsed = _parser().parse_args(args)
    if parsed.command == "rollout":
        return _rollout(parsed)
    if parsed.command == "collect":
        require_git_ignored(
            parsed.output, repository_root=Path(__file__).resolve().parents[5]
        )
        runtime = FrankaPandaSourceRuntime.from_checkpoint(
            parsed.checkpoint, device=parsed.device
        )
        dataset, summary = collect_panda_oracle_transitions(
            runtime, seeds=parsed.seeds, steps_per_episode=parsed.steps
        )
        output = dataset.write_jsonl(parsed.output)
        print(json.dumps({"output": str(output), **asdict(summary)}, sort_keys=True))
        return 0

    outputs = train_and_evaluate(
        parsed.dataset,
        output_directory=parsed.output_directory,
        validation_fraction=parsed.validation_fraction,
        group_by=parsed.group_by,
        split_seed=parsed.split_seed,
        ridge_alphas=parsed.ridge_alphas,
    )
    print(
        json.dumps(
            {
                "model_path": str(outputs.model_path),
                "report_path": str(outputs.report_path),
                "selected_ridge_alpha": outputs.selected_model.ridge_alpha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
