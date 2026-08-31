"""Command-line collection and training for xArm simulation transfer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .collection import collect_panda_oracle_transitions, default_artifact_directory
from .dataset import TransitionDataset
from .factory import (
    ForwardDynamicsConfig,
    create_forward_dynamics,
)
from .image_runtime import CrossPaintSourceWorker
from .profiles import ObservationMode
from .rollout import (
    run_cross_paint_oracle_episode,
    run_learned_replay_episode,
    run_online_oracle_episode,
    write_rollout_reports,
)
from .runtime import (
    FrankaPandaSourceRuntime,
    RobosuiteXArmAbsoluteAdapter,
    create_default_xarm_blocking_controller,
    create_xarm_target_environment,
    load_checkpoint_contract,
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
        default=None,
    )

    train = subparsers.add_parser("train", help="train and evaluate Linear/Ridge")
    train.add_argument("--dataset", required=True, type=Path)
    train.add_argument("--output-directory", type=Path, default=None)
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
        "--mode", choices=("oracle", "learned", "both"), default=None
    )
    rollout.add_argument("--model", type=Path)
    rollout.add_argument("--dataset", type=Path)
    rollout.add_argument("--seeds", type=_seeds, default=(0, 1, 2))
    rollout.add_argument(
        "--steps",
        type=int,
        default=None,
        help="policy steps; defaults to the checkpoint native rollout horizon",
    )
    rollout.add_argument(
        "--terminate-on-success",
        action="store_true",
        help="stop an oracle episode only when the target xArm task succeeds",
    )
    rollout.add_argument("--device", default="cpu")
    rollout.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    rollout.add_argument(
        "--oracle-output",
        type=Path,
        default=None,
    )
    rollout.add_argument(
        "--visual-input",
        choices=("cross_paint", "naive"),
        default="cross_paint",
        help="image-policy input; ignored by low-dimensional checkpoints",
    )
    rollout.add_argument(
        "--cross-paint-video",
        type=Path,
        default=None,
        help="optional five-panel image-policy debug MP4",
    )
    rollout.add_argument(
        "--mask-render-scale",
        type=int,
        choices=(1, 2),
        default=1,
        help="render segmentation masks at 84 or mask-only 168 resolution",
    )
    return parser


class _CrossPaintVideoWriter:
    """Write target, mask, background, source layer, and policy input panels."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._writer = imageio.get_writer(
            str(path), fps=20, codec="libx264", macro_block_size=2
        )

    @staticmethod
    def _labeled(image: np.ndarray, label: str) -> np.ndarray:
        panel = np.ascontiguousarray(image.copy())
        cv2.rectangle(panel, (0, 0), (83, 14), (0, 0, 0), thickness=-1)
        cv2.putText(
            panel,
            label,
            (2, 11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def __call__(self, _step_id, target_frame, source_step) -> None:
        mask = np.zeros_like(target_frame.rgb)
        mask[source_step.target_inpaint_mask] = (255, 255, 255)
        source = np.zeros_like(source_step.source_rgb)
        source[source_step.source_robot_mask] = source_step.source_rgb[
            source_step.source_robot_mask
        ]
        frame = np.concatenate(
            (
                self._labeled(target_frame.rgb, "target"),
                self._labeled(mask, "mask"),
                self._labeled(source_step.background_rgb, "inpaint"),
                self._labeled(source, "source"),
                self._labeled(source_step.policy_rgb, "policy"),
            ),
            axis=1,
        )
        self._writer.append_data(frame)

    def close(self) -> None:
        self._writer.close()


def _seed_video_path(path: Path, seed: int, seed_count: int) -> Path:
    if seed_count == 1:
        return path
    return path.with_name(f"{path.stem}_seed{seed}{path.suffix or '.mp4'}")


def _seed_episode(dataset: TransitionDataset, seed: int):
    records = tuple(record for record in dataset.records if record.seed == seed)
    episode_ids = {record.episode_id for record in records}
    if len(episode_ids) != 1:
        raise ValueError(
            f"dataset must contain exactly one episode for seed {seed}, "
            f"found {len(episode_ids)}"
        )
    return records


def _rollout_image(
    parsed: argparse.Namespace,
    *,
    checkpoint_contract,
    steps: int,
) -> int:
    if parsed.mode != "oracle":
        raise ValueError("image checkpoints currently support oracle rollout mode only")
    artifact_directory = default_artifact_directory(
        checkpoint_contract.task_profile,
        checkpoint_contract.observation_mode,
    )
    output_path = parsed.output or artifact_directory / "xarm_rollout_report.json"
    oracle_output_path = (
        parsed.oracle_output
        or artifact_directory / "rollout_oracle_transitions.jsonl"
    )
    repository_root = Path(__file__).resolve().parents[5]
    require_git_ignored(output_path, repository_root=repository_root)
    require_git_ignored(oracle_output_path, repository_root=repository_root)
    if parsed.cross_paint_video is not None:
        require_git_ignored(
            parsed.cross_paint_video, repository_root=repository_root
        )

    reports = []
    records = []
    for seed in parsed.seeds:
        target_environment = create_xarm_target_environment(parsed.checkpoint)
        target_adapter = RobosuiteXArmAbsoluteAdapter(target_environment)
        controller = create_default_xarm_blocking_controller(target_adapter)
        video_writer = None
        if parsed.cross_paint_video is not None:
            video_writer = _CrossPaintVideoWriter(
                _seed_video_path(
                    parsed.cross_paint_video, seed, len(parsed.seeds)
                )
            )
        try:
            with CrossPaintSourceWorker(
                parsed.checkpoint,
                device=parsed.device,
                mask_render_scale=parsed.mask_render_scale,
            ) as source_worker:
                dataset, report = run_cross_paint_oracle_episode(
                    source_worker=source_worker,
                    target_adapter=target_adapter,
                    controller=controller,
                    seed=seed,
                    steps=steps,
                    terminate_on_success=parsed.terminate_on_success,
                    visual_input=parsed.visual_input,
                    frame_callback=video_writer,
                )
            records.extend(dataset.records)
            reports.append(report)
        finally:
            if video_writer is not None:
                video_writer.close()
            target_environment.close()

    TransitionDataset(records).write_jsonl(oracle_output_path)
    output = write_rollout_reports(output_path, reports)
    summaries = []
    for report in reports:
        summary = asdict(report)
        summary.pop("steps")
        summaries.append(summary)
    print(json.dumps({"output": str(output), "reports": summaries}, sort_keys=True))
    return 0


def _rollout(parsed: argparse.Namespace) -> int:
    _, checkpoint_contract = load_checkpoint_contract(parsed.checkpoint)
    checkpoint_contract.require_runtime_supported()
    if parsed.mode is None:
        parsed.mode = (
            "oracle"
            if checkpoint_contract.observation_mode is not ObservationMode.LOW_DIM
            else "both"
        )
    steps = (
        checkpoint_contract.native_rollout_horizon
        if parsed.steps is None
        else parsed.steps
    )
    if steps <= 0:
        raise ValueError("rollout steps must be positive")
    if checkpoint_contract.observation_mode is not ObservationMode.LOW_DIM:
        return _rollout_image(
            parsed, checkpoint_contract=checkpoint_contract, steps=steps
        )
    if parsed.cross_paint_video is not None:
        raise ValueError("--cross-paint-video requires an image checkpoint")
    if parsed.mode in {"learned", "both"} and parsed.model is None:
        raise ValueError("--model is required for learned or both mode")
    if parsed.mode == "learned" and parsed.dataset is None:
        raise ValueError("--dataset is required for learned-only mode")

    artifact_directory = default_artifact_directory(
        checkpoint_contract.task_profile,
        checkpoint_contract.observation_mode,
    )
    output_path = parsed.output or artifact_directory / "xarm_rollout_report.json"
    oracle_output_path = (
        parsed.oracle_output
        or artifact_directory / "rollout_oracle_transitions.jsonl"
    )
    repository_root = Path(__file__).resolve().parents[5]
    require_git_ignored(output_path, repository_root=repository_root)
    if parsed.mode in {"oracle", "both"}:
        require_git_ignored(oracle_output_path, repository_root=repository_root)

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
            ForwardDynamicsConfig("learned_ridge", str(parsed.model)),
            checkpoint_contract=checkpoint_contract,
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
                    steps=steps,
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
                            min(steps, len(seed_records))
                            if parsed.mode == "both" and parsed.terminate_on_success
                            else steps
                        ),
                    )
                )
    finally:
        target_environment.close()
        if source_runtime is not None:
            source_runtime.core.close()

    if oracle_records:
        TransitionDataset(oracle_records).write_jsonl(oracle_output_path)
    output = write_rollout_reports(output_path, reports)
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
        _, checkpoint_contract = load_checkpoint_contract(parsed.checkpoint)
        checkpoint_contract.require_runtime_supported()
        output_path = parsed.output or (
            default_artifact_directory(
                checkpoint_contract.task_profile,
                checkpoint_contract.observation_mode,
            )
            / "panda_transitions.jsonl"
        )
        require_git_ignored(
            output_path, repository_root=Path(__file__).resolve().parents[5]
        )
        runtime = FrankaPandaSourceRuntime.from_checkpoint(
            parsed.checkpoint, device=parsed.device
        )
        dataset, summary = collect_panda_oracle_transitions(
            runtime, seeds=parsed.seeds, steps_per_episode=parsed.steps
        )
        output = dataset.write_jsonl(output_path)
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
