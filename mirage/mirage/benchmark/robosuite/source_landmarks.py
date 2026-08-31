"""Create physical calibration landmarks from a Robomimic checkpoint."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils

from mirage_state.calibration import (
    CalibrationDataset,
    CalibrationError,
    CalibrationThresholds,
    LandmarkCorrespondence,
    save_calibration_dataset,
)


SUPPORTED_PROFILE = 'pick_place_can_bin1'


def pick_place_can_bin1_dataset(
    bin1_position_m,
    *,
    source_frame: str = 'panda_sim_world',
    target_frame: str = 'link_base',
) -> CalibrationDataset:
    """Build the fixed four-fit-plus-holdout landmark geometry."""
    origin = np.asarray(bin1_position_m, dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise CalibrationError('bin1_pos must contain three finite values')
    definitions = (
        ('origin', 'fit', (0.0, 0.0, 0.0)),
        ('x_100mm', 'fit', (0.1, 0.0, 0.0)),
        ('y_100mm', 'fit', (0.0, 0.1, 0.0)),
        ('z_80mm', 'fit', (0.0, 0.0, 0.08)),
        ('xy_100mm_holdout', 'holdout', (0.1, 0.1, 0.0)),
    )
    landmarks = tuple(
        LandmarkCorrespondence(
            name=name,
            role=role,
            source_position_m=tuple(float(item) for item in origin + offset),
        )
        for name, role, offset in definitions
    )
    return CalibrationDataset(
        task='PickPlaceCan',
        profile=SUPPORTED_PROFILE,
        source_frame=source_frame,
        target_frame=target_frame,
        minimum_captures_per_landmark=3,
        thresholds=CalibrationThresholds(),
        landmarks=landmarks,
    )


def _unwrap_environment(environment):
    current = environment
    visited = set()
    while hasattr(current, 'env') and id(current) not in visited:
        visited.add(id(current))
        current = current.env
    return current


def source_dataset_from_checkpoint(
    checkpoint_path: str | Path, profile: str
) -> CalibrationDataset:
    """Instantiate the checkpoint environment and read its exact ``bin1_pos``."""
    if profile != SUPPORTED_PROFILE:
        raise CalibrationError(f'unsupported profile: {profile}')
    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_file():
        raise CalibrationError(f'checkpoint does not exist: {checkpoint}')
    try:
        checkpoint_data = FileUtils.load_dict_from_checkpoint(str(checkpoint))
    except Exception as error:
        raise CalibrationError(f'cannot load checkpoint: {error}') from error
    metadata = deepcopy(checkpoint_data.get('env_metadata'))
    if not isinstance(metadata, dict):
        raise CalibrationError('checkpoint has no environment metadata')
    if metadata.get('env_name') != 'PickPlaceCan':
        raise CalibrationError('profile requires a PickPlaceCan checkpoint')
    environment_arguments = metadata.get('env_kwargs')
    if not isinstance(environment_arguments, dict):
        raise CalibrationError('checkpoint has invalid environment arguments')
    robots = environment_arguments.get('robots')
    if robots not in (['Panda'], 'Panda'):
        raise CalibrationError('profile requires the checkpoint Panda environment')

    try:
        environment = EnvUtils.create_env_from_metadata(
            env_meta=metadata,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
            use_depth_obs=False,
        )
    except Exception as error:
        raise CalibrationError(
            f'cannot create checkpoint environment: {error}'
        ) from error
    try:
        core = _unwrap_environment(environment)
        if not hasattr(core, 'bin1_pos'):
            raise CalibrationError('checkpoint environment has no bin1_pos')
        return pick_place_can_bin1_dataset(core.bin1_pos)
    finally:
        close = getattr(environment, 'close', None)
        if callable(close):
            close()


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Extract PickPlaceCan source-frame landmarks from the exact '
            'checkpoint environment. No robot command is produced.'
        )
    )
    parser.add_argument('--agent', required=True, help='Robomimic checkpoint')
    parser.add_argument(
        '--profile', required=True, choices=[SUPPORTED_PROFILE]
    )
    parser.add_argument('--output', required=True, help='New correspondence YAML')
    return parser.parse_args(args)


def run(args=None) -> int:
    parsed = _parse_args(args)
    try:
        dataset = source_dataset_from_checkpoint(parsed.agent, parsed.profile)
        save_calibration_dataset(parsed.output, dataset)
    except CalibrationError as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    print(f'wrote {len(dataset.landmarks)} landmarks to {parsed.output}')
    return 0


def main(args=None) -> None:
    raise SystemExit(run(args))


if __name__ == '__main__':
    main(sys.argv[1:])
