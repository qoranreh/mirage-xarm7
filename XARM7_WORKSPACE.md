# Mirage xArm7 workspace status

This repository keeps the official Mirage history plus the local xArm7
integration work. Machine-generated state stays outside Git.

## What is backed up

- The official Mirage history, with BerkeleyAutomation configured as
  `upstream` and this private repository configured as `origin`.
- The custom ROS 2 package under `xarm_extensions/mirage_xarm_state`.
- A portable robosuite smoke-test template.
- The robomimic single-GPU compatibility change as a reproducible patch.

## What stays local

- `.venv/` and installed Python packages.
- `~/Mirage_xarm_ws/build`, `install`, and `log`.
- Rosbag recordings and benchmark outputs.
- `local_smoke.yaml`, because it contains paths specific to one computer.

## Verified local baseline

On 2026-07-31, the copied Linux workspace passed `pip check`, CUDA and EGL
device checks, and a 50-step Panda-to-UR5e low-dimensional robosuite smoke
run. The ROS package unit suite passed eight tests in its active workspace.

## Known xArm integration limitation

The current fake-hardware integration can supply non-finite optional effort
values. The state normalizer rejects those samples, so normalized output may
remain empty even though the unit tests pass. Treat the package as backed up
and testable, but not yet validated for experiment recording on xArm7.
