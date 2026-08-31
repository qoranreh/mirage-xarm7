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

## Verified xArm fake-hardware integration

On 2026-07-31, the State adapter was fixed and revalidated against the xArm7
fake hardware. Non-finite optional effort values are omitted while valid joint
positions and velocities continue through the pipeline. Arm state, gripper
state, timestamp-matched end-effector pose, and diagnostics all published with
healthy status. The package exits cleanly when ROS is already shutting down.

A 14.5-second rosbag check recorded 8,998 messages across the raw and
normalized topics and produced both the SQLite database and `metadata.yaml`.
The active workspace passed all 13 package tests. Actual xArm7 hardware remains
unverified and must be tested separately with low-speed safety limits.
