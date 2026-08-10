# Task-profiled Franka Panda to xArm7 simulation transfer

This package implements the simulation-only state/control transfer path. The
source is the Franka Emika Panda robot, registered by Robosuite 1.4.1 under the
identifier `Panda`. No ROS command, service, action, hardware SDK, robot IP, or
real calibration is used here.

## Static task and checkpoint contracts

`profiles.py` resolves each checkpoint into
`TaskProfile × ObservationMode × ForwardDynamics × TargetController`. The
active runtime profiles are `PickPlaceCan`, `Lift`, `Square_D0`, `Stack_D0`,
and `ThreePieceAssembly_D0`, all for low-dimensional online-oracle rollout.
The shared action layout is
`[dx, dy, dz, dRx, dRy, dRz, gripper]`. The policy runs at 20 Hz with
`OSC_POSE`, `control_delta=true`, translation limits of ±0.05 m per axis, and
rotation-vector limits of ±0.5 rad per axis. `SourceAction` retains both the
exact raw checkpoint output and the normalized value after Robosuite's
`[-1, 1]` clip. `ScaledOscCommand` stores the resulting physical delta.

Poses use metres, XYZW quaternions, nanosecond timestamps, and explicit frame
IDs. Quaternion normalization and sign canonicalization happen at every
serialization boundary. The source frame is `panda_sim_world`; the target
frame is `xarm7_sim_world`.

The static registry fixes each task's native horizon, low-dimensional object
shape, manipulated MJCF objects, transform provenance, environment registration,
and blocking-controller profile:

| Task | Object observation | Manipulated objects | Native horizon | Contact relaxation |
|---|---:|---|---:|---|
| PickPlaceCan | `[14]` | `Can` | 400 | verified contact: 0.060 m / 18° |
| Lift | `[10]` | `cube` | 400 | disabled |
| Square_D0 | `[14]` | `SquareNut` | 400 | verified contact: 0.060 m / 18° |
| Stack_D0 | `[23]` | `cubeA` | 400 | disabled |
| ThreePieceAssembly_D0 | `[42]` | `piece_1`, `piece_2` | 500 | verified contact: 0.060 m / 18° |

Checkpoints using `image_no_proprio` or `image_proprio` are classified, but
inference is deliberately unavailable in this phase. Both runtime factories
fail before simulator construction with
`image observation runtime not implemented`. Observation shapes are the
modality source of truth, so stale `use_camera_obs=true` flags in MimicGen
low-dimensional checkpoints are sanitized before environment construction.
The target is always camera-free and never inherits source policy cameras.

`SimulationTransform` is a separate type whose `kind` is always
`simulation_only` and whose `calibrated` value is always false. Version 2
explicitly composes two independent terms:

```text
T_targetWorld_targetTool =
    T_targetWorld_sourceWorld
    @ T_sourceWorld_sourceTool
    @ T_sourceTool_targetTool
```

The task/world term is left-multiplied and may transform Cartesian positions.
The tool correction is right-multiplied and changes the end-effector convention
without rotating world-frame motion directions. Panda and xArm use the same
simulator task coordinates for all registered tasks, so the task transform is
explicitly identity with task-specific provenance. Their nominal gripper axes
differ by -90° about local Z; that fixed
rotation appears only in the tool correction. Robot home poses are never used
to infer either term. Unsafe version-1 home-alignment documents and
configurations claiming `calibrated: true` are rejected.

## Collect oracle transitions

Run commands from the Python project directory:

```bash
cd mirage
../.venv/bin/python -m mirage.benchmark.robosuite.xarm_transfer.cli collect \
  --checkpoint mirage/models/can/low_dim/model_epoch_250_PickPlaceCan_success_1.0.pth \
  --seeds 0,1,2,3,4,5 \
  --steps 50 \
  --device cpu
```

The low-dimensional source runtime explicitly disables cameras and offscreen
rendering, and reads Panda gripper qpos directly from simulator state rather
than from policy observations. For every transition it stores task,
observation mode, episode and step IDs, seed,
timestamp, current pose, raw and clipped policy actions, scaled OSC command,
achieved next pose, gripper states, provider, and latency. The default JSONL
location is
`mirage/benchmark/robosuite/results/xarm_transfer/<task-slug>/<observation-mode>/`,
which is Git-ignored. Transition schema v2 records task and mode. Version-1
records are read as `PickPlaceCan` / `low_dim` for compatibility.

## Train and compare forward dynamics

```bash
cd mirage
../.venv/bin/python -m mirage.benchmark.robosuite.xarm_transfer.cli train \
  --dataset mirage/benchmark/robosuite/results/xarm_transfer/pick-place-can/low_dim/panda_transitions.jsonl \
  --group-by episode \
  --validation-fraction 0.34 \
  --split-seed 11
```

The trainer splits whole episodes or whole seeds, never transitions. It fits
LinearRegression and several Ridge candidates in a stable six-dimensional
pose-delta representation. The report compares identity, ideal delta
integration, Linear, Ridge, and the zero-error online-oracle reference with
separate position (m) and angular orientation (rad and deg) metrics. The
deployable artifact is always the best held-out Ridge candidate.

The `.npz` model contains numeric coefficients plus strict JSON metadata; load
does not enable pickle. Ridge schema v2 stores task and observation mode in
addition to frame, dimensions, scaling, and held-out bounds. Version-1 models
load only as `PickPlaceCan` / `low_dim`. Rollout refuses an artifact whose
task or mode differs from the checkpoint contract. `ForwardDynamicsConfig`
selects either `online_oracle` or `learned_ridge`. A learned artifact without
held-out bounds is refused. Lift Ridge training is outside this phase.

## Target control and safety

`AbsoluteBlockingController` repeatedly sends an absolute Cartesian target to
the Robosuite xArm7 OSC controller. It records separate position/orientation
errors, iterations, timeout, rejection reason, and latency. Before the first
controller call and before every repeat, `SimulationSafetyFilter` checks:

- simulation-only type and target frame;
- command freshness and future-clock skew at admission (an accepted blocking
  goal does not become stale solely while its own controller loop converges);
- finite pose and configured workspace;
- finite seven-joint state and model joint limits;
- held-out model position and orientation error bounds.

The normalized xArm simulation gripper convention is explicit: `-1` is open
and `+1` is closed. A rejected or out-of-bound learned prediction is never
sent to the target controller.

Both static controller profiles use a 0.025 m position tolerance and a 7°
orientation tolerance in free space. Success requires five consecutive
simulation steps inside both bounds (the first crossing plus four settling
steps), so the controller does not stop on a transient boundary crossing. A
bounded Cartesian outer loop uses position gain 1.0 with at most 0.030 m and
orientation gain 0.5 with at most 5° of world-frame correction to remove OSC
steady-state offset. The compensated command is rechecked by the full safety
filter before every simulator step. The official xArm gripper can settle
against a PickPlaceCan object before reaching the free-space bounds, so 0.060 m
and 18° contact bounds are allowed only while the simulator verifies
gripper-body contact with a task object. During that verified contact, the
simulation-only OSC gains change from `[150] * 6` to
`[300, 300, 500, 300, 300, 300]`, and bounded position compensation may reach
0.100 m. This contact tolerance, correction, and gain switch apply to the
PickPlaceCan `Can`, Square `SquareNut`, and the movable ThreePieceAssembly
`piece_1` / `piece_2` objects. Lift and Stack always use free-space bounds and
`[150] * 6` gains. Fixtures and distractors never enable relaxation. Every use
of a verified-contact exception is recorded per step and counted
in the JSON report. It does not relax workspace, joint, freshness, or
learned-model safety gates. A target task-success event terminates blocking
immediately even when the final EE pose has not converged. Timeout and
iteration-limit failures are distinct, and either makes a rollout incomplete.

`run_online_oracle_episode` implements policy → one Panda step → `T_TS` → xArm
blocking control. `run_learned_replay_episode` starts from the same source pose
and target reset, replays the same recorded actions autoregressively, and
records oracle prediction and target tracking errors. Rollout reports are JSON
and include timeout, rejection, error, iteration, latency, source task success,
and target task success per step.

`--terminate-on-success` observes only target xArm task success. Source Panda
success is still recorded, but cannot stop a rollout while the target remains
unsuccessful. Reports include task name, observation mode, native horizon, and
requested horizon; episode IDs include the task slug.

PickPlaceCan, Lift, Square, and Stack have native horizons of 400 policy steps;
ThreePieceAssembly uses 500. Omitting `--steps` uses that native value; an
explicit smaller value remains a smoke-test override. Example full rollouts:

```bash
cd mirage
MUJOCO_GL=egl ../.venv/bin/python -m \
  mirage.benchmark.robosuite.xarm_transfer.cli rollout \
  --checkpoint mirage/models/can/low_dim/model_epoch_250_PickPlaceCan_success_1.0.pth \
  --mode oracle --seeds 1 --terminate-on-success

MUJOCO_GL=egl ../.venv/bin/python -m \
  mirage.benchmark.robosuite.xarm_transfer.cli rollout \
  --checkpoint mirage/models/lift/low_dim/lift_ph_low_dim_epoch_1000_succ_100.pth \
  --mode oracle --seeds 0,1,3,7 --terminate-on-success

MUJOCO_GL=egl ../.venv/bin/python -m \
  mirage.benchmark.robosuite.xarm_transfer.cli rollout \
  --checkpoint mirage/models/square/low_dim/model_epoch_950_Square_D0_success_0.96.pth \
  --mode oracle --seeds 3 --terminate-on-success

MUJOCO_GL=egl ../.venv/bin/python -m \
  mirage.benchmark.robosuite.xarm_transfer.cli rollout \
  --checkpoint mirage/models/stack/low_dim/model_epoch_2000.pth \
  --mode oracle --seeds 0 --terminate-on-success

MUJOCO_GL=egl ../.venv/bin/python -m \
  mirage.benchmark.robosuite.xarm_transfer.cli rollout \
  --checkpoint mirage/models/two_piece/low_dim/model_epoch_1000_ThreePieceAssembly_D0_success_0.68.pth \
  --mode oracle --seeds 0 --terminate-on-success
```

Run both providers against the same held-out seeds and freshly recorded oracle
actions:

```bash
cd mirage
MUJOCO_GL=egl ../.venv/bin/python -m \
  mirage.benchmark.robosuite.xarm_transfer.cli rollout \
  --checkpoint mirage/models/can/low_dim/model_epoch_250_PickPlaceCan_success_1.0.pth \
  --mode both \
  --model mirage/benchmark/robosuite/results/xarm_transfer/pick-place-can/low_dim/ridge_forward_dynamics.npz \
  --seeds 1,3 \
  --steps 50
```

The learned provider's controller gate uses the model artifact's held-out
one-step maximum errors. The report additionally compares the 50-step
autoregressive trajectory with the oracle after the fact; accumulated rollout
drift is diagnostic and is not mislabeled as a calibrated or directly
observable online error bound.

## Validated simulation result

The reference run collected 300 transitions (six complete 50-step episodes)
and split by episode: seeds 0, 2, 4, and 5 for training, seeds 1 and 3 for
validation. The selected Ridge model used alpha 0.01.

| Provider / baseline | Held-out position RMSE | Held-out orientation RMSE |
|---|---:|---:|
| Oracle ground truth | 0 m | 0 rad / 0° |
| Ridge (selected) | 0.002394 m | 0.004427 rad / 0.2536° |
| Linear | 0.002728 m | 0.004303 rad / 0.2466° |
| Identity | 0.009627 m | 0.011400 rad / 0.6532° |
| Delta integration | 0.028905 m | 0.040744 rad / 2.3345° |

With the version-2 task/tool transform, oracle and learned xArm rollouts for
seeds 1, 3, and 7 all completed 50/50 steps with zero timeout,
iteration-limit, rejection, or contact-tolerance event. The achieved tracking
errors were:

| Provider | Seed | Position mean / max | Orientation mean / max |
|---|---:|---:|---:|
| Oracle | 1 | 0.00985 / 0.02305 m | 2.289° / 5.199° |
| Oracle | 3 | 0.00453 / 0.01012 m | 1.060° / 2.664° |
| Oracle | 7 | 0.00448 / 0.00972 m | 0.976° / 2.583° |
| Ridge | 1 | 0.00888 / 0.02473 m | 2.175° / 5.707° |
| Ridge | 3 | 0.00653 / 0.01112 m | 1.546° / 2.682° |
| Ridge | 7 | 0.00489 / 0.01075 m | 1.022° / 2.572° |

The ignored `xarm_rollout_report_task_tool_v2.json` and
`xarm_learned_rollout_report_task_tool_v2.json` retain every step. Learned
autoregressive source-pose drift is reported separately and is not confused
with target-controller tracking error.

The ignored `panda_xarm_oracle_seed1_task_tool_v2.mp4` shows the 50-step
oracle path side by side and overlays the achieved position/orientation error.

The 400-step task-mode validation for seed 1 reached actual xArm task success
at policy step 116: the xArm picked up the can and placed it in the bin. It had
zero timeout, iteration-limit, and safety-rejection events; 32 steps used the
verified-contact bounds. The source Panda would reach success at step 124, but
the rollout stopped at step 116 because the target task had already succeeded.
The ignored `panda_xarm_oracle_seed1_full_task.mp4` contains the complete
side-by-side run, and `xarm_full_task_oracle_seed1.json` retains every reported
error, including the final task-success step's unconverged EE error.

Lift full-horizon validation also used target-only early termination. All four
seeds reached xArm success with zero timeout, iteration-limit, rejection, or
contact-tolerance event:

| Seed | Source success step | Target success step | Position max | Orientation max |
|---:|---:|---:|---:|---:|
| 0 | 55 | 57 | 0.01814 m | 2.614° |
| 1 | 40 | 41 | 0.01196 m | 1.784° |
| 3 | 49 | 50 | 0.01346 m | 2.540° |
| 7 | 46 | 47 | 0.02069 m | 1.913° |

The seed-1 result specifically proves that source success at step 40 no longer
stops the rollout before target success at step 41. Ignored version-2 reports
are stored below the task-aware `pick-place-can/low_dim/` and `lift/low_dim/`
result directories.

The added MimicGen profiles were validated with full native-horizon oracle
rollouts. Square seed 3 reached target success at step 128, Stack seed 0 at
step 101, and ThreePieceAssembly seed 0 at step 166, with zero timeout,
iteration-limit, or safety-rejection event. Square and ThreePieceAssembly used
verified-object contact bounds; Stack required no contact exception.

## Validation

With pytest installed in the active environment:

```bash
cd mirage
MPLCONFIGDIR=/tmp/mirage_mpl MUJOCO_GL=egl \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  --rootdir=. --import-mode=importlib \
  mirage/benchmark/robosuite/tests/test_xarm_transfer_*.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  --rootdir=. --import-mode=importlib \
  mirage/benchmark/robosuite/tests/test_canonical_policy_checkpoint.py
```

The second command protects identical PickPlaceCan action output for the same
observation and seed through the existing CanonicalState projector. The Panda
integration test also snapshots the real checkpoint's seed-7 raw action and
asserts that the oracle executes exactly one Robosuite policy step. The xArm7
integration test verifies pinned upstream asset hashes, creates the official
target model, and executes a real 50-step Panda-oracle-to-xArm blocking path.

Generated datasets, models, and reports must stay below the ignored `results/`
directory. None of these modules may be imported into a real xArm ROS command
path.

Official asset provenance and licenses are documented in `assets/README.md`;
the repository-root `REMOTE_CHANGES.md` records the approved backport. The
Robosuite submodule itself remains unchanged.
