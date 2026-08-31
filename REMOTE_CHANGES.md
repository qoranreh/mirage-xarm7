# Remote-Sourced Changes

## 2026-08-06 — Official Robosuite xArm7 simulation backport

- Source repository: `ARISE-Initiative/robosuite`
- Upstream version and commit: v1.5.2,
  `824ac14cefcfb7ec125fe5eb2e0bad7364466154`
- Asset-origin license source: `google-deepmind/mujoco_menagerie`,
  `71f066ad0be9cd271f7ed58c030243ef157af9f4`
- User approval: explicit approval received for the official xArm7 assets and
  adapter backport.
- Reason: the repository's Robosuite v1.4.1 submodule predates the official
  xArm7 robot and gripper, but Mirage needs a simulation-only xArm target.
- Affected paths:
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/assets/`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/robosuite_xarm7.py`
  - `mirage/setup.py`
  - `mirage/.gitignore`
- Summary:
  - Copied the official xArm7 robot/gripper MJCF and the minimal 16 unique STL
    meshes without modifying their bytes.
  - Preserved the Robosuite MIT and UFactory BSD 3-Clause license texts and
    recorded fixed provenance in the asset README.
  - Backported the upstream xArm7 robot and one-action gripper classes through
    a new Mirage-owned Robosuite v1.4 registry adapter. The adapter uses the
    checkpoint's existing OSC controller configuration and has no real-robot
    transport.
  - Added package-data entries so installed Mirage packages retain the assets.
  - Added path-specific ignore exceptions for only the approved xArm7 STL
    directories; the existing general `meshes` ignore remains in force.
  - Did not modify the `robosuite/` Git submodule or its referenced commit.
- Validation commands:
  - `git hash-object` over both MJCF files and all 16 STL files; hashes matched
    the v1.5.2 Git tree.
  - `python -m pytest -q mirage/benchmark/robosuite/tests/test_xarm_transfer_xarm7_integration.py`
    (`3 passed`).
  - Full xArm transfer and checkpoint parity suite (`34 passed`).
  - `python setup.py egg_info`; the generated source manifest contained both
    MJCF files, both license files, the asset README, and all 16 STL files.
  - Simulation rollout CLI with held-out seeds 1 and 3, 50 steps each, in both
    `online_oracle` and `learned_ridge` modes; all four rollouts completed with
    zero timeout, iteration-limit, and rejected-command events.

## 2026-08-06 — Task-profiled PickPlaceCan and Lift transfer

- Source repository: `qoranreh/mirage-xarm7`
- Upstream commit before modification:
  `4e4429e8dfbd69dd224c1d3bb614c718d0ee0919`
- User approval: explicit approval received for the first-party Mirage files
  listed below. Git submodules and official xArm assets were excluded.
- Reason: remove PickPlaceCan-only coupling and support validated Lift
  low-dimensional online-oracle transfer under a static task registry.
- Affected paths:
  - `REMOTE_CHANGES.md`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/profiles.py`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/{__init__,cli,collection,contracts,dataset,factory,models,rollout,runtime,simulation_transform,training}.py`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/README.md`
  - `mirage/mirage/benchmark/robosuite/config/xarm_transfer_sim.example.yaml`
  - `mirage/mirage/benchmark/robosuite/tests/test_canonical_policy_checkpoint.py`
  - `mirage/mirage/benchmark/robosuite/tests/test_xarm_transfer_{dataset_models,panda_integration,profiles,rollout,runtime,xarm7_integration}.py`
- Summary:
  - Added static PickPlaceCan and Lift task/controller profiles, checkpoint
    observation-mode classification, native-horizon validation, and early
    image/unregistered-task rejection.
  - Split source and target environment metadata construction, moved Panda
    gripper reads to simulator qpos, and resolved only the profile-selected
    graspable object's contact geoms from `model.mujoco_objects` after reset.
  - Made controller gains/contact behavior and simulation transform provenance
    task-aware; Lift has no contact relaxation or gain switch.
  - Changed early termination to target xArm success, made report paths and
    episode IDs task-aware, and added transition/Ridge schema v2 with strict
    version-1 PickPlaceCan/low-dim compatibility.
  - Added Lift CanonicalState action parity, contract, reset-pose, controller,
    schema, and full-rollout regression coverage without changing CanonicalState.
- Validation commands:
  - Focused contract/runtime/controller/rollout/dataset suite (`62 passed`).
  - Panda checkpoint and CanonicalState action parity suite (`3 passed` before
    the added Lift Panda smoke; included again in the final full suite).
  - xArm target creation, task-object reset parity for seeds 0/1/3/7, and
    PickPlaceCan 50-step regression (`12 passed`).
  - Full target-success integration for PickPlaceCan seed 1 and Lift seeds
    0/1/3/7 (`5 passed`), with zero timeout, iteration-limit, or rejection;
    Lift also had zero contact-tolerance events.
  - CLI native-horizon PickPlaceCan seed-1 and Lift seeds-0/1/3/7 rollouts;
    all reached target success and wrote ignored task-aware version-2 reports.
  - Existing version-1 PickPlaceCan dataset and Ridge artifact 50-step learned
    replay (`completed`, zero timeout, iteration-limit, rejection, or contact
    tolerance events).
  - Final Robosuite benchmark regression suite (`89 passed`).
  - ROS 2 Humble xArm extension suite with `ROS_LOG_DIR` in `/tmp`
    (`55 passed`).

## 2026-08-10 — Complete local checkpoint task registry

- Source repository: `qoranreh/mirage-xarm7`
- Upstream commit before modification:
  `4e4429e8dfbd69dd224c1d3bb614c718d0ee0919`
- User approval: explicit approval received to implement the reviewed task
  profile and common-runtime plan. Git submodules were excluded.
- Reason: make every task represented by a local policy checkpoint executable
  through the common low-dimensional Panda-to-xArm7 simulation path.
- Affected paths:
  - `REMOTE_CHANGES.md`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/{profiles,runtime}.py`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/README.md`
  - `mirage/mirage/benchmark/robosuite/config/xarm_transfer_sim.example.yaml`
  - `mirage/mirage/benchmark/robosuite/tests/test_xarm_transfer_{profiles,runtime,xarm7_integration}.py`
- Summary:
  - Registered `Square_D0`, `Stack_D0`, and `ThreePieceAssembly_D0` with their
    exact low-dimensional shapes, native horizons, slugs, manipulated objects,
    MimicGen environment registration module, and controller profiles.
  - Made observation shapes authoritative for modality classification and
    sanitized stale camera flags before constructing low-dimensional runtimes.
  - Replaced the single-object runtime assumption with validated multi-object
    contact-geometry resolution; ThreePieceAssembly combines `piece_1` and
    `piece_2` while excluding its fixed base.
  - Reused the verified-contact controller only for PickPlaceCan, Square, and
    ThreePieceAssembly; Lift and Stack retain strict free-space control.
  - Kept image inference fail-closed and made no Robosuite, Robomimic, or
    MimicGen submodule change.
- Validation commands:
  - Contract and runtime unit suite (`42 passed`).
  - All 15 local checkpoints resolved to the expected task, observation mode,
    shape, and native horizon.
  - Source-only native-horizon validation: Square seed 3 succeeded at step 133,
    Stack seed 0 at step 106, and ThreePieceAssembly seed 0 at step 168.
  - xArm native-horizon oracle validation: Square seed 3 reached target success
    at step 128, Stack seed 0 at step 101, and ThreePieceAssembly seed 0 at step
    166; all had zero timeout, iteration-limit, and safety-rejection events.
  - Final xArm transfer regression suite (`120 passed`).
  - Canonical checkpoint regression suite (`2 passed`).

## 2026-08-10 — Image-policy cross-paint transfer runtime

- Source repository: `qoranreh/mirage-xarm7`
- Upstream commit before modification:
  `d572c81e4616259e3b96038ad87e72b9ecdeb4c6`
- User approval: explicit approval received to implement the reviewed
  cross-inpainting plan for every local `image_no_proprio` and
  `image_proprio` checkpoint. Git submodules were excluded.
- Reason: let image policies observe a Panda-equivalent camera frame while the
  target rollout remains an xArm7 simulation.
- Affected paths:
  - `REMOTE_CHANGES.md`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/{__init__,cli,cross_painting,image_runtime,profiles,rollout,runtime}.py`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/README.md`
  - `mirage/mirage/benchmark/robosuite/tests/test_cross_painting.py`
  - `mirage/mirage/benchmark/robosuite/tests/test_xarm_transfer_{profiles,rollout}.py`
- Summary:
  - Added an analytic OpenCV Telea compositor that removes a geometry-derived
    xArm mask, depth-composites a rendered Panda robot layer, and emits the
    checkpoint's normalized CHW image.
  - Added metric RGB-D rendering, robot geom-ID masks, camera matrices, and
    z-buffer reprojection for tasks whose Panda and xArm agent-view cameras
    differ.
  - Added target-to-source EEF alignment and manipulated-object state
    synchronization before each image-policy inference step.
  - Isolated the Panda policy and offscreen renderer in a spawned worker so it
    does not share an EGL context with the xArm target environment.
  - Added image-oracle CLI routing, a raw-xArm `naive` ablation, five-panel
    debug video output, and visual/alignment/latency rollout metrics.
  - Made no Robosuite, Robomimic, or MimicGen submodule change.
- Validation commands:
  - Cross-paint, task-contract, runtime, rollout, and Panda integration suite
    (`53 passed`).
  - Focused xArm7 environment/step integration suite (`9 passed`, `29
    deselected`).
  - EGL one-step rollout for all ten task/modality combinations across
    PickPlaceCan, Lift, Square, Stack, and ThreePieceAssembly; all completed
    with non-empty source and target masks and zero timeout, iteration-limit,
    or rejected-command events.
  - PickPlaceCan `image_no_proprio` recurrent smoke (`10/10` steps) and native
    rollout (`400/400` steps); both completed with zero timeout,
    iteration-limit, or rejected-command events.

## 2026-08-11 — Stabilize cross-paint gripper, masks, and source alignment

- Source repository: `qoranreh/mirage-xarm7`
- Upstream commit before modification:
  `d572c81e4616259e3b96038ad87e72b9ecdeb4c6`
- User approval: explicit approval received after the affected local runtime,
  compositor, rollout, tests, and documentation paths were listed. Git
  submodules and unrelated working-tree changes were excluded.
- Reason: prevent repeated xArm gripper integration, protect the manipulated
  object during inpainting, and restore Mirage-style source alignment while
  synchronizing dynamic object state.
- Affected paths:
  - `REMOTE_CHANGES.md`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/{__init__,cli,control,cross_painting,image_runtime,rollout,runtime}.py`
  - `mirage/mirage/benchmark/robosuite/xarm_transfer/README.md`
  - `mirage/mirage/benchmark/robosuite/tests/test_cross_painting.py`
  - `mirage/mirage/benchmark/robosuite/tests/test_xarm_transfer_{control,rollout,runtime,xarm7_integration}.py`
- Summary:
  - Split Cartesian tracking from gripper application. All OSC/settling
    physics steps hold gripper input at zero; after pose convergence the local
    xArm adapter applies one Panda-rate-calibrated actuator increment and then
    advances physics with zero gripper input.
  - Measured xArm finger-pad aperture and driver-joint closure, synchronized
    normalized opening into Panda finger qpos/qvel and actuator state before
    every image-policy observation, and recorded one-shot/aperture diagnostics.
  - Split robot, gripper, and manipulated-object segmentation. Dilation now
    applies only to the gripper, visible object pixels are excluded from the
    erase mask, and source/object occlusion remains depth-tested.
  - Added optional mask-only 168 x 168 supersampling while retaining 84 x 84
    RGB, depth, and policy inputs.
  - Restored absolute-OSC source alignment with zero gripper input, separate
    3 mm / 2 degree convergence bounds, and no arm IK/qpos teleport.
  - Extended manipulated-object transfer to qpos plus qvel, including task
    frame rotation of free-joint linear/angular velocity, and reapplied both
    after alignment physics.
  - Made no Robosuite, Robomimic, or MimicGen submodule change.
- Validation commands:
  - Focused compositor/controller/runtime/rollout/profile suite (`74 passed`).
  - PickPlaceCan image-proprio EGL one-step smoke (`1/1` completed), with one
    gripper application, 1.92 mm / 0.15 degree source alignment, and zero
    object-pixel loss.
  - PickPlaceCan image-proprio seed-0 200-step rollout (`200/200` completed),
    with 200 gripper applications, zero timeout/rejection, 634 protected
    object pixels, and zero object-pixel loss.
  - PickPlaceCan image-proprio seeds 1/2 and image-no-proprio seeds 0/1
    (`200/200` each); all completed with one gripper application per step,
    zero timeout/rejection, normalized aperture error at most 0.00146, and
    zero protected-object pixel loss.
  - Mask-only 168 x 168 EGL smoke (`1/1` completed) while RGB/depth/policy
    remained 84 x 84; normalized aperture error was below `1.6e-7`.
  - Final xArm transfer regression and integration suite (`129 passed`).
