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
