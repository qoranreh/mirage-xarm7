# xArm7 simulation assets

These files are simulation assets only. They do not contain or enable a ROS,
UFactory SDK, network, service, action, or physical-robot command path.

The two MJCF files and sixteen selected STL files are copied without content
changes from `ARISE-Initiative/robosuite` v1.5.2, commit
`824ac14cefcfb7ec125fe5eb2e0bad7364466154`:

- `robosuite/models/assets/robots/xarm7/robot.xml`
- `robosuite/models/assets/grippers/xarm7_gripper.xml`
- arm meshes `link_base`, `link1` through `link7`, and `end_tool`
- gripper meshes `base_link`, the left/right fingers, inner knuckles, and outer
  knuckles

Upstream stores a second duplicate mesh directory. This backport intentionally
keeps only one arm mesh set and one gripper mesh set. `LICENSE.robosuite`
preserves Robosuite's MIT terms. `LICENSE.ufactory` preserves the UFactory BSD
3-Clause asset terms as published by `google-deepmind/mujoco_menagerie` at
commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`.

The v1.4 Python compatibility code is outside this directory in
`robosuite_xarm7.py`; the `robosuite/` Git submodule is not modified.
