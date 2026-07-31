# mirage_xarm_state

This ROS 2 package normalizes xArm joint and gripper state by joint name, pairs
each valid arm sample with `link_base -> link_eef` at the exact sample stamp,
and publishes machine-readable diagnostics. It does not depend on a camera.

## Launch

Fake hardware with the gripper:

```bash
ros2 launch xarm_moveit_config xarm7_moveit_fake.launch.py add_gripper:=true
ros2 launch mirage_xarm_state xarm_state_recorder.launch.py \
  use_sim_time:=false record:=true
```

Gazebo with the gripper:

```bash
ros2 launch xarm_gazebo xarm7_beside_table_gazebo.launch.py add_gripper:=true
ros2 launch mirage_xarm_state xarm_state_recorder.launch.py \
  use_sim_time:=true record:=true
```

Bags are written under `~/Mirage_xarm_ws/recordings` by default.
Set `record:=false` to run only the monitor, or set `recordings_dir:=...` to
change the bag parent directory.

Normalized outputs are:

- `/mirage/xarm/arm_joint_states`
- `/mirage/xarm/ee_pose`
- `/mirage/xarm/gripper_joint_state`
- `/mirage/xarm/state_diagnostics`

The raw joint/TF/robot inputs and all normalized outputs are included in each
bag. `drive_joint` is kept in its native radian representation; conversion to
physical finger width is intentionally out of scope for this package.
