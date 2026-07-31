# xArm7 extension

`mirage_xarm_state` is the version-controlled source snapshot of the custom
ROS 2 package used by the local xArm7 workspace. Build artifacts, recordings,
and Python caches are intentionally excluded.

To restore it into a ROS 2 workspace, copy
`xarm_extensions/mirage_xarm_state` to
`~/Mirage_xarm_ws/src/mirage_xarm_state`, then run:

```bash
cd ~/Mirage_xarm_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mirage_xarm_state
```

The active development workspace remains `~/Mirage_xarm_ws`; this directory is
the durable Git-backed copy.
