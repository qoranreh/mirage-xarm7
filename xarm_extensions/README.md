# xArm7 extension

This directory contains the version-controlled ROS 2 sources used by the local
xArm7 workspace:

- `mirage_state_interfaces` defines the robot-neutral `CanonicalState` and
  nested object-state messages.
- `mirage_xarm_state` publishes normalized xArm state and adapts it to the
  canonical contract.

Build artifacts, recordings, and Python caches are intentionally excluded.

To restore the packages into a ROS 2 workspace, copy both directories to the
workspace `src` directory, then run:

```bash
cd ~/Mirage_xarm_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select mirage_state_interfaces mirage_xarm_state
```

The active development workspace remains `~/Mirage_xarm_ws`; this directory is
the durable Git-backed copy.
