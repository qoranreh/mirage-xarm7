# mirage_xarm_state

This ROS 2 package normalizes xArm joint and gripper state by joint name, pairs
each valid arm sample with `link_base -> link_eef` at the exact sample stamp,
publishes machine-readable diagnostics, and produces a read-only Gate A JSON
report. It does not depend on a camera and does not send robot commands.

## Real xArm7 read-only Gate A

The one-command real-hardware entry point starts only the xArm state driver,
robot state/TF visualization, the Mirage state monitor, the validator, and a
rosbag recorder. It does not start ros2_control, a trajectory controller, a
motion action client, or a policy bridge. `robot_ip` must be explicit:

```bash
ros2 launch mirage_xarm_state xarm7_read_only_gate_a.launch.py \
  robot_ip:=<XARM_IP> duration_sec:=30
```

Use `show_rviz:=false` for a headless run. TF remains available because the
robot description and state publisher are started independently of RViz.

The run exits automatically after the validator finishes. JSON reports are
written under `~/Mirage_xarm_ws/reports`; bags are written under
`~/Mirage_xarm_ws/recordings`. For a deliberate manual-motion observation run,
add `require_motion:=true`; this only checks observed joint travel and still
does not publish a command.

After the bag closes, verify recorded topic coverage offline:

```bash
ros2 run mirage_xarm_state xarm_gate_a_bag_check \
  ~/Mirage_xarm_ws/recordings/xarm_gate_a_TIMESTAMP \
  --require-robot-state
```

RViz alignment remains a human visual confirmation. MoveIt FK checking is
supported by the validator when `/compute_fk` exists, but it is optional in the
driver-only real-hardware launch so that no MoveIt control stack is required.

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
- `/mirage/xarm/gate_a_report` (one JSON message at validator completion)

The raw joint/TF/robot inputs and all normalized outputs are included in each
bag. `drive_joint` remains in its native radian representation on the normalized
xArm topic. The optional canonical publisher converts it to calibrated physical
finger width as described below.

`position` is required and must be finite. `velocity` and `effort` are optional:
if a fake or real driver publishes non-finite values for either optional field,
the monitor omits only that field and reports it in diagnostics instead of
discarding the valid joint position sample.

## CanonicalState contract

`mirage_state_interfaces/msg/CanonicalState` is the robot-neutral boundary used
by policy-side code. It carries:

- end-effector position in metres and an `xyzw` unit quaternion;
- physical gripper opening width in metres;
- named, shaped object-state arrays and an explicit schema identifier;
- source robot, timestamp, frame, freshness, validity, and error reasons.

The pure-Python `mirage_state` package contains immutable state models, xArm and
Robosuite Panda adapters, source/target rigid transforms, and the checkpoint
observation projector. It can be installed from this package without ROS
message generation for unit testing:

```bash
python -m pip install -e ./xarm_extensions/mirage_xarm_state
```

Start the read-only xArm adapter directly with:

```bash
ros2 launch mirage_xarm_state canonical_state.launch.py
```

It consumes `/mirage/xarm/ee_pose`,
`/mirage/xarm/gripper_joint_state`, and `/mirage/object_state`, then publishes
`/mirage/canonical_state`. Alternatively, pass `publish_canonical:=true` to
`xarm_state_recorder.launch.py` or either Gate A launch.

The default gripper calibration maps `drive_joint=0.0 rad` to `0.085 m` open
width and `0.85 rad` to `0.0 m`. Adjust the calibration and freshness limits in
`config/canonical_state.yaml` for measured hardware. Values outside the
calibration tolerance, stale or skewed components, unexpected frames, missing
object state, non-finite data, and malformed object arrays are published as an
invalid canonical sample rather than silently accepted.

The xArm adapter publishes in `link_base`. A policy trained in another task
frame must receive an explicitly transformed CanonicalState; changing only the
frame label is unsafe and the policy projector rejects mismatches.

## Source/target transform T_TS

`RigidTransform` defines `T_TS` as the mapping from source frame `S` to target
frame `T`:

```plain text
p_T  = R_TS p_S + t_TS
q_TE = q_TS * q_SE
```

Quaternion multiplication is the Hamilton product in `xyzw` order. Translation
is always expressed in metres. `inverse()` returns `T_ST`, and pose conversion
rejects a source `frame_id` mismatch instead of relabeling the pose.

Copy `config/source_target_transform.example.yaml` and replace its placeholder
translation and rotation with measured values. Keep `calibrated: false` until
the measurement is complete. The supported physical workflow estimates these
values rather than asking the operator to type them.

First install both editable packages. Then generate the five source landmarks
from the exact PickPlaceCan Panda checkpoint environment:

```bash
mirage-source-landmarks \
  --agent PATH_TO_PICK_PLACE_CAN_CHECKPOINT \
  --profile pick_place_can_bin1 \
  --output config/calibration_correspondences.local.yaml
```

The profile uses `bin1_pos` as its origin, four fit points (`O`, `O+100 mm X`,
`O+100 mm Y`, `O+80 mm Z`) and one independent holdout
(`O+100 mm X+100 mm Y`). Copy `config/calibration_pointer.example.yaml` to a
`.local.yaml` file, measure the rigid `link_eef -> calibration_tip` offset, and
only then set its `calibrated` field to true.

With `xarm_state_monitor` already publishing `/mirage/xarm/ee_pose`, move the
rigid pointer to a named physical landmark using only the pendant in low-speed
mode. Capture each landmark three times:

```bash
ros2 run mirage_xarm_state xarm_transform_capture \
  --draft config/calibration_correspondences.local.yaml \
  --landmark origin \
  --tool-config config/calibration_pointer.local.yaml \
  --samples 50
```

Repeat for `x_100mm`, `y_100mm`, `z_80mm`, and `xy_100mm_holdout`. The capture
executable only subscribes to `/mirage/xarm/ee_pose`; it has no publisher,
service, action client, controller, or robot motion API. It rejects wrong
frames, zero, duplicate, future or stale timestamps, non-finite poses, unstable
sample bursts, and unmeasured tool offsets. Each accepted attempt is appended
atomically without replacing prior evidence.

Estimate with Kabsch/SVD and create the transform, version-2 fixtures, and JSON
report at new paths:

```bash
mirage-transform-calibrate \
  --correspondences config/calibration_correspondences.local.yaml \
  --output config/source_target_transform.local.yaml \
  --fixtures config/transform_validation.local.yaml \
  --report reports/transform_calibration.json \
  --mode smoke
```

Smoke mode requires at most 5 mm position error and 1 degree orientation error;
strict mode requires 2 mm and 0.5 degree and should pass before any controller
integration. Four points fit `T_TS`; the fifth never participates in the fit.
Reflection, rank-deficient or near-duplicate geometry, missing captures, and
failed repeatability or holdout gates remain fail-closed with
`calibrated: false`. Exit codes are 0 for pass, 1 for a valid measurement that
misses a gate, and 2 for invalid input. All measured `.local.yaml` files and
reports are ignored by Git.

The validator deliberately rejects an uncalibrated configuration:

```bash
mirage-transform-validate \
  --config PATH_TO_MEASURED_T_TS.yaml \
  --fixtures PATH_TO_KNOWN_POSES.yaml \
  --report transform_report.json
```

The fixture format is demonstrated in
`config/transform_validation.example.yaml`. Version 2 separates physical pose
and axis tolerances from tight numerical `T_TS -> T_ST` round-trip tolerances;
the validator still accepts existing version-1 fixtures. Exit status is zero
only when every configured tolerance passes.

Camera extrinsics must live in a separate configuration. The T_TS loader uses a
strict schema and rejects camera fields or any other unknown setting. This
transform API applies only to Cartesian poses; it does not transform or relabel
the opaque Robomimic `object_state` arrays because their spatial meaning depends
on the observation schema.

## Read-only policy probe

Install both editable Python packages and source the ROS workspace, then run:

```bash
mirage-canonical-policy-probe \
  --agent PATH_TO_CHECKPOINT \
  --expected-frame TASK_FRAME \
  --object-schema PickPlaceCan:robomimic-v1
```

The probe derives required observation shapes from the Robomimic checkpoint and
only runs inference for a valid, fresh, schema-matching CanonicalState. It
publishes the resulting vector to `/mirage/policy_action_preview`; it has no
controller or robot-command connection.
