import json
import time
from types import SimpleNamespace

import numpy as np

from mirage.benchmark.robosuite.xarm_transfer.contracts import (
    CartesianPose,
    OscActionScaler,
    SourceAction,
    SourceDesiredPose,
)
from mirage.benchmark.robosuite.xarm_transfer.control import (
    AbsoluteBlockingController,
    SafetyLimits,
    SimulationSafetyFilter,
)
from mirage.benchmark.robosuite.xarm_transfer.dataset import TransitionRecord
from mirage.benchmark.robosuite.xarm_transfer.dynamics import (
    ForwardDynamics,
    ForwardDynamicsRequest,
    ForwardDynamicsResult,
    PandaOnlineOracle,
)
from mirage.benchmark.robosuite.xarm_transfer.geometry import (
    rotvec_to_quaternion_xyzw,
)
from mirage.benchmark.robosuite.xarm_transfer.rollout import (
    run_cross_paint_oracle_episode,
    run_online_oracle_episode,
    run_learned_replay_episode,
    write_rollout_reports,
)
from mirage.benchmark.robosuite.xarm_transfer.image_runtime import (
    CrossPaintSourceStep,
    CrossPaintStepDiagnostics,
)
from mirage.benchmark.robosuite.xarm_transfer.profiles import (
    CheckpointContract,
    ObservationMode,
    TASK_PROFILES,
)
from mirage.benchmark.robosuite.xarm_transfer.runtime import (
    GripperApertureState,
    ManipulatedObjectState,
)


SOURCE_FRAME = "panda_sim_world"
TARGET_FRAME = "xarm7_sim_world"


def oracle_episode(length=50):
    records = []
    scaler = OscActionScaler()
    for step in range(length):
        timestamp = 1_000_000_000 + step * 50_000_000
        current = CartesianPose(
            (step * 0.001, 0.0, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            SOURCE_FRAME,
            timestamp,
        )
        action = SourceAction((0.0,) * 7, SOURCE_FRAME, timestamp + 1)
        next_pose = CartesianPose(
            ((step + 1) * 0.001, 0.0, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            SOURCE_FRAME,
            timestamp + 50_000_000,
        )
        records.append(
            TransitionRecord(
                "panda-seed-42",
                step,
                42,
                action.timestamp_ns,
                current,
                action,
                scaler.scale(action),
                next_pose,
                (0.02, -0.02),
                (0.02, -0.02),
                "panda_online_oracle",
                100,
            )
        )
    return tuple(records)


class PerfectLearnedProvider(ForwardDynamics):
    def __init__(self):
        self.scaler = OscActionScaler()

    def predict(self, request: ForwardDynamicsRequest):
        next_pose = CartesianPose(
            (
                request.current_pose.position_m[0] + 0.001,
                request.current_pose.position_m[1],
                request.current_pose.position_m[2],
            ),
            request.current_pose.quaternion_xyzw,
            request.current_pose.frame_id,
            request.action.timestamp_ns + 50_000_000,
        )
        return ForwardDynamicsResult(
            request.current_pose,
            request.action,
            self.scaler.scale(request.action),
            SourceDesiredPose(
                next_pose,
                request.action.gripper_normalized,
                request.action.timestamp_ns,
            ),
            (),
            (),
            "learned_ridge",
            100,
            0.001,
            0.01,
        )


class PerfectXArmTarget:
    def __init__(self):
        self.pose = None
        self.joints = np.zeros(7)
        self.absolute = False
        self.step_count = 0
        self.gripper_commands = []
        self.task_profile = TASK_PROFILES["PickPlaceCan"]

    def reset(self, seed=None):
        self.pose = CartesianPose(
            (0.0, 0.0, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            TARGET_FRAME,
            time.time_ns(),
        )
        self.absolute = False
        self.step_count = 0
        self.gripper_commands = []
        return {}

    def set_absolute_mode(self):
        self.absolute = True

    def read_pose(self):
        return self.pose

    def read_joint_positions(self):
        return self.joints.copy()

    def step_absolute(self, action):
        assert self.absolute
        self.pose = CartesianPose(
            tuple(action[:3]),
            tuple(rotvec_to_quaternion_xyzw(action[3:6])),
            TARGET_FRAME,
            time.time_ns(),
        )
        self.step_count += 1
        return {}, 0.0, False, {}

    def apply_gripper_command_once(self, normalized_command):
        self.gripper_commands.append(float(normalized_command))

    def task_succeeded(self):
        return False


class SourceSucceedsFirst:
    def __init__(self):
        profile = TASK_PROFILES["PickPlaceCan"]
        self.checkpoint_contract = CheckpointContract(
            profile,
            ObservationMode.LOW_DIM,
            tuple(profile.low_dim_observation_shapes),
            profile.native_horizon,
        )
        self.task_profile = profile
        self.policy_step_count = 0
        self.pose = None

    @staticmethod
    def clock_ns():
        return time.time_ns()

    def reset(self, seed):
        self.policy_step_count = 0
        self.pose = CartesianPose(
            (0.0, 0.0, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            SOURCE_FRAME,
            time.time_ns(),
        )
        return {}

    def read_pose(self):
        return self.pose

    def policy_action(self, *, timestamp_ns):
        return SourceAction((0.0,) * 7, SOURCE_FRAME, timestamp_ns)

    def step_policy_once(self, _action):
        self.policy_step_count += 1
        self.pose = CartesianPose(
            (self.policy_step_count * 0.001, 0.0, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            SOURCE_FRAME,
            time.time_ns(),
        )
        return {}, 0.0, False, {}

    @staticmethod
    def read_gripper_state():
        return (0.02, -0.02)

    def make_oracle(self):
        return PandaOnlineOracle(
            step_policy_once=self.step_policy_once,
            read_pose=self.read_pose,
            read_gripper_state=self.read_gripper_state,
        )

    def task_succeeded(self):
        return self.policy_step_count >= 1


class TargetSucceedsSecond(PerfectXArmTarget):
    def task_succeeded(self):
        return self.step_count >= 4


def test_learned_replay_completes_50_step_blocking_path_and_writes_report(tmp_path):
    target = PerfectXArmTarget()
    safety = SimulationSafetyFilter(
        SafetyLimits(
            target_frame=TARGET_FRAME,
            workspace_min_m=(-0.5, -0.5, 0.4),
            workspace_max_m=(0.5, 0.5, 1.2),
            joint_min_rad=(-2.0,) * 7,
            joint_max_rad=(2.0,) * 7,
        )
    )
    controller = AbsoluteBlockingController(
        adapter=target,
        safety_filter=safety,
        position_tolerance_m=1e-9,
        orientation_tolerance_rad=1e-9,
        max_iterations=5,
        timeout_sec=1.0,
    )

    report = run_learned_replay_episode(
        oracle_transitions=oracle_episode(),
        learned_dynamics=PerfectLearnedProvider(),
        target_adapter=target,
        controller=controller,
        seed=42,
        steps=50,
    )

    assert report.completed
    assert report.completed_steps == 50
    assert report.timeout_count == 0
    assert report.rejected_command_count == 0
    assert target.step_count == 100
    assert len(target.gripper_commands) == 50
    assert report.position_error_max_m <= 1e-9
    assert report.orientation_error_max_rad <= 1e-9
    output = write_rollout_reports(tmp_path / "rollout.json", (report,))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["reports"][0]["completed_steps"] == 50
    assert len(payload["reports"][0]["steps"]) == 50
    assert report.task_name == "PickPlaceCan"
    assert report.observation_mode == "low_dim"
    assert report.native_rollout_horizon == 400
    assert report.episode_id.startswith("pick-place-can-")


def test_terminate_on_success_waits_for_target_and_records_source_success():
    source = SourceSucceedsFirst()
    target = TargetSucceedsSecond()
    safety = SimulationSafetyFilter(
        SafetyLimits(
            target_frame=TARGET_FRAME,
            workspace_min_m=(-0.5, -0.5, 0.4),
            workspace_max_m=(0.5, 0.5, 1.2),
            joint_min_rad=(-2.0,) * 7,
            joint_max_rad=(2.0,) * 7,
        )
    )
    controller = AbsoluteBlockingController(
        adapter=target,
        safety_filter=safety,
        position_tolerance_m=1e-9,
        orientation_tolerance_rad=1e-9,
        max_iterations=5,
        timeout_sec=1.0,
    )

    dataset, report = run_online_oracle_episode(
        source_runtime=source,
        target_adapter=target,
        controller=controller,
        seed=7,
        steps=5,
        terminate_on_success=True,
    )

    assert report.completed
    assert report.completed_steps == 2
    assert report.source_success_step == 1
    assert report.target_success_step == 2
    assert source.policy_step_count == 2
    assert len(dataset) == 2
    assert {record.task_name for record in dataset.records} == {"PickPlaceCan"}
    assert {record.observation_mode for record in dataset.records} == {
        ObservationMode.LOW_DIM
    }


class FakeImageWorker:
    def __init__(self):
        profile = TASK_PROFILES["PickPlaceCan"]
        self.checkpoint_contract = CheckpointContract(
            profile,
            ObservationMode.IMAGE_NO_PROPRIO,
            (("agentview_image", (3, 84, 84)),),
            profile.native_horizon,
        )
        self.task_profile = profile
        self.pose = None

    def reset(self, _seed):
        self.pose = CartesianPose(
            (0.0, 0.0, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            SOURCE_FRAME,
            time.time_ns(),
        )
        return self.pose

    def step(self, **_kwargs):
        timestamp = self.pose.timestamp_ns + 1
        action = SourceAction((0.0,) * 7, SOURCE_FRAME, timestamp)
        next_pose = CartesianPose(
            (self.pose.position_m[0] + 0.001, 0.0, 0.8),
            self.pose.quaternion_xyzw,
            SOURCE_FRAME,
            timestamp + 1,
        )
        result = ForwardDynamicsResult(
            self.pose,
            action,
            OscActionScaler().scale(action),
            SourceDesiredPose(next_pose, 0.0, timestamp),
            (0.02, -0.02),
            (0.02, -0.02),
            "panda_online_oracle",
            100,
            0.0,
            0.0,
        )
        current = self.pose
        self.pose = next_pose
        image = np.zeros((84, 84, 3), dtype=np.uint8)
        mask = np.zeros((84, 84), dtype=bool)
        mask[0, 0] = True
        return CrossPaintSourceStep(
            current,
            action,
            result,
            False,
            image,
            image,
            mask,
            image,
            mask,
            CrossPaintStepDiagnostics(
                "cross_paint", 2_000_000, 500_000, 10, 11, 9, 0.001, 0.002, 1
            ),
        )


class FakeImageTarget(PerfectXArmTarget):
    def render_robot_camera_frame(self, **_kwargs):
        return SimpleNamespace(rgb=np.zeros((84, 84, 3), dtype=np.uint8))

    @staticmethod
    def read_manipulated_object_joint_states():
        return {
            "Can_joint0": ManipulatedObjectState(np.zeros(7), np.zeros(6))
        }

    @staticmethod
    def read_gripper_aperture_state():
        return GripperApertureState(0.08, 1.0)


def test_cross_paint_rollout_records_visual_metrics_and_callback():
    worker = FakeImageWorker()
    target = FakeImageTarget()
    safety = SimulationSafetyFilter(
        SafetyLimits(
            target_frame=TARGET_FRAME,
            workspace_min_m=(-0.5, -0.5, 0.4),
            workspace_max_m=(0.5, 0.5, 1.2),
            joint_min_rad=(-2.0,) * 7,
            joint_max_rad=(2.0,) * 7,
        )
    )
    controller = AbsoluteBlockingController(
        adapter=target,
        safety_filter=safety,
        position_tolerance_m=1e-9,
        orientation_tolerance_rad=1e-9,
        max_iterations=5,
        timeout_sec=1.0,
    )
    frames = []

    dataset, report = run_cross_paint_oracle_episode(
        source_worker=worker,
        target_adapter=target,
        controller=controller,
        seed=0,
        steps=2,
        frame_callback=lambda *frame: frames.append(frame),
    )

    assert report.completed
    assert report.observation_mode == "image_no_proprio"
    assert report.visual_input == "cross_paint"
    assert report.perception_latency_mean_ms == 2.0
    assert report.composition_latency_max_ms == 0.5
    assert report.alignment_position_error_max_m == 0.001
    assert len(dataset) == 2
    assert len(frames) == 2
