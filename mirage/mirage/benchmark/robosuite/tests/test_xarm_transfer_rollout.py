import json
import time

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
)
from mirage.benchmark.robosuite.xarm_transfer.geometry import (
    rotvec_to_quaternion_xyzw,
)
from mirage.benchmark.robosuite.xarm_transfer.rollout import (
    run_learned_replay_episode,
    write_rollout_reports,
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

    def reset(self, seed=None):
        self.pose = CartesianPose(
            (0.0, 0.0, 0.8),
            (0.0, 0.0, 0.0, 1.0),
            TARGET_FRAME,
            time.time_ns(),
        )
        self.absolute = False
        self.step_count = 0
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
    assert target.step_count == 50
    assert report.position_error_max_m <= 1e-9
    assert report.orientation_error_max_rad <= 1e-9
    output = write_rollout_reports(tmp_path / "rollout.json", (report,))
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["reports"][0]["completed_steps"] == 50
    assert len(payload["reports"][0]["steps"]) == 50
