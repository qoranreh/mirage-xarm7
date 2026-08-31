"""Read-only xArm pointer capture for physical source/target calibration."""

import argparse
import math
import sys
import time

from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile

from mirage_state.calibration import (
    CalibrationError,
    ToolOffset,
    append_capture,
    capture_evidence,
    load_calibration_dataset,
    load_tool_offset,
    tool_tip_pose,
)
from mirage_state.transforms import CartesianPose, TransformError


class PoseCaptureBuffer:
    """Validate and collect distinct, fresh pointer-tip pose samples."""

    def __init__(
        self,
        *,
        expected_frame: str,
        tool: ToolOffset,
        sample_count: int,
        max_age_sec: float,
    ) -> None:
        if not expected_frame:
            raise CalibrationError('expected_frame must not be empty')
        if sample_count < 2:
            raise CalibrationError('sample_count must be at least 2')
        if not math.isfinite(max_age_sec) or max_age_sec <= 0.0:
            raise CalibrationError('max_age_sec must be finite and positive')
        self.expected_frame = expected_frame
        self.tool = tool
        self.sample_count = sample_count
        self.max_age_sec = max_age_sec
        self.poses: list[CartesianPose] = []
        self.ages_sec: list[float] = []
        self.last_stamp_ns: int | None = None
        self.last_rejection = 'waiting for a pose'

    @property
    def complete(self) -> bool:
        return len(self.poses) >= self.sample_count

    def add(
        self, ee_pose: CartesianPose, *, stamp_ns: int, now_ns: int
    ) -> bool:
        """Add a valid sample; return false with ``last_rejection`` otherwise."""
        if self.complete:
            self.last_rejection = 'capture is already complete'
            return False
        if ee_pose.frame_id != self.expected_frame:
            self.last_rejection = (
                f'frame {ee_pose.frame_id!r} does not match '
                f'{self.expected_frame!r}'
            )
            return False
        if stamp_ns <= 0:
            self.last_rejection = 'timestamp is zero'
            return False
        if self.last_stamp_ns is not None and stamp_ns <= self.last_stamp_ns:
            self.last_rejection = 'timestamp is not newer than the last sample'
            return False
        age_sec = (now_ns - stamp_ns) / 1e9
        if age_sec < -0.05:
            self.last_rejection = 'timestamp is in the future'
            return False
        if age_sec > self.max_age_sec:
            self.last_rejection = f'pose is stale by {age_sec:.3f}s'
            return False
        try:
            tip_pose = tool_tip_pose(ee_pose, self.tool)
        except TransformError as error:
            self.last_rejection = str(error)
            return False
        self.poses.append(tip_pose)
        self.ages_sec.append(max(0.0, age_sec))
        self.last_stamp_ns = stamp_ns
        self.last_rejection = ''
        return True

    def evidence(self):
        if not self.complete:
            raise CalibrationError(
                f'capture has {len(self.poses)}/{self.sample_count} samples'
            )
        return capture_evidence(
            self.poses,
            max_sample_age_sec=max(self.ages_sec),
        )


class XArmTransformCapture(Node):
    """Subscribe to EE poses without creating any command-capable endpoint."""

    def __init__(self, *, topic: str, buffer: PoseCaptureBuffer) -> None:
        super().__init__('xarm_transform_capture')
        self.buffer = buffer
        self.rejected_samples = 0
        self.create_subscription(
            PoseStamped,
            topic,
            self._pose_callback,
            QoSProfile(depth=10),
        )
        self.get_logger().warning(
            'read-only calibration capture started; move the robot only with '
            'the pendant in low-speed mode'
        )

    def _pose_callback(self, message: PoseStamped) -> None:
        try:
            pose = CartesianPose(
                position_m=(
                    message.pose.position.x,
                    message.pose.position.y,
                    message.pose.position.z,
                ),
                quaternion_xyzw=(
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                    message.pose.orientation.w,
                ),
                frame_id=message.header.frame_id,
            )
        except TransformError as error:
            self.rejected_samples += 1
            self.buffer.last_rejection = str(error)
            return
        stamp_ns = (
            int(message.header.stamp.sec) * 1_000_000_000
            + int(message.header.stamp.nanosec)
        )
        accepted = self.buffer.add(
            pose,
            stamp_ns=stamp_ns,
            now_ns=self.get_clock().now().nanoseconds,
        )
        if not accepted:
            self.rejected_samples += 1


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Capture one landmark from /mirage/xarm/ee_pose. This executable '
            'does not publish, call a service, create an action client, or move '
            'the robot.'
        )
    )
    parser.add_argument('--draft', required=True, help='Correspondence YAML')
    parser.add_argument('--landmark', required=True)
    parser.add_argument('--tool-config', required=True)
    parser.add_argument('--topic', default='/mirage/xarm/ee_pose')
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--max-age-sec', type=float, default=0.2)
    parser.add_argument('--timeout-sec', type=float, default=20.0)
    return parser.parse_known_args(args)


def run(args=None) -> int:
    parsed, ros_args = _parse_args(args)
    try:
        dataset = load_calibration_dataset(parsed.draft)
        if parsed.landmark not in {item.name for item in dataset.landmarks}:
            raise CalibrationError(f'unknown landmark: {parsed.landmark}')
        tool = load_tool_offset(parsed.tool_config)
        buffer = PoseCaptureBuffer(
            expected_frame=dataset.target_frame,
            tool=tool,
            sample_count=parsed.samples,
            max_age_sec=parsed.max_age_sec,
        )
        if not math.isfinite(parsed.timeout_sec) or parsed.timeout_sec <= 0.0:
            raise CalibrationError('timeout_sec must be finite and positive')
    except CalibrationError as error:
        print(f'error: {error}', file=sys.stderr)
        return 2

    rclpy.init(args=ros_args)
    node = XArmTransformCapture(topic=parsed.topic, buffer=buffer)
    deadline = time.monotonic() + parsed.timeout_sec
    try:
        while not buffer.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if not buffer.complete:
        print(
            f'error: captured {len(buffer.poses)}/{buffer.sample_count}; '
            f'last rejection: {buffer.last_rejection}',
            file=sys.stderr,
        )
        return 2
    try:
        evidence = buffer.evidence()
        if (
            max(evidence.position_std_m)
            > dataset.thresholds.capture_position_std_m
        ):
            raise CalibrationError('capture position standard deviation exceeds gate')
        if (
            evidence.orientation_spread_rad
            > dataset.thresholds.capture_orientation_spread_rad
        ):
            raise CalibrationError('capture orientation spread exceeds gate')
        append_capture(parsed.draft, parsed.landmark, evidence)
    except CalibrationError as error:
        print(f'error: {error}', file=sys.stderr)
        return 1
    print(
        f'appended attempt for {parsed.landmark}: '
        f'{evidence.sample_count} samples, '
        f'max std={max(evidence.position_std_m):.6f} m'
    )
    return 0


def main(args=None) -> None:
    raise SystemExit(run(args))


if __name__ == '__main__':
    main(sys.argv[1:])
