"""Read-only Robomimic inference from CanonicalState ROS messages."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import robomimic.utils.file_utils as FileUtils
from std_msgs.msg import Float64MultiArray

from mirage_state.policy import (
    CanonicalPolicyObservationProjector,
    CanonicalStateError,
)
from mirage_state.ros import canonical_state_from_ros
from mirage_state_interfaces.msg import CanonicalState


class CanonicalPolicyProbe(Node):
    """Run policy inference and publish previews without commanding a robot."""

    def __init__(
        self,
        *,
        policy: Any,
        projector: CanonicalPolicyObservationProjector,
        canonical_topic: str,
        preview_topic: str,
    ) -> None:
        super().__init__('canonical_policy_probe')
        self._policy = policy
        self._projector = projector
        self._last_timestamp_ns: int | None = None
        self._preview_publisher = self.create_publisher(
            Float64MultiArray, preview_topic, 10
        )
        self.create_subscription(
            CanonicalState,
            canonical_topic,
            self._canonical_callback,
            qos_profile_sensor_data,
        )
        self._policy.start_episode()
        self.get_logger().warning(
            'read-only policy probe started; actions are previews and are not '
            'connected to a controller'
        )

    def _canonical_callback(self, message: CanonicalState) -> None:
        state = canonical_state_from_ros(message)
        if state.timestamp_ns == self._last_timestamp_ns:
            return
        try:
            observation = self._projector.project(
                state, now_ns=self.get_clock().now().nanoseconds
            )
        except CanonicalStateError as error:
            self.get_logger().warning(f'policy inference blocked: {error}')
            return

        action = np.asarray(self._policy(ob=observation), dtype=np.float64)
        preview = Float64MultiArray()
        preview.data = action.reshape(-1).tolist()
        self._preview_publisher.publish(preview)
        self._last_timestamp_ns = state.timestamp_ns


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description=(
            'Run read-only policy inference from /mirage/canonical_state. '
            'No action is sent to a robot.'
        )
    )
    parser.add_argument('--agent', required=True, help='Robomimic checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--canonical-topic', default='/mirage/canonical_state')
    parser.add_argument(
        '--preview-topic', default='/mirage/policy_action_preview'
    )
    parser.add_argument('--expected-frame', required=True)
    parser.add_argument('--object-schema', required=True)
    parser.add_argument('--max-state-age-sec', type=float, default=0.15)
    return parser.parse_known_args(args)


def main(args=None) -> None:
    parsed, ros_args = _parse_args(args)
    checkpoint = Path(parsed.agent).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint does not exist: {checkpoint}')
    policy, checkpoint_data = FileUtils.policy_from_checkpoint(
        ckpt_path=str(checkpoint), device=parsed.device, verbose=True
    )
    projector = CanonicalPolicyObservationProjector.from_shape_metadata(
        checkpoint_data['shape_metadata'],
        expected_frame_id=parsed.expected_frame,
        expected_object_schema=parsed.object_schema,
        max_state_age_sec=parsed.max_state_age_sec,
    )

    rclpy.init(args=ros_args)
    node = CanonicalPolicyProbe(
        policy=policy,
        projector=projector,
        canonical_topic=parsed.canonical_topic,
        preview_topic=parsed.preview_topic,
    )
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
