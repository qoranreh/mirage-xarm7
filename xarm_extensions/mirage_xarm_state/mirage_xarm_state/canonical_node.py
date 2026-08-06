"""Publish CanonicalState from the read-only normalized xArm topics."""

from typing import Optional

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from mirage_state_interfaces.msg import CanonicalState, ObjectState
from sensor_msgs.msg import JointState

from mirage_state.adapters import (
    GripperSample,
    ObjectSample,
    XArm7CanonicalAdapter,
    XArmAdapterConfig,
)
from mirage_state.ros import (
    canonical_state_to_ros,
    gripper_sample_from_ros,
    object_sample_from_ros,
    pose_sample_from_ros,
)


class XArmCanonicalStateNode(Node):
    """Join pose, gripper, and object samples without issuing commands."""

    def __init__(self) -> None:
        super().__init__('xarm_canonical_state')
        self.declare_parameter('pose_topic', '/mirage/xarm/ee_pose')
        self.declare_parameter(
            'gripper_topic', '/mirage/xarm/gripper_joint_state'
        )
        self.declare_parameter('object_topic', '/mirage/object_state')
        self.declare_parameter('output_topic', '/mirage/canonical_state')
        self.declare_parameter('gripper_joint_name', 'drive_joint')
        self.declare_parameter('source_robot', 'xarm7')
        self.declare_parameter('expected_frame_id', 'link_base')
        self.declare_parameter('max_pose_age_sec', 0.15)
        self.declare_parameter('max_gripper_age_sec', 0.30)
        self.declare_parameter('max_object_age_sec', 0.30)
        self.declare_parameter('max_component_skew_sec', 0.10)
        self.declare_parameter('open_angle_rad', 0.0)
        self.declare_parameter('closed_angle_rad', 0.85)
        self.declare_parameter('open_width_m', 0.085)
        self.declare_parameter('closed_width_m', 0.0)
        self.declare_parameter('angle_tolerance_rad', 0.02)

        self._gripper_joint_name = str(
            self.get_parameter('gripper_joint_name').value
        )
        self._adapter = XArm7CanonicalAdapter(
            XArmAdapterConfig(
                source_robot=str(self.get_parameter('source_robot').value),
                expected_frame_id=str(
                    self.get_parameter('expected_frame_id').value
                ),
                max_pose_age_sec=float(
                    self.get_parameter('max_pose_age_sec').value
                ),
                max_gripper_age_sec=float(
                    self.get_parameter('max_gripper_age_sec').value
                ),
                max_object_age_sec=float(
                    self.get_parameter('max_object_age_sec').value
                ),
                max_component_skew_sec=float(
                    self.get_parameter('max_component_skew_sec').value
                ),
                open_angle_rad=float(
                    self.get_parameter('open_angle_rad').value
                ),
                closed_angle_rad=float(
                    self.get_parameter('closed_angle_rad').value
                ),
                open_width_m=float(
                    self.get_parameter('open_width_m').value
                ),
                closed_width_m=float(
                    self.get_parameter('closed_width_m').value
                ),
                angle_tolerance_rad=float(
                    self.get_parameter('angle_tolerance_rad').value
                ),
            )
        )
        self._gripper: Optional[GripperSample] = None
        self._object: Optional[ObjectSample] = None
        self._publisher = self.create_publisher(
            CanonicalState,
            str(self.get_parameter('output_topic').value),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            JointState,
            str(self.get_parameter('gripper_topic').value),
            self._gripper_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            ObjectState,
            str(self.get_parameter('object_topic').value),
            self._object_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter('pose_topic').value),
            self._pose_callback,
            qos_profile_sensor_data,
        )

    def _gripper_callback(self, message: JointState) -> None:
        try:
            self._gripper = gripper_sample_from_ros(
                message, self._gripper_joint_name
            )
        except ValueError as error:
            self.get_logger().warning(str(error))
            self._gripper = None

    def _object_callback(self, message: ObjectState) -> None:
        self._object = object_sample_from_ros(message)

    def _pose_callback(self, message: PoseStamped) -> None:
        state = self._adapter.adapt(
            pose=pose_sample_from_ros(message),
            gripper=self._gripper,
            object_sample=self._object,
            now_ns=self.get_clock().now().nanoseconds,
        )
        self._publisher.publish(canonical_state_to_ros(state))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = XArmCanonicalStateNode()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
