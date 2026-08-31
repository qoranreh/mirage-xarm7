"""Start the read-only xArm CanonicalState publisher."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory('mirage_xarm_state'))
    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='false'),
            DeclareLaunchArgument(
                'config_file',
                default_value=str(
                    package_share / 'config' / 'canonical_state.yaml'
                ),
            ),
            Node(
                package='mirage_xarm_state',
                executable='xarm_canonical_state',
                name='xarm_canonical_state',
                output='screen',
                parameters=[
                    LaunchConfiguration('config_file'),
                    {
                        'use_sim_time': ParameterValue(
                            LaunchConfiguration('use_sim_time'),
                            value_type=bool,
                        )
                    },
                ],
            ),
        ]
    )
