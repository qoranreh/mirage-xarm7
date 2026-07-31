"""Launch the xArm state monitor and an optional rosbag recorder."""

from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


RECORDED_TOPICS = (
    '/joint_states',
    '/tf',
    '/tf_static',
    '/clock',
    '/xarm/robot_states',
    '/mirage/xarm/arm_joint_states',
    '/mirage/xarm/ee_pose',
    '/mirage/xarm/gripper_joint_state',
    '/mirage/xarm/state_diagnostics',
)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _make_recorder(context):
    if not _as_bool(LaunchConfiguration('record').perform(context)):
        return [LogInfo(msg='xArm state rosbag recording disabled')]

    recordings_dir = Path(
        LaunchConfiguration('recordings_dir').perform(context)
    ).expanduser()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_path = recordings_dir / f'xarm_state_{timestamp}'

    command = [
        'ros2',
        'bag',
        'record',
        '--storage',
        'sqlite3',
        '--output',
        str(output_path),
    ]
    if _as_bool(LaunchConfiguration('use_sim_time').perform(context)):
        command.append('--use-sim-time')
    command.extend(RECORDED_TOPICS)

    return [
        LogInfo(msg=f'Recording xArm state to {output_path}'),
        ExecuteProcess(
            cmd=command,
            output='screen',
            sigterm_timeout='10',
            sigkill_timeout='5',
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('mirage_xarm_state')
    default_config = str(Path(package_share) / 'config' / 'xarm_state.yaml')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description=(
                    'Use /clock for the monitor and rosbag timestamps.'
                ),
            ),
            DeclareLaunchArgument(
                'record',
                default_value='true',
                description='Start ros2 bag record alongside the monitor.',
            ),
            DeclareLaunchArgument(
                'recordings_dir',
                default_value='~/Mirage_xarm_ws/recordings',
                description='Parent directory for xArm state bags.',
            ),
            DeclareLaunchArgument(
                'config_file',
                default_value=default_config,
                description='Monitor parameter YAML file.',
            ),
            Node(
                package='mirage_xarm_state',
                executable='xarm_state_monitor',
                name='xarm_state_monitor',
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
            OpaqueFunction(function=_make_recorder),
        ]
    )
