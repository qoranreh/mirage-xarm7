"""Run the state monitor, Gate A validator, and optional rosbag recorder."""

from datetime import datetime
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
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
    '/mirage/xarm/gate_a_report',
    '/mirage/canonical_state',
)


def _as_bool(value: str) -> bool:
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context):
    use_sim_time = _as_bool(
        LaunchConfiguration('use_sim_time').perform(context)
    )
    record = _as_bool(LaunchConfiguration('record').perform(context))
    publish_canonical = _as_bool(
        LaunchConfiguration('publish_canonical').perform(context)
    )
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    recordings_dir = Path(
        LaunchConfiguration('recordings_dir').perform(context)
    ).expanduser()
    reports_dir = Path(
        LaunchConfiguration('reports_dir').perform(context)
    ).expanduser()
    recordings_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    bag_path = recordings_dir / f'xarm_gate_a_{timestamp}'
    report_path = reports_dir / f'gate_a_{timestamp}.json'

    monitor = Node(
        package='mirage_xarm_state',
        executable='xarm_state_monitor',
        name='xarm_state_monitor',
        output='screen',
        parameters=[
            LaunchConfiguration('monitor_config_file'),
            {'use_sim_time': use_sim_time},
        ],
    )
    validator = Node(
        package='mirage_xarm_state',
        executable='xarm_gate_a_validator',
        name='xarm_gate_a_validator',
        output='screen',
        parameters=[
            LaunchConfiguration('validator_config_file'),
            {
                'use_sim_time': use_sim_time,
                'duration_sec': ParameterValue(
                    LaunchConfiguration('duration_sec'), value_type=float
                ),
                'report_path': str(report_path),
                'require_gripper': ParameterValue(
                    LaunchConfiguration('require_gripper'), value_type=bool
                ),
                'require_robot_state': ParameterValue(
                    LaunchConfiguration('require_robot_state'),
                    value_type=bool,
                ),
                'require_motion': ParameterValue(
                    LaunchConfiguration('require_motion'), value_type=bool
                ),
                'check_moveit_fk': ParameterValue(
                    LaunchConfiguration('check_moveit_fk'), value_type=bool
                ),
                'require_moveit_fk': ParameterValue(
                    LaunchConfiguration('require_moveit_fk'), value_type=bool
                ),
            },
        ],
    )
    actions = [
        LogInfo(msg=f'Gate A JSON report: {report_path}'),
        monitor,
    ]
    if publish_canonical:
        actions.append(
            Node(
                package='mirage_xarm_state',
                executable='xarm_canonical_state',
                name='xarm_canonical_state',
                output='screen',
                parameters=[
                    LaunchConfiguration('canonical_config_file'),
                    {'use_sim_time': use_sim_time},
                ],
            )
        )
    if record:
        command = [
            'ros2',
            'bag',
            'record',
            '--storage',
            'sqlite3',
            '--output',
            str(bag_path),
        ]
        if use_sim_time:
            command.append('--use-sim-time')
        command.extend(RECORDED_TOPICS)
        actions.extend(
            [
                LogInfo(msg=f'Gate A rosbag: {bag_path}'),
                ExecuteProcess(
                    cmd=command,
                    output='screen',
                    sigterm_timeout='10',
                    sigkill_timeout='5',
                ),
            ]
        )
    else:
        actions.append(LogInfo(msg='Gate A rosbag recording disabled'))

    actions.extend(
        [
            TimerAction(
                period=float(
                    LaunchConfiguration('validator_start_delay_sec').perform(
                        context
                    )
                ),
                actions=[validator],
            ),
            RegisterEventHandler(
                OnProcessExit(
                    target_action=validator,
                    on_exit=[
                        EmitEvent(
                            event=Shutdown(
                                reason='Gate A validator completed'
                            )
                        )
                    ],
                )
            ),
        ]
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    package_share = Path(
        get_package_share_directory('mirage_xarm_state')
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='false'),
            DeclareLaunchArgument('record', default_value='true'),
            DeclareLaunchArgument('publish_canonical', default_value='false'),
            DeclareLaunchArgument('duration_sec', default_value='30.0'),
            DeclareLaunchArgument(
                'validator_start_delay_sec', default_value='2.0'
            ),
            DeclareLaunchArgument('require_gripper', default_value='true'),
            DeclareLaunchArgument(
                'require_robot_state', default_value='false'
            ),
            DeclareLaunchArgument('require_motion', default_value='false'),
            DeclareLaunchArgument('check_moveit_fk', default_value='true'),
            DeclareLaunchArgument(
                'require_moveit_fk', default_value='false'
            ),
            DeclareLaunchArgument(
                'recordings_dir',
                default_value='~/Mirage_xarm_ws/recordings',
            ),
            DeclareLaunchArgument(
                'reports_dir', default_value='~/Mirage_xarm_ws/reports'
            ),
            DeclareLaunchArgument(
                'monitor_config_file',
                default_value=str(package_share / 'config' / 'xarm_state.yaml'),
            ),
            DeclareLaunchArgument(
                'validator_config_file',
                default_value=str(package_share / 'config' / 'gate_a.yaml'),
            ),
            DeclareLaunchArgument(
                'canonical_config_file',
                default_value=str(
                    package_share / 'config' / 'canonical_state.yaml'
                ),
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
