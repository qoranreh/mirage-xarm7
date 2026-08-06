"""Connect to xArm7 state only and run Gate A without controllers."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    robot_ip = LaunchConfiguration('robot_ip')
    add_gripper = LaunchConfiguration('add_gripper')
    show_rviz = LaunchConfiguration('show_rviz')

    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('xarm_api'), 'launch', 'xarm7_driver.launch.py']
            )
        ),
        launch_arguments={
            'robot_ip': robot_ip,
            'report_type': 'rich',
            'add_gripper': add_gripper,
            'show_rviz': 'false',
        }.items(),
    )
    robot_state = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('xarm_description'),
                    'launch',
                    '_robot_joint_state.launch.py',
                ]
            )
        ),
        launch_arguments={
            'dof': '7',
            'robot_type': 'xarm',
            'hw_ns': 'xarm',
            'add_gripper': add_gripper,
        }.items(),
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare('xarm_description'),
                    'launch',
                    '_rviz_display.launch.py',
                ]
            )
        ),
        condition=IfCondition(show_rviz),
    )
    gate = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare('mirage_xarm_state'), 'launch', 'gate_a.launch.py']
            )
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'record': LaunchConfiguration('record'),
            'publish_canonical': LaunchConfiguration('publish_canonical'),
            'duration_sec': LaunchConfiguration('duration_sec'),
            'require_gripper': add_gripper,
            'require_robot_state': 'true',
            'require_motion': LaunchConfiguration('require_motion'),
            'check_moveit_fk': 'false',
            'require_moveit_fk': 'false',
            'recordings_dir': LaunchConfiguration('recordings_dir'),
            'reports_dir': LaunchConfiguration('reports_dir'),
        }.items(),
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'robot_ip',
                description='Explicit IP of the xArm7 to read from.',
            ),
            DeclareLaunchArgument('add_gripper', default_value='true'),
            DeclareLaunchArgument('show_rviz', default_value='true'),
            DeclareLaunchArgument('record', default_value='true'),
            DeclareLaunchArgument('publish_canonical', default_value='false'),
            DeclareLaunchArgument('duration_sec', default_value='30.0'),
            DeclareLaunchArgument('require_motion', default_value='false'),
            DeclareLaunchArgument(
                'recordings_dir',
                default_value='~/Mirage_xarm_ws/recordings',
            ),
            DeclareLaunchArgument(
                'reports_dir', default_value='~/Mirage_xarm_ws/reports'
            ),
            driver,
            robot_state,
            rviz,
            gate,
        ]
    )
