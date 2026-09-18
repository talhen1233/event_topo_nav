import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    package_name = 'drone_simulation'
    pkg_share = get_package_share_directory(package_name)

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time from /clock',
    )
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_sim_time_param = ParameterValue(use_sim_time, value_type=bool)

    default_world = os.path.join(pkg_share, 'worlds', 'crazyflie_final_prelim_03.sdf')
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='World to load',
    )
    world = LaunchConfiguration('world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'),
        ]),
        launch_arguments=[('gz_args', ['-r ', world]), ('on_exit_shutdown', 'true')],
    )

    bridge_params = os.path.join(pkg_share, 'config', 'crazyflie_gz_bridge.yaml')
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time_param}],
    )

    sim_time_enforcer_params = os.path.join(pkg_share, 'config', 'sim_time_enforcer.yaml')
    sim_time_enforcer = Node(
        package='drone_simulation',
        executable='sim_time_enforcer',
        name='sim_time_enforcer',
        output='screen',
        parameters=[sim_time_enforcer_params, {'use_sim_time': use_sim_time_param}],
    )

    hover_params = os.path.join(pkg_share, 'config', 'kinematic_hover_shim.yaml')
    hover_shim = Node(
        package='drone_simulation',
        executable='kinematic_hover_shim',
        output='screen',
        emulate_tty=True,
        parameters=[hover_params, {'use_sim_time': use_sim_time_param}],
    )

    crazyflie_tf = Node(
        package='drone_simulation',
        executable='crazyflie_odom_to_tf',
        name='crazyflie_odom_to_tf',
        output='screen',
        parameters=[{
            'publish_odom_tf': False,
            'odom_topic': '/crazyflie/odometry',
            'base_link_frame': 'crazyflie/base_link',
            'use_sim_time': use_sim_time_param,
        }],
    )

    odo2imu_params = os.path.join(pkg_share, 'config', 'odometry_to_imu.yaml')
    odometry_to_imu = Node(
        package='drone_simulation',
        executable='odometry_to_imu',
        name='odometry_to_imu',
        output='screen',
        parameters=[odo2imu_params, {'use_sim_time': use_sim_time_param}],
    )

    controller_params = os.path.join(pkg_share, 'config', 'crazyflie_geometric_controller.yaml')
    crazyflie_controller = Node(
        package='drone_simulation',
        executable='crazyflie_firmware_controller',
        name='crazyflie_firmware_controller',
        output='screen',
        parameters=[controller_params, {'use_sim_time': use_sim_time_param}],
    )

    crazyflie_camera_info = Node(
        package='drone_simulation',
        executable='crazyflie_camera_info_publisher',
        name='crazyflie_camera_info_publisher',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time_param}],
    )

    teleop_keyboard = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard',
        remappings=[('/cmd_vel', '/crazyflie/cmd_vel_user')],
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': use_sim_time_param}],
    )

    return LaunchDescription([
        use_sim_time_arg,
        world_arg,
        gazebo,
        ros_gz_bridge,
        sim_time_enforcer,
        hover_shim,
        crazyflie_tf,
        odometry_to_imu,
        crazyflie_controller,
        crazyflie_camera_info,
        teleop_keyboard,
    ])
