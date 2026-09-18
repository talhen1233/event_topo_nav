#!/usr/bin/env python3
"""Launch event system, smart navigation, and the web dashboard from one YAML."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PACKAGE_NAME = "event_navigation_system"
DEFAULT_CONFIG_FILE = os.path.join(
    get_package_share_directory(PACKAGE_NAME), "config", "params.yaml"
)


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    common_parameters = [
        params_file,
        {
            "use_sim_time": ParameterValue(
                use_sim_time,
                value_type=bool,
            ),
        },
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=DEFAULT_CONFIG_FILE,
                description="Path to the parameter YAML file.",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="Use the ROS simulation clock.",
            ),
            Node(
                package=PACKAGE_NAME,
                executable="event_system_node",
                name="event_system",
                output="screen",
                emulate_tty=True,
                parameters=common_parameters,
            ),
            Node(
                package=PACKAGE_NAME,
                executable="smart_navigation_node",
                name="smart_navigation",
                output="screen",
                emulate_tty=True,
                parameters=common_parameters,
            ),
            Node(
                package=PACKAGE_NAME,
                executable="web_dashboard_node",
                name="web_dashboard",
                output="screen",
                emulate_tty=True,
                parameters=common_parameters,
            ),
        ]
    )
