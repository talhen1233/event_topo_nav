#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PACKAGE_NAME = "of_localization"
DEFAULT_CONFIG_FILE = os.path.join(
    get_package_share_directory(PACKAGE_NAME), "config", "params.yaml"
)


def generate_launch_description() -> LaunchDescription:
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

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
                executable="of_localization_node",
                name="of_localization_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    params_file,
                    {
                        "use_sim_time": ParameterValue(
                            use_sim_time,
                            value_type=bool,
                        ),
                    },
                ],
            ),
        ]
    )
