#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

configurable_parameters = [
    {'name': 'baudrate', 'default': "460800", 'description': 'Baudrate for serial connection'},
    {'name': 'portname', 'default': "/dev/ttyAML1", 'description': 'Serial port name'},
    {'name': 'update_rate', 'default': "50", 'description': 'Update rate in Hz'},
    {'name': 'output_freq_hz', 'default': "50", 'description': 'Update rate in Hz'},
    {'name': 'use_filter', 'default': "True", 'description': 'Update rate in Hz'},
    {'name': 'verbose', 'default': "True", 'description': 'Enable verbose logging'},
]

def declare_configurable_parameters(parameters):
    return [DeclareLaunchArgument(param['name'], default_value=param['default'], description=param['description']) for param in parameters]

def set_configurable_parameters(parameters):
    return dict([(param['name'], LaunchConfiguration(param['name'])) for param in parameters])

def generate_launch_description():

    declared_parameters = declare_configurable_parameters(configurable_parameters)
    configured_parameters = set_configurable_parameters(configurable_parameters)

    namespace = LaunchConfiguration("namespace", default='')

    return LaunchDescription(
        declared_parameters +
        [
            Node(
                package='sdm15_ros2',
                executable='sdm15_ros2_node',
                output='screen',
                emulate_tty=True,
                namespace=namespace,
                parameters=[
                    configured_parameters
                ],
            ),
        ]
    )
