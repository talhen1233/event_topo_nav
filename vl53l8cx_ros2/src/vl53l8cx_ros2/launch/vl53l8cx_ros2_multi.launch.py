#!/usr/bin/env python3

from launch import LaunchDescription, LaunchContext
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

_common_params_spec = [
    ('executable_path',    '/workspaces/vl53l8cx_ros2/src/vl53l8cx_ros2/config',
                           'Path to driver executable', str),
    ('publish_depth',      'true',  'Publish depth image',    bool),
    ('publish_pointcloud', 'true',  'Publish pointcloud',     bool),
    ('valid_statuses',     '5,9',   'Comma-separated accepted target statuses', list),
    ('i2c_device_path',    '/dev/i2c-0', 'I²C device path',   str),
    ('verbose',            'false', 'Colored console output', bool),
]

_sensor_list_args = [
    ('sensor_names',      'sensor0',     'Comma-sep namespaces'),
    ('sensor_addresses',  '0x29',        'Comma-sep I²C addresses'),
    ('sensor_frame_ids',  'vl53_frame0', 'Comma-sep frame_ids'),
    ('enable_sensors',    'true',        'Comma-sep booleans'),
]


def _declare_args(spec):
    return [DeclareLaunchArgument(name, default_value=default, description=desc)
            for name, default, desc, *_ in spec]

def _declare_sensor_args(spec):
    return [DeclareLaunchArgument(name, default_value=default, description=desc)
            for name, default, desc in spec]

def _typed_value(raw: str, pytype):
    if pytype is bool:
        return raw.lower() in ('1', 'true', 'yes')
    if pytype is int:
        return int(raw, 0)
    if pytype is list:
        return [int(value.strip(), 0) for value in raw.split(',') if value.strip()]
    return raw

def _configure_nodes(context: LaunchContext, *_) -> list:
    common_params = {}
    for name, _, _, typ in _common_params_spec:
        raw = context.launch_configurations[name]
        common_params[name] = _typed_value(raw, typ)

    names  = context.launch_configurations['sensor_names'     ].split(',')
    addrs  = context.launch_configurations['sensor_addresses' ].split(',')
    frames = context.launch_configurations['sensor_frame_ids' ].split(',')
    enabs  = context.launch_configurations['enable_sensors'   ].split(',')
    max_len = max(len(names), len(addrs), len(frames), len(enabs))
    names  += [f'sensor{idx}'       for idx in range(len(names),  max_len)]
    addrs  += ['0x29']              * (max_len - len(addrs))
    frames += [f'vl53_frame{idx}'   for idx in range(len(frames), max_len)]
    enabs  += ['true']              * (max_len - len(enabs))

    nodes = []
    for idx in range(max_len):
        if enabs[idx].strip().lower() not in ('1', 'true', 'yes'):
            continue

        node = Node(
            package='vl53l8cx_ros2',
            executable='vl53l8cx_ros2_node',
            namespace=names[idx].strip(),
            name='vl53l8cx_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                common_params,
                {
                    'sensor_address': int(addrs[idx].strip(), 0),
                    'frame_id': frames[idx].strip()
                }
            ]
        )
        nodes.append(node)
    return nodes

def generate_launch_description():
    declared_common  = _declare_args(_common_params_spec)
    declared_sensors = _declare_sensor_args(_sensor_list_args)
    return LaunchDescription(
        declared_common +
        declared_sensors + [
            OpaqueFunction(function=_configure_nodes)
        ]
    )
