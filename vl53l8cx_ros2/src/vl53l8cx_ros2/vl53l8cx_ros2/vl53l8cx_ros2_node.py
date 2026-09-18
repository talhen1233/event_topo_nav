#!/usr/bin/env python3

import os
import subprocess
import re
import numpy as np
import threading
import struct
import math
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField


def is_valid_target_status(status, valid_statuses):
    """Return whether a VL53L8CX target status is explicitly accepted."""
    return int(status) in {int(value) for value in valid_statuses}

class VL53L8Node(Node):
    def __init__(self):
        super().__init__('vl53l8cx_ros2_node')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('executable_path', '/workspaces/vl53l8cx_ros2/src/vl53l8cx_ros2/config'),
                ('publish_depth', True),
                ('publish_pointcloud', True),
                ('valid_statuses', [5, 9]),  # UM3109: 5 valid, 9 usable
                ('frame_id', "vl53l8_frame"),
                ('verbose', False),
                ('sensor_address', 0x2A),
                ('i2c_device_path', '/dev/i2c-0')
            ]
        )

        self.executable_path = self.get_parameter('executable_path').value
        self.publish_depth = self.get_parameter('publish_depth').value
        self.publish_pointcloud = self.get_parameter('publish_pointcloud').value
        self.valid_statuses = tuple(
            int(value) for value in self.get_parameter('valid_statuses').value
        )
        self.frame_id = self.get_parameter('frame_id').value
        self.verbose = self.get_parameter('verbose').value
        self.sensor_address = self.get_parameter('sensor_address').value
        self.i2c_device_path = self.get_parameter('i2c_device_path').value

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_pub = self.create_publisher(Image, '~/depth_image', qos) if self.publish_depth else None
        self.cloud_pub = self.create_publisher(PointCloud2, '~/pointcloud', qos) if self.publish_pointcloud else None
        self.image_pub_color = None
        if self.verbose and self.publish_depth:
            self.image_pub_color = self.create_publisher(Image, '~/depth_image_color', qos)

        self.get_logger().info(f"Publishing Depth: {self.publish_depth}, PointCloud: {self.publish_pointcloud}, Verbose: {self.verbose}")

        env = dict(os.environ)
        env["VL53L8CX_I2C_DEV"] = self.i2c_device_path
        addr_8bit = self.sensor_address << 1  # driver expects 8-bit address
        env["VL53L8CX_I2C_ADDRESS"] = hex(addr_8bit)

        sensor_executable = os.path.join(self.executable_path, 'vl53l8_test')
        self.get_logger().info(f"Starting sensor with: {sensor_executable}")
        self.get_logger().info(f"Env: VL53L8CX_I2C_DEV={env['VL53L8CX_I2C_DEV']}, VL53L8CX_I2C_ADDRESS={env['VL53L8CX_I2C_ADDRESS']}")

        self.process = subprocess.Popen(
            [sensor_executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env
        )
        self.get_logger().info("Sensor process started.")

        self.line_re = re.compile(r"Zone\s*:\s*(\d+),\s*Status\s*:\s*(\d+),\s*Distance\s*:\s*(\d+)\s*mm")
        self.current_distances = np.full(64, np.nan, dtype=np.float32)
        self.zone_count = 0

        threading.Thread(target=self.read_from_process, daemon=True).start()

        self.width = self.height = 8
        self.hfov = math.radians(65.0)
        self.vfov = math.radians(65.0)

    def read_from_process(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                self.get_logger().warn("Sensor process terminated or no more data.")
                break

            match = self.line_re.match(line.strip())
            if match:
                zone, status, distance = map(int, match.groups())
                if is_valid_target_status(status, self.valid_statuses):
                    self.current_distances[zone] = distance / 1000.0

                self.zone_count += 1
                if self.zone_count == 64:
                    if self.publish_depth:
                        self.publish_depth_image()
                    if self.verbose and self.publish_depth:
                        self.publish_color_image()
                    if self.publish_pointcloud:
                        self.publish_point_cloud()

                    self.current_distances[:] = np.nan
                    self.zone_count = 0

    def publish_depth_image(self):
        msg = Image()
        distances = np.nan_to_num(self.current_distances, nan=0.0).reshape((self.height, self.width))
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "32FC1"
        msg.is_bigendian = False
        msg.step = self.width * 4
        msg.data = distances.tobytes()
        self.image_pub.publish(msg)
        self.get_logger().debug("Depth image published.")

    def publish_color_image(self):
        distances = np.nan_to_num(self.current_distances, nan=0.0).reshape((self.height, self.width))
        scaled = np.clip(distances / 4.0, 0, 1) * 255
        color_img = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_JET)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = self.width * 3
        msg.data = color_img.tobytes()
        self.image_pub_color.publish(msg)
        self.get_logger().debug("Color-mapped depth image published.")

    def publish_point_cloud(self):
        distances = self.current_distances.reshape((self.height, self.width))
        points = []
        center = ((self.height - 1) / 2.0, (self.width - 1) / 2.0)

        for r in range(self.height):
            for c in range(self.width):
                dist = distances[r, c]
                angle_x = (c - center[1]) * (self.hfov / (self.width - 1))
                angle_y = (r - center[0]) * (self.vfov / (self.height - 1))
                X = dist * math.tan(angle_x)
                Y = dist * math.tan(angle_y)
                points.append((X, Y, dist))

        if not points:
            self.get_logger().debug("No points to publish.")
            return

        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.height = 1
        msg.width = len(points)
        msg.is_bigendian = False
        msg.is_dense = False
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1)
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        packed_data = []
        for p in points:
            packed_data.append(struct.pack('fff', *p))
        msg.data = b''.join(packed_data)
        self.cloud_pub.publish(msg)
        self.get_logger().debug("Point cloud published.")

    def destroy_node(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write("stop\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
                self.get_logger().info("Sensor process terminated.")
            except Exception as e:
                self.get_logger().error(f"Error stopping sensor: {e}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VL53L8Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
