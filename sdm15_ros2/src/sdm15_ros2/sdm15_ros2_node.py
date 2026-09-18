#!/usr/bin/env python3

import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16, UInt8

from sdm15_ros2.sdm15 import BaudRate, FilterHex, OutputFreqHex, SDM15


class SDM15Node(Node):
    def __init__(self):
        super().__init__('sdm15_node')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.declare_parameter('portname', '/dev/ttyAML1')
        self.declare_parameter('baudrate', 460800)
        self.declare_parameter('update_rate', 50)
        self.declare_parameter('output_freq_hz', 50)
        self.declare_parameter('verbose', True)
        self.declare_parameter('use_filter', False)

        self.portname = self.get_parameter('portname').get_parameter_value().string_value
        self.baudrate_val = self.get_parameter('baudrate').get_parameter_value().integer_value
        self.update_rate = self.get_parameter('update_rate').get_parameter_value().integer_value
        self.output_freq = self.get_parameter('output_freq_hz').get_parameter_value().integer_value
        self.verbose = self.get_parameter('verbose').get_parameter_value().bool_value
        self.use_filter = self.get_parameter('use_filter').get_parameter_value().bool_value

        if self.baudrate_val == 230400:
            chosen_baud = BaudRate.BAUD_230400
        elif self.baudrate_val == 460800:
            chosen_baud = BaudRate.BAUD_460800
        elif self.baudrate_val == 921600:
            chosen_baud = BaudRate.BAUD_921600
        elif self.baudrate_val == 1500000:
            chosen_baud = BaudRate.BAUD_1500000
        else:
            self.get_logger().warn(f"Unsupported baudrate {self.baudrate_val}, default to 460800.")
            chosen_baud = BaudRate.BAUD_460800

        try:
            self.lidar = SDM15(self.portname, chosen_baud)
            self.get_logger().info(f"Connected to SDM15 on {self.portname} at {chosen_baud} baud.")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to LiDAR: {e}")
            return

        try:
            version_info = self.lidar.obtain_version_info()
            self.get_logger().info(f"LiDAR Version Info: {version_info}")
            self.lidar.lidar_self_test()
            self.get_logger().info("Self-test succeeded.")
        except Exception as e:
            self.get_logger().error(f"Version/Self-test failed: {e}")
            return

        if self.output_freq <= 10:
            freq_hex = OutputFreqHex.Freq_10Hz
        elif self.output_freq <= 100:
            freq_hex = OutputFreqHex.Freq_100Hz
        elif self.output_freq <= 200:
            freq_hex = OutputFreqHex.Freq_200Hz
        elif self.output_freq <= 500:
            freq_hex = OutputFreqHex.Freq_500Hz
        elif self.output_freq <= 1000:
            freq_hex = OutputFreqHex.Freq_1000Hz
        else:
            freq_hex = OutputFreqHex.Freq_1800Hz

        try:
            self.lidar.set_output_freq(freq_hex)
            self.get_logger().info(f"Set LiDAR output frequency to ~{self.output_freq} Hz (hardware).")
        except Exception as e:
            self.get_logger().warn(f"Failed to set hardware frequency. {e}")

        try:
            if self.use_filter:
                self.lidar.set_filter(FilterHex.On)
                self.get_logger().info("Enabled built-in filter.")
            else:
                self.lidar.set_filter(FilterHex.Off)
                self.get_logger().info("Disabled built-in filter.")
        except Exception as e:
            self.get_logger().warn(f"Failed to set filter mode. {e}")

        try:
            self.lidar.start_scan()
            self.get_logger().info("LiDAR scanning started.")
        except Exception as e:
            self.get_logger().error(f"Failed to start LiDAR scan: {e}")
            return

        self.laser_pub = self.create_publisher(LaserScan, '~/scan', qos_profile)
        self.distance_pub = self.create_publisher(Int16, '~/distance', qos_profile)
        self.intensity_pub = self.create_publisher(UInt8, '~/intensity', qos_profile)
        self.disturb_pub = self.create_publisher(UInt8, '~/disturb', qos_profile)

        self._latest_distance = 0
        self._latest_intensity = 0
        self._latest_disturb = 0
        self._data_lock = threading.Lock()
        self._stop_thread = False
        self.reader_thread = threading.Thread(target=self._continuous_read)
        self.reader_thread.start()

        self.timer = self.create_timer(1.0 / float(self.update_rate), self.publish_data)

    def _continuous_read(self):
        """Drain the serial buffer so publish_data always has the newest sample."""
        while not self._stop_thread:
            try:
                while self.lidar.ser.in_waiting > 0:
                    distance, intensity, disturb = self.lidar.get_distance()
                    with self._data_lock:
                        self._latest_distance = distance
                        self._latest_intensity = intensity
                        self._latest_disturb = disturb
                time.sleep(0.001)
            except Exception as e:
                self.get_logger().error(f"Error reading LiDAR data: {e}")
                time.sleep(0.1)

    def publish_data(self):
        with self._data_lock:
            distance = self._latest_distance
            intensity = self._latest_intensity
            disturb = self._latest_disturb

        if self.verbose:
            self.get_logger().info(f"Distance: {distance} mm, Intensity: {intensity}, Disturb: {disturb}")

        dist_msg = Int16()
        dist_msg.data = distance
        self.distance_pub.publish(dist_msg)

        intens_msg = UInt8()
        intens_msg.data = intensity
        self.intensity_pub.publish(intens_msg)

        distb_msg = UInt8()
        distb_msg.data = disturb
        self.disturb_pub.publish(distb_msg)

        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = "laser_frame"
        scan_msg.angle_min = 0.0
        scan_msg.angle_max = 0.0
        scan_msg.angle_increment = 0.0
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 1.0 / float(self.update_rate)
        scan_msg.range_min = 0.05
        scan_msg.range_max = 1.5
        scan_msg.ranges = [float(distance) * 1e-3]
        scan_msg.intensities = [float(intensity)]
        self.laser_pub.publish(scan_msg)

    def destroy_node(self):
        self._stop_thread = True
        if self.reader_thread.is_alive():
            self.reader_thread.join(1.0)
        try:
            self.lidar.stop_scan()
        except Exception as e:
            self.get_logger().warn(f"Error stopping scan: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SDM15Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down SDM15 node (Ctrl-C).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
