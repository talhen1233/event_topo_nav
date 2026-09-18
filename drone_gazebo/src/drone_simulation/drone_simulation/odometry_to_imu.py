#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Tuple

import math
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> Tuple[Tuple[float, float, float],
                                                                                   Tuple[float, float, float],
                                                                                   Tuple[float, float, float]]:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x /= n
    y /= n
    z /= n
    w /= n

    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)

    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)

    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    return (
        (r00, r10, r20),
        (r01, r11, r21),
        (r02, r12, r22),
    )


def mat3_mul_vec3(m: Tuple[Tuple[float, float, float],
                           Tuple[float, float, float],
                           Tuple[float, float, float]],
                  v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


class OdometryToImu(Node):
    def __init__(self) -> None:
        super().__init__('odometry_to_imu')

        self.declare_parameter('odom_topic', '/crazyflie/odometry')
        self.declare_parameter('imu_out_topic', '/crazyflie/imu/acc_derived')
        self.declare_parameter('imu_in_topic', '/crazyflie/imu/data')
        self.declare_parameter('frame_id', 'crazyflie/base_link')
        self.declare_parameter('copy_orientation_from_imu', True)
        self.declare_parameter('copy_angular_velocity_from_imu', True)
        self.declare_parameter('copy_orientation_from_odom', False)
        self.declare_parameter('use_odom_angular_velocity', False)
        self.declare_parameter('gravity_m_s2', 9.81)
        self.declare_parameter('enable_acc_noise', False)
        self.declare_parameter('acc_noise_stddev', 1.7e-2)
        self.declare_parameter('acc_bias_mean', 0.0)
        self.declare_parameter('acc_bias_stddev', 1.0e-9)
        self.declare_parameter('noise_seed', 0)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self._on_odom,
            qos,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            self.get_parameter('imu_in_topic').get_parameter_value().string_value,
            self._on_imu,
            qos,
        )
        self.imu_pub = self.create_publisher(
            Imu,
            self.get_parameter('imu_out_topic').get_parameter_value().string_value,
            qos,
        )

        self._last_vel_w: Optional[Tuple[float, float, float]] = None
        self._last_stamp_ns: Optional[int] = None
        self._last_imu: Optional[Imu] = None

        seed = int(self.get_parameter('noise_seed').get_parameter_value().integer_value)
        self._rng = random.Random(seed)
        bias_mean = float(self.get_parameter('acc_bias_mean').get_parameter_value().double_value)
        bias_std = float(self.get_parameter('acc_bias_stddev').get_parameter_value().double_value)
        self._acc_bias = (
            self._rng.gauss(bias_mean, bias_std),
            self._rng.gauss(bias_mean, bias_std),
            self._rng.gauss(bias_mean, bias_std),
        )

        self.get_logger().info('odometry_to_imu started')

    def _on_odom(self, msg: Odometry) -> None:
        v_w = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.linear.y),
            float(msg.twist.twist.linear.z),
        )
        t_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

        if self._last_vel_w is None or self._last_stamp_ns is None:
            self._last_vel_w = v_w
            self._last_stamp_ns = t_ns
            return

        dt = max(1e-6, (t_ns - self._last_stamp_ns) * 1e-9)
        ax_w = (v_w[0] - self._last_vel_w[0]) / dt
        ay_w = (v_w[1] - self._last_vel_w[1]) / dt
        az_w = (v_w[2] - self._last_vel_w[2]) / dt

        self._last_vel_w = v_w
        self._last_stamp_ns = t_ns

        g = float(self.get_parameter('gravity_m_s2').get_parameter_value().double_value)
        g_w = (0.0, 0.0, -g)
        f_w = (ax_w - g_w[0], ay_w - g_w[1], az_w - g_w[2])

        q = msg.pose.pose.orientation
        R_bw = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        f_b = mat3_mul_vec3(R_bw, f_w)

        if bool(self.get_parameter('enable_acc_noise').get_parameter_value().bool_value):
            std = float(self.get_parameter('acc_noise_stddev').get_parameter_value().double_value)
            f_b = (
                f_b[0] + self._acc_bias[0] + self._rng.gauss(0.0, std),
                f_b[1] + self._acc_bias[1] + self._rng.gauss(0.0, std),
                f_b[2] + self._acc_bias[2] + self._rng.gauss(0.0, std),
            )

        out = Imu()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        if bool(self.get_parameter('copy_orientation_from_imu').get_parameter_value().bool_value) and self._last_imu is not None:
            out.orientation = self._last_imu.orientation
            out.orientation_covariance[0] = -1.0
        elif bool(self.get_parameter('copy_orientation_from_odom').get_parameter_value().bool_value):
            out.orientation = msg.pose.pose.orientation
            out.orientation_covariance[0] = -1.0

        if bool(self.get_parameter('copy_angular_velocity_from_imu').get_parameter_value().bool_value) and self._last_imu is not None:
            out.angular_velocity = self._last_imu.angular_velocity
            out.angular_velocity_covariance[0] = -1.0
        elif bool(self.get_parameter('use_odom_angular_velocity').get_parameter_value().bool_value):
            out.angular_velocity = msg.twist.twist.angular
            out.angular_velocity_covariance[0] = -1.0

        out.linear_acceleration.x = f_b[0]
        out.linear_acceleration.y = f_b[1]
        out.linear_acceleration.z = f_b[2]
        out.linear_acceleration_covariance[0] = -1.0

        self.imu_pub.publish(out)

    def _on_imu(self, msg: Imu) -> None:
        self._last_imu = msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdometryToImu()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
