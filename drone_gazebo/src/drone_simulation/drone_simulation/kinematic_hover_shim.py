#!/usr/bin/env python3
from typing import Optional
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class KinematicHoverShim(Node):
    def __init__(self):
        super().__init__('kinematic_hover_shim')

        self.declare_parameter('input_cmd_vel_topic', '/crazyflie/cmd_vel_user')
        self.declare_parameter('output_cmd_vel_topic', '/crazyflie/cmd_vel')
        self.declare_parameter('odom_topic', '/crazyflie/odometry')
        self.declare_parameter('z_hold_kp', 0.8)
        self.declare_parameter('z_hold_kd', 0.5)
        self.declare_parameter('z_deadband', 0.01)
        self.declare_parameter('max_z_speed', 2.0)
        self.declare_parameter('allow_z_cmd', True)
        self.declare_parameter('enable_pitch_visual', True)
        self.declare_parameter('pitch_per_ms_deg', 5.4)
        self.declare_parameter('pitch_max_deg', 5.4)
        self.declare_parameter('pitch_kp', 6.0)
        self.declare_parameter('pitch_kd', 0.8)
        self.declare_parameter('pitch_rate_limit_deg', 90.0)
        self.declare_parameter('roll_hold_kp', 6.0)
        self.declare_parameter('roll_hold_kd', 0.8)
        self.declare_parameter('roll_rate_limit_deg', 90.0)
        self.declare_parameter('update_rate_hz', 30.0)
        self.declare_parameter('cmd_timeout', 0.5)

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._on_odom,
            qos,
        )
        self.cmd_sub = self.create_subscription(
            Twist,
            self.get_parameter('input_cmd_vel_topic').value,
            self._on_cmd,
            qos,
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            self.get_parameter('output_cmd_vel_topic').value,
            qos,
        )

        self._last_cmd: Optional[Twist] = None
        self._z_ref: Optional[float] = None
        self._z_hold_active: bool = False
        self._z_vel_meas: float = 0.0
        self._z_meas: float = 0.0
        self._pitch_meas: float = 0.0
        self._pitch_rate_meas: float = 0.0
        self._roll_meas: float = 0.0
        self._roll_rate_meas: float = 0.0

        self._last_cmd_msg: Optional[Twist] = None
        self._last_out: Optional[Twist] = None
        self._last_cmd_stamp = self.get_clock().now()

        rate_hz = float(self.get_parameter('update_rate_hz').value)
        self._pub_timer = self.create_timer(max(1e-3, 1.0 / max(1.0, rate_hz)), self._on_timer)

        self.get_logger().info('Kinematic hover shim started.')

    def _on_odom(self, msg: Odometry) -> None:
        self._z_meas = float(msg.pose.pose.position.z)
        self._z_vel_meas = float(msg.twist.twist.linear.z)
        q = msg.pose.pose.orientation
        roll, pitch, _ = self._quat_to_rpy(q.x, q.y, q.z, q.w)
        self._pitch_meas = float(pitch)
        self._pitch_rate_meas = float(msg.twist.twist.angular.y)
        self._roll_meas = float(roll)
        self._roll_rate_meas = float(msg.twist.twist.angular.x)

    def _on_cmd(self, msg: Twist) -> None:
        self._last_cmd_msg = msg
        self._last_cmd_stamp = self.get_clock().now()
        out = self._compute_output_from_cmd(msg)
        self._last_out = out
        self.cmd_pub.publish(out)

    def _compute_output_from_cmd(self, msg: Twist) -> Twist:
        kp = float(self.get_parameter('z_hold_kp').value)
        kd = float(self.get_parameter('z_hold_kd').value)
        deadband = float(self.get_parameter('z_deadband').value)
        max_z_speed = float(self.get_parameter('max_z_speed').value)
        allow_z_cmd = bool(self.get_parameter('allow_z_cmd').value)

        out = Twist()
        out.linear.x = msg.linear.x
        out.linear.y = msg.linear.y
        out.angular = msg.angular

        z_input = msg.linear.z if allow_z_cmd else 0.0

        if (not allow_z_cmd) or abs(z_input) <= deadband:
            if not self._z_hold_active or self._z_ref is None:
                self._z_ref = self._z_meas
                self._z_hold_active = True
            z_err = (self._z_ref - self._z_meas)
            z_cmd = kp * z_err - kd * self._z_vel_meas
            if z_cmd > max_z_speed:
                z_cmd = max_z_speed
            elif z_cmd < -max_z_speed:
                z_cmd = -max_z_speed
            out.linear.z = float(z_cmd)
        else:
            self._z_hold_active = False
            self._z_ref = None
            out.linear.z = float(z_input)

        if bool(self.get_parameter('enable_pitch_visual').value):
            slope_deg = float(self.get_parameter('pitch_per_ms_deg').value)
            pitch_max_deg = float(self.get_parameter('pitch_max_deg').value)
            kp_p = float(self.get_parameter('pitch_kp').value)
            kd_p = float(self.get_parameter('pitch_kd').value)
            max_rate_deg = float(self.get_parameter('pitch_rate_limit_deg').value)

            slope = math.radians(slope_deg)
            pitch_max = math.radians(pitch_max_deg)
            max_rate = math.radians(max_rate_deg)

            pitch_des = max(-pitch_max, min(pitch_max, msg.linear.x * slope))
            e_pitch = pitch_des - self._pitch_meas
            wy_cmd = kp_p * e_pitch - kd_p * self._pitch_rate_meas
            if wy_cmd > max_rate:
                wy_cmd = max_rate
            elif wy_cmd < -max_rate:
                wy_cmd = -max_rate
            out.angular.y = float(wy_cmd)

        rk = float(self.get_parameter('roll_hold_kp').value)
        rd = float(self.get_parameter('roll_hold_kd').value)
        rmax = math.radians(float(self.get_parameter('roll_rate_limit_deg').value))
        e_roll = -self._roll_meas
        wx_cmd = rk * e_roll - rd * self._roll_rate_meas
        if wx_cmd > rmax:
            wx_cmd = rmax
        elif wx_cmd < -rmax:
            wx_cmd = -rmax
        out.angular.x = float(wx_cmd)

        return out

    def _on_timer(self) -> None:
        timeout = float(self.get_parameter('cmd_timeout').value)
        now = self.get_clock().now()

        cmd = self._last_cmd_msg if self._last_cmd_msg is not None else Twist()

        if self._last_cmd_msg is not None:
            dt = (now - self._last_cmd_stamp).nanoseconds * 1e-9
            if dt > timeout:
                cmd = Twist()

        out = self._compute_output_from_cmd(cmd)
        self._last_out = out
        self.cmd_pub.publish(out)

    @staticmethod
    def _quat_to_rpy(x: float, y: float, z: float, w: float):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return roll, pitch, yaw


def main(args=None):
    rclpy.init(args=args)
    node = KinematicHoverShim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
