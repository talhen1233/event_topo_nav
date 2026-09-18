#!/usr/bin/env python3
import math
import numpy as np
from typing import Optional, Dict

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from actuator_msgs.msg import Actuators


R2D = 180.0 / math.pi
D2R = math.pi / 180.0

def wrap_pi(a: float) -> float:
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a

class CrazyflieFirmwareController(Node):
    def __init__(self):
        super().__init__('crazyflie_firmware_controller')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/crazyflie/odometry')
        self.declare_parameter('imu_topic', '/crazyflie/imu/data')
        self.declare_parameter('actuators_topic', '/crazyflie/command/motor_speed')
        self.declare_parameter('frame_id', 'crazyflie/base_link')

        self.declare_parameter('mass', 0.027)
        self.declare_parameter('g', 9.81)
        self.declare_parameter('Ix', 1.43e-5)
        self.declare_parameter('Iy', 1.43e-5)
        self.declare_parameter('Iz', 2.89e-5)
        self.declare_parameter('arm_length', 0.046)
        self.declare_parameter('layout', 'x')
        self.declare_parameter('kf', 2.3e-8)
        self.declare_parameter('km', 7.8e-10)
        self.declare_parameter('motor_min', 0.0)
        self.declare_parameter('motor_max', 4000.0)

        self.declare_parameter('kp_pos_xy', 2.0)
        self.declare_parameter('ki_pos_xy', 0.05)
        self.declare_parameter('kd_pos_xy', 0.6)
        self.declare_parameter('kp_pos_z', 8.0)
        self.declare_parameter('ki_pos_z', 0.2)
        self.declare_parameter('kd_pos_z', 2.5)

        self.declare_parameter('kp_vel_xy', 2.0)
        self.declare_parameter('ki_vel_xy', 0.0)
        self.declare_parameter('kd_vel_xy', 0.1)
        self.declare_parameter('kp_vel_z', 3.0)
        self.declare_parameter('ki_vel_z', 0.0)
        self.declare_parameter('kd_vel_z', 0.2)

        self.declare_parameter('kp_att_rp', 8.0)
        self.declare_parameter('kd_att_rp', 0.2)
        self.declare_parameter('kp_att_yaw', 4.0)
        self.declare_parameter('kd_att_yaw', 0.1)

        self.declare_parameter('max_tilt_deg', 25.0)
        self.declare_parameter('max_vel_xy', 2.0)
        self.declare_parameter('max_vel_z', 1.5)
        self.declare_parameter('cmd_timeout', 0.25)
        self.declare_parameter('enable_hover_hold', True)

        self.declare_parameter('i_limit_pos', 0.5)
        self.declare_parameter('i_limit_vel', 0.5)

        self.declare_parameter('gyro_lpf_hz', 40.0)

        self.last_cmd: Optional[Twist] = None
        self.last_cmd_stamp = self.get_clock().now()

        self.has_odom = False
        self.has_imu = False

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.att_rpy = np.zeros(3)
        self.gyro = np.zeros(3)

        self.pos_ref = np.zeros(3)
        self.yaw_ref = 0.0

        self.phi_des = 0.0
        self.theta_des = 0.0
        self.az_cmd = 0.0

        self.i_pos = np.zeros(3)
        self.i_vel = np.zeros(3)

        self.gyro_filt = np.zeros(3)
        self.inner_prev_stamp = self.get_clock().now()

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE)

        self.cmd_sub = self.create_subscription(
            Twist,
            self.get_parameter('cmd_vel_topic').get_parameter_value().string_value,
            self.on_cmd_vel,
            qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self.on_odom,
            qos)

        self.imu_sub = self.create_subscription(
            Imu,
            self.get_parameter('imu_topic').get_parameter_value().string_value,
            self.on_imu,
            qos)

        self.act_pub = self.create_publisher(
            Actuators,
            self.get_parameter('actuators_topic').get_parameter_value().string_value,
            qos)

        self.outer_timer = self.create_timer(0.01, self.outer_loop_cb)
        self.inner_timer = self.create_timer(0.0025, self.inner_loop_cb)

        self.get_logger().info("Crazyflie controller started.")

    def on_cmd_vel(self, msg: Twist):
        self.last_cmd = msg
        self.last_cmd_stamp = self.get_clock().now()

    def on_odom(self, msg: Odometry):
        self.pos[0] = msg.pose.pose.position.x
        self.pos[1] = msg.pose.pose.position.y
        self.pos[2] = msg.pose.pose.position.z

        self.vel[0] = msg.twist.twist.linear.x
        self.vel[1] = msg.twist.twist.linear.y
        self.vel[2] = msg.twist.twist.linear.z

        q = msg.pose.pose.orientation
        roll, pitch, yaw = self.quat_to_rpy(q.x, q.y, q.z, q.w)
        self.att_rpy[:] = (roll, pitch, yaw)

        self.gyro[:] = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z
        ]
        self.has_odom = True

    def on_imu(self, msg: Imu):
        q = msg.orientation
        roll, pitch, yaw = self.quat_to_rpy(q.x, q.y, q.z, q.w)
        self.att_rpy[:] = (roll, pitch, yaw)

        self.gyro[:] = [msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z]
        self.has_imu = True

    def outer_loop_cb(self):
        if not (self.has_odom and (self.has_imu or True)):
            return

        now = self.get_clock().now()
        dt = 0.01

        enable_hold = self.get_parameter('enable_hover_hold').value
        timeout_s = self.get_parameter('cmd_timeout').value
        zero_cmd = False
        timed_out = (now - self.last_cmd_stamp) > Duration(seconds=timeout_s)

        if self.last_cmd is None:
            zero_cmd = True
        else:
            lc = self.last_cmd.linear
            ac = self.last_cmd.angular
            zero_cmd = (abs(lc.x) < 1e-3 and abs(lc.y) < 1e-3 and
                        abs(lc.z) < 1e-3 and abs(ac.z) < 1e-3)

        roll, pitch, yaw = self.att_rpy
        m = self.get_parameter('mass').value
        g = self.get_parameter('g').value

        if enable_hold and (zero_cmd or timed_out):
            self.i_pos *= 0.95
            self.i_vel *= 0.95
        else:
            vx_b = float(self.last_cmd.linear.x)
            vy_b = float(self.last_cmd.linear.y)
            vz   = float(self.last_cmd.linear.z)
            wz   = float(self.last_cmd.angular.z)

            c, s = math.cos(yaw), math.sin(yaw)
            vx_w =  c * vx_b - s * vy_b
            vy_w =  s * vx_b + c * vy_b

            vmax_xy = self.get_parameter('max_vel_xy').value
            vmax_z  = self.get_parameter('max_vel_z').value
            vx_w = np.clip(vx_w, -vmax_xy, vmax_xy)
            vy_w = np.clip(vy_w, -vmax_xy, vmax_xy)
            vz   = np.clip(vz,   -vmax_z,  vmax_z)

            self.pos_ref[0] += vx_w * dt
            self.pos_ref[1] += vy_w * dt
            self.pos_ref[2] += vz   * dt
            self.yaw_ref     = wrap_pi(self.yaw_ref + wz * dt)

        kp_px = self.get_parameter('kp_pos_xy').value
        ki_px = self.get_parameter('ki_pos_xy').value
        kd_px = self.get_parameter('kd_pos_xy').value
        kp_pz = self.get_parameter('kp_pos_z').value
        ki_pz = self.get_parameter('ki_pos_z').value
        kd_pz = self.get_parameter('kd_pos_z').value

        e_pos = self.pos_ref - self.pos
        self.i_pos += np.array([e_pos[0], e_pos[1], e_pos[2]]) * dt
        self.apply_i_clamp(self.i_pos, self.get_parameter('i_limit_pos').value)

        v_ref = np.zeros(3)
        v_ref[0] = kp_px * e_pos[0] + kd_px * (0.0 - self.vel[0]) + ki_px * self.i_pos[0]
        v_ref[1] = kp_px * e_pos[1] + kd_px * (0.0 - self.vel[1]) + ki_px * self.i_pos[1]
        v_ref[2] = kp_pz * e_pos[2] + kd_pz * (0.0 - self.vel[2]) + ki_pz * self.i_pos[2]

        kp_vx = self.get_parameter('kp_vel_xy').value
        ki_vx = self.get_parameter('ki_vel_xy').value
        kd_vx = self.get_parameter('kd_vel_xy').value
        kp_vz = self.get_parameter('kp_vel_z').value
        ki_vz = self.get_parameter('ki_vel_z').value
        kd_vz = self.get_parameter('kd_vel_z').value

        e_vel = v_ref - self.vel
        self.i_vel += np.array([e_vel[0], e_vel[1], e_vel[2]]) * dt
        self.apply_i_clamp(self.i_vel, self.get_parameter('i_limit_vel').value)

        ax_des = kp_vx * e_vel[0] + kd_vx * (0.0) + ki_vx * self.i_vel[0]
        ay_des = kp_vx * e_vel[1] + kd_vx * (0.0) + ki_vx * self.i_vel[1]
        az_des = kp_vz * e_vel[2] + kd_vz * (0.0) + ki_vz * self.i_vel[2]

        theta_des = ( ax_des * math.cos(yaw) + ay_des * math.sin(yaw) ) / g
        phi_des   = ( ax_des * math.sin(yaw) - ay_des * math.cos(yaw) ) / g

        max_tilt = self.get_parameter('max_tilt_deg').value * D2R
        self.theta_des = float(np.clip(theta_des, -max_tilt, max_tilt))
        self.phi_des   = float(np.clip(phi_des,   -max_tilt, max_tilt))

        self.az_cmd = float(az_des)

    def inner_loop_cb(self):
        if not self.has_odom:
            return

        now = self.get_clock().now()
        dt = max(1e-4, (now - self.inner_prev_stamp).nanoseconds * 1e-9)
        self.inner_prev_stamp = now

        fc = max(0.1, self.get_parameter('gyro_lpf_hz').value)
        alpha = math.exp(-2.0 * math.pi * fc * dt)
        self.gyro_filt = alpha * self.gyro_filt + (1.0 - alpha) * self.gyro
        p, q, r = self.gyro_filt.tolist()

        roll, pitch, yaw = self.att_rpy
        m  = self.get_parameter('mass').value
        g  = self.get_parameter('g').value
        Ix = self.get_parameter('Ix').value
        Iy = self.get_parameter('Iy').value
        Iz = self.get_parameter('Iz').value

        kp_rp = self.get_parameter('kp_att_rp').value
        kd_rp = self.get_parameter('kd_att_rp').value
        kp_y  = self.get_parameter('kp_att_yaw').value
        kd_y  = self.get_parameter('kd_att_yaw').value

        e_roll  = wrap_pi(self.phi_des   - roll)
        e_pitch = wrap_pi(self.theta_des - pitch)
        e_yaw   = wrap_pi(self.yaw_ref   - yaw)

        tau_x = Ix * (kp_rp * e_roll  - kd_rp * p)
        tau_y = Iy * (kp_rp * e_pitch - kd_rp * q)
        tau_z = Iz * (kp_y  * e_yaw   - kd_y  * r)

        denom = max(1e-3, math.cos(roll) * math.cos(pitch))
        thrust = m * (g + self.az_cmd) / denom

        w = self.allocate_motors(thrust, tau_x, tau_y, tau_z)
        self.publish_actuators(w)

    @staticmethod
    def quat_to_rpy(x, y, z, w):
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

    @staticmethod
    def apply_i_clamp(vec: np.ndarray, limit: float):
        for i in range(3):
            vec[i] = float(np.clip(vec[i], -limit, limit))

    def allocate_motors(self, thrust_N: float, tau_x: float, tau_y: float, tau_z: float):
        l  = float(self.get_parameter('arm_length').value)
        layout = self.get_parameter('layout').value.lower()
        kf = float(self.get_parameter('kf').value)
        km = float(self.get_parameter('km').value)
        w_min = float(self.get_parameter('motor_min').value)
        w_max = float(self.get_parameter('motor_max').value)

        if layout == 'x':
            le = l / math.sqrt(2.0)
        else:
            le = l

        A = np.array([
            [ kf,     kf,     kf,     kf    ],
            [-kf*le,  kf*le,  kf*le, -kf*le ],
            [ kf*le,  kf*le, -kf*le, -kf*le ],
            [ km,    -km,     km,    -km    ]
        ])

        y = np.array([thrust_N, tau_x, tau_y, tau_z])
        try:
            w2 = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            y = np.array([thrust_N, 0.5*tau_x, 0.5*tau_y, 0.5*tau_z])
            w2 = np.linalg.lstsq(A, y, rcond=None)[0]

        w2 = np.clip(w2, w_min**2, w_max**2)
        w = np.sqrt(w2)
        return w.tolist()

    def publish_actuators(self, w_rad_s):
        msg = Actuators()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value

        msg.velocity = w_rad_s

        w_min = float(self.get_parameter('motor_min').value)
        w_max = float(self.get_parameter('motor_max').value)
        span = max(1e-6, (w_max - w_min))
        norm = [ 2.0 * (wi - w_min) / span - 1.0 for wi in w_rad_s ]
        msg.normalized = norm

        self.act_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieFirmwareController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
