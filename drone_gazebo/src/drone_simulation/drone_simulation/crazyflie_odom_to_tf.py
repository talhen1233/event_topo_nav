#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
import math
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2


class CrazyflieOdomToTfNode(Node):
    def __init__(self) -> None:
        super().__init__('crazyflie_odom_to_tf')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        self.declare_parameter('publish_odom_tf', True)
        self.declare_parameter('odom_topic', '/crazyflie/odometry')
        self.declare_parameter('base_link_frame', 'crazyflie/base_link')

        self.cv_bridge: CvBridge = CvBridge()

        self.bottom_compressed_pub = self.create_publisher(
            CompressedImage, '/crazyflie/camera/bottom/compressed', 10
        )
        self.bottom_image_sub = self.create_subscription(
            Image, '/crazyflie/camera/bottom', self._on_bottom_image, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').get_parameter_value().string_value,
            self._on_odom,
            10,
        )

        self._publish_static_transforms()

        self.get_logger().info('crazyflie_odom_to_tf started')

    def _euler_to_quaternion(self, roll: float, pitch: float, yaw: float):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        return (
            cy * cp * sr - sy * sp * cr,
            sy * cp * sr + cy * sp * cr,
            sy * cp * cr - cy * sp * sr,
            cy * cp * cr + sy * sp * sr,
        )

    def _publish_static_transforms(self) -> None:
        base_link = self.get_parameter('base_link_frame').get_parameter_value().string_value
        current_time = rclpy.time.Time().to_msg()
        transforms = []

        def add_tf(
            parent: str,
            child: str,
            x: float,
            y: float,
            z: float,
            rr: float,
            pp: float,
            yy: float,
        ) -> None:
            t = TransformStamped()
            t.header.stamp = current_time
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = float(z)
            qx, qy, qz, qw = self._euler_to_quaternion(rr, pp, yy)
            t.transform.rotation.x = qx
            t.transform.rotation.y = qy
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            transforms.append(t)

        add_tf(base_link, 'front_depth_link', 0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
        add_tf(base_link, 'back_depth_link', -0.1, 0.0, 0.0, 0.0, 0.0, 3.14159)
        add_tf(base_link, 'left_depth_link', 0.0, 0.1, 0.0, 0.0, 0.0, 1.57)
        add_tf(base_link, 'right_depth_link', 0.0, -0.1, 0.0, 0.0, 0.0, -1.57)
        add_tf(base_link, 'top_depth_link', 0.0, 0.0, 0.05, 0.0, -1.57, 0.0)
        add_tf(base_link, 'bottom_depth_link', 0.0, 0.0, -0.05, 0.0, 1.57, 0.0)
        add_tf(base_link, 'imu_link', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        add_tf(base_link, 'bottom_camera_link', 0.0, 0.0, -0.05, 0.0, 1.57, 0.0)
        add_tf(base_link, 'third_person_camera_link', -0.8, 0.0, 0.35, 0.0, 0.3, 0.0)
        add_tf(base_link, 'height_lidar_link', 0.0, 0.0, -0.05, 0.0, 1.57, 0.0)

        optical_roll = -math.pi / 2.0
        optical_pitch = 0.0
        optical_yaw = -math.pi / 2.0
        add_tf('bottom_camera_link', 'bottom_camera_optical_frame', 0.0, 0.0, 0.0, optical_roll, optical_pitch, optical_yaw)
        add_tf(
            'third_person_camera_link',
            'third_person_camera_optical_frame',
            0.0,
            0.0,
            0.0,
            optical_roll,
            optical_pitch,
            optical_yaw,
        )

        add_tf(base_link, 'crazyflie/rotor_0', 0.0326, -0.0326, 0.005, 0.0, 0.0, 0.0)
        add_tf(base_link, 'crazyflie/rotor_1', -0.0326, 0.0326, 0.005, 0.0, 0.0, 0.0)
        add_tf(base_link, 'crazyflie/rotor_2', 0.0326, 0.0326, 0.005, 0.0, 0.0, 0.0)
        add_tf(base_link, 'crazyflie/rotor_3', -0.0326, -0.0326, 0.005, 0.0, 0.0, 0.0)

        self.static_tf_broadcaster.sendTransform(transforms)
        self.get_logger().info(f'Published {len(transforms)} static transforms for crazyflie sensors')

    def _on_odom(self, msg: Odometry) -> None:
        if not self.get_parameter('publish_odom_tf').get_parameter_value().bool_value:
            return
        base_link = self.get_parameter('base_link_frame').get_parameter_value().string_value
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = 'odom'
        tf_msg.child_frame_id = base_link
        tf_msg.transform.translation.x = msg.pose.pose.position.x
        tf_msg.transform.translation.y = msg.pose.pose.position.y
        tf_msg.transform.translation.z = msg.pose.pose.position.z
        tf_msg.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(tf_msg)

    def _on_bottom_image(self, image_msg: Image) -> None:
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='mono8')
            ok, encoded = cv2.imencode('.jpg', cv_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                self.get_logger().warn('cv2.imencode returned success=False')
                return
        except Exception as exc:
            self.get_logger().warn(f'Bottom image convert/encode failed: {exc}')
            return

        out = CompressedImage()
        out.header = image_msg.header
        out.format = 'jpeg'
        out.data = encoded.tobytes()
        self.bottom_compressed_pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CrazyflieOdomToTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
