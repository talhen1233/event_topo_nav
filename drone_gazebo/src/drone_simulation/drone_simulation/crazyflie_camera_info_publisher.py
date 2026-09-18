#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Dict, Optional
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo


@dataclass
class CameraConfig:
    name: str
    frame_id: str
    optical_frame_id: str
    image_topic: str
    camera_info_topic: str
    optical_image_topic: str
    optical_camera_info_topic: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float = 0.0
    legacy_camera_info_topic: Optional[str] = None


def _fx_from_hfov(width: int, hfov_rad: float) -> float:
    return width / (2.0 * math.tan(hfov_rad / 2.0))


class CrazyflieCameraInfoPublisher(Node):
    def __init__(self) -> None:
        super().__init__('crazyflie_camera_info_publisher')

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        third_width = 640
        third_height = 480
        third_hfov = 1.57
        third_fx = _fx_from_hfov(third_width, third_hfov)
        third_fy = third_fx
        third_cx = third_width / 2.0
        third_cy = third_height / 2.0

        bottom_width = 640
        bottom_height = 400
        bottom_hfov = math.radians(103.0)
        bottom_fx = _fx_from_hfov(bottom_width, bottom_hfov)
        bottom_fy = bottom_fx
        bottom_cx = bottom_width / 2.0
        bottom_cy = bottom_height / 2.0

        self._configs: Dict[str, CameraConfig] = {
            'bottom': CameraConfig(
                name='bottom',
                frame_id='bottom_camera_link',
                optical_frame_id='bottom_camera_optical_frame',
                image_topic='/crazyflie/camera/bottom',
                camera_info_topic='/crazyflie/camera/bottom/camera_info',
                optical_image_topic='/crazyflie/camera/bottom_optical',
                optical_camera_info_topic='/crazyflie/camera/bottom_optical/camera_info',
                width=bottom_width,
                height=bottom_height,
                fx=bottom_fx,
                fy=bottom_fy,
                cx=bottom_cx,
                cy=bottom_cy,
            ),
            'third_person': CameraConfig(
                name='third_person',
                frame_id='third_person_camera_link',
                optical_frame_id='third_person_camera_optical_frame',
                image_topic='/crazyflie/camera/third_person',
                camera_info_topic='/crazyflie/camera/third_person/camera_info',
                legacy_camera_info_topic='/crazyflie/camera/camera_info',
                optical_image_topic='/crazyflie/camera/third_person_optical',
                optical_camera_info_topic='/crazyflie/camera/third_person_optical/camera_info',
                width=third_width,
                height=third_height,
                fx=third_fx,
                fy=third_fy,
                cx=third_cx,
                cy=third_cy,
            ),
        }

        self._camera_info_publishers: Dict[str, list[rclpy.publisher.Publisher]] = {}
        self._optical_image_publishers: Dict[str, rclpy.publisher.Publisher] = {}
        self._optical_camera_info_publishers: Dict[str, rclpy.publisher.Publisher] = {}

        for cam_name, cfg in self._configs.items():
            pubs: list[rclpy.publisher.Publisher] = [
                self.create_publisher(CameraInfo, cfg.camera_info_topic, qos),
            ]
            if cfg.legacy_camera_info_topic:
                pubs.append(self.create_publisher(CameraInfo, cfg.legacy_camera_info_topic, qos))
            self._camera_info_publishers[cam_name] = pubs

            self._optical_image_publishers[cam_name] = self.create_publisher(Image, cfg.optical_image_topic, qos)
            self._optical_camera_info_publishers[cam_name] = self.create_publisher(
                CameraInfo,
                cfg.optical_camera_info_topic,
                qos,
            )

            self.create_subscription(
                Image,
                cfg.image_topic,
                lambda msg, cam=cam_name: self._on_image(msg, cam),
                qos,
            )

        self.get_logger().info('Crazyflie camera_info publisher started.')

    def _on_image(self, msg: Image, cam_name: str) -> None:
        cfg = self._configs[cam_name]
        info = CameraInfo()

        info.header = msg.header
        if not info.header.frame_id:
            info.header.frame_id = cfg.frame_id

        info.width = cfg.width
        info.height = cfg.height
        info.distortion_model = 'plumb_bob'
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        fx = cfg.fx
        fy = cfg.fy
        cx = cfg.cx
        cy = cfg.cy
        s = cfg.skew

        info.k = [
            fx, s, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]
        info.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
        info.p = [
            fx, s, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        info.binning_x = 1
        info.binning_y = 1
        info.roi.x_offset = 0
        info.roi.y_offset = 0
        info.roi.height = 0
        info.roi.width = 0
        info.roi.do_rectify = False

        for pub in self._camera_info_publishers[cam_name]:
            pub.publish(info)

        optical_info = CameraInfo()
        optical_info.header = msg.header
        optical_info.header.frame_id = cfg.optical_frame_id
        optical_info.width = info.width
        optical_info.height = info.height
        optical_info.distortion_model = info.distortion_model
        optical_info.d = list(info.d)
        optical_info.k = list(info.k)
        optical_info.r = list(info.r)
        optical_info.p = list(info.p)
        optical_info.binning_x = info.binning_x
        optical_info.binning_y = info.binning_y
        optical_info.roi = info.roi
        self._optical_camera_info_publishers[cam_name].publish(optical_info)

        optical_image = Image()
        optical_image.header = msg.header
        optical_image.header.frame_id = cfg.optical_frame_id
        optical_image.height = msg.height
        optical_image.width = msg.width
        optical_image.encoding = msg.encoding
        optical_image.is_bigendian = msg.is_bigendian
        optical_image.step = msg.step
        optical_image.data = msg.data
        self._optical_image_publishers[cam_name].publish(optical_image)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CrazyflieCameraInfoPublisher()
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
