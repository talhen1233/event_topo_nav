#!/usr/bin/env python3

"""Capture calibration images from a compressed ROS image topic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class CalibrationImageCapture(Node):
    """Capture calibration frames from a compressed image stream."""

    def __init__(
        self,
        image_topic: str,
        output_dir: Path,
        filename_prefix: str,
        headless: bool,
        auto_save_every_nth_frame: int,
        max_images: int,
    ) -> None:
        """Create the image capture node."""
        super().__init__("calibration_image_capture")
        self._output_dir = output_dir
        self._filename_prefix = filename_prefix
        self._image_index = 0
        self._received_frame_count = 0
        self._headless = headless
        self._display_available = not headless
        self._auto_save_every_nth_frame = max(0, auto_save_every_nth_frame)
        self._max_images = max_images
        self._window_created = False
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.create_subscription(CompressedImage, image_topic, self._image_callback, 10)
        if self._headless:
            self.get_logger().info("Calibration image capture ready in headless mode.")
        else:
            self.get_logger().info(
                "Calibration image capture ready. "
                "Press 's' to save a frame and 'q' to quit."
            )
        self.get_logger().info(f"Listening on {image_topic}")
        self.get_logger().info(f"Saving images to {self._output_dir}")
        if self._auto_save_every_nth_frame > 0:
            self.get_logger().info(
                "Headless autosave will save one frame every "
                f"{self._auto_save_every_nth_frame} frames."
            )
        if self._max_images > 0:
            self.get_logger().info(f"Capture will stop after {self._max_images} images.")

    def _image_callback(self, msg: CompressedImage) -> None:
        """Decode, display, and optionally save the incoming frame."""
        frame_buffer = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("Failed to decode compressed image frame.")
            return

        self._received_frame_count += 1
        if self._display_available:
            self._handle_display(frame)
            return

        self._handle_headless_autosave(frame)

    def _save_frame(self, frame: np.ndarray) -> None:
        """Persist the current frame to disk."""
        image_path = self._output_dir / f"{self._filename_prefix}_{self._image_index:04d}.jpg"
        if not cv2.imwrite(str(image_path), frame):
            self.get_logger().error(f"Failed to save image to {image_path}")
            return

        self.get_logger().info(f"Saved {image_path}")
        self._image_index += 1
        if self._max_images > 0 and self._image_index >= self._max_images:
            self.get_logger().info("Reached requested number of calibration images; stopping.")
            rclpy.shutdown()

    def _handle_display(self, frame: np.ndarray) -> None:
        """Render the current frame and react to key presses."""
        try:
            cv2.imshow("Calibration Capture", frame)
            self._window_created = True
            key = cv2.waitKey(1) & 0xFF
        except cv2.error as error:
            self._display_available = False
            self.get_logger().warning(
                "OpenCV display is unavailable; switching to headless autosave mode. "
                f"Original error: {error}"
            )
            self._handle_headless_autosave(frame)
            return

        if key == ord("s"):
            self._save_frame(frame)
        elif key == ord("q"):
            self.get_logger().info("Stopping calibration image capture.")
            rclpy.shutdown()

    def _handle_headless_autosave(self, frame: np.ndarray) -> None:
        """Save images automatically when running without an OpenCV display."""
        if self._auto_save_every_nth_frame <= 0:
            if self._received_frame_count == 1:
                self.get_logger().error(
                    "Headless mode requires --auto-save-every-nth-frame > 0."
                )
                rclpy.shutdown()
            return

        if self._received_frame_count % self._auto_save_every_nth_frame == 0:
            self._save_frame(frame)

    def close(self) -> None:
        """Release OpenCV windows when they were actually created."""
        if not self._window_created:
            return
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, Sequence[str]]:
    """Parse script arguments while preserving ROS arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-topic",
        default="/image_raw/compressed",
        help="Compressed image topic used for calibration capture.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/home/talhen/docker_ws/tiny_drone/"
            "tiny_drone/config/usb_cam_driver/radtan/camera_bottom_vga_images"
        ),
        help="Directory where captured images will be stored.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="camera_bottom_vga",
        help="Filename prefix for captured images.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable OpenCV window rendering and save images automatically.",
    )
    parser.add_argument(
        "--auto-save-every-nth-frame",
        type=int,
        default=30,
        help="In headless mode, save one image every N frames.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Stop automatically after saving this many images. Non-positive disables the limit.",
    )
    return parser.parse_known_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the calibration image capture tool."""
    cli_args, ros_args = _parse_args(argv)
    rclpy.init(args=list(ros_args))
    node = CalibrationImageCapture(
        image_topic=cli_args.image_topic,
        output_dir=Path(cli_args.output_dir).expanduser(),
        filename_prefix=cli_args.filename_prefix,
        headless=cli_args.headless,
        auto_save_every_nth_frame=cli_args.auto_save_every_nth_frame,
        max_images=cli_args.max_images,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
