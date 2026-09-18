from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import signal
import subprocess
import sys

try:
    from of_localization.offline_debug.common import load_ros_params
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from of_localization.offline_debug.common import load_ros_params


def build_topic_list(args: argparse.Namespace, params: dict[str, object]) -> list[str]:
    """Build the list of topics to record."""
    topics = [
        str(args.image_topic or params.get("image_topic", "/crazyflie/camera/bottom/compressed")),
        str(args.camera_info_topic or params.get("camera_info_topic", "/crazyflie/camera/bottom/camera_info")),
        str(args.imu_topic or params.get("imu_topic", "/crazyflie/imu/acc_derived")),
        str(args.lidar_topic or params.get("lidar_topic", "/crazyflie/lidar/height")),
        str(args.depth_topic or params.get("bottom_depth_points_topic", "/crazyflie/depth_camera/bottom/points")),
        str(args.third_person_image_topic or params.get("third_person_image_topic", "/crazyflie/camera/third_person/compressed")),
    ]
    for extra_topic in args.extra_topic:
        topics.append(str(extra_topic))
    return [topic for topic in dict.fromkeys(topics) if topic]


def build_record_command(args: argparse.Namespace, topics: list[str]) -> list[str]:
    """Build the `ros2 bag record` command."""
    command = ["ros2", "bag", "record", "-o", str(args.output)]
    if args.storage_id:
        command.extend(["-s", str(args.storage_id)])
    if args.compression_mode:
        command.extend(["--compression-mode", str(args.compression_mode)])
    if args.compression_format:
        command.extend(["--compression-format", str(args.compression_format)])
    command.extend(topics)
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Record the OF debug topics required for offline image export.",
    )
    parser.add_argument("--params-file", default=None, help="Path to the OF params YAML used to resolve default topics.")
    parser.add_argument(
        "--output",
        default=f"of_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output bag directory.",
    )
    parser.add_argument("--image-topic", default=None, help="Override the bottom compressed image topic.")
    parser.add_argument("--camera-info-topic", default=None, help="Override the CameraInfo topic.")
    parser.add_argument("--imu-topic", default=None, help="Override the IMU topic.")
    parser.add_argument("--lidar-topic", default=None, help="Override the scalar height topic.")
    parser.add_argument("--depth-topic", default=None, help="Override the bottom depth PointCloud2 topic.")
    parser.add_argument("--third-person-image-topic", default=None, help="Override the third-person compressed image topic.")
    parser.add_argument("--extra-topic", action="append", default=[], help="Additional topic to include in the bag.")
    parser.add_argument("--storage-id", default="sqlite3", help="rosbag2 storage plugin.")
    parser.add_argument("--compression-mode", default=None, help="Optional rosbag2 compression mode.")
    parser.add_argument("--compression-format", default=None, help="Optional rosbag2 compression format.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the bag recorder wrapper."""
    args = parse_args(argv)
    params = load_ros_params(args.params_file)
    topics = build_topic_list(args, params)
    output_dir = Path(args.output).expanduser()
    args.output = output_dir
    command = build_record_command(args, topics)

    print("Recording topics:")
    for topic in topics:
        print(f"  - {topic}")
    print("Command:")
    print("  " + " ".join(shlex.quote(token) for token in command))

    process = subprocess.Popen(command)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        return process.wait()


if __name__ == "__main__":
    sys.exit(main())
