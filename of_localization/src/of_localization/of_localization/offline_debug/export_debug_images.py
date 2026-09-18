from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

try:
    from of_localization.offline_debug.common import load_ros_params
    from of_localization.offline_debug.processor import OfflineDebugProcessor
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from of_localization.offline_debug.common import load_ros_params
    from of_localization.offline_debug.processor import OfflineDebugProcessor


def decode_compressed_image(msg) -> object:
    """Decode a ROS `CompressedImage` message into BGR."""
    buffer = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def stamp_from_msg(msg) -> float | None:
    """Extract message header time in seconds."""
    header = getattr(msg, "header", None)
    if header is None or getattr(header, "stamp", None) is None:
        return None
    stamp = header.stamp
    stamp_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
    if stamp_s <= 0.0:
        return None
    return stamp_s


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export thesis-ready OF debug images from a saved rosbag2 recording.",
    )
    parser.add_argument("bag_path", help="Path to the rosbag2 directory.")
    parser.add_argument(
        "--output-dir",
        default="of_debug_frames",
        help="Directory where rendered PNG images will be written.",
    )
    parser.add_argument("--params-file", default=None, help="Path to the OF params YAML.")
    parser.add_argument("--storage-id", default="sqlite3", help="rosbag2 storage plugin.")
    parser.add_argument("--frame-interval-sec", type=float, default=0.1, help="Minimum time between exported frames.")
    parser.add_argument("--min-speed-mps", type=float, default=0.02, help="Only export frames whose selected OF speed exceeds this threshold.")
    parser.add_argument("--max-images", type=int, default=0, help="Stop after exporting this many images (0 = unlimited).")
    parser.add_argument("--render-scale", type=float, default=None, help="Override the debug render scale.")
    parser.add_argument("--calibration-file", default=None, help="Override the fallback calibration XML path.")
    parser.add_argument(
        "--use-camera-info",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether CameraInfo is used.",
    )
    return parser.parse_args(argv)


def iter_bag_messages(
    bag_path: Path,
    storage_id: str,
):
    """Iterate serialized rosbag2 messages in chronological order."""
    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_path),
        storage_id=storage_id,
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    type_map = {entry.name: entry.type for entry in topics_and_types}
    while reader.has_next():
        topic, data, stamp_ns = reader.read_next()
        msg_type = get_message(type_map[topic])
        yield topic, deserialize_message(data, msg_type), float(stamp_ns) * 1e-9


def build_processor_params(args: argparse.Namespace) -> dict[str, object]:
    """Load and override OF params for offline export."""
    params = load_ros_params(args.params_file)
    if args.render_scale is not None:
        params["debug_render_scale"] = float(args.render_scale)
    if args.calibration_file is not None:
        params["calibration_file"] = str(args.calibration_file)
    if args.use_camera_info is not None:
        params["use_camera_info"] = bool(args.use_camera_info)
    return params


def main(argv: list[str] | None = None) -> int:
    """Export PNG debug frames from a saved rosbag2 recording."""
    args = parse_args(argv)
    bag_path = Path(args.bag_path).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = OfflineDebugProcessor(build_processor_params(args))
    params = processor.params
    interesting_topics = {
        str(params.get("image_topic", "/crazyflie/camera/bottom/compressed")),
        str(params.get("camera_info_topic", "/crazyflie/camera/bottom/camera_info")),
        str(params.get("imu_topic", "/crazyflie/imu/acc_derived")),
        str(params.get("lidar_topic", "/crazyflie/lidar/height")),
        str(params.get("bottom_depth_points_topic", "/crazyflie/depth_camera/bottom/points")),
        str(params.get("third_person_image_topic", "/crazyflie/camera/third_person/compressed")),
    }
    third_person_topic = str(
        params.get("third_person_image_topic", "/crazyflie/camera/third_person/compressed")
    )

    exported = 0
    last_export_stamp_s: float | None = None
    latest_third_person_image = None
    latest_third_person_stamp_s: float | None = None
    for topic, msg, _ in iter_bag_messages(bag_path, args.storage_id):
        if topic not in interesting_topics:
            continue
        if topic == third_person_topic:
            latest_third_person_image = decode_compressed_image(msg)
            latest_third_person_stamp_s = stamp_from_msg(msg)
            continue
        frame = processor.ingest(topic, msg)
        if frame is None:
            continue
        if frame.speed_mps < args.min_speed_mps:
            continue
        if (
            last_export_stamp_s is not None
            and (frame.stamp_s - last_export_stamp_s) < args.frame_interval_sec
        ):
            continue

        filename = output_dir / f"frame_{exported:04d}_{frame.stamp_s:010.3f}_{frame.mode}.png"
        cv2.imwrite(str(filename), frame.image_bgr)
        print(f"saved {filename}")
        if latest_third_person_image is not None:
            third_filename = output_dir / f"frame_{exported:04d}_{frame.stamp_s:010.3f}_{frame.mode}_third_person.png"
            cv2.imwrite(str(third_filename), latest_third_person_image)
            if latest_third_person_stamp_s is not None:
                delta_t = abs(frame.stamp_s - latest_third_person_stamp_s)
                print(f"saved {third_filename} (dt={delta_t:.3f}s)")
            else:
                print(f"saved {third_filename}")
        exported += 1
        last_export_stamp_s = frame.stamp_s
        if args.max_images > 0 and exported >= args.max_images:
            break

    print(f"exported {exported} frames to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
