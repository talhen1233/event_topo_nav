from argparse import Namespace

from of_localization.offline_debug.processor import OfflineDebugProcessor
from of_localization.offline_debug.record_debug_bag import (
    build_record_command,
    build_topic_list,
)


def test_build_topic_list_uses_params_and_extra_topics():
    args = Namespace(
        image_topic=None,
        camera_info_topic=None,
        imu_topic=None,
        lidar_topic=None,
        depth_topic=None,
        third_person_image_topic=None,
        extra_topic=["/foo", "/bar"],
    )
    params = {
        "image_topic": "/img",
        "camera_info_topic": "/cam_info",
        "imu_topic": "/imu",
        "lidar_topic": "/lidar",
        "bottom_depth_points_topic": "/depth",
        "third_person_image_topic": "/third_person",
    }
    topics = build_topic_list(args, params)
    assert topics == ["/img", "/cam_info", "/imu", "/lidar", "/depth", "/third_person", "/foo", "/bar"]


def test_build_record_command_includes_selected_storage_and_topics(tmp_path):
    args = Namespace(
        output=tmp_path / "bag",
        storage_id="sqlite3",
        compression_mode=None,
        compression_format=None,
    )
    command = build_record_command(args, ["/img", "/imu"])
    assert command == ["ros2", "bag", "record", "-o", str(tmp_path / "bag"), "-s", "sqlite3", "/img", "/imu"]


def test_offline_debug_processor_initializes_from_params():
    params = {
        "scale_factor": 0.5,
        "use_camera_info": True,
        "distortion_model": "equidistant",
    }
    processor = OfflineDebugProcessor(params)
    assert processor.scale_factor == 0.5
