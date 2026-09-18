"""State-transition tests for sequential return navigation."""

import json
from types import MethodType, SimpleNamespace

import pytest
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import String

from radical_event_navigation_system.core.event_types import NavigationMode
from radical_event_navigation_system.core.route_progress import LocalizationMode
from radical_event_navigation_system.core.navigation_event import Pose3D
from radical_event_navigation_system.event_navigation_node import SmartNavigationNode


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _Logger:
    def info(self, message: str) -> None:
        del message

    def warn(self, message: str) -> None:
        del message

    def error(self, message: str) -> None:
        del message


def _bind_history_helpers(fake) -> None:
    fake._find_previous_junction_before = MethodType(
        SmartNavigationNode._find_previous_junction_before,
        fake,
    )
    fake._find_junction_entry = MethodType(
        SmartNavigationNode._find_junction_entry,
        fake,
    )


def test_successful_match_advances_to_previous_target_with_handover() -> None:
    starts = []
    updates = []
    fake = SimpleNamespace(
        home_event_id='HOME',
        return_target_event_id='B',
        returning_to_prev_junction=True,
        junction_history=[
            {'event_id': 'A', 'route_distance_m': 4.0},
            {'event_id': 'B', 'route_distance_m': 8.0},
        ],
        route_distance=10.0,
        _return_anchor_route_distance_m=10.0,
        _return_handover_from_event_id=None,
        debug_mode=False,
        return_viz_pub=_Publisher(),
        get_logger=lambda: _Logger(),
        _publish_entry_arm_commit=lambda *args, **kwargs: True,
        _publish_expected_event_id=lambda event_id: None,
        _start_return_search_for_current_target=(
            lambda *, reset_anchor: starts.append(reset_anchor)
        ),
        _update_return_progress_and_maybe_skip=lambda: updates.append(True),
    )
    _bind_history_helpers(fake)

    SmartNavigationNode._handle_return_to_base_match(
        fake,
        'B',
        {'observed_junction': {'center_xy': [8.0, 0.0]}},
    )

    assert fake.return_target_event_id == 'A'
    assert fake._return_handover_from_event_id == 'B'
    assert fake._return_anchor_route_distance_m == 8.0
    assert fake.route_distance == 8.0
    assert starts == [True]
    assert updates == [True]


def test_skipped_target_does_not_create_false_handover_anchor() -> None:
    starts = []
    fake = SimpleNamespace(
        return_target_event_id='B',
        current_pose=Pose3D(7.0, 0.0, 0.0),
        junction_history=[
            {'event_id': 'A', 'route_distance_m': 4.0},
            {'event_id': 'B', 'route_distance_m': 8.0},
        ],
        current_mode=NavigationMode.RETURN_TO_BASE,
        home_event_id='HOME',
        returning_to_prev_junction=True,
        _return_handover_from_event_id='STALE',
        _distance_since_localization_anchor_m=23.0,
        debug_mode=False,
        get_logger=lambda: _Logger(),
        set_mode=lambda mode: None,
        _start_return_search_for_current_target=(
            lambda *, reset_anchor: starts.append(reset_anchor)
        ),
    )
    _bind_history_helpers(fake)

    SmartNavigationNode._skip_return_target(fake, 'passed_route_window')

    assert fake.return_target_event_id == 'A'
    assert fake._return_handover_from_event_id is None
    assert fake._distance_since_localization_anchor_m == 23.0
    assert starts == [False]


def test_real_match_resets_anchor_but_waits_for_estimator_correction_ack() -> None:
    handled = []
    fake = SimpleNamespace(
        returning_to_prev_junction=True,
        return_target_event_id='B',
        current_mode=NavigationMode.RETURN_TO_BASE,
        _distance_since_localization_anchor_m=18.0,
        _localization_anchor_kind='mission_start',
        _localization_anchor_event_id=None,
        _handle_return_to_base_match=(
            lambda event_id, data: handled.append((event_id, data))
        ),
    )
    message = String()
    message.data = json.dumps({
        'event_id': 'B',
        'pose_correction_published': True,
    })

    SmartNavigationNode.match_found_callback(fake, message)

    assert fake._distance_since_localization_anchor_m == 0.0
    assert fake._localization_anchor_kind == 'match'
    assert fake._localization_anchor_event_id == 'B'
    assert handled[0][0] == 'B'


def test_applied_pose_correction_ack_registers_application_stamp() -> None:
    stamps = []
    fake = SimpleNamespace(
        _distance_tracker=SimpleNamespace(
            acknowledge_correction=lambda stamp: stamps.append(stamp)
        )
    )
    message = PoseWithCovarianceStamped()
    message.header.stamp.sec = 12
    message.header.stamp.nanosec = 500_000_000

    SmartNavigationNode._pose_correction_callback(fake, message)

    assert stamps == [12.5]


def test_unexpected_match_does_not_reset_localization_anchor() -> None:
    fake = SimpleNamespace(
        returning_to_prev_junction=True,
        return_target_event_id='B',
        current_mode=NavigationMode.RETURN_TO_BASE,
        _distance_since_localization_anchor_m=18.0,
        _localization_anchor_kind='match',
        _localization_anchor_event_id='C',
    )
    message = String()
    message.data = json.dumps({
        'event_id': 'STALE',
        'pose_correction_published': False,
    })

    SmartNavigationNode.match_found_callback(fake, message)

    assert fake._distance_since_localization_anchor_m == 18.0
    assert fake._localization_anchor_kind == 'match'
    assert fake._localization_anchor_event_id == 'C'


@pytest.mark.parametrize(
    ("last_update_s", "degraded", "expected"),
    [
        (-1.0, False, LocalizationMode.OF_DEGRADED),
        (9.9, False, LocalizationMode.NORMAL),
        (9.9, True, LocalizationMode.OF_DEGRADED),
        (9.0, False, LocalizationMode.OF_DEGRADED),
        (10.1, False, LocalizationMode.OF_DEGRADED),
    ],
)
def test_localization_mode_fails_safe_for_missing_stale_or_future_flag(
    last_update_s, degraded, expected
) -> None:
    fake = SimpleNamespace(
        _of_degraded_update_time_s=last_update_s,
        _of_degraded=degraded,
        of_degraded_timeout_s=0.5,
        current_pose_cov=None,
        odom_sigma_degraded_threshold_m=0.75,
        _sigma_xy_from_pose_cov=lambda covariance: None,
    )

    assert SmartNavigationNode._localization_mode(fake, 10.0) == expected


def _start_event_message(event_id: str = "HOME") -> String:
    message = String()
    message.data = json.dumps({
        "event_id": event_id,
        "event_type": "START",
        "timestamp": "2026-01-01T00:00:00",
        "decision": "HOVER",
        "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0},
        "pose_uncertainty": [[0.0, 0.0, 0.0]] * 3,
        "odometry_distance": 0.0,
        "tunnel_geometry": {
            "width": 2.0,
            "height": 2.0,
            "shape_descriptor": [],
        },
        "junction_config": None,
        "radial_descriptor": None,
    })
    return message


def test_atomic_start_event_releases_navigation_once_and_is_idempotent() -> None:
    published = _Publisher()
    fake = SimpleNamespace(
        mission_active=True,
        home_event_id=None,
        _start_ready=False,
        _pending_start_event_id=None,
        _pending_start_event_time_s=-1.0,
        current_mode=NavigationMode.WAIT_START,
        navigation_ready_pub=published,
        _now_s=lambda: 10.0,
        set_mode=lambda mode: setattr(fake, "current_mode", mode),
        get_logger=lambda: _Logger(),
    )
    fake._activate_exploration_if_start_ready = MethodType(
        SmartNavigationNode._activate_exploration_if_start_ready, fake
    )
    fake._publish_navigation_ready = MethodType(
        SmartNavigationNode._publish_navigation_ready, fake
    )

    SmartNavigationNode._start_event_callback(fake, _start_event_message())
    SmartNavigationNode._start_event_callback(fake, _start_event_message())

    assert fake.home_event_id == "HOME"
    assert fake.current_mode == NavigationMode.EXPLORE
    assert [message.data for message in published.messages] == [True]
