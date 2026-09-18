"""Focused tests for compact return-gate dashboard telemetry."""

import json
import threading
from types import SimpleNamespace

from std_msgs.msg import String

from radical_event_navigation_system.web_dashboard_node import (
    WebDashboardNode,
    _html_page,
)


def test_dashboard_contains_compact_gate_and_pose_guard_rows() -> None:
    html = _html_page(poll_hz=5.0).decode("utf-8")

    assert 'id="return-gate-row"' in html
    assert 'id="return-gate-range"' in html
    assert 'id="return-gate-state"' in html
    assert 'id="pose-guard-row"' in html
    assert 'id="pose-guard-limit"' in html
    assert 'id="return-uncertainty"' not in html


def test_event_map_exposes_professional_title_and_arm_states() -> None:
    html = _html_page(poll_hz=5.0).decode("utf-8")

    assert "Navigation Event Map · Odometry Frame" in html
    assert "ODOM Debug Map" not in html
    assert "Available arm" in html
    assert "Selected arm" in html
    assert "Rejected arm" in html
    assert "Array.isArray(e.dead_end_paths)" in html
    assert "drawMapCross(ctx, ex, ey" in html


def test_dashboard_accepts_valid_latched_correction_policy() -> None:
    fake = SimpleNamespace(
        _lock=threading.RLock(),
        _telemetry=SimpleNamespace(pose_correction_policy={}),
    )
    message = String()
    message.data = json.dumps({
        "enabled": True,
        "base_m": 1.0,
        "growth_per_meter": 0.12,
        "hard_cap_m": 8.0,
    })

    WebDashboardNode._on_pose_correction_policy(fake, message)

    assert fake._telemetry.pose_correction_policy == {
        "enabled": True,
        "base_m": 1.0,
        "growth_per_meter": 0.12,
        "hard_cap_m": 8.0,
    }


def test_dashboard_rejects_invalid_correction_policy() -> None:
    original = {"enabled": True, "base_m": 1.0}
    fake = SimpleNamespace(
        _lock=threading.RLock(),
        _telemetry=SimpleNamespace(pose_correction_policy=original.copy()),
    )
    message = String()
    message.data = json.dumps({
        "enabled": True,
        "base_m": 1.0,
        "growth_per_meter": -0.12,
        "hard_cap_m": 8.0,
    })

    WebDashboardNode._on_pose_correction_policy(fake, message)

    assert fake._telemetry.pose_correction_policy == original


def test_dashboard_tracks_selected_and_rejected_junction_arms() -> None:
    telemetry = SimpleNamespace(
        total_events_discovered=0,
        junction_events_discovered=0,
        dead_ends_encountered=0,
        event_map={},
        matched_event_ids=set(),
    )
    fake = SimpleNamespace(
        _lock=threading.RLock(),
        _telemetry=telemetry,
        _known_stored_junction_ids=set(),
        _get_short_id=lambda event_id: "J1",
        _add_activity=lambda *args: None,
    )
    message = String()
    message.data = json.dumps({
        "status": "stored",
        "event_id": "junction-1",
        "event_type": "JUNCTION",
        "junction_type": "T_JUNCTION",
        "x": 1.0,
        "y": 2.0,
        "num_paths": 3,
        "path_angles": [0.0, 1.57, 3.14],
        "selected_path_index": 1,
        "dead_end_paths": [2],
    })

    WebDashboardNode._on_event_stored(fake, message)

    event = telemetry.event_map["junction-1"]
    assert event.selected_path_index == 1
    assert event.path_angles == [0.0, 1.57, 3.14]
    assert event.dead_end_paths == [2]
