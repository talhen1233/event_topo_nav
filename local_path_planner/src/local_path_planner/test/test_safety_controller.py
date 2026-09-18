import math
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from local_path_planner.scripts.safety_controller import (  # noqa: E402
    SafetyController,
    SafetyControllerParams,
)


def test_dynamic_stop_distance_contains_full_budget():
    controller = SafetyController(
        SafetyControllerParams(
            max_decel_mps2=2.0,
            reaction_time_s=0.25,
            vehicle_radius_m=0.15,
            stop_margin_m=0.10,
            slow_margin_m=0.40,
        )
    )

    stop, slow = controller._stopping_distances(1.5)

    expected = 1.5**2 / (2.0 * 2.0) + 1.5 * 0.25 + 0.15 + 0.10
    assert stop == pytest.approx(expected)
    assert slow == pytest.approx(expected + 0.40)


def test_dynamic_gate_stops_inside_required_distance():
    controller = SafetyController(SafetyControllerParams())

    output = controller.update(
        d_front=1.0,
        d_left=0.8,
        d_right=0.8,
        graph_has_continuation=True,
        dead_end_candidate=False,
        drone_yaw=0.0,
        now_s=1.0,
        forward_speed_mps=1.5,
    )

    assert output.collision_stop
    assert output.vx_scale == 0.0
    assert controller.active_stop_dist_m > 1.0


def test_no_occupied_front_voxel_allows_motion_with_a_fresh_map():
    controller = SafetyController(SafetyControllerParams())

    output = controller.update(
        d_front=None,
        d_left=None,
        d_right=None,
        graph_has_continuation=True,
        dead_end_candidate=False,
        drone_yaw=0.0,
        now_s=1.0,
    )

    assert not output.collision_stop
    assert output.vx_scale == 1.0
    assert output.mode == "NONE"
    assert output.yaw_rate_override is None


def test_missing_front_during_turn_does_not_fake_an_open_route():
    controller = SafetyController(SafetyControllerParams())
    controller._enter_turn_to_opening(
        [("left", 2.0)],
        d_left=2.0,
        d_right=0.5,
        drone_yaw=0.0,
    )

    output = controller.update(
        d_front=None,
        d_left=2.0,
        d_right=0.5,
        graph_has_continuation=False,
        dead_end_candidate=True,
        drone_yaw=0.1,
        now_s=1.0,
    )

    assert output.mode == "TURN_TO_OPENING"


def test_persistent_unknown_never_matures_into_blocked_or_dead_end():
    controller = SafetyController(SafetyControllerParams())

    for now_s in (1.0, 1.5, 2.0, 3.0):
        output = controller.update(
            d_front=None,
            d_left=None,
            d_right=None,
            graph_has_continuation=False,
            dead_end_candidate=True,
            drone_yaw=0.0,
            now_s=now_s,
        )
        assert output.mode == "NONE"


def test_side_opening_requires_a_measured_sector_distance():
    controller = SafetyController(SafetyControllerParams(open_dist_m=0.5))

    assert controller._find_opening(None, None) is None
    assert controller._find_opening(1.0, None) == [("left", 1.0)]


def test_low_speed_keeps_configured_minimum_stop_distance():
    controller = SafetyController(
        SafetyControllerParams(
            front_stop_dist_m=0.30,
            vehicle_radius_m=0.0,
            stop_margin_m=0.0,
            reaction_time_s=0.0,
        )
    )

    stop, _ = controller._stopping_distances(0.0)

    assert math.isclose(stop, 0.30)


def test_block_threshold_never_falls_inside_dynamic_stop_distance():
    controller = SafetyController(
        SafetyControllerParams(
            front_block_dist_m=0.35,
            front_clear_dist_m=0.75,
            front_block_hysteresis_m=0.20,
            block_debounce_s=0.25,
        )
    )

    first = controller.update(
        d_front=1.0,
        d_left=0.8,
        d_right=0.8,
        graph_has_continuation=True,
        dead_end_candidate=False,
        drone_yaw=0.0,
        now_s=1.0,
        forward_speed_mps=1.5,
    )
    stop_at_entry = controller.active_stop_dist_m

    assert first.collision_stop
    assert controller.active_front_block_dist_m >= stop_at_entry
    assert (
        controller.active_front_clear_dist_m
        >= controller.active_front_block_dist_m + 0.20
    )

    # The command is now zero, but the entry distance remains latched long
    # enough for the BLOCKED debounce to complete.
    second = controller.update(
        d_front=1.0,
        d_left=0.8,
        d_right=0.8,
        graph_has_continuation=True,
        dead_end_candidate=False,
        drone_yaw=0.0,
        now_s=1.3,
        forward_speed_mps=0.0,
    )
    assert second.mode == "BLOCKED"
    assert controller.active_front_block_dist_m >= stop_at_entry


def test_graph_terminal_at_measured_wall_reaches_dead_end():
    controller = SafetyController(
        SafetyControllerParams(
            front_block_dynamic_max_m=1.2,
            front_block_hysteresis_m=0.20,
            open_dist_m=1.5,
            block_debounce_s=0.25,
            dead_end_time_s=1.0,
        )
    )

    common = dict(
        d_front=0.78,
        d_left=1.42,
        d_right=0.98,
        graph_has_continuation=False,
        dead_end_candidate=True,
        drone_yaw=0.0,
        forward_speed_mps=0.0,
    )
    assert controller.update(now_s=1.0, **common).mode == "NONE"
    assert controller.update(now_s=1.3, **common).mode == "BLOCKED"
    assert controller.update(now_s=1.4, **common).mode == "BLOCKED"
    assert controller.update(now_s=2.5, **common).mode == "DEAD_END"


def test_blocked_hysteresis_prevents_threshold_chatter():
    controller = SafetyController(
        SafetyControllerParams(
            front_block_dist_m=0.35,
            front_clear_dist_m=0.40,
            front_block_hysteresis_m=0.20,
            block_debounce_s=0.0,
        )
    )
    common = dict(
        d_left=0.8,
        d_right=0.8,
        graph_has_continuation=True,
        dead_end_candidate=False,
        drone_yaw=0.0,
        forward_speed_mps=0.0,
    )

    assert controller.update(d_front=0.30, now_s=1.0, **common).mode == "BLOCKED"
    assert controller.update(d_front=0.45, now_s=1.1, **common).mode == "BLOCKED"
    assert controller.update(d_front=0.56, now_s=1.2, **common).mode == "NONE"


def test_turn_search_needs_graph_continuation_before_clearance_release():
    controller = SafetyController(
        SafetyControllerParams(
            front_block_dynamic_max_m=1.2,
            front_block_hysteresis_m=0.20,
            open_dist_m=1.5,
            block_debounce_s=0.0,
        )
    )
    terminal = dict(
        d_left=1.8,
        d_right=0.9,
        graph_has_continuation=False,
        dead_end_candidate=True,
        forward_speed_mps=0.0,
    )

    assert controller.update(
        d_front=0.8, drone_yaw=0.0, now_s=1.0, **terminal
    ).mode == "BLOCKED"
    assert controller.update(
        d_front=0.8, drone_yaw=0.0, now_s=1.1, **terminal
    ).mode == "TURN_TO_OPENING"
    assert controller.update(
        d_front=1.8, drone_yaw=0.1, now_s=1.2, **terminal
    ).mode == "TURN_TO_OPENING"

    recovered = dict(terminal)
    recovered["graph_has_continuation"] = True
    recovered["dead_end_candidate"] = False
    assert controller.update(
        d_front=1.8, drone_yaw=0.2, now_s=1.3, **recovered
    ).mode == "NONE"


def test_terminal_pending_trigger_is_cancelled_when_graph_recovers():
    controller = SafetyController(SafetyControllerParams(block_debounce_s=0.25))

    first = controller.update(
        d_front=0.78,
        d_left=1.42,
        d_right=0.98,
        graph_has_continuation=False,
        dead_end_candidate=True,
        drone_yaw=0.0,
        now_s=1.0,
    )
    assert first.mode == "NONE"

    recovered = controller.update(
        d_front=0.78,
        d_left=1.42,
        d_right=0.98,
        graph_has_continuation=True,
        dead_end_candidate=False,
        drone_yaw=0.0,
        now_s=1.1,
    )
    assert recovered.mode == "NONE"
