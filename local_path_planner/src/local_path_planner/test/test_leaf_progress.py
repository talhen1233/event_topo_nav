import math

import pytest

from local_path_planner.scripts.leaf_progress import LeafProgressTracker
from local_path_planner.scripts.safety_controller import SafetyController, SafetyControllerParams


def tick(tracker, t, xy=(0.0, 0.0), **overrides):
    args = dict(eligible=True, remaining_m=0.2, position_xy=xy,
                heading_rad=0.0, now_s=t)
    args.update(overrides)
    return tracker.update(**args)


def test_lateral_motion_and_endpoint_jitter_are_not_forward_progress():
    tracker = LeafProgressTracker()
    for i in range(20):
        assert not tick(tracker, i * 0.1, (0.01 * (i % 2), 0.08 * (i % 2)),
                        remaining_m=0.2 + 0.03 * (i % 2))
    assert tick(tracker, 2.0)


def test_real_forward_progress_restarts_window():
    tracker = LeafProgressTracker()
    for i in range(60):
        assert not tick(tracker, i * 0.1, (i * 0.012, 0.0))


def test_motion_is_projected_onto_edge_direction():
    tracker = LeafProgressTracker()
    for i in range(60):
        assert not tick(tracker, i * 0.1, (0.0, i * 0.012), heading_rad=math.pi / 2)


@pytest.mark.parametrize('overrides', [dict(eligible=False), dict(remaining_m=0.5)])
def test_holds_or_graph_extension_cancel_pending_stall(overrides):
    tracker = LeafProgressTracker()
    for i in range(19):
        tick(tracker, i * 0.1)
    assert not tick(tracker, 1.9, **overrides)
    assert not tick(tracker, 2.0)
    assert tracker.wait_s == 0.0


def test_pose_jump_clock_gap_and_direction_change_restart_observation():
    tracker = LeafProgressTracker()
    for i in range(19):
        tick(tracker, i * 0.1)
    assert not tick(tracker, 1.9, (1.0, 0.0))
    assert tracker.wait_s == 0.0
    assert not tick(tracker, 5.0, (1.0, 0.0))
    assert tracker.wait_s == 0.0
    assert not tick(tracker, 5.1, (1.0, 0.0), heading_rad=math.pi)
    assert tracker.wait_s == 0.0


@pytest.mark.parametrize('front', [None, 1.0, 3.0])
def test_leaf_stall_searches_then_reverses_without_nearby_wall(front):
    tracker = LeafProgressTracker()
    safety = SafetyController(SafetyControllerParams())
    args = dict(d_front=front, d_left=0.6, d_right=0.6,
                graph_has_continuation=False, dead_end_candidate=True)
    for i in range(21):
        stalled = tick(tracker, i * 0.1)
        result = safety.update(**args, drone_yaw=0.0, now_s=i * 0.1, graph_stalled=stalled)
        assert result.mode == ("TURN_TO_OPENING" if i == 20 else "NONE")
    assert result.vx_scale == 0.0
    # The sweep is anchored once, not restarted by each graph update.
    result = safety.update(**args, drone_yaw=math.radians(71), now_s=9.0)
    assert result.mode == "TURN_TO_OPENING"
    assert result.yaw_rate_override < 0.0
    result = safety.update(**args, drone_yaw=-math.radians(71), now_s=22.0)
    assert result.mode == "DEAD_END"
    assert safety.update(**args, drone_yaw=-1.5, now_s=22.1).mode == "DEAD_END"
    safety.reset()  # Planner completes its latched reverse turn.
    assert not safety.graph_recovery


def test_search_releases_when_graph_returns_but_not_through_obstacle():
    safety = SafetyController(SafetyControllerParams())
    args = dict(d_left=0.6, d_right=0.6, dead_end_candidate=True,
                drone_yaw=0.0)
    safety.update(**args, d_front=2.0, graph_has_continuation=False,
                  graph_stalled=True, now_s=0.0)
    assert safety.update(**args, d_front=0.2, graph_has_continuation=True,
                         now_s=0.1).mode == "TURN_TO_OPENING"
    assert safety.update(**args, d_front=2.0, graph_has_continuation=True,
                         now_s=0.2).mode == "NONE"
    assert not safety.graph_recovery
