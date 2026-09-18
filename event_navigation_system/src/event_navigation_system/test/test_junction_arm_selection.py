"""Tests for exhausted junction-arm selection."""

import math
from types import SimpleNamespace

import numpy as np

from radical_event_navigation_system.core.event_types import NavigationMode
from radical_event_navigation_system.event_navigation_node import SmartNavigationNode


def test_reidentified_junction_retreats_through_entry_when_exits_are_dead() -> None:
    event_id = "t-junction"
    event = {
        "event_id": event_id,
        "entry_angle": math.pi,
        "junction_config": {
            "path_angles": [math.pi, 0.5 * math.pi, -0.5 * math.pi],
            "dead_end_paths": [1, 2],
        },
    }
    fake = SimpleNamespace(
        attempted_paths_per_event={event_id: {1, 2}},
        current_pose=SimpleNamespace(yaw=0.0),
        last_selected_direction=-0.5 * math.pi,
        inertia_penalty_weight=0.3,
        _angle_diff=lambda first, second: SmartNavigationNode._angle_diff(
            fake,
            first,
            second,
        ),
        _reidentified_arm_constraints=lambda payload: (
            SmartNavigationNode._reidentified_arm_constraints(fake, payload)
        ),
        _select_preferred_arm=SmartNavigationNode._select_preferred_arm,
    )

    selected = SmartNavigationNode._select_next_arm_after_reid(fake, event)

    assert selected == 0


def test_unexplored_exit_still_has_priority_over_entry_arm() -> None:
    scores = np.array([1.0, 0.2, 0.8], dtype=np.float32)

    selected = SmartNavigationNode._select_preferred_arm(
        scores,
        forbidden={0, 1},
        entry_idx=0,
    )

    assert selected == 2


def _junction_event() -> dict:
    return {
        "event_id": "t-junction",
        "event_type": "JUNCTION",
        "timestamp": "2026-01-01T00:00:00",
        "decision": "HOVER",
        "pose": {"x": 4.0, "y": 0.0, "z": 1.0, "yaw": 0.0},
        "pose_uncertainty": np.eye(3).tolist(),
        "odometry_distance": 4.0,
        "tunnel_geometry": {
            "width": 2.0,
            "height": 2.0,
            "shape_descriptor": [],
        },
        "junction_config": {
            "junction_type": "T_JUNCTION",
            "num_paths": 3,
            "path_angles": [math.pi, 0.5 * math.pi, -0.5 * math.pi],
            "selected_path_index": 2,
            "dead_end_paths": [1, 2],
            "path_widths": [],
        },
        "entry_angle": math.pi,
        "previous_event_id": "previous-junction",
        "radial_descriptor": None,
    }


def test_exhausted_reidentified_junction_dispatches_to_backtracking() -> None:
    retreats = []
    event = _junction_event()
    fake = SimpleNamespace(
        debug_mode=False,
        _pending_match_observation={
            "center_xy": [4.0, 0.0],
            "path_angles": event["junction_config"]["path_angles"],
        },
        current_pose=SimpleNamespace(x=4.0, y=0.0, yaw=0.0),
        attempted_paths_per_event={"t-junction": {1, 2}},
        _align_stored_angles_to_current=lambda stored, selected, observed: stored,
        _angle_diff=lambda first, second: SmartNavigationNode._angle_diff(
            fake,
            first,
            second,
        ),
        _reidentified_arm_constraints=lambda payload: (
            SmartNavigationNode._reidentified_arm_constraints(fake, payload)
        ),
        _backtrack_from_exhausted_junction=lambda junction, observation: (
            retreats.append((junction.event_id, observation)) or True
        ),
    )

    SmartNavigationNode._handle_reidentified_junction(fake, event)

    assert retreats[0][0] == "t-junction"
    assert fake.at_junction


def test_exhausted_junction_targets_previous_event_in_backtrack_mode() -> None:
    published = []
    modes = []
    searches = []
    current_entry = {"event_id": "t-junction"}
    previous_entry = {"event_id": "previous-junction"}
    fake = SimpleNamespace(
        home_event_id="HOME",
        debug_mode=False,
        returning_to_prev_junction=False,
        return_target_event_id=None,
        _return_handover_from_event_id="stale",
        _find_junction_entry=lambda event_id: current_entry,
        _find_previous_junction_before=lambda event_id: previous_entry,
        _publish_entry_arm_commit=lambda entry, *, mode, observation: (
            published.append((entry, mode, observation)) or True
        ),
        set_mode=lambda mode: modes.append(mode),
        _start_return_search_for_current_target=lambda *, reset_anchor: (
            searches.append(reset_anchor)
        ),
        _update_return_progress_and_maybe_skip=lambda: False,
    )
    event = SimpleNamespace(event_id="t-junction")

    result = SmartNavigationNode._backtrack_from_exhausted_junction(
        fake,
        event,
        {"center_xy": [4.0, 0.0]},
    )

    assert result
    assert published[0][1] == "backtrack"
    assert modes == [NavigationMode.BACKTRACK]
    assert fake.return_target_event_id == "previous-junction"
    assert fake.returning_to_prev_junction
    assert searches == [True]


def test_dead_end_clears_commit_before_reverse_handover() -> None:
    calls = []
    fake = SimpleNamespace(
        at_junction=True,
        _exit_junction_context=lambda: calls.append("clear"),
        _send_reverse_command=lambda *, mode: calls.append(
            f"reverse:{mode}"
        ),
    )

    SmartNavigationNode._handoff_dead_end_reverse(fake)

    assert calls == ["clear", "reverse:backtrack"]
