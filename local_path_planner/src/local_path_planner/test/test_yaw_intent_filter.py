import math

import numpy as np

from local_path_planner.scripts.arm_commit_tracker import CommittedArmIntent
from local_path_planner.scripts.graph_tracker import GraphArmCandidate, GraphTrackingState
from local_path_planner.scripts.yaw_intent_filter import (
    YawIntentFilter,
    YawIntentFilterParams,
)


def test_uncommitted_junction_jitter_does_not_repeatedly_block_translation():
    filt = YawIntentFilter(YawIntentFilterParams())
    straight = GraphTrackingState(
        valid=True, current_heading_rad=0.0, ahead_path_length_m=2.0,
    )
    for _ in range(100):
        filt.update(straight, None, dt=0.05)

    for degrees in [25.0, -25.0] * 20:
        graph = GraphTrackingState(
            valid=True,
            current_heading_rad=math.radians(degrees),
            ahead_path_length_m=2.0,
            approaching_junction=True,
        )
        intent = filt.update(graph, None, dt=0.05)
        assert abs(intent.gate_heading_rad) < 0.35
        assert abs(intent.yaw_ref_rad) <= 0.045 + 1e-6


def test_uncommitted_partial_junction_holds_incoming_heading_and_position():
    filt = YawIntentFilter(YawIntentFilterParams())
    straight = GraphTrackingState(
        valid=True,
        current_heading_rad=0.0,
        ahead_path_length_m=2.0,
    )
    filt.update(straight, None, dt=0.05)
    left_arm = GraphArmCandidate(
        node_id=2,
        angle_rad=0.5 * math.pi,
        edge_length_m=2.0,
        lookahead_xy=np.array([0.0, 0.8], dtype=np.float32),
        lookahead_heading_rad=0.5 * math.pi,
    )
    partial_junction = GraphTrackingState(
        valid=True,
        current_heading_rad=0.0,
        dist_to_ahead_node_m=0.4,
        approaching_junction=True,
        arm_candidates=[left_arm],
        ahead_path_length_m=0.4,
    )

    for _ in range(20):
        intent = filt.update(partial_junction, None, dt=0.05)

    assert math.isclose(intent.yaw_ref_rad, 0.0, abs_tol=1e-6)
    assert math.isclose(intent.speed_scale, 0.0, abs_tol=1e-6)
    assert intent.selected_arm_angle_rad is None


def test_uncommitted_hold_falls_back_to_controlled_creep() -> None:
    filt = YawIntentFilter(
        YawIntentFilterParams(
            uncommitted_hold_timeout_s=3.0,
            uncommitted_creep_speed_scale=0.15,
        )
    )
    junction = GraphTrackingState(
        valid=True,
        current_heading_rad=0.0,
        dist_to_ahead_node_m=0.2,
        approaching_junction=True,
        ahead_path_length_m=0.2,
    )

    for _ in range(61):
        intent = filt.update(junction, None, dt=0.05)

    assert math.isclose(intent.speed_scale, 0.15, abs_tol=1e-6)


def test_persistent_uncommitted_bend_still_requires_alignment():
    filt = YawIntentFilter(YawIntentFilterParams())
    filt.update(GraphTrackingState(valid=True, current_heading_rad=0.0), None, 0.05)
    bend = GraphTrackingState(
        valid=True, current_heading_rad=math.radians(60.0),
        ahead_path_length_m=2.0,
    )
    for _ in range(40):
        intent = filt.update(bend, None, dt=0.05)
    assert intent.gate_heading_rad > 0.35
    assert math.isclose(intent.yaw_ref_rad, math.radians(60.0), abs_tol=1e-5)


def test_large_committed_turn_gates_against_final_heading_not_slewed_reference():
    selected_heading = 0.5 * math.pi
    arm = GraphArmCandidate(
        node_id=2,
        angle_rad=selected_heading,
        edge_length_m=2.0,
        lookahead_xy=np.array([0.0, 0.8], dtype=np.float32),
        lookahead_heading_rad=selected_heading,
    )
    graph = GraphTrackingState(
        valid=True,
        current_heading_rad=0.0,
        dist_to_ahead_node_m=0.5,
        approaching_junction=True,
        arm_candidates=[arm],
        ahead_path_length_m=0.5,
    )
    commit = CommittedArmIntent(
        event_id="junction",
        mode="explore",
        event_center_xy=np.zeros(2, dtype=np.float32),
        gate_radius_m=2.0,
        selected_arm_angle_rad=selected_heading,
    )
    filt = YawIntentFilter(YawIntentFilterParams(slew_rate_rps=0.9))

    # Build a strong, persistent potential around the old corridor heading.
    # A newly committed arm must override it immediately rather than waiting
    # for the old evidence to decay.
    old_graph = GraphTrackingState(
        valid=True,
        current_heading_rad=0.0,
        dist_to_ahead_node_m=0.5,
        approaching_junction=True,
        arm_candidates=[],
        ahead_path_length_m=2.0,
    )
    for _ in range(100):
        filt.update(old_graph, None, dt=0.05)

    intent = filt.update(graph, commit, dt=0.05)

    assert intent.gate_heading_rad is not None
    assert math.isclose(intent.gate_heading_rad, selected_heading, abs_tol=1e-5)
    assert math.isclose(intent.yaw_ref_rad, 0.045, abs_tol=1e-6)
    assert abs(intent.gate_heading_rad) > math.radians(20.0)
    assert abs(intent.yaw_ref_rad) < math.radians(20.0)


def test_locked_commit_remains_authoritative_after_junction_disappears():
    selected_heading = 0.5 * math.pi
    graph = GraphTrackingState(
        valid=True,
        current_heading_rad=math.radians(35.0),
        approaching_junction=False,
        ahead_path_length_m=2.0,
    )
    commit = CommittedArmIntent(
        event_id="junction",
        mode="explore",
        event_center_xy=np.zeros(2, dtype=np.float32),
        gate_radius_m=2.0,
        selected_arm_angle_rad=selected_heading,
        branch_locked=True,
    )
    filt = YawIntentFilter(YawIntentFilterParams(slew_rate_rps=0.9))
    filt.update(
        GraphTrackingState(valid=True, current_heading_rad=0.0),
        None,
        dt=0.05,
    )

    intent = filt.update(graph, commit, dt=0.05)

    assert math.isclose(intent.gate_heading_rad, selected_heading, abs_tol=1e-6)
    assert intent.yaw_ref_rad > 0.0
    assert math.isclose(
        intent.selected_arm_angle_rad,
        selected_heading,
        abs_tol=1e-6,
    )
