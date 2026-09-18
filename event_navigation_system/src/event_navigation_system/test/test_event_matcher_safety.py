"""Focused integration tests for matcher handover and correction guards."""

import math
from types import SimpleNamespace

import numpy as np
from builtin_interfaces.msg import Time

from radical_event_navigation_system.core.event_matcher import (
    EventMatcher,
    MatchCandidate,
)
from radical_event_navigation_system.core.event_repository import EventRepository
from radical_event_navigation_system.core.event_types import EventType
from radical_event_navigation_system.core.navigation_event import Pose3D


class _Logger:
    def info(self, message: str) -> None:
        del message

    def warn(self, message: str) -> None:
        del message


class _Stamp:
    def to_msg(self):
        return Time()


class _Clock:
    def now(self):
        return _Stamp()


class _RadialMatcher:
    def compute_similarity(self, current, stored):
        del current
        score = 0.88 if stored == 'from_descriptor' else 0.90
        return {
            'statistical_similarity': score,
            'spectral_similarity': score,
            'combined_score': score,
        }


def _event(event_id: str, x: float, descriptor: str):
    return SimpleNamespace(
        event_id=event_id,
        event_type=EventType.JUNCTION,
        junction_config=None,
        pose=Pose3D(x, 0.0, 1.0),
        radial_descriptor=descriptor,
        confidence=1.0,
    )


def _matcher(current_x: float) -> tuple[EventMatcher, object, object]:
    source = _event('A', 0.0, 'from_descriptor')
    target = _event('B', 4.0, 'target_descriptor')
    repository = SimpleNamespace(
        stored_events={'A': source, 'B': target},
        return_viz_state={
            'target_event_id': 'B',
            'status': 'returning',
            'handover_from_event_id': 'A',
            'expected_distance_m': 4.0,
            'progress_m': current_x,
            'distance_since_anchor_m': 5.0,
        },
        _compute_event_cov_xy_from_descriptor=lambda descriptor: np.eye(2) * 0.25,
    )
    node = SimpleNamespace(
        current_pose=Pose3D(current_x, 0.0, 1.0),
        pose_covariance=np.eye(3) * 0.01,
        get_logger=lambda: _Logger(),
        get_clock=lambda: _Clock(),
    )
    matcher = object.__new__(EventMatcher)
    matcher._node = node
    matcher._repository = repository
    matcher.current_descriptor = 'current_descriptor'
    matcher.radial_matcher = _RadialMatcher()
    matcher.debug_mode = False
    matcher._last_handover_log_key = None
    matcher.config = {
        'descriptor_score_threshold': 0.8,
        'return_handover_enabled': True,
        'return_handover_max_distance_m': 5.0,
        'return_handover_margin_base_m': 0.2,
        'return_handover_margin_sigma_mult': 1.0,
        'return_handover_margin_min_m': 0.1,
        'return_handover_margin_max_fraction': 0.4,
        'return_handover_competition_delta': 0.05,
        'pose_correction_enabled': True,
        'pose_correction_min_descriptor_score': 0.85,
        'pose_correction_detection_noise_m': 0.3,
        'pose_correction_cov_scale': 1.0,
        'pose_correction_min_variance_m2': 0.0225,
        'pose_correction_translation_base_m': 1.0,
        'pose_correction_translation_growth_per_meter': 0.12,
        'pose_correction_translation_hard_cap_m': 8.0,
        'pose_correction_event_confidence_min': 0.2,
        'pose_correction_event_cov_trace_max_xy': 0.5,
    }
    return matcher, source, target


def test_matcher_blocks_target_while_still_near_previous_event() -> None:
    matcher, _, target = _matcher(0.2)
    candidate = MatchCandidate(target, 0.9, 0.9, 0.9)
    assert not matcher._passes_nearby_event_handover(candidate)


def test_matcher_competition_requires_clear_descriptor_win() -> None:
    matcher, _, target = _matcher(2.0)
    candidate = MatchCandidate(target, 0.9, 0.9, 0.9)
    assert not matcher._passes_nearby_event_handover(candidate)


def test_matcher_competition_accepts_clear_descriptor_win() -> None:
    matcher, _, target = _matcher(2.0)
    candidate = MatchCandidate(target, 0.95, 0.95, 0.95)
    assert matcher._passes_nearby_event_handover(candidate)


def test_matcher_allows_target_in_target_only_region() -> None:
    matcher, _, target = _matcher(3.8)
    candidate = MatchCandidate(target, 0.9, 0.9, 0.9)
    assert matcher._passes_nearby_event_handover(candidate)


def test_route_progress_blocks_early_target_when_xy_is_drifted() -> None:
    matcher, _, target = _matcher(3.8)
    matcher._repository.return_viz_state['progress_m'] = 0.1
    candidate = MatchCandidate(target, 0.95, 0.95, 0.95)
    assert not matcher._passes_nearby_event_handover(candidate)


def test_matcher_requires_observed_center_for_pose_correction() -> None:
    matcher, _, target = _matcher(3.8)
    assert matcher._build_pose_correction(target, 0.9, None) is None


def test_matcher_builds_relative_center_pose_correction() -> None:
    matcher, _, target = _matcher(3.8)
    correction = matcher._build_pose_correction(
        target,
        0.9,
        {'center_xy': [3.5, 0.0]},
    )
    assert correction is not None
    assert correction.pose.pose.position.x == 4.3
    assert correction.pose.pose.position.y == 0.0
    assert correction.pose.covariance[0] >= 0.0225


def test_matcher_dynamic_correction_limit_uses_distance_since_anchor() -> None:
    matcher, _, target = _matcher(3.8)
    observation = {'center_xy': [0.0, 0.0]}

    # 5m from the anchor allows 1.6m, so a 4m correction is rejected.
    assert matcher._build_pose_correction(target, 0.9, observation) is None

    # After 30m, the same strongly matched correction is inside the 4.6m limit.
    matcher._repository.return_viz_state['distance_since_anchor_m'] = 30.0
    assert matcher._build_pose_correction(target, 0.9, observation) is not None


def test_gate_ring_is_closed_and_uses_requested_radius() -> None:
    event = SimpleNamespace(pose=Pose3D(2.0, -1.0, 0.5))
    marker = EventRepository._gate_ring(
        event=event,
        now=Time(),
        marker_id=7,
        namespace='test_gate',
        radius_m=1.25,
        color=(1.0, 0.5, 0.0),
        alpha=0.9,
        z_offset_m=0.05,
    )
    assert len(marker.points) == 65
    assert math.isclose(marker.points[0].x, marker.points[-1].x)
    assert math.isclose(marker.points[0].y, marker.points[-1].y, abs_tol=1e-12)
    assert marker.points[0].x == 3.25
