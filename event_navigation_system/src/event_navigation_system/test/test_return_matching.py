"""Tests for nearby-event handover and guarded pose correction."""

import math

import numpy as np

from radical_event_navigation_system.core.return_matching import (
    HandoverRegion,
    correction_covariance_xy,
    estimate_pose_correction,
    evaluate_geometric_handover,
    event_quality_allows_correction,
    pose_correction_translation_limit,
)
from radical_event_navigation_system.core.route_progress import (
    LocalizationMode,
    RouteProgressAccumulator,
    VarianceRates,
)


def _handover(current_x: float, *, route_gap: float = 4.0):
    return evaluate_geometric_handover(
        current_xy=(current_x, 0.0),
        from_xy=(0.0, 0.0),
        target_xy=(4.0, 0.0),
        expected_route_gap_m=route_gap,
        odom_sigma_xy_m=0.1,
        max_distance_m=5.0,
        margin_base_m=0.2,
        margin_sigma_multiplier=1.0,
        margin_min_m=0.1,
        margin_max_fraction=0.4,
    )


def test_handover_has_three_regions_for_close_events() -> None:
    assert _handover(0.5).region == HandoverRegion.FROM_ONLY
    assert _handover(2.0).region == HandoverRegion.COMPETE
    assert _handover(3.5).region == HandoverRegion.TARGET_ONLY


def test_handover_is_inactive_for_long_route_gap() -> None:
    decision = _handover(0.5, route_gap=8.0)
    assert decision.region == HandoverRegion.INACTIVE
    assert decision.reason == "route_gap_too_far"


def test_handover_requires_a_measured_route_gap() -> None:
    decision = evaluate_geometric_handover(
        current_xy=(1.0, 0.0),
        from_xy=(0.0, 0.0),
        target_xy=(4.0, 0.0),
        expected_route_gap_m=None,
        odom_sigma_xy_m=0.1,
        max_distance_m=5.0,
        margin_base_m=0.2,
        margin_sigma_multiplier=1.0,
        margin_min_m=0.1,
        margin_max_fraction=0.4,
    )
    assert decision.region == HandoverRegion.INACTIVE
    assert decision.reason == "route_gap_unavailable"


def test_handover_margin_is_capped_by_pair_fraction() -> None:
    decision = evaluate_geometric_handover(
        current_xy=(2.0, 0.0),
        from_xy=(0.0, 0.0),
        target_xy=(4.0, 0.0),
        expected_route_gap_m=4.0,
        odom_sigma_xy_m=10.0,
        max_distance_m=5.0,
        margin_base_m=0.2,
        margin_sigma_multiplier=1.0,
        margin_min_m=0.1,
        margin_max_fraction=0.4,
    )
    assert math.isclose(decision.margin_m, 1.6)


def test_route_window_and_handover_protect_close_event_transition() -> None:
    rates = {
        mode: VarianceRates(0.0, 0.0)
        for mode in LocalizationMode
    }
    progress = RouteProgressAccumulator(rates)
    window = progress.window(
        expected_distance_m=4.0,
        event_sigma_m=1.5,
        sigma_multiplier=3.0,
    )

    assert window.contains(0.0)
    assert _handover(0.1).region == HandoverRegion.FROM_ONLY
    assert _handover(3.9).region == HandoverRegion.TARGET_ONLY


def test_pose_correction_uses_relative_junction_centers() -> None:
    estimate = estimate_pose_correction(
        robot_xy=(10.0, 2.0),
        stored_center_xy=(8.0, 1.0),
        observed_center_xy=(9.5, 1.5),
        max_translation_m=2.0,
    )
    assert estimate.accepted
    assert estimate.corrected_xy == (8.5, 1.5)
    assert math.isclose(estimate.translation_norm_m, math.sqrt(2.5))


def test_pose_correction_rejects_large_translation_instead_of_clamping() -> None:
    estimate = estimate_pose_correction(
        robot_xy=(0.0, 0.0),
        stored_center_xy=(4.0, 0.0),
        observed_center_xy=(0.0, 0.0),
        max_translation_m=2.0,
    )
    assert not estimate.accepted
    assert estimate.reason == "translation_limit"


def test_pose_correction_limit_grows_with_distance_since_anchor() -> None:
    short_limit = pose_correction_translation_limit(
        distance_since_anchor_m=5.0,
        base_m=1.0,
        growth_per_meter=0.12,
        hard_cap_m=8.0,
    )
    long_limit = pose_correction_translation_limit(
        distance_since_anchor_m=30.0,
        base_m=1.0,
        growth_per_meter=0.12,
        hard_cap_m=8.0,
    )
    assert math.isclose(short_limit, 1.6)
    assert math.isclose(long_limit, 4.6)


def test_pose_correction_limit_never_exceeds_catastrophic_cap() -> None:
    limit = pose_correction_translation_limit(
        distance_since_anchor_m=1000.0,
        base_m=1.0,
        growth_per_meter=0.12,
        hard_cap_m=8.0,
    )
    assert limit == 8.0


def test_weak_diffuse_event_cannot_correct_pose() -> None:
    assert not event_quality_allows_correction(
        event_confidence=0.1,
        covariance_trace_xy=1.2,
        confidence_min=0.2,
        covariance_trace_max_xy=0.5,
    )
    assert not event_quality_allows_correction(
        event_confidence=0.8,
        covariance_trace_xy=1.2,
        confidence_min=0.2,
        covariance_trace_max_xy=0.5,
    )
    assert not event_quality_allows_correction(
        event_confidence=0.1,
        covariance_trace_xy=0.1,
        confidence_min=0.2,
        covariance_trace_max_xy=0.5,
    )
    assert event_quality_allows_correction(
        event_confidence=0.8,
        covariance_trace_xy=0.1,
        confidence_min=0.2,
        covariance_trace_max_xy=0.5,
    )


def test_pose_covariance_is_conservative_near_match_threshold() -> None:
    event_covariance = np.eye(2) * 0.25
    marginal = correction_covariance_xy(
        event_covariance_xy=event_covariance,
        detection_noise_m=0.3,
        match_score=0.85,
        acceptance_threshold=0.8,
        covariance_scale=1.0,
        minimum_variance_m2=0.0225,
    )
    strong = correction_covariance_xy(
        event_covariance_xy=event_covariance,
        detection_noise_m=0.3,
        match_score=1.0,
        acceptance_threshold=0.8,
        covariance_scale=1.0,
        minimum_variance_m2=0.0225,
    )
    assert marginal[0, 0] > strong[0, 0]
    assert strong[0, 0] >= 0.25 + 0.3 ** 2
