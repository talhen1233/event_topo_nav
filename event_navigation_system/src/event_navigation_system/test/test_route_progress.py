"""Tests for one-dimensional return-progress uncertainty."""

import math

from radical_event_navigation_system.core.route_progress import (
    CorrectionAwareDistanceTracker,
    DistanceDeadbandTracker,
    LocalizationMode,
    RouteProgressAccumulator,
    VarianceRates,
)


def _accumulator() -> RouteProgressAccumulator:
    return RouteProgressAccumulator({
        LocalizationMode.NORMAL: VarianceRates(0.001, 0.00001),
        LocalizationMode.DEGRADED: VarianceRates(0.005, 0.0001),
        LocalizationMode.OF_DEGRADED: VarianceRates(0.02, 0.001),
    })


def test_variance_integrates_distance_time_and_mode() -> None:
    acc = _accumulator()
    acc.update(2.0, 1.0, LocalizationMode.NORMAL)
    acc.update(3.0, 2.0, LocalizationMode.OF_DEGRADED)

    expected = 0.001 * 2.0 + 0.00001 * 1.0
    expected += 0.02 * 3.0 + 0.001 * 2.0
    assert math.isclose(acc.progress_m, 5.0)
    assert math.isclose(acc.elapsed_s, 3.0)
    assert math.isclose(acc.integration_variance_m2, expected)


def test_window_combines_event_and_integration_variance() -> None:
    acc = _accumulator()
    acc.update(10.0, 5.0, LocalizationMode.DEGRADED)
    window = acc.window(
        expected_distance_m=20.0,
        event_sigma_m=0.3,
        sigma_multiplier=3.0,
    )

    expected_sigma = math.sqrt(0.3 ** 2 + 0.005 * 10.0 + 0.0001 * 5.0)
    assert math.isclose(window.sigma_m, expected_sigma)
    assert math.isclose(window.lower_m, 20.0 - 3.0 * expected_sigma)
    assert math.isclose(window.upper_m, 20.0 + 3.0 * expected_sigma)


def test_window_honors_minimum_half_width() -> None:
    acc = _accumulator()
    window = acc.window(
        expected_distance_m=10.0,
        event_sigma_m=0.3,
        sigma_multiplier=3.0,
        min_half_width_m=2.0,
    )

    assert math.isclose(window.lower_m, 8.0)
    assert math.isclose(window.upper_m, 12.0)


def test_reset_establishes_a_new_anchor() -> None:
    acc = _accumulator()
    acc.update(4.0, 3.0, LocalizationMode.OF_DEGRADED)
    acc.reset()

    assert acc.progress_m == 0.0
    assert acc.elapsed_s == 0.0
    assert acc.integration_variance_m2 == 0.0


def test_deadband_rejects_stationary_jitter() -> None:
    tracker = DistanceDeadbandTracker(deadband_m=0.03, max_step_m=0.75)
    tracker.update((0.0, 0.0, 0.0))

    for offset in (0.005, -0.004, 0.008, -0.006, 0.002):
        assert tracker.update((offset, 0.0, 0.0)).distance_m == 0.0


def test_deadband_preserves_accumulated_slow_motion() -> None:
    tracker = DistanceDeadbandTracker(deadband_m=0.03, max_step_m=0.75)
    tracker.update((0.0, 0.0, 0.0))

    assert tracker.update((0.01, 0.0, 0.0)).distance_m == 0.0
    assert tracker.update((0.02, 0.0, 0.0)).distance_m == 0.0
    accepted = tracker.update((0.031, 0.0, 0.0))
    assert math.isclose(accepted.distance_m, 0.031)


def test_deadband_rebases_pose_correction_jump() -> None:
    tracker = DistanceDeadbandTracker(deadband_m=0.03, max_step_m=0.75)
    tracker.update((0.0, 0.0, 0.0))
    tracker.rebase((1.5, -0.5, 0.0))

    assert tracker.update((1.51, -0.5, 0.0)).distance_m == 0.0
    accepted = tracker.update((1.54, -0.5, 0.0))
    assert math.isclose(accepted.distance_m, 0.04)


def test_deadband_rejects_unannounced_discontinuity() -> None:
    tracker = DistanceDeadbandTracker(deadband_m=0.03, max_step_m=0.75)
    tracker.update((0.0, 0.0, 0.0))

    jump = tracker.update((2.0, 0.0, 0.0))
    assert jump.distance_m == 0.0
    assert jump.rebased
    assert jump.reason == "discontinuity"


def _correction_tracker() -> CorrectionAwareDistanceTracker:
    return CorrectionAwareDistanceTracker(
        deadband_m=0.001,
        max_step_m=2.0,
        reorder_window_s=0.1,
    )


def test_correction_ack_before_corrected_odometry_excludes_state_jump() -> None:
    tracker = _correction_tracker()
    total = tracker.update((0.0, 0.0, 0.0), 0.0).distance_m
    total += tracker.update((0.1, 0.0, 0.0), 0.1).distance_m
    tracker.acknowledge_correction(0.15)
    total += tracker.update((0.6, 0.0, 0.0), 0.2).distance_m
    total += tracker.update((0.7, 0.0, 0.0), 0.31).distance_m
    total += tracker.update((0.8, 0.0, 0.0), 0.42).distance_m

    assert math.isclose(total, 0.2)


def test_corrected_odometry_before_ack_is_reordered_and_excludes_jump() -> None:
    tracker = _correction_tracker()
    total = tracker.update((0.0, 0.0, 0.0), 0.0).distance_m
    total += tracker.update((0.1, 0.0, 0.0), 0.1).distance_m
    total += tracker.update((0.6, 0.0, 0.0), 0.2).distance_m
    tracker.acknowledge_correction(0.15)
    total += tracker.update((0.7, 0.0, 0.0), 0.31).distance_m
    total += tracker.update((0.8, 0.0, 0.0), 0.42).distance_m

    assert math.isclose(total, 0.2)
