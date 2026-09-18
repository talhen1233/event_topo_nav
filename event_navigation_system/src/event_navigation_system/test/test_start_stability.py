import numpy as np

from radical_event_navigation_system.core.start_stability import (
    StartStabilityParams,
    StartStabilityTracker,
)


def _tracker() -> StartStabilityTracker:
    return StartStabilityTracker(
        StartStabilityParams(
            min_frames=3,
            max_normalized_change=0.05,
            max_speed_mps=0.05,
        )
    )


def test_requires_consecutive_frames() -> None:
    tracker = _tracker()
    profile = np.full(64, 2.0, dtype=np.float32)

    assert not tracker.update(profile, 0.0, 0.0).stable
    assert not tracker.update(profile, 0.4, 0.0).stable
    status = tracker.update(profile, 0.8, 0.0)

    assert status.stable
    assert status.consecutive_frames == 3
    np.testing.assert_allclose(status.representative_profile, profile)


def test_geometry_change_restarts_streak() -> None:
    tracker = _tracker()
    baseline = np.full(64, 2.0, dtype=np.float32)
    changed = np.full(64, 2.4, dtype=np.float32)

    tracker.update(baseline, 0.0, 0.0)
    tracker.update(baseline, 0.6, 0.0)
    status = tracker.update(changed, 1.0, 0.0)

    assert not status.stable
    assert status.consecutive_frames == 1
    assert status.normalized_change is not None
    assert status.normalized_change > 0.05


def test_motion_clears_streak() -> None:
    tracker = _tracker()
    profile = np.full(64, 2.0, dtype=np.float32)

    tracker.update(profile, 0.0, 0.0)
    assert tracker.update(profile, 0.5, 0.1).consecutive_frames == 0


def test_no_support_percentage_is_required() -> None:
    tracker = _tracker()
    # The tracker consumes the complete filtered profile and intentionally has
    # no observed-bin quota that could deadlock a sparse four-sensor rig.
    profile = np.full(64, 4.0, dtype=np.float32)
    tracker.update(profile, 0.0, 0.0)
    tracker.update(profile, 0.5, 0.0)

    assert tracker.update(profile, 1.0, 0.0).stable
