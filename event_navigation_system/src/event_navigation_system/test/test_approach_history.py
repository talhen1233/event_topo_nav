import math

import numpy as np

from radical_event_navigation_system.core.approach_history import ApproachHistory


def test_entry_arm_uses_pre_turn_trajectory() -> None:
    history = ApproachHistory(
        max_distance_m=3.0,
        sample_distance_m=1.5,
        min_sample_spacing_m=0.05,
        max_arm_error_rad=math.radians(35.0),
    )
    history.update(np.array([0.0, -2.0], dtype=np.float32))
    history.update(np.array([0.0, -1.5], dtype=np.float32))
    history.update(np.array([0.0, -0.5], dtype=np.float32))
    history.update(np.array([-0.3, 0.0], dtype=np.float32))
    arms = [-0.5 * math.pi, 0.0, 0.5 * math.pi, math.pi]

    entry = history.estimate_entry_arm(
        np.zeros(2, dtype=np.float32),
        arms,
    )

    assert entry is not None
    assert math.isclose(entry, -0.5 * math.pi, abs_tol=1e-6)


def test_entry_arm_requires_meaningful_historical_displacement() -> None:
    history = ApproachHistory(
        max_distance_m=3.0,
        sample_distance_m=1.5,
        min_sample_spacing_m=0.05,
        max_arm_error_rad=math.radians(35.0),
    )
    history.update(np.array([0.1, 0.0], dtype=np.float32))

    assert history.estimate_entry_arm(
        np.zeros(2, dtype=np.float32),
        [0.0, math.pi],
    ) is None


def test_entry_history_is_not_evicted_during_slow_approach() -> None:
    history = ApproachHistory(
        max_distance_m=3.0,
        sample_distance_m=1.5,
        min_sample_spacing_m=0.05,
        max_arm_error_rad=math.radians(35.0),
    )
    for distance_m in np.linspace(2.0, 0.0, 41):
        history.update(
            np.array([-distance_m, 0.0], dtype=np.float32)
        )

    entry = history.estimate_entry_arm(
        np.zeros(2, dtype=np.float32),
        [0.0, math.pi],
    )

    assert entry is not None
    assert math.isclose(abs(entry), math.pi, abs_tol=1e-6)


def test_entry_history_rejects_ambiguous_arm_alignment() -> None:
    history = ApproachHistory(
        max_distance_m=3.0,
        sample_distance_m=1.5,
        min_sample_spacing_m=0.05,
        max_arm_error_rad=math.radians(20.0),
    )
    history.update(np.array([-1.0, -1.0], dtype=np.float32))

    assert history.estimate_entry_arm(
        np.zeros(2, dtype=np.float32),
        [math.pi, -0.5 * math.pi],
    ) is None
