import math

import numpy as np

from radical_event_navigation_system.core.arm_direction_estimator import (
    estimate_arm_direction,
)


def test_event_arm_fit_uses_full_skeleton_segment() -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [0.4, 0.28],
            [0.8, 0.25],
            [1.2, 0.15],
            [1.6, 0.03],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )

    estimate = estimate_arm_direction(
        points,
        np.zeros(2, dtype=np.float32),
        min_distance_m=0.35,
        max_distance_m=2.0,
        min_span_m=0.8,
        fallback_angle_rad=math.radians(35.0),
    )

    assert abs(estimate.angle_rad) < math.radians(10.0)
    assert estimate.confidence > 0.9
