import math

import numpy as np

from local_path_planner.scripts.arm_direction_estimator import (
    SmoothedArmDirection,
    estimate_arm_direction,
)


def test_weighted_line_fit_rejects_misleading_local_tangent() -> None:
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


def test_direction_smoothing_respects_refinement_rate() -> None:
    estimate = SmoothedArmDirection(
        math.radians(20.0),
        smoothing_tau_s=0.35,
        max_rate_rad_s=math.radians(25.0),
    )

    first = estimate.update(0.0, confidence=1.0, dt=0.05)

    assert math.isclose(first, math.radians(18.75), abs_tol=1e-6)
    for _ in range(30):
        estimate.update(0.0, confidence=1.0, dt=0.05)
    assert abs(estimate.angle_rad) < math.radians(1.0)
