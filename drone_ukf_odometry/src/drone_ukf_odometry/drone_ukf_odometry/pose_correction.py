"""Pure validation helpers for external XY pose corrections."""

import numpy as np


def innovation_mahalanobis_squared(
    innovation_xy: np.ndarray,
    state_covariance_xy: np.ndarray,
    measurement_covariance_xy: np.ndarray,
) -> float:
    """Return the two-dimensional normalized pose-correction innovation."""
    innovation = np.asarray(innovation_xy, dtype=float).reshape(2)
    state_cov = np.asarray(state_covariance_xy, dtype=float).reshape(2, 2)
    measurement_cov = np.asarray(
        measurement_covariance_xy, dtype=float
    ).reshape(2, 2)
    innovation_cov = state_cov + measurement_cov
    innovation_cov = 0.5 * (innovation_cov + innovation_cov.T)
    try:
        solved = np.linalg.solve(
            innovation_cov + np.eye(2, dtype=float) * 1e-9,
            innovation,
        )
    except np.linalg.LinAlgError:
        return float("inf")
    return float(innovation.T @ solved)
