import numpy as np

from of_localization.flow_stats import (
    compensate_planar_motion,
    compute_flow_inlier_mask,
    covariance2x2,
    lk_error_weights,
    robust_velocity_mask,
)


def test_compute_flow_inlier_mask_rejects_large_outlier():
    flow = np.asarray(
        [
            [1.0, 0.5],
            [1.1, 0.55],
            [0.95, 0.45],
            [8.0, -7.0],
        ],
        dtype=np.float32,
    )

    mask, median_flow = compute_flow_inlier_mask(
        flow,
        distance_threshold_sq=1.0,
        cos_direction_threshold=0.95,
    )

    np.testing.assert_allclose(median_flow, np.asarray([1.05, 0.475], dtype=np.float32), atol=1e-6)
    assert mask.tolist() == [True, True, True, False]


def test_compensate_planar_motion_removes_pure_divergence():
    prev_points = np.asarray(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    flow = 0.2 * prev_points
    result = compensate_planar_motion(
        prev_points,
        flow,
        center_xy=np.zeros(2, dtype=np.float32),
    )

    np.testing.assert_allclose(result.corrected_flow, np.zeros_like(flow), atol=1e-6)
    assert abs(result.yaw_rate) < 1e-6
    assert abs(result.divergence_rate - 0.2) < 1e-6


def test_velocity_stats_helpers_downweight_noisy_tracks():
    velocity_samples = np.asarray(
        [
            [0.3, 0.1],
            [0.31, 0.11],
            [0.29, 0.09],
            [2.0, -1.5],
        ],
        dtype=np.float32,
    )

    keep_mask = robust_velocity_mask(velocity_samples)
    assert keep_mask.tolist() == [True, True, True, False]

    weights = lk_error_weights(np.asarray([0.1, 0.2, 0.15], dtype=np.float32))
    mean, cov = covariance2x2(velocity_samples[keep_mask], weights)

    np.testing.assert_allclose(mean, np.asarray([0.3, 0.1], dtype=np.float32), atol=2e-2)
    assert cov.shape == (2, 2)
    assert np.all(np.diag(cov) >= 0.0)
