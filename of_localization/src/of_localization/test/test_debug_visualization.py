import numpy as np

from of_localization.debug_visualization import (
    FeatureDepthDebug,
    VelocityComparisonDebug,
    render_depth_debug_canvas,
)


def test_render_depth_debug_canvas_returns_wide_bgr_image():
    gray = np.full((120, 160), 40, dtype=np.uint8)
    grid = np.full((8, 8), np.nan, dtype=np.float32)
    grid[3, 3] = 0.6
    grid[4, 4] = 0.8
    grid_centers = np.zeros((8, 8, 2), dtype=np.float32)
    for row in range(8):
        for col in range(8):
            grid_centers[row, col] = np.asarray([30 + col * 10, 20 + row * 10], dtype=np.float32)

    feature_debug = FeatureDepthDebug(
        prev_points_xy=np.asarray([[50.0, 50.0], [90.0, 70.0]], dtype=np.float32),
        next_points_xy=np.asarray([[56.0, 54.0], [96.0, 74.0]], dtype=np.float32),
        valid_mask=np.asarray([True, False]),
        selected_mask=np.asarray([True, False]),
        matched_depths_m=np.asarray([0.62, np.nan], dtype=np.float32),
        matched_rows=np.asarray([3, -1], dtype=np.int32),
        matched_cols=np.asarray([3, -1], dtype=np.int32),
        pointcloud_age_sec=0.03,
        valid_ratio=0.5,
    )
    comparison = VelocityComparisonDebug(
        local_depth_velocity_xy=np.asarray([0.2, 0.1], dtype=np.float32),
        single_height_velocity_xy=np.asarray([0.12, 0.05], dtype=np.float32),
        scalar_height_m=0.75,
        single_height_m=0.68,
    )

    canvas = render_depth_debug_canvas(
        gray,
        (20, 140, 15, 105),
        grid,
        grid_centers,
        feature_debug,
        comparison,
    )

    assert canvas.ndim == 3
    assert canvas.shape[0] > gray.shape[0]
    assert canvas.shape[1] > gray.shape[1]
    assert canvas.shape[2] == 3
    assert np.any(canvas[:, :, 0] != canvas[:, :, 1])
