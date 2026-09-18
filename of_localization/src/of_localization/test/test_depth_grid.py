import numpy as np
import pytest

from of_localization.depth_grid import BottomDepthGrid, DepthGridConfig


def _make_depth_points(config: DepthGridConfig) -> tuple[np.ndarray, np.ndarray]:
    half_hfov = 0.5 * config.horizontal_fov_rad
    half_vfov = 0.5 * config.vertical_fov_rad
    cols = np.linspace(-half_hfov, half_hfov, config.grid_width, dtype=np.float32)
    rows = np.linspace(-half_vfov, half_vfov, config.grid_height, dtype=np.float32)

    points = []
    expected_depths = np.zeros((config.grid_height, config.grid_width), dtype=np.float32)
    for row_idx, angle_y in enumerate(rows):
        for col_idx, angle_x in enumerate(cols):
            depth = 0.5 + 0.01 * row_idx + 0.02 * col_idx
            expected_depths[row_idx, col_idx] = depth
            points.append(
                (
                    depth,
                    -depth * np.tan(angle_x),
                    -depth * np.tan(angle_y),
                )
            )
    return np.asarray(points, dtype=np.float32), expected_depths


def test_depth_grid_sampling_matches_expected_cell_depths():
    config = DepthGridConfig(timeout_sec=0.2)
    depth_grid = BottomDepthGrid(config)
    points_xyz, expected_depths = _make_depth_points(config)
    depth_grid.update_from_points(points_xyz, stamp_s=10.0)

    fx = 120.0
    fy = 120.0
    cx = 80.0
    cy = 60.0
    sample_cells = [(0, 0), (3, 4), (7, 7)]
    image_points = []
    expected = []
    rows = np.linspace(-0.5 * config.vertical_fov_rad, 0.5 * config.vertical_fov_rad, config.grid_height)
    cols = np.linspace(-0.5 * config.horizontal_fov_rad, 0.5 * config.horizontal_fov_rad, config.grid_width)

    for row_idx, col_idx in sample_cells:
        image_points.append(
            (
                cx + fx * np.tan(cols[col_idx]),
                cy + fy * np.tan(rows[row_idx]),
            )
        )
        expected.append(expected_depths[row_idx, col_idx])

    samples = depth_grid.sample_depths(
        np.asarray(image_points, dtype=np.float32),
        fx,
        fy,
        cx,
        cy,
        reference_stamp_s=10.05,
    )

    np.testing.assert_allclose(samples.depths_m[samples.valid_mask], np.asarray(expected), atol=1e-3)
    assert samples.valid_mask.tolist() == [True, True, True]
    assert samples.overlap_mask.tolist() == [True, True, True]
    assert samples.age_sec == pytest.approx(0.05)


def test_depth_grid_reports_freshness_and_invalid_outside_overlap():
    config = DepthGridConfig(timeout_sec=0.1)
    depth_grid = BottomDepthGrid(config)
    points_xyz, _ = _make_depth_points(config)
    depth_grid.update_from_points(points_xyz, stamp_s=2.0)

    assert depth_grid.is_fresh(2.05)
    assert not depth_grid.is_fresh(2.25)

    samples = depth_grid.sample_depths(
        np.asarray([[500.0, 500.0]], dtype=np.float32),
        fx=120.0,
        fy=120.0,
        cx=80.0,
        cy=60.0,
        reference_stamp_s=2.05,
    )

    assert not samples.valid_mask[0]
    assert not samples.overlap_mask[0]
    assert np.isnan(samples.depths_m[0])


def test_depth_grid_projects_cell_centers_and_nearest_indices():
    config = DepthGridConfig(timeout_sec=0.1)
    depth_grid = BottomDepthGrid(config)
    fx = 120.0
    fy = 120.0
    cx = 80.0
    cy = 60.0

    centers = depth_grid.grid_cell_centers_image(fx, fy, cx, cy)
    assert centers.shape == (config.grid_height, config.grid_width, 2)

    test_points = np.asarray(
        [
            centers[0, 0],
            centers[3, 4],
            [1000.0, 1000.0],
        ],
        dtype=np.float32,
    )
    rows, cols, overlap_mask = depth_grid.nearest_cell_indices(test_points, fx, fy, cx, cy)

    assert overlap_mask.tolist() == [True, True, False]
    assert rows[:2].tolist() == [0, 3]
    assert cols[:2].tolist() == [0, 4]
    assert rows[2] == -1
    assert cols[2] == -1
