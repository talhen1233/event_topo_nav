from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

_EPS = 1e-6


def point_cloud_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Decode XYZ points from a `PointCloud2` message."""
    gen = point_cloud2.read_points(
        msg,
        field_names=("x", "y", "z"),
        skip_nans=True,
    )
    rec = np.fromiter(
        gen,
        dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)],
        count=-1,
    )
    if rec.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return rec.view(np.float32).reshape(-1, 3)


def stamp_to_seconds(stamp) -> Optional[float]:
    """Convert a ROS stamp to seconds."""
    if stamp is None:
        return None
    sec = float(getattr(stamp, "sec", 0.0))
    nanosec = float(getattr(stamp, "nanosec", 0.0))
    stamp_s = sec + 1e-9 * nanosec
    if stamp_s <= 0.0:
        return None
    return stamp_s


@dataclass(frozen=True)
class DepthGridConfig:
    """Static configuration for the bottom depth grid."""

    grid_width: int = 8
    grid_height: int = 8
    horizontal_fov_rad: float = 0.7854
    vertical_fov_rad: float = 0.7854
    min_range_m: float = 0.02
    max_range_m: float = 10.0
    timeout_sec: float = 0.15


@dataclass(frozen=True)
class DepthSamples:
    """Sampled local depths under tracked image features."""

    depths_m: np.ndarray
    valid_mask: np.ndarray
    overlap_mask: np.ndarray
    age_sec: float


class BottomDepthGrid:
    """Cache and sample a tiny angular depth grid."""

    def __init__(self, config: DepthGridConfig) -> None:
        self.config = config
        self._grid = np.full(
            (config.grid_height, config.grid_width),
            np.nan,
            dtype=np.float32,
        )
        self._stamp_s: Optional[float] = None
        self._half_hfov = 0.5 * float(config.horizontal_fov_rad)
        self._half_vfov = 0.5 * float(config.vertical_fov_rad)
        self._step_x = float(config.horizontal_fov_rad) / max(1, config.grid_width - 1)
        self._step_y = float(config.vertical_fov_rad) / max(1, config.grid_height - 1)

    @property
    def stamp_s(self) -> Optional[float]:
        """Return the latest cloud timestamp in seconds."""
        return self._stamp_s

    @property
    def grid(self) -> np.ndarray:
        """Return the latest cached depth grid."""
        return self._grid

    def update_from_points(
        self,
        points_xyz: np.ndarray,
        stamp_s: Optional[float] = None,
    ) -> None:
        """Refresh the cached grid from raw XYZ points."""
        points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        self.update_from_scalar_values(points, points[:, 0], stamp_s)

    def update_from_scalar_values(
        self,
        points_xyz: np.ndarray,
        scalar_values: np.ndarray,
        stamp_s: Optional[float] = None,
    ) -> None:
        """Refresh the cached grid from raw XYZ points and per-point scalar values."""
        self._grid.fill(np.nan)
        self._stamp_s = stamp_s
        if points_xyz.size == 0:
            return

        points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
        values = np.asarray(scalar_values, dtype=np.float32).reshape(-1)
        if values.shape[0] != points.shape[0]:
            raise ValueError("scalar_values must match the number of points")

        axial_depth = points[:, 0]
        ranges = np.linalg.norm(points, axis=1)
        finite_mask = np.isfinite(points).all(axis=1)
        valid_mask = (
            finite_mask
            & (axial_depth > 0.0)
            & np.isfinite(values)
            & (ranges >= float(self.config.min_range_m))
            & (ranges <= float(self.config.max_range_m))
        )
        if not np.any(valid_mask):
            return

        valid_points = points[valid_mask]
        valid_values = values[valid_mask]

        # Link-frame X is the optical axis (down).
        angle_x = np.arctan2(-valid_points[:, 1], valid_points[:, 0])
        angle_y = np.arctan2(-valid_points[:, 2], valid_points[:, 0])

        col_idx = np.rint((angle_x + self._half_hfov) / self._step_x).astype(np.int32)
        row_idx = np.rint((angle_y + self._half_vfov) / self._step_y).astype(np.int32)
        inside_mask = (
            (col_idx >= 0)
            & (col_idx < self.config.grid_width)
            & (row_idx >= 0)
            & (row_idx < self.config.grid_height)
        )
        if not np.any(inside_mask):
            return

        row_idx = row_idx[inside_mask]
        col_idx = col_idx[inside_mask]
        valid_values = valid_values[inside_mask]

        flat_idx = row_idx * self.config.grid_width + col_idx
        sums = np.bincount(
            flat_idx,
            weights=valid_values.astype(np.float64),
            minlength=self.config.grid_width * self.config.grid_height,
        )
        counts = np.bincount(
            flat_idx,
            minlength=self.config.grid_width * self.config.grid_height,
        )

        flat_grid = np.full(self.config.grid_width * self.config.grid_height, np.nan, dtype=np.float32)
        count_mask = counts > 0
        flat_grid[count_mask] = (sums[count_mask] / counts[count_mask]).astype(np.float32)
        self._grid = flat_grid.reshape(self.config.grid_height, self.config.grid_width)

    def age_seconds(self, reference_stamp_s: Optional[float]) -> float:
        """Return cloud age relative to a reference timestamp."""
        if self._stamp_s is None or reference_stamp_s is None:
            return float("inf")
        return abs(float(reference_stamp_s) - float(self._stamp_s))

    def is_fresh(self, reference_stamp_s: Optional[float]) -> bool:
        """Report whether the cached cloud is fresh enough."""
        return self.age_seconds(reference_stamp_s) <= float(self.config.timeout_sec)

    def overlap_bounds(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int]:
        """Return the pixel bounding box covered by the depth sensor."""
        if fx <= 0.0 or fy <= 0.0:
            return 0, image_width, 0, image_height

        x_extent = fx * np.tan(self._half_hfov)
        y_extent = fy * np.tan(self._half_vfov)
        u_min = int(np.floor(max(0.0, cx - x_extent)))
        u_max = int(np.ceil(min(float(image_width), cx + x_extent)))
        v_min = int(np.floor(max(0.0, cy - y_extent)))
        v_max = int(np.ceil(min(float(image_height), cy + y_extent)))
        return u_min, u_max, v_min, v_max

    def grid_cell_centers_image(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray:
        """Project the center of each depth cell into the image."""
        col_angles = np.linspace(
            -self._half_hfov,
            self._half_hfov,
            self.config.grid_width,
            dtype=np.float32,
        )
        row_angles = np.linspace(
            -self._half_vfov,
            self._half_vfov,
            self.config.grid_height,
            dtype=np.float32,
        )
        u = float(cx) + float(fx) * np.tan(col_angles)
        v = float(cy) + float(fy) * np.tan(row_angles)
        uu, vv = np.meshgrid(u, v)
        return np.stack((uu, vv), axis=-1).astype(np.float32)

    def nearest_cell_indices(
        self,
        image_points_xy: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the nearest depth-cell row/col for each image point."""
        points = np.asarray(image_points_xy, dtype=np.float32).reshape(-1, 2)
        count = points.shape[0]
        rows = np.full(count, -1, dtype=np.int32)
        cols = np.full(count, -1, dtype=np.int32)
        overlap_mask = np.zeros(count, dtype=bool)
        if count == 0 or fx <= 0.0 or fy <= 0.0:
            return rows, cols, overlap_mask

        angle_x = np.arctan((points[:, 0] - float(cx)) / float(fx))
        angle_y = np.arctan((points[:, 1] - float(cy)) / float(fy))
        overlap_mask = (
            (angle_x >= (-self._half_hfov - _EPS))
            & (angle_x <= (self._half_hfov + _EPS))
            & (angle_y >= (-self._half_vfov - _EPS))
            & (angle_y <= (self._half_vfov + _EPS))
        )
        if not np.any(overlap_mask):
            return rows, cols, overlap_mask

        col = (angle_x[overlap_mask] + self._half_hfov) / self._step_x
        row = (angle_y[overlap_mask] + self._half_vfov) / self._step_y
        cols[overlap_mask] = np.clip(
            np.rint(col).astype(np.int32),
            0,
            self.config.grid_width - 1,
        )
        rows[overlap_mask] = np.clip(
            np.rint(row).astype(np.int32),
            0,
            self.config.grid_height - 1,
        )
        return rows, cols, overlap_mask

    def sample_depths(
        self,
        image_points_xy: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        reference_stamp_s: Optional[float],
    ) -> DepthSamples:
        """Sample depth under image points using bilinear interpolation."""
        points = np.asarray(image_points_xy, dtype=np.float32).reshape(-1, 2)
        count = points.shape[0]
        depths = np.full(count, np.nan, dtype=np.float32)
        valid_mask = np.zeros(count, dtype=bool)
        overlap_mask = np.zeros(count, dtype=bool)
        age_sec = self.age_seconds(reference_stamp_s)

        if count == 0 or fx <= 0.0 or fy <= 0.0 or not np.isfinite(self._grid).any():
            return DepthSamples(depths, valid_mask, overlap_mask, age_sec)

        angle_x = np.arctan((points[:, 0] - float(cx)) / float(fx))
        angle_y = np.arctan((points[:, 1] - float(cy)) / float(fy))
        overlap_mask = (
            (angle_x >= (-self._half_hfov - _EPS))
            & (angle_x <= (self._half_hfov + _EPS))
            & (angle_y >= (-self._half_vfov - _EPS))
            & (angle_y <= (self._half_vfov + _EPS))
        )
        if not np.any(overlap_mask):
            return DepthSamples(depths, valid_mask, overlap_mask, age_sec)

        col = (angle_x[overlap_mask] + self._half_hfov) / self._step_x
        row = (angle_y[overlap_mask] + self._half_vfov) / self._step_y

        col0 = np.floor(col).astype(np.int32)
        row0 = np.floor(row).astype(np.int32)
        col1 = np.clip(col0 + 1, 0, self.config.grid_width - 1)
        row1 = np.clip(row0 + 1, 0, self.config.grid_height - 1)
        col0 = np.clip(col0, 0, self.config.grid_width - 1)
        row0 = np.clip(row0, 0, self.config.grid_height - 1)

        wx = (col - col0).astype(np.float32)
        wy = (row - row0).astype(np.float32)

        q00 = self._grid[row0, col0]
        q01 = self._grid[row0, col1]
        q10 = self._grid[row1, col0]
        q11 = self._grid[row1, col1]

        weights = np.stack(
            (
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
            ),
            axis=1,
        ).astype(np.float32)
        values = np.stack((q00, q01, q10, q11), axis=1).astype(np.float32)
        finite = np.isfinite(values)
        weighted_values = np.where(finite, values * weights, 0.0)
        weight_sum = np.where(finite, weights, 0.0).sum(axis=1)
        interp_depths = weighted_values.sum(axis=1) / np.maximum(weight_sum, _EPS)
        interp_valid = weight_sum > _EPS

        overlap_idx = np.flatnonzero(overlap_mask)
        depths[overlap_idx[interp_valid]] = interp_depths[interp_valid]
        valid_mask[overlap_idx[interp_valid]] = True
        return DepthSamples(depths, valid_mask, overlap_mask, age_sec)

    def sample_values(
        self,
        image_points_xy: np.ndarray,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        reference_stamp_s: Optional[float],
    ) -> DepthSamples:
        """Sample the cached scalar grid under image points."""
        return self.sample_depths(
            image_points_xy,
            fx,
            fy,
            cx,
            cy,
            reference_stamp_s,
        )
