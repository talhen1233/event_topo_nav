from __future__ import annotations

import math

import numpy as np

_EPS = 1e-6


def quaternion_to_euler_xyz(
    qx: float,
    qy: float,
    qz: float,
    qw: float,
) -> tuple[float, float, float]:
    """Convert quaternion orientation to XYZ Euler angles."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rotation_matrix_world_from_body(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """Return the XYZ body-to-world rotation matrix."""
    sr = math.sin(roll)
    cr = math.cos(roll)
    sp = math.sin(pitch)
    cp = math.cos(pitch)
    sy = math.sin(yaw)
    cy = math.cos(yaw)
    return np.asarray(
        [
            [cp * cy, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [cp * sy, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def gravity_direction_sensor_frame(
    roll: float,
    pitch: float,
    yaw: float,
    sensor_from_body: np.ndarray,
) -> np.ndarray:
    """Return the world-down direction expressed in the sensor frame."""
    world_down = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    rot_wb = rotation_matrix_world_from_body(roll, pitch, yaw)
    down_body = rot_wb.T @ world_down
    down_sensor = np.asarray(sensor_from_body, dtype=np.float32) @ down_body
    norm = float(np.linalg.norm(down_sensor))
    if norm <= _EPS:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return (down_sensor / norm).astype(np.float32)


def feature_ray_directions_sensor(
    image_points_xy: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Return unnormalized sensor-frame rays for image points."""
    points = np.asarray(image_points_xy, dtype=np.float32).reshape(-1, 2)
    x_norm = (points[:, 0] - float(cx)) / float(fx)
    y_norm = (points[:, 1] - float(cy)) / float(fy)
    return np.column_stack(
        (
            np.ones(points.shape[0], dtype=np.float32),
            -x_norm.astype(np.float32),
            -y_norm.astype(np.float32),
        )
    ).astype(np.float32)


def heights_from_sensor_points(
    points_xyz: np.ndarray,
    gravity_dir_sensor: np.ndarray,
) -> np.ndarray:
    """Project sensor-frame points onto the gravity direction."""
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    gravity_dir = np.asarray(gravity_dir_sensor, dtype=np.float32).reshape(3)
    return (points @ gravity_dir).astype(np.float32)


def axial_depths_from_heights(
    heights_m: np.ndarray,
    image_points_xy: np.ndarray,
    gravity_dir_sensor: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_projection: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert gravity-aligned heights to sensor axial depths for each feature ray."""
    heights = np.asarray(heights_m, dtype=np.float32).reshape(-1)
    rays = feature_ray_directions_sensor(image_points_xy, fx, fy, cx, cy)
    projections = (rays @ np.asarray(gravity_dir_sensor, dtype=np.float32).reshape(3)).astype(np.float32)
    valid_mask = np.isfinite(heights) & np.isfinite(projections) & (heights > 0.0) & (projections > float(min_projection))
    depths = np.full(heights.shape[0], np.nan, dtype=np.float32)
    if np.any(valid_mask):
        depths[valid_mask] = heights[valid_mask] / projections[valid_mask]
    return depths, valid_mask, projections
