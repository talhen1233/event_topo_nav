from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class FlowCompensationResult:
    """Corrected optical-flow samples after planar motion compensation."""

    corrected_flow: np.ndarray
    flow_mean: np.ndarray
    flow_covariance: np.ndarray
    yaw_rate: float
    divergence_rate: float


def covariance2x2(
    samples: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a 2D mean and covariance with optional weights."""
    values = np.asarray(samples, dtype=np.float32).reshape(-1, 2)
    if values.shape[0] == 0:
        return np.zeros(2, dtype=np.float32), np.zeros((2, 2), dtype=np.float32)

    if weights is None:
        mean = values.mean(axis=0).astype(np.float32)
        if values.shape[0] < 2:
            return mean, np.zeros((2, 2), dtype=np.float32)
        centered = values - mean
        cov = (centered.T @ centered) / float(values.shape[0] - 1)
        return mean, cov.astype(np.float32)

    sample_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if sample_weights.shape[0] != values.shape[0]:
        raise ValueError("weights and samples must have the same length")

    sample_weights = np.clip(sample_weights, 0.0, None)
    weight_sum = float(sample_weights.sum())
    if weight_sum <= _EPS:
        return covariance2x2(values)

    sample_weights = sample_weights / weight_sum
    mean = (values * sample_weights[:, None]).sum(axis=0).astype(np.float32)
    centered = values - mean
    cov = (centered * sample_weights[:, None]).T @ centered
    return mean, cov.astype(np.float32)


def compute_flow_inlier_mask(
    flow: np.ndarray,
    distance_threshold_sq: float,
    cos_direction_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject large and directionally inconsistent flow outliers."""
    flow_xy = np.asarray(flow, dtype=np.float32).reshape(-1, 2)
    if flow_xy.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.zeros(2, dtype=np.float32)

    median_flow = np.median(flow_xy, axis=0).astype(np.float32)
    diff = flow_xy - median_flow
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    mask_distance = dist_sq < float(distance_threshold_sq)

    median_norm = math.hypot(float(median_flow[0]), float(median_flow[1]))
    if median_norm <= 1e-6:
        return mask_distance, median_flow

    flow_norm_sq = np.einsum("ij,ij->i", flow_xy, flow_xy)
    cos_theta = (flow_xy @ median_flow) / (np.sqrt(flow_norm_sq) * median_norm + _EPS)
    mask_direction = cos_theta > float(cos_direction_threshold)
    return mask_distance & mask_direction, median_flow


def compensate_planar_motion(
    prev_points_xy: np.ndarray,
    flow: np.ndarray,
    center_xy: np.ndarray,
) -> FlowCompensationResult:
    """Remove yaw-like curl and divergence from tracked flow."""
    prev_xy = np.asarray(prev_points_xy, dtype=np.float32).reshape(-1, 2)
    flow_xy = np.asarray(flow, dtype=np.float32).reshape(-1, 2)
    center = np.asarray(center_xy, dtype=np.float32).reshape(2)

    radii = prev_xy - center
    radius_sq = np.einsum("ij,ij->i", radii, radii) + _EPS
    yaw_terms = (radii[:, 0] * flow_xy[:, 1] - radii[:, 1] * flow_xy[:, 0]) / radius_sq
    yaw_rate = float(np.median(yaw_terms))
    rot_vec = np.stack((-radii[:, 1], radii[:, 0]), axis=1) * yaw_rate

    div_terms = np.einsum("ij,ij->i", radii, flow_xy) / radius_sq
    divergence_rate = float(np.median(div_terms))
    zoom_vec = radii * divergence_rate

    corrected_flow = (flow_xy - rot_vec - zoom_vec).astype(np.float32)
    flow_mean, flow_covariance = covariance2x2(corrected_flow)
    return FlowCompensationResult(
        corrected_flow=corrected_flow,
        flow_mean=flow_mean,
        flow_covariance=flow_covariance,
        yaw_rate=yaw_rate,
        divergence_rate=divergence_rate,
    )


def lk_error_weights(errors: np.ndarray | None) -> np.ndarray:
    """Convert LK tracking errors into soft sample weights."""
    if errors is None:
        return np.ones(0, dtype=np.float32)

    values = np.asarray(errors, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.ones(0, dtype=np.float32)

    finite_mask = np.isfinite(values) & (values >= 0.0)
    if not np.any(finite_mask):
        return np.ones(values.shape[0], dtype=np.float32)

    scale = float(np.median(values[finite_mask])) + 1.0
    weights = np.ones(values.shape[0], dtype=np.float32)
    weights[finite_mask] = 1.0 / (1.0 + values[finite_mask] / scale)
    return np.clip(weights, 1e-3, 1.0)


def robust_velocity_mask(velocity_samples: np.ndarray) -> np.ndarray:
    """Reject velocity outliers using median distance in velocity space."""
    velocities = np.asarray(velocity_samples, dtype=np.float32).reshape(-1, 2)
    count = velocities.shape[0]
    if count <= 3:
        return np.ones(count, dtype=bool)

    median_velocity = np.median(velocities, axis=0)
    diff = velocities - median_velocity
    dist_sq = np.einsum("ij,ij->i", diff, diff)
    mad = float(np.median(dist_sq))
    if mad <= 1e-6:
        return np.ones(count, dtype=bool)

    keep_mask = dist_sq <= (9.0 * mad)
    if int(keep_mask.sum()) >= 2:
        return keep_mask

    keep_mask = np.zeros(count, dtype=bool)
    closest_idx = np.argsort(dist_sq)[:2]
    keep_mask[closest_idx] = True
    return keep_mask
