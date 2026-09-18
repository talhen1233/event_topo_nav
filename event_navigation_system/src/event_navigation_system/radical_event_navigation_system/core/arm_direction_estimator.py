"""Estimate an outward junction-arm direction from skeleton edge points."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmDirectionEstimate:
    """Contain fitted arm angle, straightness confidence, and observed span."""

    angle_rad: float
    confidence: float
    span_m: float


def estimate_arm_direction(
    points_xy: np.ndarray,
    junction_xy: np.ndarray,
    *,
    min_distance_m: float,
    max_distance_m: float,
    min_span_m: float,
    fallback_angle_rad: float,
) -> ArmDirectionEstimate:
    """Fit a weighted line through a junction using a small 2x2 covariance."""
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    center = np.asarray(junction_xy, dtype=np.float32).reshape(2)
    if points.shape[0] < 2:
        return ArmDirectionEstimate(fallback_angle_rad, 0.0, 0.0)

    offsets = points - center
    distances = np.linalg.norm(offsets, axis=1)
    lower = max(0.0, float(min_distance_m))
    upper = max(lower + 1e-3, float(max_distance_m))
    mask = (distances >= lower) & (distances <= upper)
    selected = offsets[mask]
    selected_distances = distances[mask]
    if selected.shape[0] < 2:
        return ArmDirectionEstimate(
            fallback_angle_rad,
            0.0,
            float(np.max(distances, initial=0.0)),
        )

    span_m = float(np.max(selected_distances, initial=0.0))
    if span_m < max(0.0, float(min_span_m)):
        return ArmDirectionEstimate(fallback_angle_rad, 0.0, span_m)

    weights = np.clip(selected_distances / upper, 0.2, 1.0)
    weighted = selected * np.sqrt(weights)[:, None]
    covariance = weighted.T @ weighted / max(float(np.sum(weights)), 1e-6)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    mean_offset = np.average(selected, axis=0, weights=weights)
    if float(np.dot(direction, mean_offset)) < 0.0:
        direction = -direction

    largest = float(max(eigenvalues[-1], 0.0))
    smallest = float(max(eigenvalues[0], 0.0))
    confidence = (largest - smallest) / max(largest + smallest, 1e-6)
    angle = math.atan2(float(direction[1]), float(direction[0]))
    return ArmDirectionEstimate(
        angle_rad=float(angle),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        span_m=span_m,
    )
