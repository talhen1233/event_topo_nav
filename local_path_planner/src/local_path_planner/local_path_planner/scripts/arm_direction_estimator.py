"""Estimate a stable outward arm direction from skeleton edge points."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArmDirectionEstimate:
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


class SmoothedArmDirection:
    """Refine one arm angle with confidence-weighted circular smoothing."""

    def __init__(
        self,
        initial_angle_rad: float,
        *,
        smoothing_tau_s: float,
        max_rate_rad_s: float,
    ) -> None:
        self._angle_rad = _wrap(initial_angle_rad)
        self._smoothing_tau_s = max(1e-3, float(smoothing_tau_s))
        self._max_rate_rad_s = max(0.0, float(max_rate_rad_s))

    @property
    def angle_rad(self) -> float:
        return self._angle_rad

    def update(
        self,
        measurement_rad: float,
        confidence: float,
        dt: float,
    ) -> float:
        step_dt = max(0.0, float(dt))
        alpha = (
            1.0 - math.exp(-step_dt / self._smoothing_tau_s)
        ) * float(np.clip(confidence, 0.0, 1.0))
        error = _wrap(float(measurement_rad) - self._angle_rad)
        requested_step = alpha * error
        max_step = self._max_rate_rad_s * step_dt
        self._angle_rad = _wrap(
            self._angle_rad
            + float(np.clip(requested_step, -max_step, max_step))
        )
        return self._angle_rad


def _wrap(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))
