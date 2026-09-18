"""Pure safety helpers for return-event handover and pose correction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence, Tuple

import numpy as np


class HandoverRegion(str, Enum):
    """Active geometric region between consecutive nearby events."""

    INACTIVE = "inactive"
    FROM_ONLY = "from_only"
    COMPETE = "compete"
    TARGET_ONLY = "target_only"


@dataclass(frozen=True)
class HandoverDecision:
    """Result of the nearby-event geometric handover check."""

    region: HandoverRegion
    pair_distance_m: float
    distance_from_m: float
    distance_target_m: float
    margin_m: float
    reason: str = ""

    @property
    def active(self) -> bool:
        return self.region != HandoverRegion.INACTIVE


@dataclass(frozen=True)
class PoseCorrectionEstimate:
    """Translation-only event pose correction and its safety decision."""

    accepted: bool
    corrected_xy: Tuple[float, float]
    translation_xy: Tuple[float, float]
    translation_norm_m: float
    reason: str = ""


def pose_correction_translation_limit(
    *,
    distance_since_anchor_m: float,
    base_m: float,
    growth_per_meter: float,
    hard_cap_m: float,
) -> float:
    """Grow the correction allowance with travel since the last trusted anchor; hard_cap is independent of route length."""
    distance = float(distance_since_anchor_m)
    if not math.isfinite(distance):
        distance = 0.0
    distance = max(0.0, distance)
    base = max(0.0, float(base_m))
    growth = max(0.0, float(growth_per_meter))
    limit = base + growth * distance
    hard_cap = max(0.0, float(hard_cap_m))
    if hard_cap > 0.0:
        limit = min(limit, hard_cap)
    return limit


def _xy(values: Sequence[float]) -> Tuple[float, float]:
    xy = tuple(float(value) for value in values)
    if len(xy) < 2 or not all(math.isfinite(value) for value in xy[:2]):
        raise ValueError("XY coordinates must contain two finite values")
    return xy[0], xy[1]


def evaluate_geometric_handover(
    *,
    current_xy: Sequence[float],
    from_xy: Sequence[float],
    target_xy: Sequence[float],
    expected_route_gap_m: Optional[float],
    odom_sigma_xy_m: float,
    max_distance_m: float,
    margin_base_m: float,
    margin_sigma_multiplier: float,
    margin_min_m: float,
    margin_max_fraction: float,
) -> HandoverDecision:
    """Classify A-only, competition, and B-only; both Euclidean pair distance and route gap must be short."""
    current = _xy(current_xy)
    source = _xy(from_xy)
    target = _xy(target_xy)
    pair_distance = math.hypot(target[0] - source[0], target[1] - source[1])
    distance_from = math.hypot(current[0] - source[0], current[1] - source[1])
    distance_target = math.hypot(current[0] - target[0], current[1] - target[1])

    limit = max(0.0, float(max_distance_m))
    if pair_distance <= 1e-6:
        return HandoverDecision(
            HandoverRegion.INACTIVE,
            pair_distance,
            distance_from,
            distance_target,
            0.0,
            "coincident_events",
        )
    if limit <= 0.0 or pair_distance > limit:
        return HandoverDecision(
            HandoverRegion.INACTIVE,
            pair_distance,
            distance_from,
            distance_target,
            0.0,
            "pair_too_far",
        )
    if expected_route_gap_m is None:
        return HandoverDecision(
            HandoverRegion.INACTIVE,
            pair_distance,
            distance_from,
            distance_target,
            0.0,
            "route_gap_unavailable",
        )
    route_gap = float(expected_route_gap_m)
    if not math.isfinite(route_gap) or route_gap < 0.0 or route_gap > limit:
        return HandoverDecision(
            HandoverRegion.INACTIVE,
            pair_distance,
            distance_from,
            distance_target,
            0.0,
            "route_gap_too_far",
        )

    sigma = max(0.0, float(odom_sigma_xy_m))
    margin = max(
        max(0.0, float(margin_min_m)),
        max(0.0, float(margin_base_m))
        + max(0.0, float(margin_sigma_multiplier)) * sigma,
    )
    max_fraction = min(0.95, max(0.0, float(margin_max_fraction)))
    margin = min(margin, max_fraction * pair_distance)

    if distance_from + margin <= distance_target:
        region = HandoverRegion.FROM_ONLY
    elif distance_target + margin <= distance_from:
        region = HandoverRegion.TARGET_ONLY
    else:
        region = HandoverRegion.COMPETE
    return HandoverDecision(
        region,
        pair_distance,
        distance_from,
        distance_target,
        margin,
    )


def estimate_pose_correction(
    *,
    robot_xy: Sequence[float],
    stored_center_xy: Sequence[float],
    observed_center_xy: Sequence[float],
    max_translation_m: float,
) -> PoseCorrectionEstimate:
    """Compute a bounded center-to-center translation correction."""
    robot = _xy(robot_xy)
    stored = _xy(stored_center_xy)
    observed = _xy(observed_center_xy)
    delta = (stored[0] - observed[0], stored[1] - observed[1])
    norm = math.hypot(delta[0], delta[1])
    corrected = (robot[0] + delta[0], robot[1] + delta[1])
    limit = max(0.0, float(max_translation_m))
    if limit > 0.0 and norm > limit:
        return PoseCorrectionEstimate(
            False,
            corrected,
            delta,
            norm,
            "translation_limit",
        )
    return PoseCorrectionEstimate(True, corrected, delta, norm)


def event_quality_allows_correction(
    *,
    event_confidence: float,
    covariance_trace_xy: float,
    confidence_min: float,
    covariance_trace_max_xy: float,
) -> bool:
    """Allow a high-impact correction only when both quality gates pass."""
    return (
        float(event_confidence) >= float(confidence_min)
        and float(covariance_trace_xy) <= float(covariance_trace_max_xy)
    )


def correction_covariance_xy(
    *,
    event_covariance_xy: np.ndarray,
    detection_noise_m: float,
    match_score: float,
    acceptance_threshold: float,
    covariance_scale: float,
    minimum_variance_m2: float,
) -> np.ndarray:
    """Build a conservative match-quality-weighted XY measurement covariance."""
    event_cov = np.asarray(event_covariance_xy, dtype=np.float64).reshape(2, 2)
    event_cov = 0.5 * (event_cov + event_cov.T)
    score = float(np.clip(match_score, 0.0, 1.0))
    threshold = float(np.clip(acceptance_threshold, 0.0, 1.0))
    quality = float(np.clip(
        (score - threshold) / max(1e-6, 1.0 - threshold),
        0.0,
        1.0,
    ))
    weight = 0.25 + 0.75 * quality * quality
    noise_variance = max(0.0, float(detection_noise_m)) ** 2
    covariance = (
        event_cov + np.eye(2, dtype=np.float64) * noise_variance
    ) / weight
    scale = max(0.0, float(covariance_scale))
    covariance *= scale
    floor = max(1e-8, float(minimum_variance_m2))
    covariance[0, 0] = max(floor, float(covariance[0, 0]))
    covariance[1, 1] = max(floor, float(covariance[1, 1]))
    return covariance
