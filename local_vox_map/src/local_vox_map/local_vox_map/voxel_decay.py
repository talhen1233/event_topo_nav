"""Pure temporal-confidence helpers used by the rolling voxel map."""

from __future__ import annotations

import math
from typing import Iterable, Tuple


def acquisition_age_s(now_ros_s: float, stamp_ros_s: float) -> float:
    """Return acquisition age; zero stamp falls back to receipt time."""
    now = float(now_ros_s)
    stamp = float(stamp_ros_s)
    if not math.isfinite(now) or not math.isfinite(stamp):
        return math.inf
    if stamp <= 0.0:
        return 0.0
    age = now - stamp
    return age if age >= 0.0 else math.inf


def cloud_input_is_fresh(
    *,
    last_input_s: float | None,
    now_s: float,
    timeout_s: float,
) -> bool:
    """Return whether the mapper has received a sensor cloud recently."""
    if last_input_s is None:
        return False
    timeout = float(timeout_s)
    if timeout <= 0.0:
        return True
    age = max(0.0, float(now_s) - float(last_input_s))
    return age <= timeout


def voxel_is_expired(
    *,
    last_hit_s: float,
    now_s: float,
    ttl_s: float,
) -> bool:
    """Return whether an unobserved voxel exceeded its configured lifetime."""
    ttl = float(ttl_s)
    if ttl <= 0.0:
        return False
    age = float(now_s) - float(last_hit_s)
    return math.isfinite(age) and age >= 0.0 and age >= ttl


def active_voxel_ttl_s(
    *,
    nominal_ttl_s: float,
    degraded_ttl_s: float,
    of_degraded: bool,
) -> float:
    """Select a bounded TTL, using the shorter value while OF is degraded."""
    nominal = max(0.0, float(nominal_ttl_s))
    degraded = max(0.0, float(degraded_ttl_s))
    if not of_degraded or degraded <= 0.0:
        return nominal
    if nominal <= 0.0:
        return degraded
    return min(nominal, degraded)


def fresh_cloud_input_count(
    *,
    last_inputs_s: Iterable[float],
    now_s: float,
    timeout_s: float,
) -> int:
    """Count independently fresh sensor-cloud sources."""
    return sum(
        cloud_input_is_fresh(
            last_input_s=stamp,
            now_s=now_s,
            timeout_s=timeout_s,
        )
        for stamp in last_inputs_s
    )


def tf_fallback_is_fresh(
    *,
    sample_stamp_s: float,
    transform_stamp_s: float,
    max_lag_s: float,
) -> bool:
    """Allow a latest-TF fallback only when it closely precedes the sample."""
    sample = float(sample_stamp_s)
    transform = float(transform_stamp_s)
    max_lag = max(0.0, float(max_lag_s))
    if not math.isfinite(sample) or not math.isfinite(transform):
        return False
    lag = sample - transform
    return 0.0 <= lag <= max_lag


def decay_voxel_state(
    *,
    vote: float,
    consistency: float,
    last_hit_s: float,
    last_decay_s: float,
    now_s: float,
    vote_decay_rate: float,
    consistency_decay_rate: float,
    decay_delay_s: float,
    confidence_delay_per_unit_s: float,
    confidence_decay_scale: float,
) -> Tuple[float, float, float]:
    """Decay a voxel once over the time interval not consumed previously."""
    now = max(float(now_s), float(last_decay_s))
    effective_delay = max(0.0, float(decay_delay_s)) + max(
        0.0, float(confidence_delay_per_unit_s)
    ) * max(0.0, float(consistency))
    decay_start = max(float(last_decay_s), float(last_hit_s) + effective_delay)
    elapsed = max(0.0, now - decay_start)
    if elapsed <= 0.0:
        return float(vote), float(consistency), now

    scale = 1.0 / (
        1.0 + max(0.0, float(confidence_decay_scale)) * max(0.0, float(consistency))
    )
    vote_new = float(vote) * math.exp(
        -max(0.0, float(vote_decay_rate)) * scale * elapsed
    )
    consistency_new = float(consistency) * math.exp(
        -max(0.0, float(consistency_decay_rate)) * scale * elapsed
    )
    return vote_new, consistency_new, now
