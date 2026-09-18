"""Pure state logic for degraded optical flow and conservative ZUPT arming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class OpticalFlowHealthParams:
    stale_timeout_s: float = 1.0
    recovery_frames: int = 3
    mismatch_threshold_mps: float = 0.25
    slow_speed_max_mps: float = 0.3
    mismatch_dwell_s: float = 0.75


class OpticalFlowHealthTracker:
    """Track whether OF is computationally usable, with recovery hysteresis."""

    def __init__(self, params: OpticalFlowHealthParams) -> None:
        self.p = params
        self.degraded = True
        self.last_usable_time_s: Optional[float] = None
        self._recovery_count = 0
        self._mismatch_start_s: Optional[float] = None

    def observe(
        self,
        *,
        now_s: float,
        measurement_usable: bool,
        of_speed_mps: float,
        command_fresh: bool,
        command_speed_mps: float,
        command_mismatch_mps: float,
    ) -> bool:
        now = float(now_s)
        if not measurement_usable:
            # A single rejected frame is not a failure; tick() uses the timeout.
            self._recovery_count = 0
            self._mismatch_start_s = None
            return self.degraded

        self.last_usable_time_s = now
        mismatch = (
            command_fresh
            and float(of_speed_mps) <= max(0.0, self.p.slow_speed_max_mps)
            and float(command_mismatch_mps)
            >= max(0.0, self.p.mismatch_threshold_mps)
            and float(command_speed_mps)
            >= max(0.0, self.p.mismatch_threshold_mps)
        )
        if mismatch:
            self._recovery_count = 0
            if self._mismatch_start_s is None:
                self._mismatch_start_s = now
            if now - self._mismatch_start_s >= max(0.0, self.p.mismatch_dwell_s):
                self.degraded = True
        else:
            self._mismatch_start_s = None
            self._recovery_count += 1
            if self._recovery_count >= max(1, int(self.p.recovery_frames)):
                self.degraded = False
        return self.degraded

    def tick(self, now_s: float) -> bool:
        timeout = max(0.0, float(self.p.stale_timeout_s))
        stale = (
            self.last_usable_time_s is None
            or (timeout > 0.0 and float(now_s) - self.last_usable_time_s > timeout)
        )
        if stale:
            self.degraded = True
            self._recovery_count = 0
            self._mismatch_start_s = None
        return self.degraded


def apply_command_velocity_prior(
    *,
    state: np.ndarray,
    covariance: np.ndarray,
    command_world_xy: np.ndarray,
    gain: float,
    sigma_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend XY velocity toward the command without shrinking covariance."""
    x = np.asarray(state, dtype=float).copy()
    p = np.asarray(covariance, dtype=float).copy()
    z = np.asarray(command_world_xy, dtype=float).reshape(2)
    if x.size < 5 or p.shape[0] < 5 or p.shape[1] < 5:
        raise ValueError("state/covariance do not contain horizontal velocity")
    if not np.all(np.isfinite(z)):
        raise ValueError("command velocity must be finite")

    alpha = float(np.clip(gain, 0.0, 1.0))
    x[3:5] += alpha * (z - x[3:5])

    variance_floor = max(1e-6, float(sigma_mps) ** 2)
    for index in (3, 4):
        if p[index, index] < variance_floor:
            p[index, index] = variance_floor
    p = 0.5 * (p + p.T)
    return x, p


def apply_zero_velocity_constraint(
    *, state: np.ndarray, covariance: np.ndarray, measurement_variance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 3-D zero-velocity constraint without changing position mean."""
    x = np.asarray(state, dtype=float).copy()
    p = np.asarray(covariance, dtype=float).copy()
    if x.size < 6 or p.shape[0] < 6 or p.shape[1] < 6:
        raise ValueError("state/covariance do not contain 3-D position and velocity")

    h = np.zeros((3, x.size), dtype=float)
    h[:, 3:6] = np.eye(3, dtype=float)
    r = max(1e-9, float(measurement_variance)) * np.eye(3, dtype=float)
    s = h @ p @ h.T + r
    try:
        k = p @ h.T @ np.linalg.inv(s)
    except np.linalg.LinAlgError:
        return x, p

    k[0:3, :] = 0.0
    x += k @ (-(h @ x))
    i_kh = np.eye(x.size, dtype=float) - k @ h
    p = i_kh @ p @ i_kh.T + k @ r @ k.T
    p = 0.5 * (p + p.T)
    return x, p


def zupt_candidate(
    *,
    command_fresh: bool,
    command_speed_mps: float,
    imu_fresh: bool,
    imu_acceleration_mps2: float,
    tilt_rad: float,
    imu_acceleration_threshold_mps2: float,
    tilt_threshold_rad: float,
    of_degraded: bool,
    of_measurement_usable: bool,
    of_speed_mps: Optional[float],
    speed_threshold_mps: float,
) -> bool:
    """True only when command, IMU, and (if healthy) OF all support rest."""
    threshold = max(0.0, float(speed_threshold_mps))
    base = (
        command_fresh
        and float(command_speed_mps) <= threshold
        and imu_fresh
        and float(imu_acceleration_mps2)
        <= max(0.0, float(imu_acceleration_threshold_mps2))
        and float(tilt_rad) <= max(0.0, float(tilt_threshold_rad))
    )
    if not base:
        return False
    if of_degraded:
        return True
    return (
        of_measurement_usable
        and of_speed_mps is not None
        and float(of_speed_mps) <= threshold
    )
