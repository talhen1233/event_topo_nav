"""Stable mission-start profile detection, independent of ROS for unit tests."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np


@dataclass(frozen=True)
class StartStabilityParams:
    """Configuration for the START-profile stability gate."""

    min_frames: int = 10
    max_normalized_change: float = 0.05
    max_speed_mps: float = 0.05
    normalization_floor_m: float = 0.5
    max_buffer_frames: int = 200


@dataclass(frozen=True)
class StartStabilityStatus:
    """Result of one profile observation."""

    stable: bool
    consecutive_frames: int
    elapsed_s: float
    normalized_change: Optional[float]
    representative_profile: Optional[np.ndarray]


class StartStabilityTracker:
    """Require both temporal persistence and geometric consistency."""

    def __init__(self, params: StartStabilityParams) -> None:
        self.p = params
        self._samples: Deque[np.ndarray] = deque(
            maxlen=max(1, int(params.max_buffer_frames))
        )
        self._streak_start_s: Optional[float] = None

    def reset(self) -> None:
        self._samples.clear()
        self._streak_start_s = None

    def update(
        self,
        profile_m: np.ndarray,
        now_s: float,
        speed_mps: float,
    ) -> StartStabilityStatus:
        """Observe a filtered radial profile and update the stability streak."""
        profile = np.asarray(profile_m, dtype=np.float64).reshape(-1)
        eligible = (
            profile.size > 0
            and bool(np.all(np.isfinite(profile)))
            and float(speed_mps) <= max(0.0, float(self.p.max_speed_mps))
        )
        if not eligible:
            self.reset()
            return StartStabilityStatus(False, 0, 0.0, None, None)

        now = float(now_s)
        if not self._samples:
            self._begin_streak(profile, now)
            return self._status(now, None)

        reference = np.median(np.stack(tuple(self._samples), axis=0), axis=0)
        if reference.shape != profile.shape:
            self._begin_streak(profile, now)
            return self._status(now, None)

        floor = max(1e-6, float(self.p.normalization_floor_m))
        scale = np.maximum(np.abs(reference), floor)
        normalized_change = float(np.mean(np.abs(profile - reference) / scale))

        if normalized_change > max(0.0, float(self.p.max_normalized_change)):
            self._begin_streak(profile, now)
            return self._status(now, normalized_change)

        self._samples.append(profile.copy())
        return self._status(now, normalized_change)

    def _begin_streak(self, profile: np.ndarray, now_s: float) -> None:
        self._samples.clear()
        self._samples.append(profile.copy())
        self._streak_start_s = float(now_s)

    def _status(
        self, now_s: float, normalized_change: Optional[float]
    ) -> StartStabilityStatus:
        elapsed = (
            0.0
            if self._streak_start_s is None
            else max(0.0, float(now_s) - self._streak_start_s)
        )
        count = len(self._samples)
        stable = count >= max(1, int(self.p.min_frames))
        representative = None
        if stable:
            representative = np.median(
                np.stack(tuple(self._samples), axis=0), axis=0
            ).astype(np.float32)
        return StartStabilityStatus(
            stable=stable,
            consecutive_frames=count,
            elapsed_s=elapsed,
            normalized_change=normalized_change,
            representative_profile=representative,
        )
