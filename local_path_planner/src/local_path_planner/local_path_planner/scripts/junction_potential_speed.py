"""Convert nearby event potential into a bounded forward-speed scale."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class JunctionPotentialSpeedParams:
    slowdown_start: float
    full_slowdown: float
    min_speed_scale: float
    timeout_s: float


class JunctionPotentialSpeed:
    def __init__(self, params: JunctionPotentialSpeedParams) -> None:
        self.p = params
        self._potential = 0.0
        self._last_update_s: Optional[float] = None
        self._suppressed_until_clear = False

    @property
    def potential(self) -> float:
        return self._potential

    def reset(self) -> None:
        self._potential = 0.0
        self._last_update_s = None
        self._suppressed_until_clear = False

    def suppress_until_clear(self) -> None:
        """Ignore the current junction after physical branch entry."""
        self._suppressed_until_clear = True

    def update(self, potential: float, now_s: float) -> None:
        if not math.isfinite(potential):
            return
        self._potential = float(np.clip(potential, 0.0, 1.0))
        self._last_update_s = float(now_s)
        if self._potential <= max(0.0, self.p.slowdown_start):
            self._suppressed_until_clear = False

    def speed_scale(self, now_s: float) -> float:
        if self._last_update_s is None:
            return 1.0
        if self._suppressed_until_clear:
            return 1.0
        age_s = float(now_s) - self._last_update_s
        if age_s < 0.0 or age_s > max(0.0, self.p.timeout_s):
            return 1.0

        start = float(np.clip(self.p.slowdown_start, 0.0, 1.0 - 1e-3))
        full = float(
            np.clip(self.p.full_slowdown, start + 1e-3, 1.0)
        )
        if self._potential <= start:
            return 1.0
        progress = float(
            np.clip((self._potential - start) / (full - start), 0.0, 1.0)
        )
        minimum = float(np.clip(self.p.min_speed_scale, 0.0, 1.0))
        return 1.0 - progress * (1.0 - minimum)
