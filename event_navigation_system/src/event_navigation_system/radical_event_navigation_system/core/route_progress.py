"""One-dimensional return-progress uncertainty for event navigation."""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from enum import Enum
import math
from typing import Deque, Dict, List, Optional, Sequence, Tuple


class LocalizationMode(str, Enum):
    """Localization quality classes used by the return-distance model."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    OF_DEGRADED = "of_degraded"


@dataclass(frozen=True)
class VarianceRates:
    """Variance growth rates per traveled meter and elapsed second."""

    q_s_m: float
    q_t_m2_per_s: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.q_s_m) or self.q_s_m < 0.0:
            raise ValueError("q_s_m must be finite and non-negative")
        if not math.isfinite(self.q_t_m2_per_s) or self.q_t_m2_per_s < 0.0:
            raise ValueError("q_t_m2_per_s must be finite and non-negative")


@dataclass(frozen=True)
class ProgressWindow:
    """Current k-sigma search window around an expected route distance."""

    lower_m: float
    upper_m: float
    sigma_m: float

    def contains(self, progress_m: float) -> bool:
        return self.lower_m <= float(progress_m) <= self.upper_m


@dataclass(frozen=True)
class DistanceUpdate:
    """Accepted displacement from a deadband-filtered pose stream."""

    distance_m: float
    rebased: bool = False
    reason: str = ""


class DistanceDeadbandTracker:
    """Reject pose jitter while accumulating slow motion against the last accepted pose."""

    def __init__(self, deadband_m: float, max_step_m: float = 0.0):
        deadband = float(deadband_m)
        max_step = float(max_step_m)
        if not math.isfinite(deadband) or deadband < 0.0:
            raise ValueError("deadband_m must be finite and non-negative")
        if not math.isfinite(max_step) or max_step < 0.0:
            raise ValueError("max_step_m must be finite and non-negative")
        if max_step > 0.0 and max_step <= deadband:
            raise ValueError("max_step_m must exceed deadband_m when enabled")
        self.deadband_m = deadband
        self.max_step_m = max_step
        self._anchor_xyz: Optional[Tuple[float, float, float]] = None

    @staticmethod
    def _xyz(position: Sequence[float]) -> Tuple[float, float, float]:
        values = tuple(float(value) for value in position)
        if len(values) < 2:
            raise ValueError("position must contain at least X and Y")
        xyz = (
            values[0],
            values[1],
            values[2] if len(values) >= 3 else 0.0,
        )
        if not all(math.isfinite(value) for value in xyz):
            raise ValueError("position must be finite")
        return xyz

    def reset(self) -> None:
        self._anchor_xyz = None

    def rebase(self, position: Sequence[float]) -> None:
        """Move the distance anchor without reporting physical travel."""
        self._anchor_xyz = self._xyz(position)

    def update(self, position: Sequence[float]) -> DistanceUpdate:
        xyz = self._xyz(position)
        if self._anchor_xyz is None:
            self._anchor_xyz = xyz
            return DistanceUpdate(0.0, rebased=True, reason="initialized")

        dx = xyz[0] - self._anchor_xyz[0]
        dy = xyz[1] - self._anchor_xyz[1]
        dz = xyz[2] - self._anchor_xyz[2]
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance < self.deadband_m:
            return DistanceUpdate(0.0)
        if self.max_step_m > 0.0 and distance > self.max_step_m:
            self._anchor_xyz = xyz
            return DistanceUpdate(0.0, rebased=True, reason="discontinuity")

        self._anchor_xyz = xyz
        return DistanceUpdate(distance)


class CorrectionAwareDistanceTracker:
    """Integrate stamped odometry while excluding acknowledged jumps that may arrive out of order."""

    def __init__(
        self,
        deadband_m: float,
        max_step_m: float,
        reorder_window_s: float = 0.1,
    ) -> None:
        self._tracker = DistanceDeadbandTracker(deadband_m, max_step_m)
        self.reorder_window_s = max(0.0, float(reorder_window_s))
        self._samples: Deque[Tuple[float, Tuple[float, float, float]]] = deque()
        self._correction_times_s: List[float] = []

    def reset(self) -> None:
        self._tracker.reset()
        self._samples.clear()
        self._correction_times_s.clear()

    def rebase(self, position: Sequence[float]) -> None:
        self._tracker.rebase(position)
        self._samples.clear()
        self._correction_times_s.clear()

    def acknowledge_correction(self, stamp_s: float) -> None:
        stamp = float(stamp_s)
        if not math.isfinite(stamp):
            return
        self._correction_times_s.append(stamp)
        self._correction_times_s.sort()

    def update(self, position: Sequence[float], stamp_s: float) -> DistanceUpdate:
        xyz = DistanceDeadbandTracker._xyz(position)
        stamp = float(stamp_s)
        if not math.isfinite(stamp):
            raise ValueError("stamp_s must be finite")

        if self._samples and stamp < self._samples[0][0] - 1e-9:
            # A simulation-clock reset starts a new, unrelated time sequence.
            self.reset()

        samples = list(self._samples)
        samples.append((stamp, xyz))
        samples.sort(key=lambda item: item[0])
        deduplicated: List[Tuple[float, Tuple[float, float, float]]] = []
        for item in samples:
            if deduplicated and abs(item[0] - deduplicated[-1][0]) <= 1e-9:
                deduplicated[-1] = item
            else:
                deduplicated.append(item)
        self._samples = deque(deduplicated)

        if len(self._samples) == 1:
            self._tracker.rebase(self._samples[0][1])
            return DistanceUpdate(0.0, rebased=True, reason="buffering")

        newest_stamp = self._samples[-1][0]
        cutoff = newest_stamp - self.reorder_window_s
        total_distance = 0.0
        rebased = False
        reasons: List[str] = []
        while len(self._samples) >= 2 and self._samples[1][0] <= cutoff + 1e-12:
            start_stamp, _start_xyz = self._samples[0]
            end_stamp, end_xyz = self._samples[1]
            crosses_correction = any(
                start_stamp < correction_stamp <= end_stamp
                for correction_stamp in self._correction_times_s
            )
            if crosses_correction:
                self._tracker.rebase(end_xyz)
                update = DistanceUpdate(0.0, rebased=True, reason="pose_correction")
            else:
                update = self._tracker.update(end_xyz)
            total_distance += float(update.distance_m)
            rebased = rebased or bool(update.rebased)
            if update.reason:
                reasons.append(update.reason)
            self._samples.popleft()
            self._correction_times_s = [
                correction_stamp
                for correction_stamp in self._correction_times_s
                if correction_stamp > end_stamp
            ]

        return DistanceUpdate(
            total_distance,
            rebased=rebased,
            reason=",".join(dict.fromkeys(reasons)),
        )


class RouteProgressAccumulator:
    """Integrate route progress and mode-dependent distance variance."""

    def __init__(self, rates: Dict[LocalizationMode, VarianceRates]):
        missing = set(LocalizationMode) - set(rates)
        if missing:
            names = ", ".join(sorted(mode.value for mode in missing))
            raise ValueError(f"missing variance rates for: {names}")
        self._rates = dict(rates)
        self.reset()

    def reset(self) -> None:
        """Start measuring from a newly established route anchor."""
        self.progress_m = 0.0
        self.elapsed_s = 0.0
        self.integration_variance_m2 = 0.0

    def update(self, distance_m: float, dt_s: float, mode: LocalizationMode) -> None:
        """Apply ``q_s ds + q_t dt`` for one odometry interval."""
        ds = float(distance_m)
        dt = float(dt_s)
        if not math.isfinite(ds) or not math.isfinite(dt):
            return
        ds = max(0.0, ds)
        dt = max(0.0, dt)
        rates = self._rates[LocalizationMode(mode)]
        self.progress_m += ds
        self.elapsed_s += dt
        self.integration_variance_m2 += rates.q_s_m * ds
        self.integration_variance_m2 += rates.q_t_m2_per_s * dt

    def window(
        self,
        *,
        expected_distance_m: float,
        event_sigma_m: float,
        sigma_multiplier: float,
        min_half_width_m: float = 0.0,
    ) -> ProgressWindow:
        """Return the search interval from event and integrated variance."""
        expected = max(0.0, float(expected_distance_m))
        event_sigma = max(0.0, float(event_sigma_m))
        k = max(0.0, float(sigma_multiplier))
        min_half_width = max(0.0, float(min_half_width_m))
        total_variance = event_sigma * event_sigma + self.integration_variance_m2
        sigma = math.sqrt(max(0.0, total_variance))
        half_width = max(min_half_width, k * sigma)
        return ProgressWindow(
            lower_m=max(0.0, expected - half_width),
            upper_m=expected + half_width,
            sigma_m=sigma,
        )
