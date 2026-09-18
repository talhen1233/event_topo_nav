from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphCenteringParams:
    enabled: bool
    k_y: float
    deadband_m: float
    vy_max: float
    filter_tau_s: float


@dataclass(frozen=True)
class GraphCenteringOutput:
    vy_correction: float = 0.0
    body_y_error_m: float = 0.0


class GraphCenteringController:
    """Generate a gentle body-Y pull toward the projected graph centerline."""

    def __init__(self, params: GraphCenteringParams) -> None:
        self.p = params
        self._vy_filtered: float = 0.0

    def reset(self) -> None:
        self._vy_filtered = 0.0

    def compute(self, *, body_y_error_m: float, dt: float, active: bool) -> GraphCenteringOutput:
        err = float(body_y_error_m)
        if not self.p.enabled or not active or not math.isfinite(err):
            self.reset()
            return GraphCenteringOutput(vy_correction=0.0, body_y_error_m=err if math.isfinite(err) else 0.0)

        deadband = max(0.0, float(self.p.deadband_m))
        if abs(err) <= deadband:
            raw = 0.0
        else:
            eff = abs(err) - deadband
            raw = math.copysign(float(self.p.k_y) * eff, err)

        raw = float(max(-self.p.vy_max, min(self.p.vy_max, raw)))
        tau = max(0.0, float(self.p.filter_tau_s))
        if tau > 0.0:
            alpha = float(max(0.0, min(1.0, dt / (tau + max(dt, 1e-6)))))
            self._vy_filtered = (1.0 - alpha) * self._vy_filtered + alpha * raw
        else:
            self._vy_filtered = raw

        if abs(self._vy_filtered) < 1e-4:
            self._vy_filtered = 0.0

        return GraphCenteringOutput(vy_correction=float(self._vy_filtered), body_y_error_m=err)
