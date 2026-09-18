"""Magnetic floor/ceiling repulsion from the point cloud."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HeightControllerParams:
    xy_max_range: float = 2.0
    up_pitch_min: float = math.radians(30.0)
    down_pitch_min: float = math.radians(30.0)
    d0_ceiling: float = 0.5
    d0_floor: float = 1.0
    k_ceiling: float = 1.0
    k_floor: float = 1.0
    vz_max: float = 0.5
    vertical_dist_percentile: float = 15.0
    vz_filter_tau: float = 0.3
    vz_deadband: float = 0.02
    control_rate: float = 20.0


class HeightController:
    def __init__(self, params: HeightControllerParams) -> None:
        self.p = params
        self._vz_filtered: float = 0.0

    def reset(self) -> None:
        self._vz_filtered = 0.0

    def compute(
        self,
        pos_w: np.ndarray,
        R_bw: np.ndarray,
        obstacles_w: Optional[np.ndarray],
    ) -> float:
        if obstacles_w is None or obstacles_w.size == 0:
            return self._apply_filter(0.0)

        pts = obstacles_w.astype(np.float32, copy=False)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return self._apply_filter(0.0)

        w = pts - pos_w[None, :]
        dist3 = np.linalg.norm(w, axis=1).astype(np.float32)
        valid = dist3 > 1e-3
        if not np.any(valid):
            return self._apply_filter(0.0)

        w = w[valid]
        dist3 = dist3[valid]

        R_wb = R_bw.T
        v_body = (R_wb @ w.T).T.astype(np.float32)
        xb, yb, zb = v_body[:, 0], v_body[:, 1], v_body[:, 2]

        r_xy = np.hypot(xb, yb)
        r_xy_safe = np.where(r_xy < 1e-3, 1e-3, r_xy)
        pitch = np.arctan2(zb, r_xy_safe)

        mask_z = r_xy <= self.p.xy_max_range
        pitch_z = pitch[mask_z]
        dist3_z = dist3[mask_z]

        if pitch_z.size == 0:
            return self._apply_filter(0.0)

        up_mask = pitch_z >= self.p.up_pitch_min
        down_mask = pitch_z <= -self.p.down_pitch_min

        d_up_soft = _percentile_where(up_mask, dist3_z, self.p.vertical_dist_percentile)
        d_down_soft = _percentile_where(down_mask, dist3_z, self.p.vertical_dist_percentile)
        d_up_hard = _min_where(up_mask, dist3_z)
        d_down_hard = _min_where(down_mask, dist3_z)

        rep_up_soft = _rep_smooth(d_up_soft, self.p.d0_ceiling)
        rep_down_soft = _rep_smooth(d_down_soft, self.p.d0_floor)
        rep_up_hard = _rep_smooth(d_up_hard, 0.35 * self.p.d0_ceiling)
        rep_down_hard = _rep_smooth(d_down_hard, 0.35 * self.p.d0_floor)

        rep_up = max(rep_up_soft, rep_up_hard)
        rep_down = max(rep_down_soft, rep_down_hard)

        vz_raw = float(self.p.k_floor * rep_down - self.p.k_ceiling * rep_up)
        return self._apply_filter(vz_raw)

    def _apply_filter(self, vz_raw: float) -> float:
        dt = 1.0 / max(self.p.control_rate, 1e-3)
        alpha = dt / (self.p.vz_filter_tau + dt) if self.p.vz_filter_tau > 0.0 else 1.0
        self._vz_filtered = (1.0 - alpha) * self._vz_filtered + alpha * vz_raw
        if abs(self._vz_filtered) < self.p.vz_deadband:
            self._vz_filtered = 0.0
        return float(np.clip(self._vz_filtered, -self.p.vz_max, self.p.vz_max))


def _min_where(mask: np.ndarray, vals: np.ndarray) -> Optional[float]:
    if mask.size == 0 or not np.any(mask):
        return None
    return float(np.min(vals[mask]))


def _percentile_where(mask: np.ndarray, vals: np.ndarray, q: float) -> Optional[float]:
    if mask.size == 0 or not np.any(mask):
        return None
    return float(np.percentile(vals[mask], float(np.clip(q, 0.0, 100.0))))


def _rep_smooth(d: Optional[float], d0: float) -> float:
    if d is None or d0 <= 0.0 or d >= d0:
        return 0.0
    x = float(np.clip((d0 - d) / d0, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)
