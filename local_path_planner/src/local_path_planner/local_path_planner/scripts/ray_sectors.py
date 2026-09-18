from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RaySectorsParams:
    xy_max_range: float
    horiz_pitch_half: float
    front_yaw_half: float
    back_yaw_half: float
    side_yaw_half: float
    up_pitch_min: float
    down_pitch_min: float
    d0_side: float
    k_y: float
    center_deadband: float
    side_dist_percentile: float
    min_points_per_dir: int
    dist_filter_tau: float
    vy_filter_tau: float
    control_rate: float
    steer_enable: bool
    steer_yaw_max: float
    steer_yaw_step: float
    steer_yaw_half: float
    steer_dist_percentile: float
    steer_min_points: int


@dataclass
class RaySectorsState:
    d_left_f: Optional[float] = None
    d_right_f: Optional[float] = None
    vy_f: float = 0.0


@dataclass(frozen=True)
class RaySectorsOutput:
    distances: dict[str, Optional[float]]
    vy_rep: float
    debug_points_world: dict[str, np.ndarray]
    steering: Optional["SteeringScan"]


@dataclass(frozen=True)
class SteeringScan:
    """Coarse body-X yaw scan for lightweight arc selection."""

    angles: np.ndarray
    clearances: np.ndarray


class RaySectors:
    """First-hit sector ranges so voxels behind the same ray do not bias distance."""

    def __init__(self, params: RaySectorsParams) -> None:
        self.p = params
        self.s = RaySectorsState()

    def reset_filters(self) -> None:
        self.s = RaySectorsState()

    @staticmethod
    def _dir_stat(mask: np.ndarray, dvals: np.ndarray, q: float, min_pts: int) -> Optional[float]:
        if mask.size == 0 or not np.any(mask):
            return None
        sel = dvals[mask]
        if int(sel.size) < int(min_pts):
            return None
        return float(np.percentile(sel, q))

    @staticmethod
    def _ema(prev: Optional[float], new: Optional[float], alpha: float) -> Optional[float]:
        if new is None:
            return prev
        if prev is None:
            return float(new)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        return float((1.0 - alpha) * float(prev) + alpha * float(new))

    @staticmethod
    def _ray_first_hit_mask_yaw_pitch(
        yaw: np.ndarray,
        pitch: np.ndarray,
        dist: np.ndarray,
        yaw_res_rad: float = math.radians(2.0),
        pitch_res_rad: float = math.radians(2.0),
    ) -> np.ndarray:
        n = int(yaw.size)
        if n == 0:
            return np.zeros(0, dtype=bool)

        yaw_bin = np.round(yaw / float(yaw_res_rad)).astype(np.int32)
        pitch_bin = np.round(pitch / float(pitch_res_rad)).astype(np.int32)
        # Key multiplier must exceed the pitch_bin span.
        keys = yaw_bin.astype(np.int64) * 100000 + pitch_bin.astype(np.int64)

        order = np.lexsort((dist.astype(np.float64), keys))
        keys_sorted = keys[order]
        unique = np.ones_like(keys_sorted, dtype=bool)
        unique[1:] = keys_sorted[1:] != keys_sorted[:-1]
        first_idx = order[unique]

        mask = np.zeros(n, dtype=bool)
        mask[first_idx] = True
        return mask

    @staticmethod
    def _rep_smooth(d: Optional[float], d0: float) -> float:
        if d is None or d0 <= 0.0:
            return 0.0
        if float(d) >= float(d0):
            return 0.0
        x = float(np.clip((float(d0) - float(d)) / float(d0), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def compute(
        self,
        pos_w: np.ndarray,
        R_bw: np.ndarray,
        obstacles_w: np.ndarray,
        *,
        debug: bool,
    ) -> RaySectorsOutput:
        empty_dists: dict[str, Optional[float]] = {
            "front": None,
            "back": None,
            "left": None,
            "right": None,
            "up": None,
            "down": None,
        }

        pts = obstacles_w.astype(np.float32, copy=False)
        if pts.ndim != 2 or pts.shape[1] != 3:
            return RaySectorsOutput(empty_dists, 0.0, {} if not debug else {}, None)

        w = pts - pos_w[None, :]
        dist3 = np.linalg.norm(w, axis=1).astype(np.float32)
        valid = dist3 > 1e-3
        if not np.any(valid):
            return RaySectorsOutput(empty_dists, 0.0, {} if not debug else {}, None)

        w = w[valid]
        dist3 = dist3[valid]
        pts_valid = pts[valid]

        R_wb = R_bw.T
        v_body = (R_wb @ w.T).T.astype(np.float32)
        xb = v_body[:, 0]
        yb = v_body[:, 1]
        zb = v_body[:, 2]

        r_xy = np.hypot(xb, yb)
        r_xy_safe = np.where(r_xy < 1e-3, 1e-3, r_xy)
        yaw = np.arctan2(yb, xb)
        pitch = np.arctan2(zb, r_xy_safe)

        mask_xy = (np.abs(pitch) <= self.p.horiz_pitch_half) & (r_xy <= self.p.xy_max_range)
        yaw_xy_all = yaw[mask_xy]
        pitch_xy_all = pitch[mask_xy]
        r_xy_all = r_xy[mask_xy]
        pts_xy_all = pts_valid[mask_xy]

        if yaw_xy_all.size > 0:
            ray_xy = self._ray_first_hit_mask_yaw_pitch(
                yaw_xy_all,
                pitch_xy_all,
                r_xy_all,
            )
            yaw_xy = yaw_xy_all[ray_xy]
            r_xy_xy = r_xy_all[ray_xy]
            pts_xy = pts_xy_all[ray_xy]
        else:
            yaw_xy = np.zeros(0, dtype=np.float32)
            r_xy_xy = np.zeros(0, dtype=np.float32)
            pts_xy = np.empty((0, 3), dtype=np.float32)

        if yaw_xy.size == 0:
            d_front = d_back = d_left = d_right = None
            front_mask = back_mask = left_mask = right_mask = np.zeros(0, dtype=bool)
        else:
            front_mask = np.abs(yaw_xy) <= self.p.front_yaw_half
            back_mask = np.abs(np.abs(yaw_xy) - math.pi) <= self.p.back_yaw_half
            left_mask = np.abs(yaw_xy - 0.5 * math.pi) <= self.p.side_yaw_half
            right_mask = np.abs(yaw_xy + 0.5 * math.pi) <= self.p.side_yaw_half

            # Front/back use the minimum for safety.
            d_front = self._dir_stat(front_mask, r_xy_xy, 0.0, 1)
            d_back = self._dir_stat(back_mask, r_xy_xy, 0.0, 1)

            # Prefer a robust side percentile; fall back to min if sparse.
            d_left_min = self._dir_stat(left_mask, r_xy_xy, 0.0, 1)
            d_right_min = self._dir_stat(right_mask, r_xy_xy, 0.0, 1)
            d_left_q = self._dir_stat(
                left_mask, r_xy_xy, self.p.side_dist_percentile, self.p.min_points_per_dir
            )
            d_right_q = self._dir_stat(
                right_mask, r_xy_xy, self.p.side_dist_percentile, self.p.min_points_per_dir
            )
            d_left = d_left_q if d_left_q is not None else d_left_min
            d_right = d_right_q if d_right_q is not None else d_right_min

        mask_z = r_xy <= self.p.xy_max_range
        pitch_z = pitch[mask_z]
        dist3_z = dist3[mask_z]

        if pitch_z.size == 0:
            d_up = d_down = None
            up_mask = down_mask = np.zeros(0, dtype=bool)
        else:
            up_mask = pitch_z >= self.p.up_pitch_min
            down_mask = pitch_z <= -self.p.down_pitch_min
            d_up = self._dir_stat(up_mask, dist3_z, 0.0, 1)
            d_down = self._dir_stat(down_mask, dist3_z, 0.0, 1)

        rep_left = self._rep_smooth(d_left, self.p.d0_side)
        rep_right = self._rep_smooth(d_right, self.p.d0_side)

        if self.p.dist_filter_tau > 0.0:
            dt_est = max(1.0 / max(self.p.control_rate, 1e-3), 1e-3)
            alpha_d = dt_est / (self.p.dist_filter_tau + dt_est)
            self.s.d_left_f = self._ema(self.s.d_left_f, d_left, alpha_d)
            self.s.d_right_f = self._ema(self.s.d_right_f, d_right, alpha_d)
            d_left_eff = self.s.d_left_f
            d_right_eff = self.s.d_right_f
        else:
            d_left_eff = d_left
            d_right_eff = d_right

        rep_left = self._rep_smooth(d_left_eff, self.p.d0_side)
        rep_right = self._rep_smooth(d_right_eff, self.p.d0_side)
        vy_rep = float(self.p.k_y * (rep_right - rep_left))

        # Deadband only when centered and away from walls, to avoid sticking.
        if (
            d_left_eff is not None
            and d_right_eff is not None
            and abs(float(d_left_eff) - float(d_right_eff)) < self.p.center_deadband
            and min(float(d_left_eff), float(d_right_eff)) > 0.75 * self.p.d0_side
        ):
            vy_rep = 0.0

        if self.p.vy_filter_tau > 0.0:
            dt_est = max(1.0 / max(self.p.control_rate, 1e-3), 1e-3)
            alpha_v = dt_est / (self.p.vy_filter_tau + dt_est)
            self.s.vy_f = float(self._ema(float(self.s.vy_f), float(vy_rep), alpha_v) or 0.0)
            vy_rep = float(self.s.vy_f)

        steering: Optional[SteeringScan] = None
        if (
            self.p.steer_enable
            and yaw_xy.size > 0
            and float(self.p.steer_yaw_max) > 1e-6
            and float(self.p.steer_yaw_step) > 1e-6
            and float(self.p.steer_yaw_half) > 1e-6
        ):
            step = float(self.p.steer_yaw_step)
            max_yaw = float(self.p.steer_yaw_max)
            angles = np.arange(-max_yaw, max_yaw + 0.5 * step, step, dtype=np.float32)
            angles = angles[np.abs(angles) <= max_yaw + 1e-6]
            if angles.size > 0:
                q = float(np.clip(float(self.p.steer_dist_percentile), 0.0, 100.0))
                min_pts = int(max(1, int(self.p.steer_min_points)))
                half = float(self.p.steer_yaw_half)
                clearances = np.empty_like(angles, dtype=np.float32)
                for i, ang in enumerate(angles):
                    dyaw = np.abs(np.arctan2(np.sin(yaw_xy - ang), np.cos(yaw_xy - ang)))
                    mask = dyaw <= half
                    d = self._dir_stat(mask, r_xy_xy, q, min_pts)
                    clearances[i] = float(self.p.xy_max_range if d is None else d)
                steering = SteeringScan(angles=angles, clearances=clearances)

        debug_points: dict[str, np.ndarray] = {}
        if debug:
            if yaw_xy.size > 0:
                debug_points["front"] = pts_xy[front_mask]
                debug_points["back"] = pts_xy[back_mask]
                debug_points["left"] = pts_xy[left_mask]
                debug_points["right"] = pts_xy[right_mask]
            else:
                debug_points["front"] = debug_points["back"] = debug_points["left"] = debug_points[
                    "right"
                ] = np.empty((0, 3), dtype=np.float32)

            # Up/down use all valid points, not the XY first-hit mask.
            if pitch_z.size > 0:
                pts_world = pts_valid
                idx_z = np.nonzero(mask_z)[0]

                def _pts_from_mask_z(mask_dir: np.ndarray) -> np.ndarray:
                    if mask_dir.size == 0 or not np.any(mask_dir):
                        return np.empty((0, 3), dtype=np.float32)
                    idx = idx_z[mask_dir]
                    return pts_world[idx]

                debug_points["up"] = _pts_from_mask_z(up_mask)
                debug_points["down"] = _pts_from_mask_z(down_mask)
            else:
                debug_points["up"] = debug_points["down"] = np.empty((0, 3), dtype=np.float32)

        distances = {
            "front": d_front,
            "back": d_back,
            "left": d_left,
            "right": d_right,
            "up": d_up,
            "down": d_down,
        }
        return RaySectorsOutput(distances, float(vy_rep), debug_points, steering)


