"""Front stop, opening search, and dead-end escalation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SafetyControllerParams:
    front_block_dist_m: float = 0.35
    front_clear_dist_m: float = 0.75
    front_block_hysteresis_m: float = 0.20
    front_block_width_margin_m: float = 0.075
    front_block_dynamic_max_m: float = 1.2
    front_width_filter_tau_s: float = 0.25
    front_stop_dist_m: float = 0.15
    front_slow_dist_m: float = 0.5
    max_decel_mps2: float = 2.0
    reaction_time_s: float = 0.25
    vehicle_radius_m: float = 0.15
    stop_margin_m: float = 0.10
    slow_margin_m: float = 0.40
    open_dist_m: float = 1.5
    block_debounce_s: float = 0.25
    dead_end_time_s: float = 1.0
    turn_search_max_yaw_rad: float = math.radians(70.0)
    yaw_rate_max: float = 0.4
    xy_max_range: float = 2.0


@dataclass
class SafetyOutput:
    collision_stop: bool = False
    vx_scale: float = 1.0
    mode: str = "NONE"
    yaw_rate_override: Optional[float] = None


class SafetyController:
    def __init__(self, params: SafetyControllerParams) -> None:
        self.p = params
        self._mode: str = "NONE"
        self._graph_recovery = False
        self._front_block_start_s: Optional[float] = None
        self._front_block_dist_active_m: float = params.front_block_dist_m
        self._front_clear_dist_active_m: float = max(
            params.front_clear_dist_m,
            params.front_block_dist_m + params.front_block_hysteresis_m,
        )
        # Latch the entry threshold so a zero vx command cannot shrink it.
        self._front_block_latched_m: Optional[float] = None
        self._front_block_latched_terminal: bool = False
        self._no_opening_start_s: Optional[float] = None
        self._turn_side: Optional[str] = None
        self._turn_ref_yaw: Optional[float] = None
        self._turn_allow_left: bool = False
        self._turn_allow_right: bool = False
        self._turn_switched: bool = False
        self._passage_width_estimate_m: Optional[float] = None
        self._last_width_update_s: Optional[float] = None
        self._active_stop_dist_m: float = params.front_stop_dist_m
        self._active_slow_dist_m: float = params.front_slow_dist_m

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def graph_recovery(self) -> bool:
        return self._graph_recovery

    @property
    def active_front_block_dist_m(self) -> float:
        return float(self._front_block_dist_active_m)

    @property
    def active_front_clear_dist_m(self) -> float:
        return float(self._front_clear_dist_active_m)

    @property
    def passage_width_estimate_m(self) -> Optional[float]:
        if self._passage_width_estimate_m is None:
            return None
        return float(self._passage_width_estimate_m)

    @property
    def active_stop_dist_m(self) -> float:
        return float(self._active_stop_dist_m)

    @property
    def active_slow_dist_m(self) -> float:
        return float(self._active_slow_dist_m)

    def reset(self) -> None:
        self._mode = "NONE"
        self._graph_recovery = False
        self._front_block_start_s = None
        self._front_block_dist_active_m = self.p.front_block_dist_m
        self._front_clear_dist_active_m = max(
            self.p.front_clear_dist_m,
            self.p.front_block_dist_m + self.p.front_block_hysteresis_m,
        )
        self._front_block_latched_m = None
        self._front_block_latched_terminal = False
        self._no_opening_start_s = None
        self._turn_side = None
        self._turn_ref_yaw = None
        self._turn_allow_left = False
        self._turn_allow_right = False
        self._turn_switched = False
        self._passage_width_estimate_m = None
        self._last_width_update_s = None
        self._active_stop_dist_m = self.p.front_stop_dist_m
        self._active_slow_dist_m = self.p.front_slow_dist_m

    def update(
        self,
        d_front: Optional[float],
        d_left: Optional[float],
        d_right: Optional[float],
        graph_has_continuation: bool,
        dead_end_candidate: bool,
        drone_yaw: float,
        now_s: float,
        forward_speed_mps: float = 0.0,
        graph_stalled: bool = False,
    ) -> SafetyOutput:
        self._active_stop_dist_m, self._active_slow_dist_m = self._stopping_distances(
            forward_speed_mps
        )

        configured_block_dist = self._effective_front_block_dist(
            d_left=d_left,
            d_right=d_right,
            dead_end_candidate=dead_end_candidate,
            now_s=now_s,
        )
        computed_block_dist = max(
            configured_block_dist,
            self._active_stop_dist_m,
        )

        if (
            self._mode == "NONE"
            and self._front_block_latched_terminal
            and not dead_end_candidate
        ):
            ordinary_block_dist = max(
                float(self.p.front_block_dist_m),
                self._active_stop_dist_m,
            )
            ordinary_blocked = (
                d_front is not None and d_front < ordinary_block_dist
            )
            if not ordinary_blocked:
                self._front_block_start_s = None
                self._front_block_latched_m = None
                self._front_block_latched_terminal = False

        self._front_block_dist_active_m = max(
            computed_block_dist,
            float(self._front_block_latched_m or 0.0),
        )
        self._front_clear_dist_active_m = max(
            float(self.p.front_clear_dist_m),
            self._front_block_dist_active_m
            + max(0.0, float(self.p.front_block_hysteresis_m)),
        )

        collision_stop = (
            d_front is not None
            and d_front < self._active_stop_dist_m
        )

        vx_scale = self._forward_speed_scale(d_front)

        front_blocked = (
            d_front is not None
            and d_front < self._front_block_dist_active_m
        )
        front_missing = d_front is None
        front_clear = (
            d_front is not None
            and d_front > self._front_clear_dist_active_m
        )

        # Graph-end stall reuses the yaw search; rotation never authorizes vx.
        if graph_stalled and not self._graph_recovery and self._mode in ("NONE", "BLOCKED"):
            self._graph_recovery = True
            sides = [("left", float(d_left or 0.0)), ("right", float(d_right or 0.0))]
            self._enter_turn_to_opening(sides, d_left, d_right, drone_yaw)

        if self._graph_recovery:
            ordinary_block = max(self.p.front_block_dist_m, self._active_stop_dist_m)
            if self._mode != "DEAD_END" and graph_has_continuation and (
                d_front is None or d_front > ordinary_block
            ):
                self.reset()
                return SafetyOutput(collision_stop=collision_stop, vx_scale=vx_scale)
            if self._mode == "DEAD_END":
                return SafetyOutput(collision_stop=True, vx_scale=0.0, mode="DEAD_END")
            yaw_rate = self._turn_to_opening_step(drone_yaw, d_left, d_right, now_s)
            return SafetyOutput(
                collision_stop=True, vx_scale=0.0, mode=self._mode,
                yaw_rate_override=yaw_rate if self._mode == "TURN_TO_OPENING" else None,
            )

        if self._mode == "DEAD_END":
            if front_missing or front_clear:
                self.reset()
                return SafetyOutput(
                    collision_stop=collision_stop,
                    vx_scale=vx_scale,
                    mode="NONE",
                )
            return SafetyOutput(
                collision_stop=True,
                vx_scale=0.0,
                mode="DEAD_END",
                yaw_rate_override=None,
            )

        if self._mode == "TURN_TO_OPENING":
            if front_clear and graph_has_continuation:
                self.reset()
                return SafetyOutput(
                    collision_stop=collision_stop,
                    vx_scale=vx_scale,
                    mode="NONE",
                )

            yaw_rate = self._turn_to_opening_step(drone_yaw, d_left, d_right, now_s)
            if self._mode == "DEAD_END":
                return SafetyOutput(
                    collision_stop=True,
                    vx_scale=0.0,
                    mode="DEAD_END",
                    yaw_rate_override=None,
                )
            return SafetyOutput(
                collision_stop=True,
                vx_scale=0.0,
                mode="TURN_TO_OPENING",
                yaw_rate_override=yaw_rate,
            )

        if self._mode == "BLOCKED":
            if front_missing:
                self.reset()
                return SafetyOutput(
                    collision_stop=collision_stop,
                    vx_scale=vx_scale,
                    mode="NONE",
                )

            if front_clear and not self._front_block_latched_terminal:
                self.reset()
                return SafetyOutput(
                    collision_stop=collision_stop,
                    vx_scale=vx_scale,
                    mode="NONE",
                )

            ordinary_block_dist = max(
                float(self.p.front_block_dist_m),
                self._active_stop_dist_m,
            )
            if (
                graph_has_continuation
                and self._front_block_latched_terminal
                and d_front is not None
                and d_front >= ordinary_block_dist
            ):
                self.reset()
                return SafetyOutput(
                    collision_stop=collision_stop,
                    vx_scale=vx_scale,
                    mode="NONE",
                )

            if graph_has_continuation:
                self._no_opening_start_s = None
                return SafetyOutput(
                    collision_stop=True,
                    vx_scale=0.0,
                    mode="BLOCKED",
                )

            opening = self._find_opening(d_left, d_right)
            if opening:
                self._enter_turn_to_opening(opening, d_left, d_right, drone_yaw)
                return SafetyOutput(
                    collision_stop=True,
                    vx_scale=0.0,
                    mode="TURN_TO_OPENING",
                    yaw_rate_override=0.0,
                )

            if self._no_opening_start_s is None:
                self._no_opening_start_s = now_s
            if (now_s - self._no_opening_start_s) >= self.p.dead_end_time_s:
                self._mode = "DEAD_END"
                return SafetyOutput(
                    collision_stop=True,
                    vx_scale=0.0,
                    mode="DEAD_END",
                    yaw_rate_override=0.0,
                )
            return SafetyOutput(
                collision_stop=True,
                vx_scale=0.0,
                mode="BLOCKED",
            )

        if front_blocked:
            if self._front_block_start_s is None:
                self._front_block_start_s = now_s
                self._front_block_latched_m = self._front_block_dist_active_m
                self._front_block_latched_terminal = bool(dead_end_candidate)
            else:
                self._front_block_latched_m = max(
                    float(self._front_block_latched_m or 0.0),
                    self._front_block_dist_active_m,
                )
                self._front_block_latched_terminal = bool(
                    self._front_block_latched_terminal or dead_end_candidate
                )
            if (now_s - self._front_block_start_s) >= self.p.block_debounce_s:
                self._mode = "BLOCKED"
                self._no_opening_start_s = None
                return SafetyOutput(
                    collision_stop=True,
                    vx_scale=0.0,
                    mode="BLOCKED",
                )
        else:
            self._front_block_start_s = None
            self._front_block_latched_m = None
            self._front_block_latched_terminal = False

        return SafetyOutput(
            collision_stop=collision_stop,
            vx_scale=vx_scale,
            mode="NONE",
        )

    def _forward_speed_scale(self, d_front: Optional[float]) -> float:
        if d_front is None:
            return 1.0
        if d_front <= self._active_stop_dist_m:
            return 0.0
        if d_front >= self._active_slow_dist_m:
            return 1.0
        span = self._active_slow_dist_m - self._active_stop_dist_m
        if span <= 1e-6:
            return 1.0
        return float(np.clip((d_front - self._active_stop_dist_m) / span, 0.0, 1.0))

    def _stopping_distances(self, forward_speed_mps: float) -> tuple[float, float]:
        """Stop/slow distances from speed, braking, and the body footprint."""
        speed = max(0.0, float(forward_speed_mps))
        decel = max(1e-3, float(self.p.max_decel_mps2))
        reaction = max(0.0, float(self.p.reaction_time_s))
        footprint = max(0.0, float(self.p.vehicle_radius_m))
        margin = max(0.0, float(self.p.stop_margin_m))
        braking = speed * speed / (2.0 * decel)
        dynamic_stop = braking + speed * reaction + footprint + margin
        stop_dist = max(float(self.p.front_stop_dist_m), dynamic_stop)
        slow_dist = max(
            float(self.p.front_slow_dist_m),
            stop_dist + max(0.0, float(self.p.slow_margin_m)),
        )
        return stop_dist, slow_dist

    def _find_opening(
        self,
        d_left: Optional[float],
        d_right: Optional[float],
    ) -> Optional[list[tuple[str, float]]]:
        sides: list[tuple[str, float]] = []
        if d_left is not None and d_left > self.p.open_dist_m:
            sides.append(("left", float(d_left)))
        if d_right is not None and d_right > self.p.open_dist_m:
            sides.append(("right", float(d_right)))
        return sides if sides else None

    def _enter_turn_to_opening(
        self,
        sides: list[tuple[str, float]],
        d_left: Optional[float],
        d_right: Optional[float],
        drone_yaw: float,
    ) -> None:
        self._mode = "TURN_TO_OPENING"
        has_left = any(s == "left" for s, _ in sides)
        has_right = any(s == "right" for s, _ in sides)
        best_side = max(sides, key=lambda kv: kv[1])[0]
        self._turn_side = best_side
        self._turn_ref_yaw = drone_yaw
        self._turn_allow_left = has_left
        self._turn_allow_right = has_right
        self._turn_switched = False
        self._no_opening_start_s = None

    def _turn_to_opening_step(
        self,
        drone_yaw: float,
        d_left: Optional[float],
        d_right: Optional[float],
        now_s: float,
    ) -> float:
        max_yaw = self.p.turn_search_max_yaw_rad
        if max_yaw > 0.0 and self._turn_ref_yaw is not None:
            delta = _wrap(drone_yaw - self._turn_ref_yaw)
            if self._turn_side == "left" and delta >= max_yaw:
                if self._turn_allow_right and not self._turn_switched:
                    self._turn_side = "right"
                    self._turn_switched = True
                else:
                    self._mode = "DEAD_END"
                    return 0.0
            elif self._turn_side == "right" and delta <= -max_yaw:
                if self._turn_allow_left and not self._turn_switched:
                    self._turn_side = "left"
                    self._turn_switched = True
                else:
                    self._mode = "DEAD_END"
                    return 0.0

        rate = 0.5 * self.p.yaw_rate_max
        if self._turn_side == "left":
            return rate
        return -rate

    def _effective_front_block_dist(
        self,
        *,
        d_left: Optional[float],
        d_right: Optional[float],
        dead_end_candidate: bool,
        now_s: float,
    ) -> float:
        base = float(self.p.front_block_dist_m)
        if not dead_end_candidate:
            return base

        width_estimate = self._update_passage_width_estimate(d_left, d_right, now_s)
        if width_estimate is None:
            return base

        dynamic = 0.5 * width_estimate + float(self.p.front_block_width_margin_m)
        max_dynamic = max(base, float(self.p.front_block_dynamic_max_m))
        return float(np.clip(dynamic, base, max_dynamic))

    def _update_passage_width_estimate(
        self,
        d_left: Optional[float],
        d_right: Optional[float],
        now_s: float,
    ) -> Optional[float]:
        width_raw: Optional[float] = None
        if d_left is not None and d_right is not None:
            width_raw = float(np.clip(d_left + d_right, 0.0, 2.0 * self.p.xy_max_range))

        if width_raw is None:
            return self.passage_width_estimate_m

        if self._passage_width_estimate_m is None or self._last_width_update_s is None:
            self._passage_width_estimate_m = width_raw
            self._last_width_update_s = now_s
            return width_raw

        dt = max(0.0, now_s - self._last_width_update_s)
        self._last_width_update_s = now_s
        tau = max(1e-3, float(self.p.front_width_filter_tau_s))
        alpha = 1.0 - math.exp(-dt / tau)
        self._passage_width_estimate_m = float(
            (1.0 - alpha) * self._passage_width_estimate_m + alpha * width_raw
        )
        return self._passage_width_estimate_m


def _wrap(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))
