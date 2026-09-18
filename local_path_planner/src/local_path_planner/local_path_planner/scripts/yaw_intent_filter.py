from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .arm_commit_tracker import CommittedArmIntent
from .graph_tracker import GraphArmCandidate, GraphTrackingState


@dataclass
class YawIntentFilterParams:
    num_bins: int = 36
    potential_tau_s: float = 1.5
    candidate_sigma_rad: float = math.radians(18.0)
    switch_margin: float = 0.35
    slew_rate_rps: float = 0.9
    slowdown_distance_m: float = 1.5
    uncommitted_hold_distance_m: float = 0.4
    uncommitted_hold_timeout_s: float = 3.0
    uncommitted_creep_speed_scale: float = 0.15
    uncommitted_hold_reset_s: float = 1.0
    min_arm_length_m: float = 0.3
    commit_support_angle_rad: float = math.radians(40.0)


@dataclass(frozen=True)
class YawIntentOutput:
    yaw_ref_rad: float = 0.0
    gate_heading_rad: Optional[float] = None
    selected_arm_angle_rad: Optional[float] = None
    speed_scale: float = 1.0
    committed: bool = False


@dataclass(frozen=True)
class _Candidate:
    heading_rad: float
    weight: float
    arm_angle_rad: Optional[float]


class YawIntentFilter:
    """Maintain a sticky low-compute heading intent from graph candidates."""

    def __init__(self, params: YawIntentFilterParams) -> None:
        self.p = params
        n = max(8, int(params.num_bins))
        self._bin_angles = np.linspace(-math.pi, math.pi, n, endpoint=False, dtype=np.float32)
        self._potentials = np.zeros(n, dtype=np.float32)
        self._active_bin: Optional[int] = None
        self._yaw_ref_rad: float = 0.0
        self._initialized: bool = False
        self._commit_key: Optional[tuple[str, str, float, bool]] = None
        self._uncommitted_hold_elapsed_s = 0.0
        self._uncommitted_hold_inactive_s = 0.0

    @property
    def reference_heading(self) -> Optional[float]:
        if not self._initialized:
            return None
        return self._yaw_ref_rad

    def reset(self) -> None:
        self._potentials.fill(0.0)
        self._active_bin = None
        self._yaw_ref_rad = 0.0
        self._initialized = False
        self._commit_key = None
        self._uncommitted_hold_elapsed_s = 0.0
        self._uncommitted_hold_inactive_s = 0.0

    def update(
        self,
        gs: GraphTrackingState,
        committed_intent: Optional[CommittedArmIntent],
        dt: float,
    ) -> YawIntentOutput:
        if not gs.valid:
            return YawIntentOutput(yaw_ref_rad=self._yaw_ref_rad)

        self._sync_commit_context(committed_intent)
        self._update_uncommitted_hold_watchdog(
            gs,
            committed_intent,
            dt,
        )
        candidates = self._build_candidates(gs, committed_intent)
        if not self._initialized:
            self._yaw_ref_rad = gs.current_heading_rad
            self._initialized = True

        self._decay_potentials(dt)
        self._accumulate_candidates(candidates)
        self._update_active_bin()

        target_heading = self._target_heading_from_candidates(candidates)
        self._yaw_ref_rad = _step_toward(
            current=self._yaw_ref_rad,
            target=target_heading,
            max_step=max(1e-3, self.p.slew_rate_rps) * max(dt, 1e-3),
        )

        return YawIntentOutput(
            yaw_ref_rad=self._yaw_ref_rad,
            # Speculative gates use the smooth ref; a committed lock gates the target immediately.
            gate_heading_rad=(
                committed_intent.selected_arm_angle_rad
                if (
                    committed_intent is not None
                    and committed_intent.branch_locked
                )
                else (
                    target_heading
                    if committed_intent is not None
                    else self._yaw_ref_rad
                )
            ),
            selected_arm_angle_rad=self._selected_arm_angle(candidates),
            speed_scale=self._speed_scale(gs, committed_intent),
            committed=committed_intent is not None,
        )

    def _sync_commit_context(
        self,
        committed_intent: Optional[CommittedArmIntent],
    ) -> None:
        """Discard stale directional evidence when authority changes."""
        new_key = None
        if committed_intent is not None:
            identity_angle = (
                committed_intent.selected_arm_angle_rad
                if committed_intent.arm_identity_angle_rad is None
                else committed_intent.arm_identity_angle_rad
            )
            new_key = (
                committed_intent.event_id,
                committed_intent.mode,
                _wrap(identity_angle),
                committed_intent.branch_locked,
            )
        if new_key == self._commit_key:
            return

        # Keep yaw_ref continuous; drop old corridor evidence so a new authority is not delayed.
        self._potentials.fill(0.0)
        self._active_bin = None
        self._commit_key = new_key

    def _build_candidates(
        self,
        gs: GraphTrackingState,
        committed_intent: Optional[CommittedArmIntent],
    ) -> list[_Candidate]:
        candidates = [self._current_heading_candidate(gs)]

        if committed_intent is not None:
            candidates.extend(self._committed_arm_candidates(gs, committed_intent))
            return candidates

        candidates.extend(self._speculative_arm_candidates(gs))
        return candidates

    def _current_heading_candidate(self, gs: GraphTrackingState) -> _Candidate:
        current_weight = 1.0 + min(1.0, gs.ahead_path_length_m / max(0.1, self.p.min_arm_length_m))
        return _Candidate(
            heading_rad=gs.current_heading_rad,
            weight=current_weight,
            arm_angle_rad=None,
        )

    def _committed_arm_candidates(
        self,
        gs: GraphTrackingState,
        committed_intent: CommittedArmIntent,
    ) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        if committed_intent.branch_locked:
            candidates.append(
                _Candidate(
                    heading_rad=committed_intent.selected_arm_angle_rad,
                    weight=5.0,
                    arm_angle_rad=committed_intent.selected_arm_angle_rad,
                )
            )

        # Wait for a local junction before treating the commit as geometric support.
        if not gs.approaching_junction:
            return candidates

        for arm in gs.arm_candidates:
            if arm.edge_length_m < self.p.min_arm_length_m:
                continue
            diff = abs(_wrap(arm.angle_rad - committed_intent.selected_arm_angle_rad))
            if diff > self.p.commit_support_angle_rad:
                continue
            candidates.append(
                self._candidate_from_arm(
                    arm=arm,
                    reference_angle_rad=committed_intent.selected_arm_angle_rad,
                    base_gain=3.5,
                )
            )
        if candidates and not committed_intent.branch_locked:
            candidates.append(
                _Candidate(
                    heading_rad=committed_intent.selected_arm_angle_rad,
                    weight=1.5,
                    arm_angle_rad=None,
                )
            )
        return candidates

    def _speculative_arm_candidates(self, gs: GraphTrackingState) -> list[_Candidate]:
        if gs.approaching_junction:
            return []
        reference = self._yaw_ref_rad if self._initialized else gs.current_heading_rad
        candidates: list[_Candidate] = []
        for arm in gs.arm_candidates:
            if arm.is_backtrack:
                continue
            if arm.edge_length_m < self.p.min_arm_length_m:
                continue
            candidates.append(self._candidate_from_arm(arm, reference, 1.5))
        return candidates

    def _candidate_from_arm(
        self,
        arm: GraphArmCandidate,
        reference_angle_rad: float,
        base_gain: float,
    ) -> _Candidate:
        diff = abs(_wrap(arm.lookahead_heading_rad - reference_angle_rad))
        continuity = max(0.0, math.cos(diff))
        length_scale = min(1.0, arm.edge_length_m / max(0.1, 3.0 * self.p.min_arm_length_m))
        weight = base_gain * (0.55 + 0.45 * continuity) * (0.55 + 0.45 * length_scale)
        return _Candidate(
            heading_rad=arm.lookahead_heading_rad,
            weight=weight,
            arm_angle_rad=arm.angle_rad,
        )

    def _decay_potentials(self, dt: float) -> None:
        tau = max(self.p.potential_tau_s, 1e-3)
        decay = math.exp(-max(dt, 0.0) / tau)
        self._potentials *= float(decay)

    def _accumulate_candidates(self, candidates: list[_Candidate]) -> None:
        sigma = max(self.p.candidate_sigma_rad, math.radians(3.0))
        for cand in candidates:
            diffs = np.arctan2(
                np.sin(self._bin_angles - float(cand.heading_rad)),
                np.cos(self._bin_angles - float(cand.heading_rad)),
            )
            kernel = np.exp(-0.5 * np.square(diffs / sigma)).astype(np.float32, copy=False)
            self._potentials += np.float32(cand.weight) * kernel

    def _update_active_bin(self) -> None:
        best_idx = int(np.argmax(self._potentials))
        if self._active_bin is None:
            self._active_bin = best_idx
            return
        current_val = float(self._potentials[self._active_bin])
        best_val = float(self._potentials[best_idx])
        if best_idx != self._active_bin and best_val <= current_val + self.p.switch_margin:
            return
        self._active_bin = best_idx

    def _target_heading_from_candidates(self, candidates: list[_Candidate]) -> float:
        if self._active_bin is None:
            return self._yaw_ref_rad
        center = float(self._bin_angles[self._active_bin])
        if not candidates:
            return center

        sin_sum = 0.0
        cos_sum = 0.0
        total = 0.0
        for cand in candidates:
            diff = abs(_wrap(cand.heading_rad - center))
            local_w = float(cand.weight) * math.exp(
                -0.5 * (diff / max(self.p.candidate_sigma_rad, math.radians(3.0))) ** 2
            )
            if local_w <= 1e-6:
                continue
            sin_sum += local_w * math.sin(cand.heading_rad)
            cos_sum += local_w * math.cos(cand.heading_rad)
            total += local_w
        if total <= 1e-6:
            return center
        return float(math.atan2(sin_sum, cos_sum))

    def _selected_arm_angle(self, candidates: list[_Candidate]) -> Optional[float]:
        if self._active_bin is None:
            return None
        center = float(self._bin_angles[self._active_bin])
        best_arm: Optional[float] = None
        best_score = -float("inf")
        for cand in candidates:
            if cand.arm_angle_rad is None:
                continue
            diff = abs(_wrap(cand.heading_rad - center))
            score = float(cand.weight) - diff
            if score > best_score:
                best_score = score
                best_arm = cand.arm_angle_rad
        return best_arm

    def _speed_scale(
        self,
        gs: GraphTrackingState,
        committed_intent: Optional[CommittedArmIntent],
    ) -> float:
        if not (gs.approaching_junction or gs.approaching_leaf):
            return 1.0
        slowdown = max(self.p.slowdown_distance_m, 1e-3)
        if gs.approaching_junction and committed_intent is None:
            hold_distance = float(
                np.clip(self.p.uncommitted_hold_distance_m, 0.0, slowdown)
            )
            travel_window = max(0.05, slowdown - hold_distance)
            approach_scale = float(
                np.clip(
                    (gs.dist_to_ahead_node_m - hold_distance) / travel_window,
                    0.0,
                    1.0,
                )
            )
            timeout_s = max(0.0, self.p.uncommitted_hold_timeout_s)
            if self._uncommitted_hold_elapsed_s < timeout_s:
                return approach_scale
            creep_scale = float(
                np.clip(
                    self.p.uncommitted_creep_speed_scale,
                    0.0,
                    1.0,
                )
            )
            return max(approach_scale, creep_scale)
        return float(np.clip(0.3 + 0.7 * gs.dist_to_ahead_node_m / slowdown, 0.3, 1.0))

    def _update_uncommitted_hold_watchdog(
        self,
        gs: GraphTrackingState,
        committed_intent: Optional[CommittedArmIntent],
        dt: float,
    ) -> None:
        if committed_intent is not None:
            self._uncommitted_hold_elapsed_s = 0.0
            self._uncommitted_hold_inactive_s = 0.0
            return

        hold_distance = max(
            0.0,
            self.p.uncommitted_hold_distance_m,
        )
        holding = (
            gs.approaching_junction
            and gs.dist_to_ahead_node_m <= hold_distance
        )
        step_s = max(0.0, float(dt))
        if holding:
            self._uncommitted_hold_elapsed_s += step_s
            self._uncommitted_hold_inactive_s = 0.0
            return

        self._uncommitted_hold_inactive_s += step_s
        if (
            self._uncommitted_hold_inactive_s
            >= max(0.0, self.p.uncommitted_hold_reset_s)
        ):
            self._uncommitted_hold_elapsed_s = 0.0
            self._uncommitted_hold_inactive_s = 0.0


def _wrap(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def _step_toward(current: float, target: float, max_step: float) -> float:
    err = _wrap(target - current)
    step = float(np.clip(err, -max_step, max_step))
    return _wrap(current + step)
