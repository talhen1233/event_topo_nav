"""Latch committed graph arms until the drone enters the selected branch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .arm_direction_estimator import SmoothedArmDirection
from .arm_commit_tracker import CommittedArmIntent
from .graph_tracker import GraphTrackingState


@dataclass(frozen=True)
class CommittedArmLockParams:
    entry_distance_m: float
    entry_cross_track_m: float
    heading_tolerance_rad: float
    max_distance_m: float
    direction_smoothing_tau_s: float
    direction_max_rate_rad_s: float
    direction_min_confidence: float
    acquisition_timeout_s: float


class CommittedArmLock:
    def __init__(self, params: CommittedArmLockParams) -> None:
        self.p = params
        self._intent: Optional[CommittedArmIntent] = None
        self._locked_heading_rad: Optional[float] = None
        self._direction_filter: Optional[SmoothedArmDirection] = None
        self._release_requested = False
        self._entry_confirmed = False
        self._acquisition_elapsed_s = 0.0

    @property
    def active_intent(self) -> Optional[CommittedArmIntent]:
        if self._intent is None:
            return None
        if self._locked_heading_rad is None:
            return self._intent
        return CommittedArmIntent(
            event_id=self._intent.event_id,
            mode=self._intent.mode,
            event_center_xy=self._intent.event_center_xy,
            gate_radius_m=self._intent.gate_radius_m,
            selected_arm_angle_rad=self._locked_heading_rad,
            branch_locked=True,
            arm_identity_angle_rad=self._intent.selected_arm_angle_rad,
        )

    @property
    def branch_locked(self) -> bool:
        return self._locked_heading_rad is not None

    @property
    def entry_confirmed(self) -> bool:
        return self._entry_confirmed

    @property
    def release_requested(self) -> bool:
        return self._release_requested

    def activate(self, intent: CommittedArmIntent) -> None:
        key = (intent.event_id, intent.mode, intent.selected_arm_angle_rad)
        current = self._intent
        current_key = (
            None
            if current is None
            else (
                current.event_id,
                current.mode,
                current.selected_arm_angle_rad,
            )
        )
        if key == current_key:
            self._release_requested = False
            return
        self._intent = intent
        self._locked_heading_rad = None
        self._direction_filter = None
        self._release_requested = False
        self._entry_confirmed = False
        self._acquisition_elapsed_s = 0.0

    def request_release(self) -> None:
        """Release now or defer until an already-latched arm is entered."""
        if self._intent is None:
            return
        self._release_requested = True
        if self._entry_confirmed:
            self.reset()

    def reset(self) -> None:
        self._intent = None
        self._locked_heading_rad = None
        self._direction_filter = None
        self._release_requested = False
        self._entry_confirmed = False
        self._acquisition_elapsed_s = 0.0

    def update(
        self,
        drone_xy: np.ndarray,
        graph: GraphTrackingState,
        dt: float,
    ) -> bool:
        if self._intent is None:
            return False

        offset = (
            np.asarray(drone_xy, dtype=np.float32)
            - self._intent.event_center_xy
        )
        radial_distance = float(np.linalg.norm(offset))

        if self._locked_heading_rad is None:
            self._try_lock_branch(graph)
            if graph.approaching_junction:
                self._acquisition_elapsed_s += max(0.0, float(dt))
            if (
                self._locked_heading_rad is None
                and self._acquisition_elapsed_s
                >= max(0.1, self.p.acquisition_timeout_s)
            ):
                self.reset()
                return False
        else:
            self._refine_locked_direction(graph, dt)
        if (
            (self._release_requested or self.branch_locked)
            and radial_distance >= max(0.1, self.p.max_distance_m)
        ):
            self.reset()
            return False
        if self._locked_heading_rad is None:
            return False

        direction = np.array(
            [
                math.cos(self._locked_heading_rad),
                math.sin(self._locked_heading_rad),
            ],
            dtype=np.float32,
        )
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        along_track = float(np.dot(offset, direction))
        cross_track = abs(float(np.dot(offset, normal)))
        heading_error = abs(
            _wrap(graph.current_heading_rad - self._locked_heading_rad)
        )
        self._entry_confirmed = (
            graph.valid
            and along_track >= max(0.0, self.p.entry_distance_m)
            and cross_track <= max(0.0, self.p.entry_cross_track_m)
            and heading_error <= max(0.0, self.p.heading_tolerance_rad)
        )
        entry_confirmed = self._entry_confirmed
        if entry_confirmed and self._release_requested:
            self.reset()
        return entry_confirmed

    def _try_lock_branch(self, graph: GraphTrackingState) -> None:
        if self._intent is None or not graph.approaching_junction:
            return
        allow_backtrack = self._allows_backtrack_candidate()
        candidates = [
            arm
            for arm in graph.arm_candidates
            if (
                (allow_backtrack or not arm.is_backtrack)
                and arm.direction_confidence
                >= max(0.0, self.p.direction_min_confidence)
            )
        ]
        if not candidates:
            return
        selected = min(
            candidates,
            key=lambda arm: abs(
                _wrap(arm.angle_rad - self._intent.selected_arm_angle_rad)
            ),
        )
        error = abs(
            _wrap(selected.angle_rad - self._intent.selected_arm_angle_rad)
        )
        if error <= max(0.0, self.p.heading_tolerance_rad):
            self._locked_heading_rad = float(selected.angle_rad)
            self._direction_filter = SmoothedArmDirection(
                selected.angle_rad,
                smoothing_tau_s=self.p.direction_smoothing_tau_s,
                max_rate_rad_s=self.p.direction_max_rate_rad_s,
            )

    def _refine_locked_direction(
        self,
        graph: GraphTrackingState,
        dt: float,
    ) -> None:
        if (
            self._intent is None
            or self._locked_heading_rad is None
            or self._direction_filter is None
        ):
            return
        allow_backtrack = self._allows_backtrack_candidate()
        candidates = [
            arm
            for arm in graph.arm_candidates
            if (
                (allow_backtrack or not arm.is_backtrack)
                and arm.direction_confidence
                >= max(0.0, self.p.direction_min_confidence)
            )
        ]
        if not candidates:
            return
        selected = min(
            candidates,
            key=lambda arm: abs(
                _wrap(arm.angle_rad - self._locked_heading_rad)
            ),
        )
        association_error = abs(
            _wrap(selected.angle_rad - self._locked_heading_rad)
        )
        identity_error = abs(
            _wrap(
                selected.angle_rad
                - self._intent.selected_arm_angle_rad
            )
        )
        tolerance = max(0.0, self.p.heading_tolerance_rad)
        if association_error > tolerance or identity_error > tolerance:
            return
        self._locked_heading_rad = self._direction_filter.update(
            selected.angle_rad,
            selected.direction_confidence,
            dt,
        )

    def _allows_backtrack_candidate(self) -> bool:
        if self._intent is None:
            return False
        return self._intent.mode in {
            "backtrack",
            "return",
        }


def _wrap(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))
