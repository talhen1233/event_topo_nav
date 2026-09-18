from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass(frozen=True)
class CommittedArmIntent:
    event_id: str
    mode: str
    event_center_xy: np.ndarray
    gate_radius_m: float
    selected_arm_angle_rad: float
    branch_locked: bool = False
    arm_identity_angle_rad: Optional[float] = None


GraphTargetAction = Literal["ignored", "clear", "reverse", "commit"]


class ArmCommitTracker:
    """Track authoritative committed arm intent from the navigation node."""

    def __init__(self) -> None:
        self._intent: Optional[CommittedArmIntent] = None
        self._reverse_mode: Optional[str] = None

    def reset(self) -> None:
        self._intent = None
        self._reverse_mode = None

    def on_graph_target(self, json_str: str) -> GraphTargetAction:
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return "ignored"
        if not isinstance(data, dict):
            return "ignored"

        kind = str(data.get("kind", "")).strip().lower()
        if kind == "clear":
            self.reset()
            return "clear"

        if kind == "reverse":
            mode = data.get("mode")
            self._intent = None
            self._reverse_mode = str(mode).strip().lower() if isinstance(mode, str) and mode.strip() else "reverse"
            return "reverse"

        if kind != "commit":
            return "ignored"

        event_id = data.get("event_id")
        mode = data.get("mode")
        center_xy = data.get("event_center_xy")
        gate_radius_m = data.get("gate_radius_m")
        arm_angle = data.get("selected_arm_angle_rad")
        if not isinstance(event_id, str) or not event_id:
            return "ignored"
        if not isinstance(mode, str) or not mode:
            return "ignored"
        if not isinstance(center_xy, list) or len(center_xy) < 2:
            return "ignored"
        try:
            center = np.array([float(center_xy[0]), float(center_xy[1])], dtype=np.float32)
            radius = float(gate_radius_m)
            selected = _wrap(float(arm_angle))
        except (TypeError, ValueError):
            return "ignored"

        self._intent = CommittedArmIntent(
            event_id=event_id,
            mode=mode,
            event_center_xy=center,
            gate_radius_m=max(0.0, radius),
            selected_arm_angle_rad=selected,
        )
        self._reverse_mode = None
        return "commit"

    @property
    def active_intent(self) -> Optional[CommittedArmIntent]:
        return self._intent

    @property
    def has_intent(self) -> bool:
        return self._intent is not None

    @property
    def reverse_requested(self) -> bool:
        return self._reverse_mode is not None

    @property
    def reverse_mode(self) -> Optional[str]:
        return self._reverse_mode


def _wrap(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))
