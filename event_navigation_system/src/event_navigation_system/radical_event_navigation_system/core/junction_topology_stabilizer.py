"""Require stable junction degree before publishing a navigation event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable


@dataclass
class _TopologyState:
    degree: int
    stable_since_s: float


class JunctionTopologyStabilizer:
    """Track continuous topology stability independently for each map area."""

    def __init__(self, settle_duration_s: float) -> None:
        """Initialize with the required continuous-stability duration."""
        self._settle_duration_s = max(0.0, float(settle_duration_s))
        self._states: Dict[Hashable, _TopologyState] = {}

    def reset(self) -> None:
        """Clear all pending topology observations."""
        self._states.clear()

    def invalidate(self, area_key: Hashable) -> None:
        """Discard pending evidence for an area that is not a junction."""
        self._states.pop(area_key, None)

    def observe(self, area_key: Hashable, degree: int, now_s: float) -> bool:
        """Return true after the same junction degree remains continuously stable."""
        observed_degree = int(degree)
        now = float(now_s)
        previous = self._states.get(area_key)
        if previous is None or previous.degree != observed_degree:
            self._states[area_key] = _TopologyState(
                degree=observed_degree,
                stable_since_s=now,
            )
            return self._settle_duration_s <= 0.0
        elapsed_s = max(0.0, now - previous.stable_since_s)
        return elapsed_s + 1e-9 >= self._settle_duration_s
