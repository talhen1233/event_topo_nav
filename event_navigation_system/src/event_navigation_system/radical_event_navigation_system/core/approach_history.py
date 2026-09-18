"""Estimate a junction entry arm from pre-turn position history."""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Sequence

import numpy as np


class ApproachHistory:
    """Retain recent XY positions and recover the physical approach direction."""

    def __init__(
        self,
        max_distance_m: float,
        sample_distance_m: float,
        min_sample_spacing_m: float,
        max_arm_error_rad: float,
    ) -> None:
        """Initialize a distance-bounded, spatially downsampled history."""
        self._max_distance_m = max(0.2, float(max_distance_m))
        self._sample_distance_m = max(0.1, float(sample_distance_m))
        self._min_sample_spacing_m = max(
            0.01,
            float(min_sample_spacing_m),
        )
        self._max_arm_error_rad = max(0.0, float(max_arm_error_rad))
        self._samples: Deque[np.ndarray] = deque()

    def reset(self) -> None:
        """Clear all retained position samples."""
        self._samples.clear()

    def update(self, xy: np.ndarray) -> None:
        """Append meaningful XY motion and retain a bounded path distance."""
        point = np.asarray(xy, dtype=np.float32).reshape(2).copy()
        if (
            self._samples
            and np.linalg.norm(point - self._samples[-1])
            < self._min_sample_spacing_m
        ):
            return
        self._samples.append(point)
        if len(self._samples) < 3:
            return

        positions = np.vstack(self._samples)
        segment_lengths = np.linalg.norm(
            np.diff(positions, axis=0),
            axis=1,
        )
        path_from_newest = np.cumsum(segment_lengths[::-1])[::-1]
        keep_from = int(
            np.searchsorted(
                -path_from_newest,
                -self._max_distance_m,
                side="left",
            )
        )
        for _ in range(keep_from):
            self._samples.popleft()

    def estimate_entry_arm(
        self,
        junction_xy: np.ndarray,
        arm_angles_rad: Sequence[float],
    ) -> Optional[float]:
        """Return the arm nearest a historical point behind the junction."""
        if not self._samples or not arm_angles_rad:
            return None

        center = np.asarray(junction_xy, dtype=np.float32).reshape(2)
        positions = np.vstack(self._samples)
        offsets = positions - center
        distances = np.linalg.norm(offsets, axis=1)
        valid = distances >= 0.5 * self._sample_distance_m
        if not np.any(valid):
            return None

        valid_indices = np.flatnonzero(valid)
        target_index = valid_indices[
            int(
                np.argmin(
                    np.abs(
                        distances[valid_indices] - self._sample_distance_m
                    )
                )
            )
        ]
        offset = offsets[target_index]
        approach_angle = math.atan2(float(offset[1]), float(offset[0]))
        arms = np.asarray(arm_angles_rad, dtype=np.float32)
        errors = np.abs(
            np.arctan2(
                np.sin(arms - approach_angle),
                np.cos(arms - approach_angle),
            )
        )
        best_index = int(np.argmin(errors))
        if float(errors[best_index]) > self._max_arm_error_rad:
            return None
        return float(arms[best_index])
