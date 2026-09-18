"""Project leaf progress onto a fixed edge direction, not a moving node."""

import math


class LeafProgressTracker:
    def __init__(self, near_m=0.30, timeout_s=2.0, progress_m=0.10):
        self.near_m = max(0.0, float(near_m))
        self.timeout_s = max(0.1, float(timeout_s))
        self.progress_m = max(0.01, float(progress_m))
        self.reset()

    def reset(self):
        self._reference = None
        self._previous = None
        self._direction = None
        self._start_s = None
        self._last_s = None
        self.wait_s = 0.0

    def update(self, *, eligible, remaining_m, position_xy, heading_rad, now_s):
        x, y = map(float, position_xy)
        if (not eligible or not all(map(math.isfinite,
                (remaining_m, x, y, heading_rad, now_s)))
                or not 0.0 <= remaining_m <= self.near_m):
            self.reset()
            return False

        direction = (math.cos(heading_rad), math.sin(heading_rad))
        if self._previous is not None:
            dt = now_s - self._last_s
            jump = math.hypot(x - self._previous[0], y - self._previous[1])
            alignment = sum(a * b for a, b in zip(direction, self._direction))
            # Clock gaps, pose jumps, or a new edge are not a stall.
            if dt < 0.0 or dt > 1.0 or jump > max(0.30, 2.0 * dt) or alignment < 0.707:
                self.reset()

        if self._reference is None:
            self._reference = (x, y)
            self._direction = direction
            self._start_s = now_s

        progress = ((x - self._reference[0]) * self._direction[0]
                    + (y - self._reference[1]) * self._direction[1])
        if progress >= self.progress_m:
            self._reference = (x, y)
            self._direction = direction
            self._start_s = now_s

        self._previous = (x, y)
        self._last_s = now_s
        self.wait_s = max(0.0, now_s - self._start_s)
        return self.wait_s >= self.timeout_s
