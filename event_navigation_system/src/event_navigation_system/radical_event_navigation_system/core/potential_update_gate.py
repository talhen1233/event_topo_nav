"""Gate event-potential updates by translation, rotation, or elapsed time."""

from __future__ import annotations

import math
from typing import Optional

from .navigation_event import Pose3D


def potential_update_due(
    current_pose: Pose3D,
    previous_pose: Optional[Pose3D],
    *,
    previous_update_s: Optional[float],
    now_s: float,
    min_distance_m: float,
    min_yaw_deg: float,
    max_interval_s: float,
) -> bool:
    """Return whether motion or elapsed time warrants a potential update."""
    if previous_pose is None:
        return True

    dx = float(current_pose.x - previous_pose.x)
    dy = float(current_pose.y - previous_pose.y)
    if math.hypot(dx, dy) >= max(0.0, float(min_distance_m)):
        return True

    yaw_error = math.atan2(
        math.sin(current_pose.yaw - previous_pose.yaw),
        math.cos(current_pose.yaw - previous_pose.yaw),
    )
    if abs(math.degrees(yaw_error)) >= max(0.0, float(min_yaw_deg)):
        return True

    interval = max(0.0, float(max_interval_s))
    if interval <= 0.0 or previous_update_s is None:
        return False
    elapsed_s = float(now_s) - float(previous_update_s)
    return elapsed_s >= interval
