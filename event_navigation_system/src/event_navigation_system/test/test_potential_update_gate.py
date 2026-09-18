import math

from radical_event_navigation_system.core.navigation_event import Pose3D
from radical_event_navigation_system.core.potential_update_gate import (
    potential_update_due,
)


def _due(current: Pose3D, previous: Pose3D, now_s: float) -> bool:
    return potential_update_due(
        current,
        previous,
        previous_update_s=0.0,
        now_s=now_s,
        min_distance_m=0.2,
        min_yaw_deg=45.0,
        max_interval_s=1.0,
    )


def test_stationary_profile_refreshes_on_bounded_interval() -> None:
    pose = Pose3D(x=0.0, y=0.0, z=0.0, yaw=0.0)

    assert not _due(pose, pose, now_s=0.99)
    assert _due(pose, pose, now_s=1.0)


def test_rotation_triggers_update_without_translation() -> None:
    previous = Pose3D(x=0.0, y=0.0, z=0.0, yaw=0.0)
    rotated = Pose3D(
        x=0.0,
        y=0.0,
        z=0.0,
        yaw=math.radians(45.0),
    )

    assert _due(rotated, previous, now_s=0.1)
