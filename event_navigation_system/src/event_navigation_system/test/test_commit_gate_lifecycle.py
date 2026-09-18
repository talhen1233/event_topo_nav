from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from radical_event_navigation_system.core.event_types import NavigationMode
from radical_event_navigation_system.event_navigation_node import (
    SmartNavigationNode,
)


def _navigation_fake(distance_m: float) -> SimpleNamespace:
    return SimpleNamespace(
        mission_active=True,
        current_pose=SimpleNamespace(x=distance_m, y=0.0),
        current_mode=NavigationMode.BACKTRACK,
        at_junction=True,
        _active_commit_center_xy=np.zeros(2, dtype=np.float32),
        _active_commit_radius_m=1.0,
        _active_commit_gate_entered=False,
        junction_commit_exit_hysteresis_m=0.2,
        returning_to_prev_junction=False,
        return_target_event_id=None,
        _exit_junction_context=Mock(),
    )


def test_distant_commit_is_not_cleared_before_gate_entry() -> None:
    fake = _navigation_fake(distance_m=2.0)

    SmartNavigationNode.navigation_update(fake)

    fake._exit_junction_context.assert_not_called()
    assert not fake._active_commit_gate_entered


def test_commit_clears_only_after_entering_then_exiting_gate() -> None:
    fake = _navigation_fake(distance_m=0.5)
    SmartNavigationNode.navigation_update(fake)
    assert fake._active_commit_gate_entered

    fake.current_pose.x = 1.3
    SmartNavigationNode.navigation_update(fake)

    fake._exit_junction_context.assert_called_once()
