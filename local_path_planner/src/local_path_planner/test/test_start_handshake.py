from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from std_msgs.msg import Bool


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from local_path_planner.local_path_planner_node import LocalPathPlanner  # noqa: E402


class _Resettable:
    def __init__(self) -> None:
        self.count = 0

    def reset(self) -> None:
        self.count += 1

    def reset_filters(self) -> None:
        self.count += 1


class _Clock:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    def now(self):
        return SimpleNamespace(nanoseconds=int(self._seconds * 1e9))


def _planner_fake():
    resets = [_Resettable() for _ in range(9)]
    fake = SimpleNamespace(
        _mission_active=False,
        _start_ready=False,
        _last_start_ready_time_s=None,
        _mode="IDLE",
        _safety_ctrl=resets[0],
        _height_ctrl=resets[1],
        _graph_center=resets[2],
        _sectors=resets[3],
        _yaw_intent=resets[4],
        _arm_commit=resets[5],
        _committed_arm_lock=resets[6],
        _junction_potential_speed=resets[7],
        _leaf_progress=resets[8],
        _last_cmd_body=np.ones(3, dtype=np.float32),
        _dead_end_reverse_yaw=1.0,
        _last_graph_reverse_yaw=1.0,
        _reverse_turn_yaw=1.0,
        _reverse_heading_hint=1.0,
        _reverse_active_prev=True,
        get_clock=lambda: _Clock(10.0),
        _publish_state=lambda: None,
        _publish_zero_cmd=lambda: None,
        _publish_debug_state=lambda payload: None,
        _idle_debug_payload=lambda **kwargs: kwargs,
    )
    return fake, resets


def _bool(value: bool) -> Bool:
    message = Bool()
    message.data = value
    return message


def test_navigation_ready_then_start_preserves_fresh_latched_ack() -> None:
    fake, resets = _planner_fake()

    LocalPathPlanner._on_start_ready(fake, _bool(True))
    LocalPathPlanner._on_mission_start(fake, _bool(True))

    assert fake._mission_active
    assert fake._start_ready
    assert fake._mode == "IDLE"
    assert all(item.count == 1 for item in resets)

    LocalPathPlanner._on_mission_start(fake, _bool(True))
    assert all(item.count == 1 for item in resets)


def test_start_then_navigation_ready_waits_until_ack() -> None:
    fake, _resets = _planner_fake()

    LocalPathPlanner._on_mission_start(fake, _bool(True))
    assert fake._mode == "WAIT_START"
    assert not fake._start_ready

    LocalPathPlanner._on_start_ready(fake, _bool(True))
    assert fake._start_ready
    assert fake._mode == "IDLE"
