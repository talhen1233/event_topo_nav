import math

import numpy as np

from local_path_planner.scripts.arm_commit_tracker import CommittedArmIntent
from local_path_planner.scripts.committed_arm_lock import (
    CommittedArmLock,
    CommittedArmLockParams,
)
from local_path_planner.scripts.graph_tracker import (
    GraphArmCandidate,
    GraphTrackingState,
)


def _lock() -> CommittedArmLock:
    return CommittedArmLock(
        CommittedArmLockParams(
            entry_distance_m=0.8,
            entry_cross_track_m=0.6,
            heading_tolerance_rad=math.radians(40.0),
            max_distance_m=4.0,
            direction_smoothing_tau_s=0.35,
            direction_max_rate_rad_s=math.radians(25.0),
            direction_min_confidence=0.65,
            acquisition_timeout_s=3.0,
        )
    )


def _intent(mode: str = "explore") -> CommittedArmIntent:
    return CommittedArmIntent(
        event_id="junction",
        mode=mode,
        event_center_xy=np.zeros(2, dtype=np.float32),
        gate_radius_m=2.0,
        selected_arm_angle_rad=0.0,
    )


def _graph(
    *,
    approaching: bool,
    heading_rad: float = 0.0,
    arm_heading_rad: float = 0.0,
    arm_is_backtrack: bool = False,
) -> GraphTrackingState:
    arm = GraphArmCandidate(
        node_id=2,
        angle_rad=arm_heading_rad,
        edge_length_m=3.0,
        lookahead_xy=np.array([0.8, 0.0], dtype=np.float32),
        lookahead_heading_rad=arm_heading_rad,
        is_backtrack=arm_is_backtrack,
    )
    return GraphTrackingState(
        valid=True,
        current_heading_rad=heading_rad,
        approaching_junction=approaching,
        arm_candidates=[arm] if approaching else [],
    )


def test_commit_locks_only_after_graph_supports_selected_arm() -> None:
    lock = _lock()
    lock.activate(_intent())

    lock.update(
        np.array([-1.0, 0.0], dtype=np.float32),
        _graph(approaching=False),
        dt=0.05,
    )
    assert not lock.branch_locked

    lock.update(
        np.array([-0.5, 0.0], dtype=np.float32),
        _graph(approaching=True),
        dt=0.05,
    )
    assert lock.branch_locked
    assert lock.active_intent is not None
    assert lock.active_intent.branch_locked


def test_locked_arm_identity_survives_bounded_direction_refinement() -> None:
    lock = _lock()
    lock.activate(_intent())
    lock.update(
        np.array([-0.5, 0.0], dtype=np.float32),
        _graph(
            approaching=True,
            arm_heading_rad=math.radians(20.0),
        ),
        dt=0.05,
    )
    assert lock.active_intent is not None
    assert math.isclose(
        lock.active_intent.selected_arm_angle_rad,
        math.radians(20.0),
        abs_tol=1e-6,
    )

    for _ in range(30):
        lock.update(
            np.array([-0.5, 0.0], dtype=np.float32),
            _graph(approaching=True, arm_heading_rad=0.0),
            dt=0.05,
        )

    assert lock.active_intent is not None
    assert abs(lock.active_intent.selected_arm_angle_rad) < math.radians(1.0)

    locked_angle = lock.active_intent.selected_arm_angle_rad
    for _ in range(10):
        lock.update(
            np.array([-0.5, 0.0], dtype=np.float32),
            _graph(
                approaching=True,
                arm_heading_rad=0.5 * math.pi,
            ),
            dt=0.05,
        )
    assert lock.active_intent is not None
    assert math.isclose(
        lock.active_intent.selected_arm_angle_rad,
        locked_angle,
        abs_tol=1e-6,
    )


def test_release_waits_until_selected_branch_entry_is_confirmed() -> None:
    lock = _lock()
    lock.activate(_intent())
    lock.update(
        np.array([-0.5, 0.0], dtype=np.float32),
        _graph(approaching=True),
        dt=0.05,
    )
    lock.request_release()

    lock.update(
        np.array([0.9, 0.7], dtype=np.float32),
        _graph(approaching=False, heading_rad=math.radians(50.0)),
        dt=0.05,
    )
    assert lock.active_intent is not None
    assert not lock.entry_confirmed

    lock.update(
        np.array([0.9, 0.2], dtype=np.float32),
        _graph(approaching=False, heading_rad=math.radians(5.0)),
        dt=0.05,
    )
    assert lock.active_intent is None


def test_early_release_is_deferred_until_graph_arm_can_be_locked() -> None:
    lock = _lock()
    lock.activate(_intent())
    lock.request_release()

    lock.update(
        np.array([-1.0, 0.0], dtype=np.float32),
        _graph(approaching=False),
        dt=0.05,
    )
    assert lock.active_intent is not None
    assert not lock.branch_locked

    lock.update(
        np.array([-0.5, 0.0], dtype=np.float32),
        _graph(approaching=True),
        dt=0.05,
    )
    assert lock.branch_locked


def test_stale_lock_is_released_at_fail_safe_distance() -> None:
    lock = _lock()
    lock.activate(_intent())
    lock.update(
        np.array([-0.5, 0.0], dtype=np.float32),
        _graph(approaching=True),
        dt=0.05,
    )

    lock.update(
        np.array([4.1, 0.0], dtype=np.float32),
        _graph(approaching=False),
        dt=0.05,
    )

    assert lock.active_intent is None


def test_return_modes_can_lock_incoming_arm() -> None:
    for mode in ("backtrack", "return"):
        lock = _lock()
        lock.activate(_intent(mode=mode))

        lock.update(
            np.array([-0.5, 0.0], dtype=np.float32),
            _graph(approaching=True, arm_is_backtrack=True),
            dt=0.05,
        )

        assert lock.branch_locked


def test_unsupported_commit_expires_near_junction() -> None:
    lock = _lock()
    lock.activate(_intent())
    unsupported = _graph(
        approaching=True,
        arm_heading_rad=math.radians(90.0),
    )

    for _ in range(61):
        lock.update(
            np.array([-0.5, 0.0], dtype=np.float32),
            unsupported,
            dt=0.05,
        )

    assert lock.active_intent is None


def test_same_commit_rearms_deferred_release() -> None:
    lock = _lock()
    intent = _intent()
    lock.activate(intent)
    lock.request_release()

    lock.activate(intent)

    assert not lock.release_requested
