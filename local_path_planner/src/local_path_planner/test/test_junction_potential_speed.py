import math

from local_path_planner.scripts.junction_potential_speed import (
    JunctionPotentialSpeed,
    JunctionPotentialSpeedParams,
)


def _controller() -> JunctionPotentialSpeed:
    return JunctionPotentialSpeed(
        JunctionPotentialSpeedParams(
            slowdown_start=0.1,
            full_slowdown=0.8,
            min_speed_scale=0.35,
            timeout_s=1.5,
        )
    )


def test_speed_reduces_monotonically_before_event_commit() -> None:
    controller = _controller()

    controller.update(0.1, now_s=1.0)
    weak_scale = controller.speed_scale(now_s=1.0)
    controller.update(0.4, now_s=1.1)
    growing_scale = controller.speed_scale(now_s=1.1)
    controller.update(0.8, now_s=1.2)
    threshold_scale = controller.speed_scale(now_s=1.2)

    assert math.isclose(weak_scale, 1.0)
    assert threshold_scale < growing_scale < weak_scale
    assert math.isclose(threshold_scale, 0.35, abs_tol=1e-6)


def test_stale_potential_does_not_limit_speed() -> None:
    controller = _controller()
    controller.update(0.8, now_s=1.0)

    assert math.isclose(controller.speed_scale(now_s=2.6), 1.0)


def test_confirmed_entry_suppresses_current_junction_until_clear() -> None:
    controller = _controller()
    controller.update(0.8, now_s=1.0)
    controller.suppress_until_clear()

    assert math.isclose(controller.speed_scale(now_s=1.1), 1.0)

    controller.update(0.0, now_s=1.2)
    controller.update(0.8, now_s=1.3)
    assert math.isclose(controller.speed_scale(now_s=1.3), 0.35)


def test_non_finite_potential_is_ignored() -> None:
    controller = _controller()
    controller.update(float("nan"), now_s=1.0)

    assert math.isclose(controller.speed_scale(now_s=1.0), 1.0)
