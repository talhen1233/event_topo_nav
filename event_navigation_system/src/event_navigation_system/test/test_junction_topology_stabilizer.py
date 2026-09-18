from radical_event_navigation_system.core.junction_topology_stabilizer import (
    JunctionTopologyStabilizer,
)


def test_partial_t_does_not_commit_before_cross_topology_settles() -> None:
    stabilizer = JunctionTopologyStabilizer(settle_duration_s=0.8)
    area = (0, 0)

    assert not stabilizer.observe(area, degree=3, now_s=0.0)
    assert not stabilizer.observe(area, degree=3, now_s=0.3)
    assert not stabilizer.observe(area, degree=4, now_s=0.4)
    assert not stabilizer.observe(area, degree=4, now_s=1.1)
    assert stabilizer.observe(area, degree=4, now_s=1.2)


def test_non_junction_observation_discards_pending_stability() -> None:
    stabilizer = JunctionTopologyStabilizer(settle_duration_s=0.5)
    area = (1, 2)

    assert not stabilizer.observe(area, degree=3, now_s=0.0)
    stabilizer.invalidate(area)
    assert not stabilizer.observe(area, degree=3, now_s=0.6)
