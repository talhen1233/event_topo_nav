"""Tests for event-potential suppression timing."""

import networkx as nx

from radical_event_navigation_system.core.navigation_event import Pose3D
from radical_event_navigation_system.core.skeleton_graph import SkeletonGraphProcessor


def test_invalidation_suppression_uses_injected_clock() -> None:
    now = [10.0]
    processor = SkeletonGraphProcessor(
        grid_width_m=8.0,
        grid_height_m=8.0,
        grid_resolution_m=0.1,
        graph_min_edge_len_m=0.2,
        graph_node_merge_radius_m=0.2,
        event_potential_cell_size_m=0.5,
        min_event_distance_junction_m=0.5,
        event_potential_increment=0.2,
        event_potential_decay=0.05,
        event_potential_threshold=1.0,
        max_junction_detection_distance_junction_m=2.0,
        clock_s=lambda: now[0],
    )

    key = processor._area_key(1.0, -0.5)
    processor.suppress_event_potential_at(1.0, -0.5)
    assert processor._is_area_suppressed(key)

    # Wall time is irrelevant; only the injected (ROS in production) clock
    # advances the cooldown.
    now[0] = 11.99
    assert processor._is_area_suppressed(key)
    now[0] = 12.0
    assert not processor._is_area_suppressed(key)


def test_cached_junction_keeps_control_potential_until_departure() -> None:
    processor = SkeletonGraphProcessor(
        grid_width_m=8.0,
        grid_height_m=8.0,
        grid_resolution_m=0.1,
        graph_min_edge_len_m=0.2,
        graph_node_merge_radius_m=0.2,
        event_potential_cell_size_m=0.5,
        min_event_distance_junction_m=0.5,
        event_potential_increment=0.2,
        event_potential_decay=0.05,
        event_potential_threshold=0.8,
        max_junction_detection_distance_junction_m=2.0,
    )
    graph = nx.Graph()
    graph.add_node(1, o=[2.0, 2.0])
    area_key = processor._area_key(1.0, 1.0)
    processor._area_potential[area_key] = 0.8
    processor._graph_node_cache[(10, 10)] = "1"

    result = processor.analyze_junction(
        graph,
        Pose3D(x=1.0, y=0.0, z=0.0, yaw=0.0),
        origin_x=0.0,
        origin_y=0.0,
        resolution=1.0,
        last_event_pose=None,
    )

    assert result is None
    assert processor.current_junction_potential == 0.8
