"""Exercise the actual timer through stall, scan, reverse and resumed travel."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import Odometry

from local_path_planner.local_path_planner_node import LocalPathPlanner
from local_path_planner.scripts.graph_tracker import GraphTrackingState


class Stamp:
    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)

    def __sub__(self, other):
        return SimpleNamespace(nanoseconds=self.nanoseconds - other.nanoseconds)


@pytest.mark.parametrize('reverse', [False, True])
def test_timer_recovers_from_far_wall_leaf_and_resumes_along_rear_graph(reverse):
    rclpy.init()
    node = LocalPathPlanner()
    try:
        clock = SimpleNamespace(seconds=1.0)
        node.get_clock = lambda: SimpleNamespace(now=lambda: Stamp(clock.seconds))
        node._last_time = Stamp(clock.seconds - 0.1)
        node._mission_active = True
        node._start_ready = True
        node._odom = Odometry()
        node._obstacles = np.zeros((0, 3), dtype=np.float32)
        node._height_ctrl.compute = lambda *args: 0.0
        node._sectors.compute = lambda *args, **kwargs: SimpleNamespace(
            distances=dict(front=1.0, back=2.0, left=0.6, right=0.6, up=None, down=None),
            vy_rep=0.0,
        )
        node._publish_state = Mock()
        node._publish_debug = Mock()
        node._publish_debug_state = Mock()
        node._build_debug_payload = lambda **kwargs: {"speculative_arm_angle_rad": None}
        node._cmd_markers.publish = Mock()
        node._maybe_log = Mock()
        commands = []
        node._publish_command = commands.append
        if reverse:
            node._last_graph_reverse_yaw = math.pi
            node._on_graph_target(
                SimpleNamespace(data='{"kind":"reverse"}')
            )

        def graph_update(xy, heading, now):
            returning = math.cos(heading) < 0.0
            return GraphTrackingState(
                valid=True, edge_id=(0, 1), ahead_node_id=0 if returning else 1,
                tangent_rad=math.pi if returning else 0.0,
                current_heading_rad=math.pi if returning else 0.0,
                approaching_leaf=not returning,
                ahead_path_length_m=2.0 if returning else 0.1,
                dist_to_ahead_node_m=2.0 if returning else 0.1,
            )

        node._graph_tracker.update = graph_update
        node._graph_tracker.get_node_xy = lambda nid: np.array([-2.0 if nid == 0 else 0.1, 0.0])
        yaw = 0.0
        modes = []
        for _ in range(650):
            node._odom.pose.pose.orientation.z = math.sin(yaw / 2)
            node._odom.pose.pose.orientation.w = math.cos(yaw / 2)
            node._last_cloud_time = clock.seconds
            node._on_timer()
            modes.append(node._mode)
            recovered_locally = (
                not reverse
                and "DEAD_END" in modes
                and node._mode == "FOLLOW_INTENT"
            )
            following_requested_reverse = (
                reverse
                and node._mode == "REVERSE_FOLLOW"
                and abs(math.cos(yaw) + 1.0) < 0.01
            )
            if recovered_locally or following_requested_reverse:
                assert commands[-1].linear.x > 0.0
                assert not node._safety_ctrl.graph_recovery
                assert node._dead_end_reverse_yaw is None
                break
            yaw += commands[-1].angular.z * 0.1
            clock.seconds += 0.1
        else:
            pytest.fail(f"Recovery did not finish; final mode={node._mode}, yaw={yaw}")
        if reverse:
            assert "TURN_TO_OPENING" not in modes
            assert "DEAD_END" in modes
        else:
            assert "TURN_TO_OPENING" in modes
            assert "DEAD_END" in modes
    finally:
        node.destroy_node()
        rclpy.shutdown()
