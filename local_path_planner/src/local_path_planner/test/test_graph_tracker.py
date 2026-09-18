import json
import math

import numpy as np

from local_path_planner.scripts.graph_tracker import GraphTracker, GraphTrackerParams


def test_heading_fit_averages_local_s_curve_tangent() -> None:
    tracker = GraphTracker(
        GraphTrackerParams(
            graph_stale_timeout_s=1.0,
            lookahead_distance_m=0.8,
            heading_fit_distance_m=1.5,
        )
    )
    graph = {
        "nodes": [
            {"id": 0, "x": 0.0, "y": 0.0, "deg": 1},
            {"id": 1, "x": 3.0, "y": 0.0, "deg": 1},
        ],
        "edges": [
            {
                "u": 0,
                "v": 1,
                "pts": [
                    [0.0, 0.0],
                    [0.5, 0.4],
                    [1.0, 0.0],
                    [1.5, -0.4],
                    [2.0, 0.0],
                    [2.5, 0.4],
                    [3.0, 0.0],
                ],
            }
        ],
    }
    tracker.on_graph_json(json.dumps(graph), now_s=1.0)

    state = tracker.update(
        np.array([0.0, 0.0], dtype=np.float32),
        heading_hint_rad=0.0,
        now_s=1.1,
    )

    assert state.valid
    assert abs(state.current_heading_rad) < math.radians(10.0)
