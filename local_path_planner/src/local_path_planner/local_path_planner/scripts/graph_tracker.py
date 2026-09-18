"""Project the drone onto the skeleton graph."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .arm_direction_estimator import estimate_arm_direction


@dataclass(frozen=True)
class GraphArmCandidate:
    node_id: int
    angle_rad: float
    edge_length_m: float
    lookahead_xy: np.ndarray
    lookahead_heading_rad: float
    is_backtrack: bool = False
    direction_confidence: float = 1.0


@dataclass(frozen=True)
class GraphTrackingState:
    valid: bool = False
    edge_id: tuple[int, int] = (-1, -1)
    projection_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    tangent_rad: float = 0.0
    current_heading_rad: float = 0.0
    current_lookahead_xy: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    lateral_offset_m: float = 0.0
    dist_to_ahead_node_m: float = float("inf")
    ahead_node_id: int = -1
    ahead_node_deg: int = 0
    approaching_junction: bool = False
    approaching_leaf: bool = False
    arm_candidates: list[GraphArmCandidate] = field(default_factory=list)
    ahead_path_length_m: float = 0.0


@dataclass
class _GraphNode:
    __slots__ = ("x", "y", "deg")
    x: float
    y: float
    deg: int


@dataclass
class _GraphEdge:
    __slots__ = ("u", "v", "pts")
    u: int
    v: int
    pts: np.ndarray


@dataclass
class GraphTrackerParams:
    junction_approach_distance_m: float = 1.0
    edge_hysteresis_m: float = 0.3
    graph_stale_timeout_s: float = 1.0
    lookahead_distance_m: float = 0.8
    heading_fit_distance_m: float = 1.5
    arm_fit_min_distance_m: float = 0.35
    arm_fit_max_distance_m: float = 2.0
    arm_fit_min_span_m: float = 0.8


class GraphTracker:
    def __init__(self, params: GraphTrackerParams) -> None:
        self.p = params
        self._nodes: dict[int, _GraphNode] = {}
        self._edges: list[_GraphEdge] = []
        self._adjacency: dict[int, list[tuple[int, int]]] = {}
        self._current_edge: Optional[tuple[int, int]] = None
        self._last_graph_time_s: Optional[float] = None

    def on_graph_json(self, json_str: str, now_s: float) -> None:
        """Parse skeleton-graph JSON into nodes and edges."""
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict):
            return

        nodes_raw = data.get("nodes", [])
        edges_raw = data.get("edges", [])

        nodes: dict[int, _GraphNode] = {}
        for n in nodes_raw:
            nid = int(n.get("id", -1))
            nodes[nid] = _GraphNode(
                x=float(n["x"]),
                y=float(n["y"]),
                deg=int(n.get("deg", 0)),
            )

        edges: list[_GraphEdge] = []
        adjacency: dict[int, list[tuple[int, int]]] = {nid: [] for nid in nodes}
        for e in edges_raw:
            u = int(e["u"])
            v = int(e["v"])
            raw_pts = e.get("pts", [])
            if len(raw_pts) < 2:
                continue
            pts = np.asarray(raw_pts, dtype=np.float32)
            edge_idx = len(edges)
            edges.append(_GraphEdge(u=u, v=v, pts=pts))
            if u in adjacency:
                adjacency[u].append((v, edge_idx))
            if v in adjacency:
                adjacency[v].append((u, edge_idx))

        self._nodes = nodes
        self._edges = edges
        self._adjacency = adjacency
        self._last_graph_time_s = now_s

    def is_graph_available(self, now_s: float) -> bool:
        if self._last_graph_time_s is None or not self._edges:
            return False
        return (now_s - self._last_graph_time_s) < self.p.graph_stale_timeout_s

    def update(self, drone_xy: np.ndarray, heading_hint_rad: float, now_s: float) -> GraphTrackingState:
        if not self.is_graph_available(now_s):
            self._current_edge = None
            return GraphTrackingState()

        best_edge_idx, best_seg_idx, best_proj, best_dist = self._find_nearest_edge(drone_xy)
        if best_edge_idx < 0:
            self._current_edge = None
            return GraphTrackingState()

        edge = self._edges[best_edge_idx]
        eid = (edge.u, edge.v)

        if (
            self._current_edge is not None
            and self._current_edge != eid
            and self._current_edge != (edge.v, edge.u)
        ):
            prev_idx, _, prev_proj, prev_dist = self._find_edge_by_id(self._current_edge, drone_xy)
            if prev_idx >= 0 and prev_dist < best_dist + self.p.edge_hysteresis_m:
                edge = self._edges[prev_idx]
                eid = (edge.u, edge.v)
                best_proj = prev_proj
                best_seg_idx = self._find_segment_index(edge, prev_proj)
                best_dist = prev_dist

        self._current_edge = eid

        tangent = self._compute_tangent(edge, best_seg_idx, heading_hint_rad)
        lateral = self._signed_lateral_offset(drone_xy, best_proj, tangent)
        ahead_id, dist_ahead = self._ahead_node(edge, best_proj, tangent)
        ahead_deg = self._nodes[ahead_id].deg if ahead_id in self._nodes else 0
        ahead_path_len = self._path_length_to_node(edge, best_proj, best_seg_idx, ahead_id)
        current_lookahead_xy, local_heading = self._lookahead_from_projection(
            edge=edge,
            proj_xy=best_proj,
            seg_idx=best_seg_idx,
            node_id=ahead_id,
            target_dist=min(self.p.lookahead_distance_m, max(ahead_path_len, 0.0)),
            fallback_heading=tangent,
        )
        heading_fit_xy, _ = self._lookahead_from_projection(
            edge=edge,
            proj_xy=best_proj,
            seg_idx=best_seg_idx,
            node_id=ahead_id,
            target_dist=min(
                max(self.p.lookahead_distance_m, self.p.heading_fit_distance_m),
                max(ahead_path_len, 0.0),
            ),
            fallback_heading=tangent,
        )
        current_heading = _heading_between(
            best_proj,
            heading_fit_xy,
            local_heading,
        )

        approaching_junction = ahead_deg > 2 and dist_ahead < self.p.junction_approach_distance_m
        approaching_leaf = ahead_deg == 1 and dist_ahead < self.p.junction_approach_distance_m

        arm_candidates: list[GraphArmCandidate] = []
        if approaching_junction and ahead_id in self._adjacency:
            behind_id = edge.u if ahead_id == edge.v else edge.v
            node_xy = self.get_node_xy(ahead_id)
            for nbr_id, edge_idx in self._adjacency[ahead_id]:
                if nbr_id not in self._nodes or not (0 <= edge_idx < len(self._edges)):
                    continue
                arm_edge = self._edges[edge_idx]
                lookahead_xy, local_arm_heading = self._lookahead_from_node(
                    arm_edge,
                    from_node_id=ahead_id,
                    target_dist=self.p.lookahead_distance_m,
                )
                heading_fit_xy, _ = self._lookahead_from_node(
                    arm_edge,
                    from_node_id=ahead_id,
                    target_dist=max(
                        self.p.lookahead_distance_m,
                        self.p.heading_fit_distance_m,
                    ),
                )
                if node_xy is None:
                    arm_angle = local_arm_heading
                    direction_confidence = 0.0
                else:
                    fallback_angle = _heading_between(
                        node_xy,
                        heading_fit_xy,
                        local_arm_heading,
                    )
                    estimate = estimate_arm_direction(
                        arm_edge.pts,
                        node_xy,
                        min_distance_m=self.p.arm_fit_min_distance_m,
                        max_distance_m=self.p.arm_fit_max_distance_m,
                        min_span_m=self.p.arm_fit_min_span_m,
                        fallback_angle_rad=fallback_angle,
                    )
                    arm_angle = estimate.angle_rad
                    direction_confidence = estimate.confidence
                arm_candidates.append(
                    GraphArmCandidate(
                        node_id=nbr_id,
                        angle_rad=arm_angle,
                        edge_length_m=_polyline_length(arm_edge.pts),
                        lookahead_xy=lookahead_xy,
                        lookahead_heading_rad=arm_angle,
                        is_backtrack=(nbr_id == behind_id),
                        direction_confidence=direction_confidence,
                    )
                )

        return GraphTrackingState(
            valid=True,
            edge_id=eid,
            projection_xy=best_proj,
            tangent_rad=tangent,
            current_heading_rad=current_heading,
            current_lookahead_xy=current_lookahead_xy,
            lateral_offset_m=lateral,
            dist_to_ahead_node_m=dist_ahead,
            ahead_node_id=ahead_id,
            ahead_node_deg=ahead_deg,
            approaching_junction=approaching_junction,
            approaching_leaf=approaching_leaf,
            arm_candidates=arm_candidates,
            ahead_path_length_m=ahead_path_len,
        )

    def get_node_xy(self, node_id: int) -> Optional[np.ndarray]:
        node = self._nodes.get(node_id)
        if node is None:
            return None
        return np.array([node.x, node.y], dtype=np.float32)

    def get_edge_points(self, edge_id: tuple[int, int]) -> Optional[np.ndarray]:
        for edge in self._edges:
            if (edge.u, edge.v) == edge_id or (edge.v, edge.u) == edge_id:
                return edge.pts
        return None

    def _find_nearest_edge(self, xy: np.ndarray) -> tuple[int, int, np.ndarray, float]:
        best_idx = -1
        best_seg = -1
        best_proj = np.zeros(2, dtype=np.float32)
        best_d = float("inf")

        for i, edge in enumerate(self._edges):
            seg, proj, d = _nearest_on_polyline(edge.pts, xy)
            if d < best_d:
                best_d = d
                best_idx = i
                best_seg = seg
                best_proj = proj
        return best_idx, best_seg, best_proj, best_d

    def _find_edge_by_id(
        self, edge_id: tuple[int, int], xy: np.ndarray
    ) -> tuple[int, int, np.ndarray, float]:
        for i, edge in enumerate(self._edges):
            if (edge.u, edge.v) == edge_id or (edge.v, edge.u) == edge_id:
                seg, proj, d = _nearest_on_polyline(edge.pts, xy)
                return i, seg, proj, d
        return -1, -1, np.zeros(2, dtype=np.float32), float("inf")

    @staticmethod
    def _find_segment_index(edge: _GraphEdge, proj: np.ndarray) -> int:
        pts = edge.pts
        if pts.shape[0] < 2:
            return 0
        diffs = pts[:-1] - proj[None, :]
        dists = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dists))

    @staticmethod
    def _compute_tangent(edge: _GraphEdge, seg_idx: int, heading_hint_rad: float) -> float:
        pts = edge.pts
        idx = min(seg_idx, pts.shape[0] - 2)
        d = pts[idx + 1] - pts[idx]
        norm = float(np.linalg.norm(d))
        if norm < 1e-6:
            return heading_hint_rad
        tangent = float(math.atan2(float(d[1]), float(d[0])))
        flipped = _wrap(tangent + math.pi)
        if abs(_wrap(tangent - heading_hint_rad)) <= abs(_wrap(flipped - heading_hint_rad)):
            return tangent
        return flipped

    @staticmethod
    def _signed_lateral_offset(drone_xy: np.ndarray, proj_xy: np.ndarray, tangent: float) -> float:
        d = drone_xy - proj_xy
        normal_x = -math.sin(tangent)
        normal_y = math.cos(tangent)
        return float(d[0]) * normal_x + float(d[1]) * normal_y

    def _ahead_node(
        self, edge: _GraphEdge, proj_xy: np.ndarray, tangent: float
    ) -> tuple[int, float]:
        u_xy = (
            np.array([self._nodes[edge.u].x, self._nodes[edge.u].y], dtype=np.float32)
            if edge.u in self._nodes
            else edge.pts[0]
        )
        v_xy = (
            np.array([self._nodes[edge.v].x, self._nodes[edge.v].y], dtype=np.float32)
            if edge.v in self._nodes
            else edge.pts[-1]
        )

        fwd = np.array([math.cos(tangent), math.sin(tangent)], dtype=np.float32)
        dot_u = float(np.dot(u_xy - proj_xy, fwd))
        dot_v = float(np.dot(v_xy - proj_xy, fwd))

        if dot_v >= dot_u:
            return edge.v, max(0.0, float(np.linalg.norm(v_xy - proj_xy)))
        return edge.u, max(0.0, float(np.linalg.norm(u_xy - proj_xy)))

    def _lookahead_from_projection(
        self,
        *,
        edge: _GraphEdge,
        proj_xy: np.ndarray,
        seg_idx: int,
        node_id: int,
        target_dist: float,
        fallback_heading: float,
    ) -> tuple[np.ndarray, float]:
        toward_v = node_id == edge.v
        if toward_v:
            tail = edge.pts[seg_idx + 1 :]
        else:
            tail = edge.pts[: seg_idx + 1][::-1]
        pts = np.vstack((proj_xy[None, :], tail.astype(np.float32, copy=False)))
        return _sample_polyline(pts, target_dist, fallback_heading)

    def _lookahead_from_node(
        self,
        edge: _GraphEdge,
        *,
        from_node_id: int,
        target_dist: float,
    ) -> tuple[np.ndarray, float]:
        pts = edge.pts if from_node_id == edge.u else edge.pts[::-1]
        fallback_heading = self._edge_heading(pts)
        return _sample_polyline(pts, target_dist, fallback_heading)

    @staticmethod
    def _edge_heading(pts: np.ndarray) -> float:
        if pts.shape[0] < 2:
            return 0.0
        diffs = pts[1:] - pts[:-1]
        lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
        idx = int(np.argmax(lengths))
        d = diffs[idx]
        if float(np.linalg.norm(d)) < 1e-6:
            return 0.0
        return float(math.atan2(float(d[1]), float(d[0])))

    @staticmethod
    def _path_length_to_node(
        edge: _GraphEdge,
        proj_xy: np.ndarray,
        seg_idx: int,
        node_id: int,
    ) -> float:
        toward_v = node_id == edge.v
        if toward_v:
            tail = edge.pts[seg_idx + 1 :]
        else:
            tail = edge.pts[: seg_idx + 1][::-1]
        pts = np.vstack((proj_xy[None, :], tail.astype(np.float32, copy=False)))
        return _polyline_length(pts)


def _sample_polyline(pts: np.ndarray, target_dist: float, fallback_heading: float) -> tuple[np.ndarray, float]:
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros(2, dtype=np.float32), fallback_heading
    if pts.shape[0] == 1:
        return pts[0].astype(np.float32, copy=False), fallback_heading

    diffs = pts[1:] - pts[:-1]
    lengths = np.sqrt(np.sum(diffs * diffs, axis=1))
    nonzero = lengths > 1e-6
    if not np.any(nonzero):
        return pts[0].astype(np.float32, copy=False), fallback_heading

    target = float(np.clip(target_dist, 0.0, np.sum(lengths)))
    cumulative = np.cumsum(lengths)
    seg_idx = int(np.searchsorted(cumulative, target, side="left"))
    seg_idx = min(seg_idx, lengths.size - 1)
    seg_len = float(lengths[seg_idx])
    start = pts[seg_idx]
    end = pts[seg_idx + 1]

    prev_sum = 0.0 if seg_idx == 0 else float(cumulative[seg_idx - 1])
    t = 0.0 if seg_len <= 1e-6 else (target - prev_sum) / seg_len
    point = start + float(np.clip(t, 0.0, 1.0)) * (end - start)
    heading = fallback_heading
    d = end - start
    if float(np.linalg.norm(d)) > 1e-6:
        heading = float(math.atan2(float(d[1]), float(d[0])))
    return point.astype(np.float32, copy=False), heading


def _polyline_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    diffs = pts[1:] - pts[:-1]
    return float(np.sum(np.sqrt(np.sum(diffs * diffs, axis=1))))


def _nearest_on_polyline(pts: np.ndarray, xy: np.ndarray) -> tuple[int, np.ndarray, float]:
    best_seg = 0
    best_proj = pts[0].copy()
    best_d2 = float(np.sum((pts[0] - xy) ** 2))

    for i in range(pts.shape[0] - 1):
        proj, d2 = _project_onto_segment(pts[i], pts[i + 1], xy)
        if d2 < best_d2:
            best_d2 = d2
            best_seg = i
            best_proj = proj

    return best_seg, best_proj, float(math.sqrt(best_d2))


def _project_onto_segment(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, float]:
    ab = b - a
    ab2 = float(np.dot(ab, ab))
    if ab2 < 1e-12:
        d = p - a
        return a.copy(), float(np.dot(d, d))
    t = float(np.clip(float(np.dot(p - a, ab)) / ab2, 0.0, 1.0))
    proj = a + t * ab
    d = p - proj
    return proj, float(np.dot(d, d))


def _heading_between(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    fallback_heading: float,
) -> float:
    delta = np.asarray(end_xy, dtype=np.float32) - np.asarray(
        start_xy,
        dtype=np.float32,
    )
    if float(np.linalg.norm(delta)) < 1e-6:
        return float(fallback_heading)
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _wrap(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))
