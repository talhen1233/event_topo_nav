#!/usr/bin/env python3
"""Skeletonization, SKNW graph construction, pruning, and junction analysis."""

from typing import Callable, Optional, Tuple, List, Dict, Set

import numpy as np
import networkx as nx
import cv2
import math
import time

from skimage.morphology import skeletonize
from numba import njit

try:  # pragma: no cover - allow running module directly
    from .arm_direction_estimator import estimate_arm_direction
    from .event_types import JunctionType
    from .junction_topology_stabilizer import JunctionTopologyStabilizer
    from .navigation_event import Pose3D
except ImportError:  # pragma: no cover
    from core.arm_direction_estimator import estimate_arm_direction
    from core.event_types import JunctionType
    from core.junction_topology_stabilizer import JunctionTopologyStabilizer
    from core.navigation_event import Pose3D


def neighbors(shape):
    """Get neighbor offsets for N-dimensional array."""
    dim = len(shape)
    block = np.ones([3] * dim)
    block[tuple([1] * dim)] = 0
    idx = np.where(block > 0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx - [1] * dim)
    acc = np.cumprod((1,) + tuple(shape[::-1][:-1]))
    return np.dot(idx, acc[::-1])


@njit(nopython=True, cache=True, fastmath=True)
def mark(img, nbs):
    """Mark pixels based on connectivity (1=chain, 2=node)."""
    img = img.ravel()
    for p in range(len(img)):
        if img[p] == 0:
            continue
        s = 0
        for dp in nbs:
            if img[p + dp] != 0:
                s += 1
        if s == 2:
            img[p] = 1
        else:
            img[p] = 2


@njit(nopython=True, cache=True, fastmath=True)
def idx2rc(idx, acc):
    """Convert flat indices to row-column coordinates."""
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i, j] = idx[i] // acc[j]
            idx[i] -= rst[i, j] * acc[j]
    rst -= 1
    return rst


@njit(nopython=True, cache=True, fastmath=True)
def fill(img, p, num, nbs, acc, buf):
    """Fill connected node region."""
    img[p] = num
    buf[0] = p
    cur = 0
    s = 1
    iso = True
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p + dp
            if img[cp] == 2:
                img[cp] = num
                buf[s] = cp
                s += 1
            if img[cp] == 1:
                iso = False
        cur += 1
        if cur == s:
            break
    return iso, idx2rc(buf[:s], acc)


@njit(nopython=True, cache=True, fastmath=True)
def trace(img, p, nbs, acc, buf):
    """Trace edge from one node to another."""
    c1 = 0
    c2 = 0
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1 == 0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2 != 0:
            break
    return (c1 - 10, c2 - 10, idx2rc(buf[:cur + 1], acc))


@njit(nopython=True, cache=True, fastmath=True)
def parse_struc(img, nbs, acc, iso, ring):
    """Parse skeleton structure into nodes and edges."""
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in range(len(img)):
        if img[p] == 2:
            isiso, nds = fill(img, p, num, nbs, acc, buf)
            if isiso and not iso:
                continue
            num += 1
            nodes.append(nds)
    edges = []
    for p in range(len(img)):
        if img[p] < 10:
            continue
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    if not ring:
        return nodes, edges
    for p in range(len(img)):
        if img[p] != 1:
            continue
        img[p] = num
        num += 1
        nodes.append(idx2rc(np.array([p], dtype=np.int64), acc))
        for dp in nbs:
            if img[p + dp] == 1:
                edge = trace(img, p + dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges


def build_graph(nodes, edges, multi=False, full=True):
    """Build NetworkX graph from skeleton nodes and edges."""
    os = np.array([i.mean(axis=0) for i in nodes])
    if full:
        os = os.round().astype(np.uint16)
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=os[i])
    for s, e, pts in edges:
        if full:
            pts[[0, -1]] = os[[s, e]]
        l = np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()
        graph.add_edge(s, e, pts=pts, weight=l)
    return graph


def build_sknw(ske, multi=False, iso=True, ring=True, full=True):
    """Build skeleton network graph from binary skeleton image."""
    buf = np.pad(ske, (1, 1), mode='constant').astype(np.uint16)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,) + tuple(buf.shape[::-1][:-1]))[::-1]
    mark(buf, nbs)
    nodes, edges = parse_struc(buf, nbs, acc, iso, ring)
    return build_graph(nodes, edges, multi, full)


class SkeletonGraphProcessor:
    """Encapsulates rasterization, skeletonization, graph building and analysis."""

    def __init__(
        self,
        grid_width_m: float,
        grid_height_m: float,
        grid_resolution_m: float,
        graph_min_edge_len_m: float,
        graph_node_merge_radius_m: float,
        event_potential_cell_size_m: float,
        min_event_distance_junction_m: float,
        event_potential_increment: float,
        event_potential_decay: float,
        event_potential_threshold: float,
        max_junction_detection_distance_junction_m: float,
        # Potential increment is always positive; unstable arms/DEG only shrink the step.
        event_potential_arm_consistency_enabled: bool = True,
        event_potential_arm_worst_match_max_deg: float = 45.0,
        event_potential_deg_penalty_base: float = 0.7,
        event_potential_increment_min_scale: float = 0.15,
        # After invalidation, suppress potential re-growth so the same junction is not immediately recreated.
        event_potential_invalidation_suppress_s: float = 2.0,
        junction_topology_settle_s: float = 0.8,
        arm_direction_fit_min_m: float = 0.35,
        arm_direction_fit_max_m: float = 2.0,
        arm_direction_fit_min_span_m: float = 0.8,
        clock_s: Optional[Callable[[], float]] = None,
        logger=None,
    ) -> None:
        self.grid_width_m = float(grid_width_m)
        self.grid_height_m = float(grid_height_m)
        self.grid_resolution_m = float(grid_resolution_m)
        self.graph_min_edge_len_m = float(graph_min_edge_len_m)
        self.graph_node_merge_radius_m = float(graph_node_merge_radius_m)
        self.min_event_distance_junction = float(min_event_distance_junction_m)
        self.event_potential_increment = float(event_potential_increment)
        self.event_potential_decay = float(event_potential_decay)
        self.event_potential_threshold = float(event_potential_threshold)
        # Decoupled from graph_node_merge_radius so stabilization granularity can be tuned separately.
        self.event_potential_cell_size = float(max(event_potential_cell_size_m, 1e-6))
        # DEAD_END is planner-driven; this distance only gates T/CROSS skeleton analysis.
        self.max_junction_detection_distance_junction = float(max_junction_detection_distance_junction_m)
        self._area_potential: Dict[Tuple[int, int], float] = {}
        self._current_junction_potential = 0.0
        # Confirmed-event cells keep their potential so decay cannot erase the landmark.
        self._frozen_area_keys: Set[Tuple[int, int]] = set()
        self._area_junction_type: Dict[Tuple[int, int], JunctionType] = {}
        self._graph_node_cache: Dict[Tuple[int, int], str] = {}
        self._logger = logger

        self._arm_consistency_enabled = bool(event_potential_arm_consistency_enabled)
        self._arm_worst_match_max_deg = float(max(event_potential_arm_worst_match_max_deg, 1e-3))
        self._deg_penalty_base = float(np.clip(event_potential_deg_penalty_base, 0.05, 1.0))
        self._increment_min_scale = float(np.clip(event_potential_increment_min_scale, 0.0, 1.0))
        self._last_area_path_angles: Dict[Tuple[int, int], np.ndarray] = {}

        # ROS injects its clock so suppression follows sim time; unit tests keep monotonic.
        self._invalidation_suppress_s = float(max(0.0, event_potential_invalidation_suppress_s))
        self._clock_s = clock_s if clock_s is not None else time.monotonic
        self._suppressed_area_until_s: Dict[Tuple[int, int], float] = {}
        self._topology_stabilizer = JunctionTopologyStabilizer(
            junction_topology_settle_s
        )
        self._arm_direction_fit_min_m = max(
            0.0,
            float(arm_direction_fit_min_m),
        )
        self._arm_direction_fit_max_m = max(
            self._arm_direction_fit_min_m + 1e-3,
            float(arm_direction_fit_max_m),
        )
        self._arm_direction_fit_min_span_m = max(
            0.0,
            float(arm_direction_fit_min_span_m),
        )

    def reset(self) -> None:
        """Clear all stabilization/potential caches as if freshly constructed."""
        self._area_potential.clear()
        self._current_junction_potential = 0.0
        self._frozen_area_keys.clear()
        self._area_junction_type.clear()
        self._graph_node_cache.clear()
        self._last_area_path_angles.clear()
        self._suppressed_area_until_s.clear()
        self._topology_stabilizer.reset()

    def _is_area_suppressed(self, key: Tuple[int, int]) -> bool:
        until = self._suppressed_area_until_s.get(key)
        if until is None:
            return False
        now = float(self._clock_s())
        if now >= float(until):
            del self._suppressed_area_until_s[key]
            return False
        return True

    def rasterize_from_profile(
        self,
        profile,
        current_pose: Pose3D,
    ) -> Tuple[np.ndarray, float, float, float]:
        """Fill a binary raster from the filtered radial curve (255=inside, 0=outside)."""
        if current_pose is None:
            return np.zeros((1, 1), dtype=np.uint8), self.grid_resolution_m, 0.0, 0.0

        w_m = self.grid_width_m
        h_m = self.grid_height_m
        res = self.grid_resolution_m

        width_px = max(1, int(round(w_m / res)))
        height_px = max(1, int(round(h_m / res)))

        origin_x = float(current_pose.x) - 0.5 * w_m
        origin_y = float(current_pose.y) - 0.5 * h_m

        image = np.zeros((height_px, width_px), dtype=np.uint8)

        distances = getattr(profile, "filtered_distances", None)
        if distances is None:
            distances = getattr(profile, "raw_distances", None)
        if distances is None:
            distances = []

        boundary_points = []
        for angle_deg, distance in zip(profile.angles, distances):
            angle_rad = np.deg2rad(angle_deg)
            world_x = current_pose.x + distance * np.cos(angle_rad)
            world_y = current_pose.y + distance * np.sin(angle_rad)

            col = int((world_x - origin_x) / res)
            row = int((world_y - origin_y) / res)

            col = max(0, min(width_px - 1, col))
            row = max(0, min(height_px - 1, row))

            boundary_points.append([col, row])

        boundary_points = np.array(boundary_points, dtype=np.int32)
        if len(boundary_points) > 2:
            cv2.fillPoly(image, [boundary_points], 255)
        return image, res, origin_x, origin_y

    def compute_skeleton(self, filled_area: np.ndarray) -> np.ndarray:
        """Skeletonize the filled area from radial profile (returns uint8 0/255)."""
        if filled_area.size == 0 or filled_area.sum() == 0:
            return np.zeros_like(filled_area, dtype=np.uint8)

        binary = (filled_area > 0).astype(np.uint8)
        binary_img = (binary * 255).astype(np.uint8)
        ske = skeletonize(binary_img)
        return ske.astype(np.uint8)

    def build_skeleton_graph(self, skeleton: np.ndarray) -> Optional[nx.Graph]:
        """Build graph from skeleton using SKNW algorithm."""
        if skeleton.size == 0:
            return None
        ske01 = (skeleton > 0).astype(np.uint8)
        if ske01.sum() == 0:
            return nx.Graph()
        graph = build_sknw(ske01, multi=False, iso=True, ring=True, full=True)
        return graph

    def prune_and_merge_graph(
        self,
        graph: nx.Graph,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> nx.Graph:
        """Prune short edges and merge close nodes."""
        if graph is None or graph.number_of_edges() == 0:
            return graph

        to_remove = []
        for u, v, attr in graph.edges(data=True):
            weight_px = float(attr.get('weight', 0.0))
            weight_m = weight_px * resolution
            if weight_m < self.graph_min_edge_len_m:
                to_remove.append((u, v))
        for (u, v) in to_remove:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)

        graph = self._merge_close_graph_nodes(graph, origin_x, origin_y, resolution)

        to_remove = []
        for u, v, attr in graph.edges(data=True):
            weight_px = float(attr.get('weight', 0.0))
            weight_m = weight_px * resolution
            if weight_m < self.graph_min_edge_len_m:
                to_remove.append((u, v))
        for (u, v) in to_remove:
            if graph.has_edge(u, v):
                graph.remove_edge(u, v)
        return graph

    def _merge_close_graph_nodes(
        self,
        graph: nx.Graph,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> nx.Graph:
        if graph is None or graph.number_of_nodes() == 0:
            return graph

        radius_m = float(self.graph_node_merge_radius_m)
        if radius_m <= 1e-9:
            return graph

        ids = list(graph.nodes())
        if not ids:
            return graph

        positions = []
        for node_id in ids:
            attr = graph.nodes[node_id]
            o = np.asarray(attr.get('o', [0, 0]), dtype=np.float32)
            r = float(o[0] - 1.0)
            c = float(o[1] - 1.0)
            x = origin_x + c * resolution
            y = origin_y + r * resolution
            positions.append((x, y))
        positions = np.array(positions, dtype=np.float32)
        N = len(ids)

        rad_sq = radius_m * radius_m
        visited = np.zeros(N, dtype=bool)
        clusters: List[List[int]] = []

        for i in range(N):
            if visited[i]:
                continue
            visited[i] = True
            stack = [i]
            cluster = [i]
            while stack:
                k = stack.pop()
                dx = positions[:, 0] - positions[k, 0]
                dy = positions[:, 1] - positions[k, 1]
                dist_sq = dx * dx + dy * dy
                close = np.where((dist_sq <= rad_sq) & (~visited))[0]
                for j in close:
                    visited[j] = True
                    stack.append(j)
                    cluster.append(j)
            clusters.append(cluster)

        if len(clusters) == N:
            return graph

        new_graph = nx.Graph()
        old_to_new: Dict[int, int] = {}
        for cluster_id, cluster_indices in enumerate(clusters):
            cluster_node_ids = [ids[i] for i in cluster_indices]
            o_values = []
            pts_list = []
            for node_id in cluster_node_ids:
                attr = graph.nodes[node_id]
                o_values.append(np.asarray(attr.get('o', [0, 0]), dtype=np.float32))
                pts = attr.get('pts', None)
                if pts is not None:
                    pts_list.append(np.asarray(pts, dtype=np.int16))
            if o_values:
                o_mean = np.mean(np.array(o_values), axis=0)
                o_new = np.rint(o_mean).astype(np.uint16)
            else:
                o_new = np.array([0, 0], dtype=np.uint16)
            if pts_list:
                pts_agg = np.vstack(pts_list)
            else:
                pts_agg = o_new.reshape(1, 2).astype(np.int16)
            new_graph.add_node(cluster_id, pts=pts_agg, o=o_new)
            for node_id in cluster_node_ids:
                old_to_new[node_id] = cluster_id

        for u, v, attr in graph.edges(data=True):
            nu = old_to_new.get(u)
            nv = old_to_new.get(v)
            if nu is None or nv is None or nu == nv:
                continue
            pts = np.asarray(attr.get('pts', None))
            weight = float(attr.get('weight', 0.0))
            if new_graph.has_edge(nu, nv):
                cur_weight = new_graph.edges[nu, nv].get('weight', 1e9)
                if weight < cur_weight:
                    new_graph.edges[nu, nv]['pts'] = pts
                    new_graph.edges[nu, nv]['weight'] = weight
            else:
                new_graph.add_edge(nu, nv, pts=pts, weight=weight)
        return new_graph

    def analyze_junction(
        self,
        graph: nx.Graph,
        current_pose: Pose3D,
        origin_x: float,
        origin_y: float,
        resolution: float,
        last_event_pose: Optional[Pose3D],
    ) -> Optional[Tuple[JunctionType, Pose3D, List[float], float]]:
        """Detect nearby T/CROSS junctions from graph topology, or None."""
        self._current_junction_potential = 0.0
        if graph is None or graph.number_of_nodes() == 0 or current_pose is None:
            return None

        robot_pos = np.array([current_pose.x, current_pose.y], dtype=np.float32)

        nearest_node = None
        nearest_dist = float('inf')
        for node_id, attr in graph.nodes(data=True):
            o = np.asarray(attr.get('o', [0, 0]), dtype=np.float32)
            r = float(o[0] - 1.0)
            c = float(o[1] - 1.0)
            x = origin_x + c * resolution
            y = origin_y + r * resolution
            dist = np.linalg.norm(robot_pos - np.array([x, y]))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_node = (node_id, attr, x, y)
        if nearest_node is None:
            return None

        node_id, attr, node_x, node_y = nearest_node
        area_key = self._area_key(node_x, node_y)
        max_dist = float(self.max_junction_detection_distance_junction)
        if nearest_dist > max_dist:
            return None

        node_key = (int(node_x * 10), int(node_y * 10))
        if node_key in self._graph_node_cache:
            self._current_junction_potential = float(
                np.clip(
                    self._area_potential.get(
                        area_key,
                        self.event_potential_threshold,
                    ),
                    0.0,
                    1.0,
                )
            )
            return None

        self._decay_potentials_around(robot_pos, area_key)

        if self._is_area_suppressed(area_key):
            return None

        degree = int(graph.degree(node_id))
        if degree >= 4:
            junction_type = JunctionType.CROSS_JUNCTION
        elif degree == 3:
            junction_type = JunctionType.T_JUNCTION
        else:
            # DEAD_END is planner-driven; skeleton degree-1 is not treated as a dead-end.
            self._topology_stabilizer.invalidate(area_key)
            return None

        if last_event_pose is not None:
            dist_traveled = current_pose.distance_to(last_event_pose)
            if dist_traveled < float(self.min_event_distance_junction):
                self._current_junction_potential = float(
                    np.clip(
                        self._area_potential.get(area_key, 0.0),
                        0.0,
                        1.0,
                    )
                )
                return None

        junction_pose = Pose3D(x=node_x, y=node_y, z=current_pose.z, yaw=current_pose.yaw)

        path_angles = self._compute_path_angles_from_graph(
            node_id, attr, graph, origin_x, origin_y, resolution
        )
        topology_stable = self._topology_stabilizer.observe(
            area_key,
            degree,
            float(self._clock_s()),
        )
        inc_scale = self._compute_potential_increment_scale(area_key, path_angles)
        potential_value = self._update_event_potential(
            area_key, observed=True, increment_scale=inc_scale
        )
        self._current_junction_potential = float(
            np.clip(potential_value, 0.0, 1.0)
        )
        self._area_junction_type[area_key] = junction_type
        if (
            potential_value < self.event_potential_threshold
            or not topology_stable
        ):
            return None
        self._frozen_area_keys.add(area_key)

        self._graph_node_cache[node_key] = f"{node_id}"

        if self._logger is not None:
            self._logger.info(
                f"Graph topology: {junction_type.name} at ({node_x:.2f}, {node_y:.2f}) "
                f"degree={degree}, paths={len(path_angles)}, potential={potential_value:.2f}"
            )

        return junction_type, junction_pose, path_angles, float(potential_value)

    @property
    def current_junction_potential(self) -> float:
        """Return potential from the latest nearby junction observation."""
        return self._current_junction_potential

    def _area_key(self, x: float, y: float) -> Tuple[int, int]:
        cell = self.event_potential_cell_size
        return (int(math.floor(x / cell)), int(math.floor(y / cell)))

    def _update_event_potential(
        self,
        key: Tuple[int, int],
        observed: bool,
        increment_scale: float = 1.0,
    ) -> float:
        cur = float(self._area_potential.get(key, 0.0))
        if observed:
            scale = float(max(0.0, increment_scale))
            cur = min(1.0, cur + float(self.event_potential_increment) * scale)
        else:
            cur = max(0.0, cur - float(self.event_potential_decay))
        if cur <= 0.0:
            if key in self._area_potential:
                del self._area_potential[key]
            self._frozen_area_keys.discard(key)
            if key in self._area_junction_type:
                del self._area_junction_type[key]
        else:
            self._area_potential[key] = cur
        return cur

    def _compute_potential_increment_scale(
        self,
        area_key: Tuple[int, int],
        path_angles: List[float],
    ) -> float:
        """Scale [min_scale, 1] that shrinks the potential increment when arms/DEG jitter."""
        if not self._arm_consistency_enabled:
            return 1.0

        new = np.asarray(path_angles, dtype=np.float32).ravel()
        prev = self._last_area_path_angles.get(area_key)

        if prev is None:
            self._last_area_path_angles[area_key] = new
            return 1.0

        # Ignore |ΔDEG| of 1; skeleton noise commonly adds/drops a single arm.
        deg_delta = abs(int(prev.size) - int(new.size))
        if deg_delta > 1:
            deg_scale = float(self._deg_penalty_base ** (deg_delta - 1))
        else:
            deg_scale = 1.0

        if prev.size == 0 or new.size == 0:
            self._last_area_path_angles[area_key] = new
            return float(max(self._increment_min_scale, deg_scale))

        old_mat = prev.astype(np.float32)[:, None]
        new_mat = new.astype(np.float32)[None, :]
        d = np.abs(np.arctan2(np.sin(old_mat - new_mat), np.cos(old_mat - new_mat)))
        min_old = np.min(d, axis=1)
        min_new = np.min(d, axis=0)
        worst_rad = float(max(float(np.max(min_old)), float(np.max(min_new))))
        worst_deg = float(np.degrees(worst_rad))

        # sqrt decay is more forgiving of small-to-medium arm-angle jitter than linear.
        ratio = min(1.0, worst_deg / float(self._arm_worst_match_max_deg))
        angle_scale = 1.0 - math.sqrt(ratio)
        angle_scale = float(max(0.0, min(1.0, angle_scale)))

        self._last_area_path_angles[area_key] = new
        scale = float(angle_scale * deg_scale)
        return float(max(self._increment_min_scale, min(1.0, scale)))

    def get_event_potential_cells(self) -> List[Tuple[float, float, float, str]]:
        """Export (x, y, potential, type_label) at each occupied area-cell center."""
        if not self._area_potential:
            return []
        cell_size = self.event_potential_cell_size
        cells: List[Tuple[float, float, float, str]] = []
        for (ix, iy), val in self._area_potential.items():
            if val <= 0.0:
                continue
            cx = (float(ix) + 0.5) * cell_size
            cy = (float(iy) + 0.5) * cell_size
            jt = self._area_junction_type.get((ix, iy))
            label = jt.name if jt is not None else "UNKNOWN"
            cells.append((cx, cy, float(val), label))
        return cells

    def _decay_potentials_around(
        self,
        robot_pos: np.ndarray,
        current_area_key: Optional[Tuple[int, int]],
    ) -> None:
        """Decay non-frozen cells near the robot except the cell updated this cycle."""
        radius = float(self.max_junction_detection_distance_junction)
        if radius <= 0.0 or not self._area_potential:
            return
        r2 = radius * radius
        cell_size = self.event_potential_cell_size

        for key in list(self._area_potential.keys()):
            if key == current_area_key:
                continue
            if key in self._frozen_area_keys:
                continue
            ix, iy = key
            cx = (float(ix) + 0.5) * cell_size
            cy = (float(iy) + 0.5) * cell_size
            dx = cx - float(robot_pos[0])
            dy = cy - float(robot_pos[1])
            if dx * dx + dy * dy <= r2:
                self._update_event_potential(key, observed=False)

    def clear_event_potential_at(self, x: float, y: float, *, clear_graph_cache: bool = True) -> None:
        """Clear potential at (x, y) so a deleted event cannot immediately re-trigger."""
        area_key = self._area_key(x, y)
        if area_key in self._area_potential:
            del self._area_potential[area_key]
        self._frozen_area_keys.discard(area_key)
        if area_key in self._area_junction_type:
            del self._area_junction_type[area_key]
        if area_key in self._last_area_path_angles:
            del self._last_area_path_angles[area_key]
        if area_key in self._suppressed_area_until_s:
            del self._suppressed_area_until_s[area_key]
        self._topology_stabilizer.invalidate(area_key)
        if clear_graph_cache:
            node_key = (int(x * 10), int(y * 10))
            if node_key in self._graph_node_cache:
                del self._graph_node_cache[node_key]

    def suppress_event_potential_at(self, x: float, y: float, *, duration_s: Optional[float] = None) -> None:
        """Clear potential at (x, y) and block re-growth for duration_s."""
        dur = float(self._invalidation_suppress_s if duration_s is None else max(0.0, float(duration_s)))
        self.clear_event_potential_at(float(x), float(y), clear_graph_cache=True)
        if dur <= 0.0:
            return
        area_key = self._area_key(float(x), float(y))
        self._suppressed_area_until_s[area_key] = float(self._clock_s()) + dur

    def _compute_path_angles_from_graph(
        self,
        node_id: int,
        node_attr: Dict,
        graph: nx.Graph,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> List[float]:
        angles: List[float] = []
        o = np.asarray(node_attr.get('o', [0, 0]), dtype=np.float32)
        r_node = float(o[0] - 1.0)
        c_node = float(o[1] - 1.0)
        x_node = origin_x + c_node * resolution
        y_node = origin_y + r_node * resolution
        for neighbor_id in graph.neighbors(node_id):
            neighbor_attr = graph.nodes[neighbor_id]
            o_n = np.asarray(neighbor_attr.get('o', [0, 0]), dtype=np.float32)
            r_n = float(o_n[0] - 1.0)
            c_n = float(o_n[1] - 1.0)
            x_n = origin_x + c_n * resolution
            y_n = origin_y + r_n * resolution
            dx = x_n - x_node
            dy = y_n - y_node
            fallback_angle = math.atan2(dy, dx)
            edge_data = graph.get_edge_data(node_id, neighbor_id) or {}
            points_rc = np.asarray(
                edge_data.get('pts', []),
                dtype=np.float32,
            )
            if points_rc.ndim != 2 or points_rc.shape[1] < 2:
                angles.append(fallback_angle)
                continue
            points_xy = np.column_stack(
                (
                    origin_x + (points_rc[:, 1] - 1.0) * resolution,
                    origin_y + (points_rc[:, 0] - 1.0) * resolution,
                )
            ).astype(np.float32, copy=False)
            estimate = estimate_arm_direction(
                points_xy,
                np.array([x_node, y_node], dtype=np.float32),
                min_distance_m=self._arm_direction_fit_min_m,
                max_distance_m=self._arm_direction_fit_max_m,
                min_span_m=self._arm_direction_fit_min_span_m,
                fallback_angle_rad=fallback_angle,
            )
            angles.append(estimate.angle_rad)
        return angles

