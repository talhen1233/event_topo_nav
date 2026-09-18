"""Radial profile processing pipeline."""
from __future__ import annotations

import json
import math
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import networkx as nx
from geometry_msgs.msg import Point
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException


class _PerformanceProfiler:
    """Optional cycle-timing stats, enabled via ROS param profiling_enabled."""
    
    COMPONENTS = [
        'radial_profile_construction',
        'skeleton_extraction',
        'potential_filter_update',
        'descriptor_computation',
        'event_matching',
        'total_cycle',
    ]
    
    def __init__(self):
        self._timings: dict = defaultdict(list)
        self._enabled = False
        self._cycle_count = 0
        self._start_time = time.time()
        self._last_report_time = time.time()
        self._report_interval = 10.0
    
    def enable(self):
        self._enabled = True
        self._start_time = time.time()
        self._last_report_time = time.time()
    
    def record(self, component: str, time_ms: float):
        if not self._enabled:
            return
        self._timings[component].append(time_ms)
        if component == 'total_cycle':
            self._cycle_count += 1
            if time.time() - self._last_report_time >= self._report_interval:
                self._print_report()
                self._last_report_time = time.time()
    
    def _print_report(self):
        if self._cycle_count == 0:
            return
        duration = time.time() - self._start_time
        print("\n" + "=" * 70)
        print(" PROFILING REPORT - Event Navigation System")
        print("=" * 70)
        print(f" Duration: {duration:.1f}s | Cycles: {self._cycle_count} | Rate: {self._cycle_count/duration:.2f}Hz")
        print("-" * 70)
        
        display_names = {
            'radial_profile_construction': 'Radial profile construction',
            'skeleton_extraction': 'Skeleton extraction',
            'potential_filter_update': 'Potential filter update',
            'descriptor_computation': 'Descriptor computation',
            'event_matching': 'Event matching (per candidate)',
            'total_cycle': 'Total cycle (<500ms budget)',
        }
        
        print(f" {'Component':<40} {'Mean':>8} {'Std':>8} {'P95':>8}")
        print("-" * 70)
        
        for comp in self.COMPONENTS:
            name = display_names.get(comp, comp)
            times = self._timings.get(comp, [])
            if times:
                arr = np.array(times)
                print(f" {name:<40} {np.mean(arr):>6.1f}ms {np.std(arr):>6.1f}ms {np.percentile(arr,95):>6.1f}ms")
            else:
                print(f" {name:<40} {'--':>8} {'--':>8} {'--':>8}")
        
        print("-" * 70)
        print("\n LaTeX table:")
        for comp in self.COMPONENTS:
            times = self._timings.get(comp, [])
            if times and comp != 'total_cycle':
                name = display_names.get(comp, comp)
                print(f" {name} & {int(round(np.mean(times)))} \\\\")
        if self._timings.get('total_cycle'):
            print(f" Total cycle ($<$500\\,ms budget) & {int(round(np.mean(self._timings['total_cycle'])))} \\\\")
        print("=" * 70 + "\n")


_profiler = _PerformanceProfiler()

try:  # pragma: no cover
    from .event_types import Decision, EventType, JunctionType, NavigationMode
    from .navigation_event import NavigationEvent, Pose3D, TunnelGeometry, JunctionConfiguration
    from .potential_update_gate import potential_update_due
    from .radial_descriptor import create_descriptor_from_profile
    from .event_repository import EventRepository
except ImportError:  # pragma: no cover
    from core.event_types import Decision, EventType, JunctionType, NavigationMode
    from core.navigation_event import NavigationEvent, Pose3D, TunnelGeometry, JunctionConfiguration
    from core.potential_update_gate import potential_update_due
    from core.radial_descriptor import create_descriptor_from_profile
    from core.event_repository import EventRepository

class RadialProfileProcessor:
    """Encapsulates radial profile updates, skeletonization, and visualization."""

    def __init__(self, node: Node, repository: EventRepository):
        self._node = node
        self._repository = repository
        self._last_graph: Optional[nx.Graph] = None
        self._last_skeleton: Optional[np.ndarray] = None
        self._last_filled_area: Optional[np.ndarray] = None
        self._last_grid_params: Optional[Tuple[float, float, float]] = None
        self._last_voxel_occupancy: Optional[np.ndarray] = None
        self._last_potential_update_pose: Optional[Pose3D] = None
        self._last_potential_update_time_s: Optional[float] = None
        # One DEAD_END event per blocked/search episode; the planner can stay stuck for many cycles.
        self._dead_end_event_published: bool = False
        self._dead_end_event_xy: Optional[np.ndarray] = None
        self._dead_end_potential: float = 0.0
        self._turn_to_opening_start_time: Optional[float] = None
        self._dead_end_terminal_proximity_m: float = self._get_param(
            'dead_end_terminal_proximity_m', 1.5
        )
        self._dead_end_proximity_potential_max: float = self._get_param(
            'dead_end_proximity_potential_max', 0.6
        )
        self._dead_end_turn_potential_rate: float = self._get_param(
            'dead_end_turn_potential_rate', 0.15
        )
        self._dead_end_confirmation_threshold: float = self._get_param(
            'dead_end_confirmation_threshold', 0.85
        )

    def reset(self) -> None:
        """Reset cached intermediate products and dead-end detection state."""
        self._last_graph = None
        self._last_skeleton = None
        self._last_filled_area = None
        self._last_grid_params = None
        self._last_voxel_occupancy = None
        self._last_potential_update_pose = None
        self._last_potential_update_time_s = None
        self._dead_end_event_published = False
        self._dead_end_event_xy = None
        self._dead_end_potential = 0.0
        self._turn_to_opening_start_time = None
    
    def _get_param(self, name: str, default: float) -> float:
        """Safely get a ROS parameter from the node, returning default if not available."""
        if hasattr(self._node, 'get_parameter'):
            try:
                return float(self._node.get_parameter(name).value)
            except (ParameterNotDeclaredException, AttributeError, TypeError, ValueError):
                return float(default)
        return default

    def _find_nearest_terminal_node(
        self,
        graph: nx.Graph,
        origin_x: float,
        origin_y: float,
        resolution: float,
        drone_xy: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Nearest degree-1 skeleton node, or (None, inf)."""
        if graph is None or graph.number_of_nodes() == 0:
            return None, float("inf")
        
        best_xy: Optional[np.ndarray] = None
        best_dist = float("inf")
        
        for node_id, attr in graph.nodes(data=True):
            degree = graph.degree(node_id)
            if degree != 1:
                continue
            
            o = np.asarray(attr.get('o', [0, 0]), dtype=np.float32)
            if o.size < 2:
                continue
            
            r = float(o[0] - 1.0)
            c = float(o[1] - 1.0)
            x = float(origin_x + c * resolution)
            y = float(origin_y + r * resolution)
            
            dist = float(np.hypot(x - drone_xy[0], y - drone_xy[1]))
            if dist < best_dist:
                best_dist = dist
                best_xy = np.array([x, y], dtype=np.float32)
        
        return best_xy, best_dist

    def _update_dead_end_potential_from_graph(self) -> float:
        """Assist DEAD_END only while the planner is already blocked or searching."""
        node = self._node
        current_time = node.get_clock().now().nanoseconds / 1e9
        local_state = (getattr(node, "local_planner_state", "") or "").strip().upper()
        planner_dead_end_episode = local_state in ("BLOCKED", "TURN_TO_OPENING", "DEAD_END")

        if not planner_dead_end_episode:
            self._turn_to_opening_start_time = None
            self._dead_end_potential *= 0.8
            return self._dead_end_potential

        proximity_potential = 0.0
        if (self._last_graph is not None and 
            self._last_grid_params is not None and 
            node.current_pose is not None):
            
            origin_x, origin_y, resolution = self._last_grid_params
            drone_xy = np.array([node.current_pose.x, node.current_pose.y], dtype=np.float32)
            
            _, dist_to_terminal = self._find_nearest_terminal_node(
                self._last_graph, origin_x, origin_y, resolution, drone_xy
            )
            
            if dist_to_terminal < self._dead_end_terminal_proximity_m:
                proximity_potential = self._dead_end_proximity_potential_max * (
                    1.0 - dist_to_terminal / self._dead_end_terminal_proximity_m
                )
        
        turn_potential = 0.0
        if local_state == "TURN_TO_OPENING":
            if self._turn_to_opening_start_time is None:
                self._turn_to_opening_start_time = current_time
            else:
                time_in_turn = current_time - self._turn_to_opening_start_time
                turn_potential = min(1.0, time_in_turn * self._dead_end_turn_potential_rate)
        else:
            self._turn_to_opening_start_time = None

        combined_potential = max(proximity_potential, turn_potential)
        alpha = 0.3
        self._dead_end_potential = (
            alpha * combined_potential + (1.0 - alpha) * self._dead_end_potential
        )
        return self._dead_end_potential

    def process(self):
        node = self._node
        profiling = getattr(node, '_profiling_enabled', False)
        if profiling and not _profiler._enabled:
            _profiler.enable()
            node.get_logger().info("Performance profiling ENABLED - reports every 10s")
        
        cycle_start = time.perf_counter() if profiling else 0
        
        pose_used = node.get_profile_pose() if hasattr(node, "get_profile_pose") else node.current_pose
        if pose_used is None or not bool(getattr(node, "_have_voxel_map_message", True)):
            return
        setattr(node, "_last_profile_pose_used", pose_used)

        current_time = node.get_clock().now().nanoseconds / 1e9 - node.start_time
        drone_position = np.array([pose_used.x, pose_used.y, pose_used.z], dtype=np.float32)

        t0 = time.perf_counter() if profiling else 0
        node.profile.update_from_voxels(
            node.voxel_map,
            drone_position,
            pose_used.yaw,
            node.voxel_size,
            current_time,
        )
        node.profile.apply_filtering(confidence_threshold=node.filter_confidence_threshold)
        if profiling:
            _profiler.record('radial_profile_construction', (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter() if profiling else 0
        descriptor = create_descriptor_from_profile(
            node.profile,
            num_harmonics=node.fft_num_harmonics,
            store_raw_profile=node.store_raw_profile,
        )
        node.matcher.update_current_descriptor(descriptor)
        if profiling:
            _profiler.record('descriptor_computation', (time.perf_counter() - t0) * 1000)
        
        msg = String()
        msg.data = json.dumps(descriptor.to_dict())
        node.current_descriptor_pub.publish(msg)

        # START handshake needs the rebuilt profile; suppress other events until HOME is committed.
        if hasattr(node, "update_start_stability"):
            node.update_start_stability()
            if bool(getattr(node, "_mission_active", False)) and not bool(
                getattr(node, "_start_event_published", False)
            ):
                return

        t0 = time.perf_counter() if profiling else 0
        filled_area, resolution, origin_x, origin_y = node.skg.rasterize_from_profile(node.profile, pose_used)
        skeleton = node.skg.compute_skeleton(filled_area)
        graph = node.skg.build_skeleton_graph(skeleton)
        if graph is not None and graph.number_of_nodes() > 0:
            graph = node.skg.prune_and_merge_graph(graph, origin_x, origin_y, resolution)
        if profiling:
            _profiler.record('skeleton_extraction', (time.perf_counter() - t0) * 1000)
        
        self._last_graph = graph
        self._last_skeleton = skeleton
        self._last_filled_area = filled_area
        self._last_grid_params = (origin_x, origin_y, resolution)
        show_vox = bool(getattr(node, "debug_images_show_voxels", True))
        self._last_voxel_occupancy = (
            self._rasterize_voxel_occupancy(
                voxel_map=node.voxel_map,
                voxel_size=float(node.voxel_size),
                pose_z=float(pose_used.z),
                filled_shape=filled_area.shape,
                origin_x=float(origin_x),
                origin_y=float(origin_y),
                resolution=float(resolution),
            )
            if show_vox
            else None
        )
        self._publish_skeleton_graph_json()

        # Revalidate stored arms even when this cycle publishes no new junction.
        if hasattr(node, "revalidate_nearby_junction_arms"):
            node.revalidate_nearby_junction_arms()

        if graph is None or graph.number_of_nodes() == 0:
            if profiling:
                _profiler.record('total_cycle', (time.perf_counter() - cycle_start) * 1000)
            return

        t_potential = time.perf_counter() if profiling else 0
        local_state = (getattr(node, "local_planner_state", "") or "").strip().upper()
        planner_dead_end_episode = local_state in ("BLOCKED", "TURN_TO_OPENING", "DEAD_END")
        graph_dead_end_potential = self._update_dead_end_potential_from_graph()
        should_trigger_dead_end = (
            local_state == "DEAD_END"
            or (
                planner_dead_end_episode
                and graph_dead_end_potential >= self._dead_end_confirmation_threshold
            )
        )
        
        if should_trigger_dead_end:
            if not self._dead_end_event_published:
                pose = Pose3D(
                    x=float(node.current_pose.x),
                    y=float(node.current_pose.y),
                    z=float(node.current_pose.z),
                    yaw=float(node.current_pose.yaw),
                )
                event = self.create_junction_event(
                    JunctionType.DEAD_END, pose, graph_angles=[], potential=1.0
                )
                self._repository.publish_event(event)
                node.last_event_pose = event.pose
                node.distance_traveled = 0.0
                self._dead_end_event_published = True
                self._dead_end_event_xy = np.array(
                    [pose.x, pose.y],
                    dtype=np.float32,
                )
                if local_state == "DEAD_END":
                    node.get_logger().info("Dead-end triggered by local planner state")
                else:
                    node.get_logger().info(
                        f"Dead-end triggered by graph potential: {graph_dead_end_potential:.2f}"
                    )
            
            # Suppress new junctions while stuck so unstable events are not emitted at the same place.
            if profiling:
                _profiler.record('potential_filter_update', (time.perf_counter() - t_potential) * 1000)
                _profiler.record('total_cycle', (time.perf_counter() - cycle_start) * 1000)
            return
        
        if (
            self._dead_end_event_published
            and not planner_dead_end_episode
            and self._dead_end_event_xy is not None
        ):
            current_xy = np.array(
                [node.current_pose.x, node.current_pose.y],
                dtype=np.float32,
            )
            rearm_distance_m = max(
                0.1,
                float(
                    getattr(
                        node,
                        'dead_end_event_rearm_distance_m',
                        0.75,
                    )
                ),
            )
            if float(
                np.linalg.norm(current_xy - self._dead_end_event_xy)
            ) >= rearm_distance_m:
                self._dead_end_event_published = False
                self._dead_end_event_xy = None

        # Motion updates capture new views; the time bound keeps potential growing while the planner holds.
        now_s = node.get_clock().now().nanoseconds * 1e-9
        allow_stab_update = potential_update_due(
            pose_used,
            self._last_potential_update_pose,
            previous_update_s=self._last_potential_update_time_s,
            now_s=now_s,
            min_distance_m=float(node.min_potential_update_distance),
            min_yaw_deg=float(node.min_potential_update_yaw_deg),
            max_interval_s=float(
                getattr(
                    node,
                    'max_potential_update_interval_s',
                    1.0,
                )
            ),
        )
        if not allow_stab_update:
            if profiling:
                _profiler.record('potential_filter_update', (time.perf_counter() - t_potential) * 1000)
                _profiler.record('total_cycle', (time.perf_counter() - cycle_start) * 1000)
            return

        # Return matching uses stored history; new junction ids during BACKTRACK/RETURN would add noise.
        matcher = getattr(node, "matcher", None)
        nav_mode = getattr(matcher, "current_nav_mode", None) if matcher is not None else None
        if nav_mode in (NavigationMode.BACKTRACK, NavigationMode.RETURN_TO_BASE):
            if profiling:
                _profiler.record('potential_filter_update', (time.perf_counter() - t_potential) * 1000)
                _profiler.record('total_cycle', (time.perf_counter() - cycle_start) * 1000)
            return

        result = node.skg.analyze_junction(
            graph=graph,
            current_pose=pose_used,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=resolution,
            last_event_pose=node.last_event_pose,
        )
        self._last_potential_update_pose = Pose3D(
            x=pose_used.x,
            y=pose_used.y,
            z=pose_used.z,
            yaw=pose_used.yaw,
        )
        self._last_potential_update_time_s = now_s
        if result is None:
            if profiling:
                _profiler.record('potential_filter_update', (time.perf_counter() - t_potential) * 1000)
                _profiler.record('total_cycle', (time.perf_counter() - cycle_start) * 1000)
            return

        junction_type, junction_pose, path_angles, potential = result
        event = self.create_junction_event(junction_type, junction_pose, path_angles, potential)
        self._repository.publish_event(event)
        node.last_event_pose = event.pose
        node.distance_traveled = 0.0

        if hasattr(node, "revalidate_nearby_junction_arms"):
            node.revalidate_nearby_junction_arms()

        if profiling:
            _profiler.record('potential_filter_update', (time.perf_counter() - t_potential) * 1000)
            _profiler.record('total_cycle', (time.perf_counter() - cycle_start) * 1000)

    def publish_visualizations(self):
        node = self._node
        pose_used = getattr(node, "_last_profile_pose_used", None) or node.current_pose
        if pose_used is None or not node.voxel_map:
            return

        if bool(getattr(node, "debug_viz", False)):
            self._publish_radial_visualization()
        if bool(getattr(node, "debug_images_enabled", True)):
            self._publish_raw_profile_debug_image()
        if bool(getattr(node, "debug_viz", False)):
            self._publish_matched_radial_overlay()
            self._publish_skeleton_graph_visualization()
        if bool(getattr(node, "debug_images_enabled", True)):
            # viz_skeleton_overlay is an image overlay, not a MarkerArray.
            if bool(getattr(node, "viz_skeleton_overlay", True)):
                self._publish_skeleton_overlay_image()
        if bool(getattr(node, "debug_viz", False)):
            self._publish_event_potential_debug()
            self._repository.publish_event_markers()

    def _publish_raw_profile_debug_image(self) -> None:
        """Publish a compact raw-vs-filtered radial-profile debug plot."""
        node = self._node
        pub = getattr(node, "raw_profile_image_pub", None)
        if pub is None:
            return
        profile = getattr(node, "profile", None)
        if profile is None:
            return
        raw = getattr(profile, "raw_distances", None)
        conf = getattr(profile, "confidence", None)
        filt = getattr(profile, "filtered_distances", None)
        if raw is None or conf is None:
            return
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        if raw.size < 2 or conf.size != raw.size:
            return

        W = 720
        H = 220
        n = int(raw.size)
        xs = np.linspace(0.0, float(n), int(W), endpoint=False, dtype=np.float32)
        i0 = np.floor(xs).astype(np.int32)
        i1 = (i0 + 1) % n
        t = xs - i0.astype(np.float32)
        raw_s = (1.0 - t) * raw[i0] + t * raw[i1]
        conf_s = (1.0 - t) * conf[i0] + t * conf[i1]

        filt_s = None
        if filt is not None:
            filt = np.asarray(filt, dtype=np.float32).reshape(-1)
            if filt.size == n:
                filt_s = (1.0 - t) * filt[i0] + t * filt[i1]

        eff_max = getattr(profile, "_effective_max_ranges", None)
        if isinstance(eff_max, np.ndarray) and eff_max.shape == (n,):
            max_range = float(np.nanmax(np.asarray(eff_max, dtype=np.float32)))
        else:
            max_range = float(getattr(getattr(profile, "sensor_config", None), "max_range", 1.0))
        if not np.isfinite(max_range) or max_range <= 1e-6:
            max_range = 1.0

        img = np.zeros((H, W, 3), dtype=np.uint8)
        bg = (np.clip(conf_s, 0.0, 1.0) * 80.0).astype(np.uint8)
        img[:, :, 0] = bg
        img[:, :, 1] = bg
        img[:, :, 2] = bg

        top = 12
        bottom = H - 18
        span = float(max(1, bottom - top))

        def _y_from_dist(d: np.ndarray) -> np.ndarray:
            d_clip = np.clip(d, 0.0, max_range)
            y = (bottom - (d_clip / max_range) * span).astype(np.int32)
            return np.clip(y, top, bottom)

        y_raw = _y_from_dist(raw_s)
        pts_raw = np.stack([np.arange(W, dtype=np.int32), y_raw], axis=1).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_raw], isClosed=False, color=(40, 40, 255), thickness=2)

        if filt_s is not None:
            y_f = _y_from_dist(filt_s)
            pts_f = np.stack([np.arange(W, dtype=np.int32), y_f], axis=1).reshape((-1, 1, 2))
            cv2.polylines(img, [pts_f], isClosed=False, color=(255, 200, 40), thickness=2)

        cv2.line(img, (0, bottom), (W - 1, bottom), (100, 100, 100), 1)
        cv2.putText(img, "raw (red), filtered (cyan), bg=confidence", (8, H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

        jpeg = self._encode_bgr_jpeg(img, quality=90)
        if jpeg is None:
            return
        msg = CompressedImage()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "radial_profile_raw_debug"
        msg.format = "jpeg"
        msg.data = jpeg
        pub.publish(msg)

    @staticmethod
    def _rasterize_voxel_occupancy(
        voxel_map,
        voxel_size: float,
        pose_z: float,
        filled_shape: Tuple[int, int],
        origin_x: float,
        origin_y: float,
        resolution: float,
        z_half_height_m: float = 0.5,
    ) -> Optional[np.ndarray]:
        """Project occupied voxels onto the skeleton raster, keeping a ±z_half_height_m slab."""
        if voxel_map is None:
            return None
        if voxel_size <= 0.0 or resolution <= 0.0:
            return None
        H, W = int(filled_shape[0]), int(filled_shape[1])
        if H <= 0 or W <= 0:
            return None

        occ = np.zeros((H, W), dtype=np.uint8)
        z_band = float(max(0.0, z_half_height_m))
        for (ix, iy, iz), v in voxel_map.items():
            if hasattr(v, "occupancy") and not bool(v.occupancy):
                continue
            x = (float(ix) + 0.5) * float(voxel_size)
            y = (float(iy) + 0.5) * float(voxel_size)
            z = (float(iz) + 0.5) * float(voxel_size)
            if z_band > 0.0 and abs(z - float(pose_z)) > z_band:
                continue
            c = int(round((x - float(origin_x)) / float(resolution)))
            r = int(round((y - float(origin_y)) / float(resolution)))
            if 0 <= r < H and 0 <= c < W:
                occ[r, c] = 255
        return occ

    def create_junction_event(
        self,
        junction_type: JunctionType,
        pose: Pose3D,
        graph_angles: list,
        potential: float,
    ) -> NavigationEvent:
        node = self._node
        descriptor = create_descriptor_from_profile(
            node.profile,
            num_harmonics=node.fft_num_harmonics,
            store_raw_profile=node.store_raw_profile,
        )
        stats = node.profile.get_statistics()
        median_radius = float(stats.get('median_radius', stats.get('mean_radius', 0.0)))
        approx_width = max(0.0, 2.0 * median_radius)
        geometry = TunnelGeometry(
            width=approx_width,
            height=0.0,
            shape_descriptor=junction_type.name.lower(),
        )
        event = NavigationEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.JUNCTION,
            timestamp=datetime.now(),
            decision=Decision.HOVER,
            pose=pose,
            pose_uncertainty=node.pose_covariance if node.pose_covariance is not None else np.eye(3),
            odometry_distance=node.distance_traveled,
            tunnel_geometry=geometry,
        )
        event.radial_descriptor = descriptor
        event.confidence = float(max(0.0, min(1.0, potential)))
        event.previous_event_id = self._repository.last_published_event_id
        angles = [float(a) for a in (graph_angles or [])]

        entry_guess = float(pose.yaw + np.pi)
        entry_guess = float((entry_guess + np.pi) % (2.0 * np.pi) - np.pi)
        if angles:
            diffs = [
                abs(float(math.atan2(math.sin(a - entry_guess), math.cos(a - entry_guess))))
                for a in angles
            ]
            if diffs:
                event.entry_angle = float(angles[int(np.argmin(np.asarray(diffs, dtype=np.float32)))])
            else:
                event.entry_angle = entry_guess
        else:
            event.entry_angle = entry_guess

        # Downstream consumers expect junction_config even when path_angles is empty.
        event.junction_config = JunctionConfiguration(
            junction_type=junction_type,
            num_paths=len(angles),
            path_angles=angles,
        )
        arm_widths = self._estimate_arm_widths_from_dt(pose, angles)
        if arm_widths and len(arm_widths) == len(angles):
            event.junction_config.path_widths = arm_widths
        node.get_logger().info(
            f"Junction detected: {junction_type.name} at ({pose.x:.2f}, {pose.y:.2f}) "
            f"with {len(angles)} paths"
        )
        self._publish_junction_profile_preview(event)
        return event

    def _estimate_arm_widths_from_dt(
        self,
        pose: Pose3D,
        path_angles: list,
        max_length_m: float = 5.0,
        step_m: float = 0.5,
    ) -> List[float]:
        """Estimate per-arm width from DT samples along each arm, starting 1 m past the junction."""
        node = self._node
        default_width = 1.0

        if not path_angles:
            return []
        if self._last_filled_area is None or self._last_grid_params is None:
            node.get_logger().warn(
                "DT width estimation: no cached filled area; using default width "
                f"{default_width:.2f}m for all junction arms."
            )
            return [default_width] * len(path_angles)

        filled = self._last_filled_area
        origin_x, origin_y, resolution = self._last_grid_params
        if filled.size == 0 or resolution <= 0.0:
            node.get_logger().warn(
                "DT width estimation: invalid raster (empty or zero resolution); "
                f"using default width {default_width:.2f}m for all junction arms."
            )
            return [default_width] * len(path_angles)

        binary = (filled > 0).astype(np.uint8)
        dist_px = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        # Skip the widened intersection; sample along the arm starting 1 m out.
        start_offset_m = 1.0
        usable_length = max(0.0, max_length_m - start_offset_m)
        max_steps = max(1, int(usable_length / float(step_m)))
        h, w = dist_px.shape
        widths: List[float] = []

        for ang in path_angles:
            theta = float(ang)

            cos_a = math.cos(theta)
            sin_a = math.sin(theta)
            samples: List[float] = []

            for k in range(max_steps):
                s = float(start_offset_m + k * step_m)
                wx = float(pose.x + s * cos_a)
                wy = float(pose.y + s * sin_a)

                c = int(round((wx - origin_x) / resolution))
                r = int(round((wy - origin_y) / resolution))

                if r < 0 or r >= h or c < 0 or c >= w:
                    break
                if binary[r, c] == 0:
                    break

                d_m = float(dist_px[r, c] * resolution)
                if d_m > 0.0:
                    samples.append(d_m)

            if samples:
                half_width = float(np.median(np.array(samples, dtype=np.float32)))
                widths.append(max(0.0, 2.0 * half_width))
            else:
                node.get_logger().warn(
                    "DT width estimation: no valid DT samples along junction arm; "
                    f"using default width {default_width:.2f}m for this arm."
                )
                widths.append(default_width)

        return widths

    def _publish_junction_profile_preview(self, event: NavigationEvent) -> None:
        """Publish a scaled radial-profile ring at the junction pose."""
        node = self._node
        if not hasattr(node, "junction_profile_viz_pub"):
            return
        if node.junction_profile_viz_pub is None:
            return

        if node.profile is None or node.current_pose is None:
            return

        distances = getattr(node.profile, "filtered_distances", None)
        if distances is None:
            distances = getattr(node.profile, "raw_distances", None)
        angles_deg = getattr(node.profile, "angles", None)
        if distances is None or angles_deg is None:
            return

        num_rays = len(distances)
        if num_rays == 0:
            return

        step = max(1, num_rays // 16)

        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        now = node.get_clock().now().to_msg()
        ring = Marker()
        ring.header.frame_id = "odom"
        ring.header.stamp = now
        ring.ns = "junction_profile"
        ring.id = 0
        ring.type = Marker.LINE_STRIP
        ring.action = Marker.ADD
        ring.scale.x = 0.03
        ring.color.r = 0.2
        ring.color.g = 0.8
        ring.color.b = 0.2
        ring.color.a = 0.9

        cx = float(event.pose.x)
        cy = float(event.pose.y)
        cz = float(event.pose.z)
        scale = 0.2

        for i in range(0, num_rays, step):
            angle_rad = float(np.deg2rad(angles_deg[i]))
            r = float(distances[i]) * scale
            px = cx + r * float(np.cos(angle_rad))
            py = cy + r * float(np.sin(angle_rad))
            p = Point()
            p.x = px
            p.y = py
            p.z = cz
            ring.points.append(p)

        if len(ring.points) >= 2:
            ring.points.append(ring.points[0])

        ma.markers.append(ring)
        node.junction_profile_viz_pub.publish(ma)

    def _publish_radial_visualization(self):
        node = self._node
        pose = getattr(node, "_last_profile_pose_used", None) or node.current_pose
        if pose is None:
            return
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        now = node.get_clock().now().to_msg()
        marker_id = 0

        # Profile pose is often delayed relative to live odometry.
        cross = Marker()
        cross.header.frame_id = "odom"
        cross.header.stamp = now
        cross.ns = "profile_pose_used"
        cross.id = marker_id
        marker_id += 1
        cross.type = Marker.LINE_LIST
        cross.action = Marker.ADD
        cross.scale.x = 0.02
        cross.color.r = 1.0
        cross.color.g = 1.0
        cross.color.b = 1.0
        cross.color.a = 0.9
        d = 0.03
        z = float(pose.z) + 0.05
        p1 = Point(x=float(pose.x) - d, y=float(pose.y) - d, z=z)
        p2 = Point(x=float(pose.x) + d, y=float(pose.y) + d, z=z)
        p3 = Point(x=float(pose.x) - d, y=float(pose.y) + d, z=z)
        p4 = Point(x=float(pose.x) + d, y=float(pose.y) - d, z=z)
        cross.points = [p1, p2, p3, p4]
        ma.markers.append(cross)
        for i, (angle_deg, raw_dist, conf) in enumerate(zip(
            node.profile.angles,
            node.profile.raw_distances,
            node.profile.confidence,
        )):
            if conf < node.viz_min_confidence:
                continue
            angle_rad = np.deg2rad(angle_deg)
            start = Point()
            start.x = float(pose.x)
            start.y = float(pose.y)
            start.z = float(pose.z)
            end_raw = Point()
            end_raw.x = float(pose.x) + raw_dist * node.viz_ray_scale * np.cos(angle_rad)
            end_raw.y = float(pose.y) + raw_dist * node.viz_ray_scale * np.sin(angle_rad)
            end_raw.z = float(pose.z)
            m_raw = Marker()
            m_raw.header.frame_id = "odom"
            m_raw.header.stamp = now
            m_raw.ns = "rays_raw"
            m_raw.id = marker_id
            marker_id += 1
            m_raw.type = Marker.LINE_STRIP
            m_raw.action = Marker.ADD
            m_raw.points = [start, end_raw]
            m_raw.scale.x = 0.01
            m_raw.color.r = float(1.0 - conf)
            m_raw.color.g = float(conf)
            m_raw.color.b = float(0.0)
            m_raw.color.a = float(0.2 + 0.3 * conf)
            ma.markers.append(m_raw)

        m_curve = Marker()
        m_curve.header.frame_id = "odom"
        m_curve.header.stamp = now
        m_curve.ns = "filtered_curve"
        m_curve.id = marker_id
        marker_id += 1
        m_curve.type = Marker.LINE_STRIP
        m_curve.action = Marker.ADD
        m_curve.scale.x = 0.04
        m_curve.color.r = 0.2
        m_curve.color.g = 0.6
        m_curve.color.b = 1.0
        m_curve.color.a = node.viz_curve_alpha
        colors: List[ColorRGBA] = []
        for angle_deg, filt_dist, conf in zip(
            node.profile.angles, node.profile.filtered_distances, node.profile.confidence
        ):
            angle_rad = np.deg2rad(angle_deg)
            pt = Point()
            pt.x = float(pose.x) + filt_dist * np.cos(angle_rad)
            pt.y = float(pose.y) + filt_dist * np.sin(angle_rad)
            pt.z = float(pose.z)
            m_curve.points.append(pt)
            if node.viz_curve_use_gradient:
                c = float(max(0.0, min(1.0, conf)))
                r = (1.0 - c) * node.viz_curve_low[0] + c * node.viz_curve_high[0]
                g = (1.0 - c) * node.viz_curve_low[1] + c * node.viz_curve_high[1]
                b = (1.0 - c) * node.viz_curve_low[2] + c * node.viz_curve_high[2]
                colors.append(ColorRGBA(r=float(r), g=float(g), b=float(b), a=float(node.viz_curve_alpha)))
        if len(m_curve.points) > 0:
            m_curve.points.append(m_curve.points[0])
            if node.viz_curve_use_gradient and colors:
                colors.append(colors[0])
        if node.viz_curve_use_gradient and colors:
            m_curve.colors = colors
        ma.markers.append(m_curve)

        forward_cone = Marker()
        forward_cone.header.frame_id = "odom"
        forward_cone.header.stamp = now
        forward_cone.ns = "forward_sector"
        forward_cone.id = marker_id
        marker_id += 1
        forward_cone.type = Marker.LINE_LIST
        forward_cone.action = Marker.ADD
        forward_cone.scale.x = 0.02
        forward_cone.color.r = 0.0
        forward_cone.color.g = 1.0
        forward_cone.color.b = 0.0
        forward_cone.color.a = 0.4
        for sector_angle in [-45.0, 45.0]:
            angle_rad = np.deg2rad(sector_angle)
            start = Point()
            start.x = float(pose.x)
            start.y = float(pose.y)
            start.z = float(pose.z)
            end = Point()
            end.x = float(pose.x) + 3.0 * np.cos(angle_rad)
            end.y = float(pose.y) + 3.0 * np.sin(angle_rad)
            end.z = float(pose.z)
            forward_cone.points.append(start)
            forward_cone.points.append(end)
        ma.markers.append(forward_cone)
        node.radial_viz_pub.publish(ma)

    def _publish_matched_radial_overlay(self):
        node = self._node
        desc = self._repository.last_matched_descriptor
        pose = getattr(node, "_last_profile_pose_used", None) or node.current_pose
        if desc is None or pose is None:
            return
        if not hasattr(desc, 'filtered_distances') or desc.filtered_distances is None:
            return
        if len(desc.filtered_distances) != len(node.profile.filtered_distances):
            return
        ma = MarkerArray()
        now = node.get_clock().now().to_msg()
        cx = float(pose.x)
        cy = float(pose.y)
        cz = float(pose.z)
        scale = 0.2

        m_match = Marker()
        m_match.header.frame_id = "odom"
        m_match.header.stamp = now
        m_match.ns = "matched_profile_ring"
        m_match.id = 0
        m_match.type = Marker.LINE_STRIP
        m_match.action = Marker.ADD
        m_match.scale.x = 0.03
        m_match.color.r = 1.0
        m_match.color.g = 0.2
        m_match.color.b = 0.2
        m_match.color.a = 0.9
        for angle_deg, dist in zip(node.profile.angles, desc.filtered_distances):
            angle_rad = np.deg2rad(angle_deg)
            r = float(dist) * scale
            p = Point()
            p.x = cx + r * float(np.cos(angle_rad))
            p.y = cy + r * float(np.sin(angle_rad))
            p.z = cz
            m_match.points.append(p)
        if len(m_match.points) >= 2:
            m_match.points.append(m_match.points[0])
        ma.markers.append(m_match)

        m_curr = Marker()
        m_curr.header.frame_id = "odom"
        m_curr.header.stamp = now
        m_curr.ns = "current_profile_ring"
        m_curr.id = 1
        m_curr.type = Marker.LINE_STRIP
        m_curr.action = Marker.ADD
        m_curr.scale.x = 0.03
        m_curr.color.r = 0.2
        m_curr.color.g = 0.6
        m_curr.color.b = 1.0
        m_curr.color.a = 0.9
        for angle_deg, dist in zip(node.profile.angles, node.profile.filtered_distances):
            angle_rad = np.deg2rad(angle_deg)
            r = float(dist) * scale
            p = Point()
            p.x = cx + r * float(np.cos(angle_rad))
            p.y = cy + r * float(np.sin(angle_rad))
            p.z = cz
            m_curr.points.append(p)
        if len(m_curr.points) >= 2:
            m_curr.points.append(m_curr.points[0])
        ma.markers.append(m_curr)

        node.matched_profile_pub.publish(ma)

    def _publish_skeleton_graph_visualization(self):
        node = self._node
        if self._last_graph is None or self._last_grid_params is None:
            return
        origin_x, origin_y, resolution = self._last_grid_params
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        now = node.get_clock().now().to_msg()
        marker_id = 0
        for u, v, attr in self._last_graph.edges(data=True):
            pts = attr.get('pts', None)
            if pts is None:
                continue
            rc = np.asarray(pts, dtype=np.float32)
            r = rc[:, 0] - 1.0
            c = rc[:, 1] - 1.0
            xs = origin_x + c * resolution
            ys = origin_y + r * resolution
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = now
            m.ns = "skeleton_edges"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.8
            for xi, yi in zip(xs, ys):
                p = Point()
                p.x = float(xi)
                p.y = float(yi)
                p.z = node.current_pose.z if node.current_pose else 0.0
                m.points.append(p)
            ma.markers.append(m)

        for node_id, attr in self._last_graph.nodes(data=True):
            o = np.asarray(attr.get('o', [0, 0]), dtype=np.float32)
            r_node = float(o[0] - 1.0)
            c_node = float(o[1] - 1.0)
            x = origin_x + c_node * resolution
            y = origin_y + r_node * resolution
            degree = self._last_graph.degree(node_id)
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = now
            m.ns = "skeleton_nodes"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = node.current_pose.z if node.current_pose else 0.0
            if degree >= 4:
                m.scale.x = m.scale.y = m.scale.z = 0.20
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 1.0
            elif degree == 3:
                m.scale.x = m.scale.y = m.scale.z = 0.18
                m.color.r, m.color.g, m.color.b = 1.0, 0.5, 0.0
            elif degree == 1:
                m.scale.x = m.scale.y = m.scale.z = 0.15
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            else:
                m.scale.x = m.scale.y = m.scale.z = 0.10
                m.color.r, m.color.g, m.color.b = 0.5, 0.5, 0.5
            m.color.a = 0.9
            ma.markers.append(m)
            if degree != 2:
                m_text = Marker()
                m_text.header.frame_id = "odom"
                m_text.header.stamp = now
                m_text.ns = "skeleton_node_labels"
                m_text.id = marker_id
                marker_id += 1
                m_text.type = Marker.TEXT_VIEW_FACING
                m_text.action = Marker.ADD
                m_text.pose.position.x = float(x)
                m_text.pose.position.y = float(y)
                m_text.pose.position.z = (node.current_pose.z if node.current_pose else 0.0) + 0.25
                m_text.scale.z = 0.15
                m_text.color.r = 1.0
                m_text.color.g = 1.0
                m_text.color.b = 1.0
                m_text.color.a = 0.95
                m_text.text = f"Deg{degree}"
                ma.markers.append(m_text)

        node.skeleton_graph_pub.publish(ma)

    def _publish_skeleton_overlay_image(self):
        node = self._node
        pub = getattr(node, "skeleton_overlay_image_pub", None)
        if pub is None:
            return
        if self._last_skeleton is None or self._last_filled_area is None:
            return
        if self._last_grid_params is None:
            return
        filled = self._last_filled_area
        skeleton = self._last_skeleton
        show_vox = bool(getattr(node, "debug_images_show_voxels", True))
        occ = self._last_voxel_occupancy if show_vox else None
        if occ is not None and occ.shape != filled.shape:
            occ = None
        H, W = filled.shape
        overlay = np.zeros((H, W, 3), dtype=np.uint8)
        overlay[filled > 0] = [50, 50, 150]
        if occ is not None:
            overlay[occ > 0] = [170, 170, 170]
        overlay[skeleton > 0] = [255, 255, 0]

        origin_x, origin_y, resolution = self._last_grid_params
        pose = getattr(node, "_last_profile_pose_used", None) or node.current_pose
        profile = getattr(node, "profile", None)
        if pose is not None and profile is not None and float(resolution) > 0.0:
            angles = getattr(profile, "angles", None)
            filt_d = getattr(profile, "filtered_distances", None)
            if angles is not None and filt_d is not None:
                angles = np.asarray(angles, dtype=np.float32).reshape(-1)
                filt_d = np.asarray(filt_d, dtype=np.float32).reshape(-1)
                if angles.size >= 2 and filt_d.size == angles.size:
                    def _curve_to_polyline(distances: np.ndarray) -> Optional[np.ndarray]:
                        pts = []
                        for ang_deg, dist in zip(angles, distances):
                            a = float(np.deg2rad(float(ang_deg)))
                            wx = float(pose.x) + float(dist) * float(np.cos(a))
                            wy = float(pose.y) + float(dist) * float(np.sin(a))
                            c = int((wx - float(origin_x)) / float(resolution))
                            r = int((wy - float(origin_y)) / float(resolution))
                            c = max(0, min(W - 1, c))
                            r = max(0, min(H - 1, r))
                            pts.append((c, r))
                        if len(pts) < 2:
                            return None
                        return np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))

                    pts_f = _curve_to_polyline(filt_d)
                    if pts_f is not None:
                        cv2.polylines(overlay, [pts_f], isClosed=True, color=(255, 200, 40), thickness=2)

                    c0 = int((float(pose.x) - float(origin_x)) / float(resolution))
                    r0 = int((float(pose.y) - float(origin_y)) / float(resolution))
                    if 0 <= c0 < W and 0 <= r0 < H:
                        cv2.circle(overlay, (c0, r0), 4, (0, 255, 255), -1)
                    cv2.putText(
                        overlay,
                        "filled=blue, skel=yellow, vox=gray, filt=cyan",
                        (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (230, 230, 230),
                        1,
                        cv2.LINE_AA,
                    )
        if self._last_graph is not None:
            for node_id, attr in self._last_graph.nodes(data=True):
                o = np.asarray(attr.get('o', [0, 0]), dtype=np.float32)
                r = int(o[0] - 1.0)
                c = int(o[1] - 1.0)
                degree = self._last_graph.degree(node_id)
                if degree >= 4:
                    color = (255, 0, 255); radius = 5
                elif degree == 3:
                    color = (0, 128, 255); radius = 4
                elif degree == 1:
                    color = (0, 0, 255); radius = 3
                else:
                    color = (128, 128, 128); radius = 2
                if 0 <= r < H and 0 <= c < W:
                    cv2.circle(overlay, (c, r), radius, color, -1)

        # Flip so +Y in odom appears up in the image viewer.
        overlay_vis = overlay[::-1, :, :].copy()
        jpeg = self._encode_bgr_jpeg(overlay_vis, quality=90)
        if jpeg is None:
            return
        msg = CompressedImage()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "skeleton_overlay"
        msg.format = "jpeg"
        msg.data = jpeg
        pub.publish(msg)

    @staticmethod
    def _encode_bgr_jpeg(image_bgr: np.ndarray, quality: int = 90) -> Optional[bytes]:
        """Encode a BGR uint8 image as JPEG bytes."""
        if not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            return None
        if image_bgr.dtype != np.uint8:
            return None
        q = int(max(10, min(100, int(quality))))
        ok, compressed = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok or compressed is None:
            return None
        return compressed.tobytes()

    def _publish_skeleton_graph_json(self):
        node = self._node
        if self._last_graph is None or self._last_grid_params is None:
            return
        origin_x, origin_y, resolution = self._last_grid_params
        nodes_payload = []
        for node_id, attr in self._last_graph.nodes(data=True):
            o = np.asarray(attr.get('o', [0, 0]), dtype=np.float32)
            r = float(o[0] - 1.0)
            c = float(o[1] - 1.0)
            x = float(origin_x + c * resolution)
            y = float(origin_y + r * resolution)
            deg = int(self._last_graph.degree(node_id))
            nodes_payload.append({'id': int(node_id), 'x': x, 'y': y, 'deg': deg})
        edges_payload = []
        for u, v, attr in self._last_graph.edges(data=True):
            pts = attr.get('pts', None)
            if pts is None:
                continue
            rc = np.asarray(pts, dtype=np.float32)
            if rc.ndim != 2 or rc.shape[0] == 0:
                continue
            r = rc[:, 0] - 1.0
            c = rc[:, 1] - 1.0
            xs = origin_x + c * resolution
            ys = origin_y + r * resolution
            step = 3
            idxs = list(range(0, xs.shape[0], step))
            if (xs.shape[0] - 1) not in idxs:
                idxs.append(xs.shape[0] - 1)
            pts_xy = [[float(xs[i]), float(ys[i])] for i in idxs]
            edges_payload.append({'u': int(u), 'v': int(v), 'pts': pts_xy})
        payload = {'nodes': nodes_payload, 'edges': edges_payload}
        msg = String()
        msg.data = json.dumps(payload)
        node.skeleton_graph_json_pub.publish(msg)

    def _publish_event_potential_debug(self) -> None:
        """Visualize event area potentials as colored spheres in RViz."""
        node = self._node
        if not hasattr(node, "event_potential_debug_pub"):
            return
        if node.event_potential_debug_pub is None:
            return
        if not hasattr(node, "skg") or node.skg is None:
            return

        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        now = node.get_clock().now().to_msg()
        marker_id = 0

        if node.current_pose is not None and self._dead_end_potential > 0.01:
            cx = float(node.current_pose.x)
            cy = float(node.current_pose.y)
            cz = float(node.current_pose.z)
            de_pot = float(self._dead_end_potential)
            
            ring = Marker()
            ring.header.frame_id = "odom"
            ring.header.stamp = now
            ring.ns = "dead_end_ring"
            ring.id = marker_id
            marker_id += 1
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.scale.x = 0.06
            if de_pot < 0.5:
                ring.color.r = 2.0 * de_pot
                ring.color.g = 1.0
            else:
                ring.color.r = 1.0
                ring.color.g = 2.0 * (1.0 - de_pot)
            ring.color.b = 0.0
            ring.color.a = 0.8
            
            radius = 0.4 + 0.3 * de_pot
            for angle in np.linspace(0, 2 * np.pi, 20):
                p = Point()
                p.x = cx + radius * float(np.cos(angle))
                p.y = cy + radius * float(np.sin(angle))
                p.z = cz
                ring.points.append(p)
            ring.points.append(ring.points[0])
            ma.markers.append(ring)

        cells = node.skg.get_event_potential_cells()

        if node.current_pose is not None:
            max_viz_dist = float(getattr(node, "max_junction_detection_distance_junction", 0.0))
            if max_viz_dist > 0.0:
                cx = float(node.current_pose.x)
                cy = float(node.current_pose.y)
                filtered = []
                for (x, y, pot, etype) in cells:
                    dx = float(x) - cx
                    dy = float(y) - cy
                    if float(np.hypot(dx, dy)) <= max_viz_dist:
                        filtered.append((x, y, pot, etype))
                cells = filtered

        for (x, y, pot, etype) in cells:
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = now
            m.ns = "event_potential"
            m.id = marker_id
            marker_id += 1
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = node.current_pose.z if node.current_pose else 0.0
            scale = 0.1 + 0.4 * float(pot)
            m.scale.x = m.scale.y = m.scale.z = scale
            p = float(max(0.0, min(1.0, pot)))
            m.color.r = p
            m.color.g = 1.0 - max(0.0, p - 0.5) * 2.0
            m.color.b = 0.0
            m.color.a = 0.8
            ma.markers.append(m)

            t = Marker()
            t.header.frame_id = "odom"
            t.header.stamp = now
            t.ns = "event_potential_text"
            t.id = marker_id
            marker_id += 1
            t.type = Marker.TEXT_VIEW_FACING
            t.action = Marker.ADD
            t.pose.position.x = float(x)
            t.pose.position.y = float(y)
            t.pose.position.z = (node.current_pose.z if node.current_pose else 0.0) + scale * 0.8
            t.scale.z = 0.15
            t.color.r = t.color.g = t.color.b = 1.0
            t.color.a = 0.9
            t.text = f"{pot:.2f} {etype}"
            ma.markers.append(t)

        node.event_potential_debug_pub.publish(ma)

