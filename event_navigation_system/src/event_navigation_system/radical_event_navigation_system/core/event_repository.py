"""Event storage, query, and visualization helpers."""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from rclpy.node import Node

try:  # pragma: no cover - allow running module directly
    from .event_types import EventType, JunctionType
    from .navigation_event import NavigationEvent
    from .radial_descriptor import RadialDescriptorMatcher
except ImportError:  # pragma: no cover
    from core.event_types import EventType, JunctionType
    from core.navigation_event import NavigationEvent
    from core.radial_descriptor import RadialDescriptorMatcher

class EventRepository:
    """Manage detected events, merging, queries, and visualization state."""

    def __init__(self, node: Node, default_event_cov_xy: float, default_event_cov_z: float):
        self._node = node
        self.default_event_cov_xy = default_event_cov_xy
        self.default_event_cov_z = default_event_cov_z
        self.stored_events: Dict[str, NavigationEvent] = {}
        self.last_published_event_id: Optional[str] = None
        self._junction_attempts: Dict[str, Set[int]] = {}
        w_spectral = float(getattr(node, "descriptor_weight_spectral", 0.7))
        w_spectral = float(np.clip(w_spectral, 0.0, 1.0))
        w_statistical = float(1.0 - w_spectral)
        self._merge_matcher = RadialDescriptorMatcher(
            weight_statistical=w_statistical,
            weight_spectral=w_spectral,
        )
        self._return_viz_state: Dict[str, Any] = {}
        self._last_matched_descriptor = None
        self._last_merge_update_time_s: Dict[str, float] = {}
        self._event_numbers: Dict[str, int] = {}
        self._next_event_number: int = 1

    def reset(self) -> None:
        """Clear all stored events and internal bookkeeping."""
        self.stored_events.clear()
        self.last_published_event_id = None
        self._junction_attempts.clear()
        self._return_viz_state.clear()
        self._last_matched_descriptor = None
        self._last_merge_update_time_s.clear()
        self._event_numbers.clear()
        self._next_event_number = 1

    def _get_event_number(self, event_id: str) -> int:
        """Get or assign a sequential number for an event."""
        if event_id not in self._event_numbers:
            self._event_numbers[event_id] = self._next_event_number
            self._next_event_number += 1
        return self._event_numbers[event_id]

    def _get_short_label(self, event: NavigationEvent) -> str:
        """Get a short label for visualization (e.g., 'J1' for junction 1, 'S' for start)."""
        num = self._get_event_number(event.event_id)
        if event.event_type == EventType.START:
            return "S"
        if event.event_type == EventType.JUNCTION:
            jc = event.junction_config
            if jc and jc.junction_type == JunctionType.DEAD_END:
                return f"D{num}"
            return f"J{num}"
        elif event.event_type == EventType.GEOMETRY:
            return f"G{num}"
        return f"E{num}"

    def publish_event(self, event: NavigationEvent):
        merged_event = self._maybe_merge_event(event)
        if merged_event is None:
            return
        payload = merged_event.to_dict()
        msg = String()
        msg.data = json.dumps(payload)
        self._node.event_pub.publish(msg)

        self.stored_events[merged_event.event_id] = merged_event
        self.last_published_event_id = merged_event.event_id
        self._node._publish_event_stored_notification(merged_event.event_id, merged_event)

    def _maybe_merge_event(self, new_event: NavigationEvent) -> Optional[NavigationEvent]:
        node = self._node
        if new_event.event_type not in (EventType.JUNCTION, EventType.GEOMETRY):
            return new_event
        if getattr(new_event, 'radial_descriptor', None) is None:
            return new_event

        best_candidate: Optional[NavigationEvent] = None
        best_score = -1.0
        closest_mismatch: Optional[Tuple[NavigationEvent, float, float]] = None

        for old in self.stored_events.values():
            if old.event_type != new_event.event_type:
                continue
            if new_event.event_type == EventType.JUNCTION:
                if not old.junction_config or not new_event.junction_config:
                    continue
                jt_old = old.junction_config.junction_type
                jt_new = new_event.junction_config.junction_type
                if (jt_old == JunctionType.DEAD_END) or (jt_new == JunctionType.DEAD_END):
                    if jt_old != jt_new:
                        continue
            else:
                if (old.tunnel_geometry.shape_descriptor or "") != (new_event.tunnel_geometry.shape_descriptor or ""):
                    continue
            if getattr(old, 'radial_descriptor', None) is None:
                continue
            dx = float(new_event.pose.x - old.pose.x)
            dy = float(new_event.pose.y - old.pose.y)
            euclid_m = float(np.hypot(dx, dy))
            # Euclidean cap: Mahalanobis alone permits long-range merges when covariances inflate.
            max_euclid = float(getattr(node, "event_merge_max_euclidean_distance_m", 0.0) or 0.0)
            if max_euclid > 0.0 and euclid_m > max_euclid:
                continue
            d_vec = np.array([dx, dy], dtype=np.float32)
            P_event_xy = self._compute_event_cov_xy_from_descriptor(old.radial_descriptor)
            try:
                Sigma = P_event_xy + np.eye(2, dtype=np.float32) * 1e-6
                m2 = float(d_vec.T @ np.linalg.inv(Sigma) @ d_vec)
            except np.linalg.LinAlgError:
                m2 = float('inf')
            if (
                new_event.event_type == EventType.JUNCTION
                and old.junction_config is not None
                and old.junction_config.selected_path_index < 0
            ):
                if m2 <= node.event_merge_mahal_threshold:
                    self._node.get_logger().info(
                        f"Discarding new JUNCTION event at "
                        f"({new_event.pose.x:.2f}, {new_event.pose.y:.2f}) "
                        f"near untouched junction id={old.event_id} (m2={m2:.2f} <= "
                        f"{node.event_merge_mahal_threshold:.2f})."
                    )
                    return None
                continue
            if m2 > node.event_merge_mahal_threshold:
                continue
            scores = self._merge_matcher.compute_similarity(
                new_event.radial_descriptor,
                old.radial_descriptor,
            )
            combined = float(scores['combined_score'])
            if combined >= float(node.event_merge_score_threshold):
                if combined > best_score:
                    best_score = combined
                    best_candidate = old
            else:
                if closest_mismatch is None or m2 < float(closest_mismatch[1]):
                    closest_mismatch = (old, float(m2), float(combined))

        if best_candidate is None:
            discard_on_mismatch = bool(getattr(node, "event_merge_discard_on_mismatch", True))
            if discard_on_mismatch and closest_mismatch is not None:
                old, m2, score = closest_mismatch
                self._node.get_logger().info(
                    f"Discarding new {new_event.event_type.name} event at "
                    f"({new_event.pose.x:.2f}, {new_event.pose.y:.2f}) near existing id={old.event_id} "
                    f"(descriptor mismatch: score={score:.2f} < {float(node.event_merge_score_threshold):.2f}, "
                    f"m2={m2:.2f} <= {float(node.event_merge_mahal_threshold):.2f})."
                )
                return None
            return new_event

        old = best_candidate

        # Always allow updates that discover new arms; otherwise throttle noisy merge publishes.
        min_update_dt_s = float(getattr(node, "event_merge_min_update_interval_s", 0.0) or 0.0)
        if min_update_dt_s > 0.0:
            allow_bypass = False
            if (
                new_event.event_type == EventType.JUNCTION
                and old.junction_config is not None
                and new_event.junction_config is not None
            ):
                old_deg = len([float(a) for a in (getattr(old.junction_config, "path_angles", []) or [])])
                new_deg = len([float(a) for a in (getattr(new_event.junction_config, "path_angles", []) or [])])
                allow_bypass = new_deg > old_deg

            now_s = float(node.get_clock().now().nanoseconds) * 1e-9
            last_s = float(self._last_merge_update_time_s.get(old.event_id, -1.0))
            if not allow_bypass and last_s >= 0.0 and (now_s - last_s) < min_update_dt_s:
                return None
            self._last_merge_update_time_s[old.event_id] = now_s

        self._blend_event_measurements(old, new_event, best_score)

        if (
            new_event.event_type == EventType.JUNCTION
            and old.junction_config is not None
            and new_event.junction_config is not None
        ):
            if old.junction_config.junction_type != JunctionType.DEAD_END:
                old_paths = [float(a) for a in (getattr(old.junction_config, "path_angles", []) or [])]
                new_paths = [float(a) for a in (getattr(new_event.junction_config, "path_angles", []) or [])]
                old_deg = len(old_paths)
                new_deg = len(new_paths)
                
                if new_deg > old_deg:
                    # Remap selected/entry/dead-end/attempted indices onto the new arm set.
                    old_selected_idx = getattr(old.junction_config, "selected_path_index", -1)
                    old_entry_angle = getattr(old, "entry_angle", None)

                    def find_closest_arm_idx(ref_angle: float, candidates: list) -> int:
                        if not candidates:
                            return -1
                        diffs = [abs(math.atan2(math.sin(float(a) - ref_angle), 
                                                math.cos(float(a) - ref_angle))) 
                                 for a in candidates]
                        return int(np.argmin(diffs))

                    new_selected_idx = -1
                    if isinstance(old_selected_idx, int) and 0 <= old_selected_idx < old_deg:
                        old_selected_angle = float(old_paths[old_selected_idx])
                        new_selected_idx = find_closest_arm_idx(old_selected_angle, new_paths)

                    new_entry_angle = old_entry_angle
                    if old_entry_angle is not None:
                        try:
                            entry_ang_f = float(old_entry_angle)
                            closest_new_idx = find_closest_arm_idx(entry_ang_f, new_paths)
                            if closest_new_idx >= 0:
                                new_entry_angle = float(new_paths[closest_new_idx])
                        except (TypeError, ValueError):
                            pass

                    old_dead_ends = getattr(old.junction_config, "dead_end_paths", []) or []
                    new_dead_ends = []
                    for idx in old_dead_ends:
                        if isinstance(idx, int) and 0 <= idx < old_deg:
                            old_de_angle = float(old_paths[idx])
                            new_idx = find_closest_arm_idx(old_de_angle, new_paths)
                            if new_idx >= 0 and new_idx not in new_dead_ends:
                                new_dead_ends.append(new_idx)

                    old_attempts = self._junction_attempts.get(old.event_id, set())
                    new_attempts = set()
                    for idx in old_attempts:
                        if isinstance(idx, int) and 0 <= idx < old_deg:
                            old_att_angle = float(old_paths[idx])
                            new_idx = find_closest_arm_idx(old_att_angle, new_paths)
                            if new_idx >= 0:
                                new_attempts.add(new_idx)

                    old.junction_config.path_angles = new_paths
                    old.junction_config.junction_type = new_event.junction_config.junction_type
                    old.junction_config.num_paths = new_deg
                    old.junction_config.selected_path_index = new_selected_idx
                    old.junction_config.dead_end_paths = new_dead_ends
                    if new_entry_angle is not None:
                        old.entry_angle = new_entry_angle
                    if new_attempts:
                        self._junction_attempts[old.event_id] = new_attempts

                    new_widths = getattr(new_event.junction_config, "path_widths", None)
                    if new_widths and len(new_widths) == new_deg:
                        old.junction_config.path_widths = [float(w) for w in new_widths]
                else:
                    old.junction_config.num_paths = max(
                        int(old.junction_config.num_paths), int(old_deg)
                    )

        self._node.get_logger().info(
            f"Merged new {new_event.event_type.name} event into existing id={old.event_id} "
            f"at ({old.pose.x:.2f}, {old.pose.y:.2f}) (score={best_score:.2f})"
        )
        return old

    def _blend_event_measurements(
        self,
        old: NavigationEvent,
        new: NavigationEvent,
        combined_score: float,
    ) -> None:
        """Blend pose/uncertainty/descriptor without changing entry_angle/selected_path_index."""
        node = self._node
        thr = float(getattr(node, "event_merge_score_threshold", 0.0))
        alpha_max = float(getattr(node, "event_merge_update_alpha_max", 0.5))
        if alpha_max <= 0.0:
            return

        denom = max(1e-6, 1.0 - thr)
        alpha = alpha_max * max(0.0, (float(combined_score) - thr) / denom)
        alpha = float(np.clip(alpha, 0.0, alpha_max))
        if alpha <= 0.0:
            return

        use_arm_scale = bool(getattr(node, "event_merge_update_use_arm_consistency", True))
        if (
            use_arm_scale
            and old.event_type == EventType.JUNCTION
            and old.junction_config is not None
            and new.junction_config is not None
        ):
            max_deg = float(getattr(node, "event_merge_update_arm_worst_match_max_deg", 60.0))
            deg_base = float(getattr(node, "event_merge_update_deg_penalty_base", 0.7))
            arm_scale = self._compute_arm_consistency_scale(
                getattr(old.junction_config, "path_angles", []) or [],
                getattr(new.junction_config, "path_angles", []) or [],
                worst_match_max_deg=max_deg,
                deg_penalty_base=deg_base,
            )
            alpha *= float(np.clip(arm_scale, 0.0, 1.0))
            if alpha <= 0.0:
                return

        self._blend_pose3d_inplace(old.pose, new.pose, alpha)

        # Blend covariance but never shrink diagonals (keep the more conservative uncertainty).
        try:
            P_old = np.asarray(old.pose_uncertainty, dtype=np.float32).reshape(3, 3)
            P_new = np.asarray(new.pose_uncertainty, dtype=np.float32).reshape(3, 3)
            P_blend = (1.0 - float(alpha)) * P_old + float(alpha) * P_new
            diag = np.maximum(np.diag(P_old), np.diag(P_new))
            P_blend[0, 0] = diag[0]
            P_blend[1, 1] = diag[1]
            P_blend[2, 2] = diag[2]
            old.pose_uncertainty = P_blend
        except (TypeError, ValueError):
            old.pose_uncertainty = new.pose_uncertainty

        try:
            old.tunnel_geometry.width = float(
                (1.0 - alpha) * float(old.tunnel_geometry.width) + alpha * float(new.tunnel_geometry.width)
            )
            old.tunnel_geometry.height = float(
                (1.0 - alpha) * float(old.tunnel_geometry.height) + alpha * float(new.tunnel_geometry.height)
            )
        except (TypeError, ValueError):
            old.tunnel_geometry = new.tunnel_geometry

        if getattr(old, "radial_descriptor", None) is not None and getattr(new, "radial_descriptor", None) is not None:
            self._blend_radial_descriptor_inplace(old.radial_descriptor, new.radial_descriptor, alpha)
        else:
            old.radial_descriptor = new.radial_descriptor

        old_conf = getattr(old, "confidence", None)
        new_conf = getattr(new, "confidence", None)
        if isinstance(old_conf, (int, float)) and isinstance(new_conf, (int, float)):
            setattr(old, "confidence", float((1.0 - alpha) * float(old_conf) + alpha * float(new_conf)))

    @staticmethod
    def _blend_pose3d_inplace(old_pose, new_pose, alpha: float) -> None:
        old_pose.x = float((1.0 - alpha) * float(old_pose.x) + alpha * float(new_pose.x))
        old_pose.y = float((1.0 - alpha) * float(old_pose.y) + alpha * float(new_pose.y))
        old_pose.z = float((1.0 - alpha) * float(old_pose.z) + alpha * float(new_pose.z))
        try:
            dyaw = math.atan2(
                math.sin(float(new_pose.yaw) - float(old_pose.yaw)),
                math.cos(float(new_pose.yaw) - float(old_pose.yaw)),
            )
            old_pose.yaw = float(old_pose.yaw + alpha * dyaw)
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _blend_radial_descriptor_inplace(old_desc, new_desc, alpha: float) -> None:
        old_desc.mean_radius = float((1.0 - alpha) * float(old_desc.mean_radius) + alpha * float(new_desc.mean_radius))
        old_desc.std_radius = float((1.0 - alpha) * float(old_desc.std_radius) + alpha * float(new_desc.std_radius))
        old_desc.radius_range = float((1.0 - alpha) * float(old_desc.radius_range) + alpha * float(new_desc.radius_range))
        a = np.asarray(getattr(old_desc, "fourier_magnitudes", []), dtype=np.float32).ravel()
        b = np.asarray(getattr(new_desc, "fourier_magnitudes", []), dtype=np.float32).ravel()
        if a.size > 0 or b.size > 0:
            n = int(max(a.size, b.size))
            if a.size != n:
                a = np.pad(a, (0, n - int(a.size)))
            if b.size != n:
                b = np.pad(b, (0, n - int(b.size)))
            old_desc.fourier_magnitudes = ((1.0 - alpha) * a + alpha * b).astype(np.float32)
        old_desc.circularity = float(old_desc.std_radius / max(old_desc.mean_radius, 1e-6))

        for name in ("angles", "raw_distances", "filtered_distances", "confidence"):
            a_arr = getattr(old_desc, name, None)
            b_arr = getattr(new_desc, name, None)
            if isinstance(a_arr, np.ndarray) and isinstance(b_arr, np.ndarray) and a_arr.shape == b_arr.shape and a_arr.size > 0:
                setattr(old_desc, name, ((1.0 - alpha) * a_arr + alpha * b_arr).astype(np.float32))

    @staticmethod
    def _compute_arm_consistency_scale(
        old_angles: List[float],
        new_angles: List[float],
        worst_match_max_deg: float,
        deg_penalty_base: float,
    ) -> float:
        """Scale [0,1] from symmetric worst-match distance between two arm-angle sets."""
        max_deg = float(max(worst_match_max_deg, 1e-3))
        base = float(np.clip(deg_penalty_base, 0.05, 1.0))

        old = np.asarray(old_angles, dtype=np.float32).ravel()
        new = np.asarray(new_angles, dtype=np.float32).ravel()
        deg_delta = abs(int(old.size) - int(new.size))
        deg_scale = float(base ** deg_delta) if deg_delta > 0 else 1.0

        if old.size == 0 or new.size == 0:
            return float(np.clip(deg_scale, 0.0, 1.0))

        old_mat = old[:, None]
        new_mat = new[None, :]
        d = np.abs(np.arctan2(np.sin(old_mat - new_mat), np.cos(old_mat - new_mat)))
        min_old = np.min(d, axis=1)
        min_new = np.min(d, axis=0)
        worst_rad = float(max(float(np.max(min_old)), float(np.max(min_new))))
        worst_deg = float(np.degrees(worst_rad))
        angle_scale = 1.0 - (worst_deg / max_deg)
        angle_scale = float(np.clip(angle_scale, 0.0, 1.0))
        return float(np.clip(angle_scale * deg_scale, 0.0, 1.0))

    def _compute_event_cov_xy_from_descriptor(self, desc) -> np.ndarray:
        radius_range = float(getattr(desc, 'radius_range', 0.0))
        std_radius = float(getattr(desc, 'std_radius', 0.0))

        # Extent is for merge/RViz/correction covariance, not a global XY match gate.
        sigma_geom = max(self.default_event_cov_xy, std_radius, 0.25 * radius_range)
        s2 = float(sigma_geom * sigma_geom)
        return np.eye(2, dtype=np.float32) * s2

    @staticmethod
    def _gate_ring(
        *,
        event: NavigationEvent,
        now: Any,
        marker_id: int,
        namespace: str,
        radius_m: float,
        color: Tuple[float, float, float],
        alpha: float,
        z_offset_m: float,
    ) -> Marker:
        ring = Marker()
        ring.header.frame_id = "odom"
        ring.header.stamp = now
        ring.ns = namespace
        ring.id = marker_id
        ring.type = Marker.LINE_STRIP
        ring.action = Marker.ADD
        ring.pose.orientation.w = 1.0
        ring.scale.x = 0.045
        ring.color.r, ring.color.g, ring.color.b = color
        ring.color.a = alpha
        radius = max(0.1, float(radius_m))
        for index in range(65):
            angle = 2.0 * math.pi * float(index) / 64.0
            point = Point()
            point.x = float(event.pose.x + radius * math.cos(angle))
            point.y = float(event.pose.y + radius * math.sin(angle))
            point.z = float(event.pose.z + z_offset_m)
            ring.points.append(point)
        return ring

    def publish_event_markers(self):
        node = self._node
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        if not self.stored_events:
            node.event_marker_pub.publish(ma)
            return

        now = node.get_clock().now().to_msg()
        marker_id = 0
        COLOR_START = (0.37, 0.65, 0.93)
        COLOR_T_JUNCTION = (0.13, 0.82, 0.82)
        COLOR_CROSS = (0.98, 0.65, 0.0)
        COLOR_DEAD_END = (0.94, 0.27, 0.27)
        COLOR_GEOMETRY = (0.56, 0.47, 0.96)
        COLOR_SELECTED = (0.13, 0.87, 0.47)
        COLOR_ENTRY = (0.3, 0.3, 0.3)
        COLOR_DEAD_ARM = (0.94, 0.27, 0.27)
        COLOR_NEUTRAL = (0.55, 0.55, 0.55)
        
        for event in self.stored_events.values():
            short_label = self._get_short_label(event)
            
            m_sphere = Marker()
            m_sphere.header.frame_id = "odom"
            m_sphere.header.stamp = now
            m_sphere.ns = "events"
            m_sphere.id = marker_id
            marker_id += 1
            m_sphere.type = Marker.SPHERE
            m_sphere.action = Marker.ADD
            m_sphere.pose.position.x = event.pose.x
            m_sphere.pose.position.y = event.pose.y
            m_sphere.pose.position.z = event.pose.z
            scale = 0.3
            m_sphere.scale.x = m_sphere.scale.y = m_sphere.scale.z = scale

            is_start_event = (event.event_type == EventType.START)
            if is_start_event:
                m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_START
                m_sphere.scale.x = m_sphere.scale.y = m_sphere.scale.z = 0.4
            elif event.event_type == EventType.JUNCTION:
                if event.junction_config:
                    jt = event.junction_config.junction_type
                    if jt == JunctionType.T_JUNCTION:
                        m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_T_JUNCTION
                    elif jt == JunctionType.CROSS_JUNCTION:
                        m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_CROSS
                    elif jt == JunctionType.DEAD_END:
                        m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_DEAD_END
                    else:
                        m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_T_JUNCTION
                else:
                    m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_T_JUNCTION
            elif event.event_type == EventType.GEOMETRY:
                m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_GEOMETRY
            else:
                m_sphere.color.r, m_sphere.color.g, m_sphere.color.b = COLOR_NEUTRAL
            m_sphere.color.a = 1.0
            ma.markers.append(m_sphere)

            if event.event_type == EventType.JUNCTION and event.junction_config:
                length = 1.0
                if event.junction_config.path_angles:
                    line = Marker()
                    line.header.frame_id = "odom"
                    line.header.stamp = now
                    line.ns = "junction_shape"
                    line.id = marker_id
                    marker_id += 1
                    line.type = Marker.LINE_LIST
                    line.action = Marker.ADD
                    line.scale.x = 0.05
                    line.color.r, line.color.g, line.color.b = COLOR_NEUTRAL
                    line.color.a = 0.8
                    for ang_world in event.junction_config.path_angles:
                        ang = float(ang_world)
                        ex = float(event.pose.x + length * np.cos(ang))
                        ey = float(event.pose.y + length * np.sin(ang))
                        p0 = Point(); p0.x = event.pose.x; p0.y = event.pose.y; p0.z = event.pose.z
                        p1 = Point(); p1.x = ex; p1.y = ey; p1.z = event.pose.z
                        line.points.append(p0); line.points.append(p1)
                    ma.markers.append(line)

                    sel_idx = getattr(event.junction_config, 'selected_path_index', -1)
                    if isinstance(sel_idx, int) and 0 <= sel_idx < len(event.junction_config.path_angles):
                        sel = Marker()
                        sel.header.frame_id = "odom"
                        sel.header.stamp = now
                        sel.ns = "junction_selected"
                        sel.id = marker_id
                        marker_id += 1
                        sel.type = Marker.LINE_LIST
                        sel.action = Marker.ADD
                        sel.scale.x = 0.1
                        sel.color.r, sel.color.g, sel.color.b = COLOR_SELECTED
                        sel.color.a = 1.0
                        ang = float(event.junction_config.path_angles[sel_idx])
                        ex = float(event.pose.x + length * np.cos(ang))
                        ey = float(event.pose.y + length * np.sin(ang))
                        p0 = Point(); p0.x = event.pose.x; p0.y = event.pose.y; p0.z = event.pose.z + 0.01
                        p1 = Point(); p1.x = ex; p1.y = ey; p1.z = event.pose.z + 0.01
                        sel.points.append(p0); sel.points.append(p1)
                        ma.markers.append(sel)

                    if getattr(event.junction_config, 'dead_end_paths', None):
                        dead = Marker()
                        dead.header.frame_id = "odom"
                        dead.header.stamp = now
                        dead.ns = "junction_dead_ends"
                        dead.id = marker_id
                        marker_id += 1
                        dead.type = Marker.LINE_LIST
                        dead.action = Marker.ADD
                        dead.scale.x = 0.08
                        dead.color.r, dead.color.g, dead.color.b = COLOR_DEAD_ARM
                        dead.color.a = 1.0
                        for idx in event.junction_config.dead_end_paths:
                            if 0 <= idx < len(event.junction_config.path_angles):
                                ang = float(event.junction_config.path_angles[idx])
                                ex = float(event.pose.x + length * np.cos(ang))
                                ey = float(event.pose.y + length * np.sin(ang))
                                p0 = Point(); p0.x = event.pose.x; p0.y = event.pose.y; p0.z = event.pose.z + 0.02
                                p1 = Point(); p1.x = ex; p1.y = ey; p1.z = event.pose.z + 0.02
                                dead.points.append(p0); dead.points.append(p1)
                        if dead.points:
                            ma.markers.append(dead)

                entry_angle = getattr(event, 'entry_angle', None)
                if (
                    node.viz_entry_from
                    and entry_angle is not None
                    and event.junction_config.path_angles
                    and event.junction_config.junction_type != JunctionType.DEAD_END
                ):
                    entry_ang = float(entry_angle)
                    diffs = []
                    for ang in event.junction_config.path_angles:
                        d = math.atan2(
                            math.sin(float(ang) - entry_ang),
                            math.cos(float(ang) - entry_ang),
                        )
                        diffs.append(abs(d))
                    entry_idx = int(np.argmin(diffs))
                    ent = Marker()
                    ent.header.frame_id = "odom"
                    ent.header.stamp = now
                    ent.ns = "junction_entry_arm"
                    ent.id = marker_id
                    marker_id += 1
                    ent.type = Marker.LINE_LIST
                    ent.action = Marker.ADD
                    ent.scale.x = 0.07
                    ent.color.r, ent.color.g, ent.color.b = COLOR_ENTRY
                    ent.color.a = 1.0
                    ang = float(event.junction_config.path_angles[entry_idx])
                    ex = float(event.pose.x + length * np.cos(ang))
                    ey = float(event.pose.y + length * np.sin(ang))
                    z_lift = float(event.pose.z + 0.03)
                    p0 = Point(); p0.x = event.pose.x; p0.y = event.pose.y; p0.z = z_lift
                    p1 = Point(); p1.x = ex; p1.y = ey; p1.z = z_lift
                    ent.points.append(p0); ent.points.append(p1)
                    ma.markers.append(ent)

            if (
                node.viz_event_uncertainty
                and hasattr(event, 'pose_uncertainty')
                and event.pose_uncertainty is not None
            ):
                if not (
                    event.event_type == EventType.JUNCTION
                    and event.junction_config is not None
                    and event.junction_config.junction_type == JunctionType.DEAD_END
                ):
                    cov = np.asarray(event.pose_uncertainty, dtype=np.float32)
                    sx = float(np.sqrt(abs(cov[0, 0])) * 2.0)
                    sy = float(np.sqrt(abs(cov[1, 1])) * 2.0)
                    unc = Marker()
                    unc.header.frame_id = "odom"
                    unc.header.stamp = now
                    unc.ns = "event_uncertainty"
                    unc.id = marker_id
                    marker_id += 1
                    unc.type = Marker.CYLINDER
                    unc.action = Marker.ADD
                    unc.pose.position.x = event.pose.x
                    unc.pose.position.y = event.pose.y
                    unc.pose.position.z = event.pose.z - 0.02
                    unc.scale.x = max(0.2, sx)
                    unc.scale.y = max(0.2, sy)
                    unc.scale.z = 0.03
                    unc.color.r = 0.22
                    unc.color.g = 0.53
                    unc.color.b = 0.87
                    unc.color.a = 0.25
                    ma.markers.append(unc)

            m_text = Marker()
            m_text.header.frame_id = "odom"
            m_text.header.stamp = now
            m_text.ns = "event_labels"
            m_text.id = marker_id
            marker_id += 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = event.pose.x
            m_text.pose.position.y = event.pose.y
            m_text.pose.position.z = event.pose.z + 0.4
            m_text.scale.z = 0.25
            m_text.color.r = m_text.color.g = m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = short_label
            ma.markers.append(m_text)

            if (
                self._return_viz_state.get('target_event_id') == event.event_id
                and self._return_viz_state.get('status') == 'returning'
            ):
                search_active = bool(
                    self._return_viz_state.get('search_active', False)
                )
                inner_radius = self._return_viz_state.get(
                    'gate_inner_radius_m'
                )
                outer_radius = self._return_viz_state.get(
                    'gate_outer_radius_m'
                )
                try:
                    inner_radius_f = float(inner_radius)
                    outer_radius_f = float(outer_radius)
                except (TypeError, ValueError):
                    inner_radius_f = outer_radius_f = 0.0
                if inner_radius_f > 0.0:
                    ma.markers.append(self._gate_ring(
                        event=event,
                        now=now,
                        marker_id=marker_id,
                        namespace='return_gate_inner',
                        radius_m=inner_radius_f,
                        color=(0.25, 0.65, 1.0),
                        alpha=0.9,
                        z_offset_m=0.04,
                    ))
                    marker_id += 1
                if outer_radius_f > 0.0:
                    outer_color = (
                        (0.1, 0.9, 0.35)
                        if search_active
                        else (0.97, 0.75, 0.09)
                    )
                    ma.markers.append(self._gate_ring(
                        event=event,
                        now=now,
                        marker_id=marker_id,
                        namespace='return_gate_outer',
                        radius_m=outer_radius_f,
                        color=outer_color,
                        alpha=0.95,
                        z_offset_m=0.05,
                    ))
                    marker_id += 1

        self._node.event_marker_pub.publish(ma)

    def mark_dead_end_path(self, event_id: str) -> None:
        """Mark the selected arm as a dead-end so later exploration avoids it."""
        event = self.stored_events.get(event_id)
        if event is None or event.junction_config is None:
            return
        sel_idx = getattr(event.junction_config, 'selected_path_index', -1)
        if not isinstance(sel_idx, int) or sel_idx < 0:
            return
        if sel_idx not in event.junction_config.dead_end_paths:
            event.junction_config.dead_end_paths.append(int(sel_idx))
        attempts = self._junction_attempts.setdefault(event_id, set())
        attempts.add(int(sel_idx))

    def match_found_callback(self, msg: String):
        data = json.loads(msg.data)

        event_id = data.get('event_id')
        if not isinstance(event_id, str):
            return
        event = self.stored_events.get(event_id)
        if event is None or getattr(event, 'radial_descriptor', None) is None:
            return
        self._last_matched_descriptor = event.radial_descriptor

    def return_viz_callback(self, msg: String) -> None:
        """Cache return context for handover logic and RViz markers."""
        try:
            data = json.loads(msg.data)
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(data, dict):
            return
        self._return_viz_state = dict(data)
        self.publish_event_markers()

    @property
    def return_viz_state(self) -> Dict[str, Any]:
        return self._return_viz_state

    def update_last_matched_descriptor(self, descriptor):
        self._last_matched_descriptor = descriptor

    @property
    def last_matched_descriptor(self):
        return self._last_matched_descriptor

    def decision_update_callback(self, msg: String):
        data = json.loads(msg.data)
        event_id = data.get('event_id')
        sel_idx = data.get('selected_path_index')
        entry_angle = data.get('entry_angle', None)
        prev_event_id = data.get('previous_event_id', None)
        if not isinstance(event_id, str) or not isinstance(sel_idx, int):
            self._node.get_logger().warn("decision_update_callback: missing event_id or selected_path_index")
            return
        event = self.stored_events.get(event_id)
        if event is None:
            self._node.get_logger().warn(f"decision_update_callback: event_id {event_id} not found")
            return
        if event.junction_config is None or not event.junction_config.path_angles:
            self._node.get_logger().warn(f"decision_update_callback: event {event_id} has no junction_config")
            return
        num_paths = len(event.junction_config.path_angles)
        if sel_idx < 0 or sel_idx >= num_paths:
            self._node.get_logger().warn(
                f"decision_update_callback: index {sel_idx} out of range [0,{num_paths-1}] for event {event_id}"
            )
            return
        attempts = self._junction_attempts.setdefault(event_id, set())
        attempts.add(int(sel_idx))
        event.junction_config.selected_path_index = int(sel_idx)

        # Navigation owns entry_angle/previous_event_id; merge heuristics must not invent them.
        if entry_angle is not None:
            try:
                ea = float(entry_angle)
                if math.isfinite(ea):
                    event.entry_angle = ea
            except (TypeError, ValueError):
                pass
        if isinstance(prev_event_id, str) and prev_event_id:
            event.previous_event_id = str(prev_event_id)

        self._node._publish_event_stored_notification(event_id, event)

        if self._node.debug_viz:
            self.publish_event_markers()

    def dead_end_path_callback(self, msg: String) -> None:
        """Mark the selected arm of a previously visited junction as a dead-end."""
        data = json.loads(msg.data)
        event_id = data.get('event_id')
        if not isinstance(event_id, str):
            self._node.get_logger().warn("dead_end_path_callback missing event_id")
            return
        self.mark_dead_end_path(event_id)
        event = self.stored_events.get(event_id)
        if event is not None:
            self._node._publish_event_stored_notification(event_id, event)
        if self._node.debug_viz:
            self.publish_event_markers()

    def query_by_id_callback(self, msg: String):
        raw = msg.data.strip()
        if not raw:
            return
        event_ids: List[str] = []
        data = json.loads(raw)
        if isinstance(data, dict):
            if isinstance(data.get('event_ids'), list):
                event_ids = [str(eid) for eid in data['event_ids']]
            elif 'event_id' in data:
                event_ids = [str(data['event_id'])]
        elif isinstance(data, list):
            event_ids = [str(eid) for eid in data]
        elif isinstance(data, str):
            event_ids = [data]
        events_payload = []
        for eid in event_ids:
            event = self.stored_events.get(eid)
            if event is not None:
                events_payload.append(event.to_dict())
        if not events_payload:
            return
        self._node._publish_query_response({'query_type': 'by_id', 'events': events_payload})

    def query_nearest_callback(self, msg: String):
        data = json.loads(msg.data)
        if not isinstance(data, dict):
            return
        x = float(data.get('x', 0.0))
        y = float(data.get('y', 0.0))
        z = float(data.get('z', 0.0))

        max_distance = float(data.get('max_distance', 5.0))
        max_results = int(data.get('max_results', 10)) or 10
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for event in self.stored_events.values():
            if event.pose is None:
                continue
            dx = float(event.pose.x - x)
            dy = float(event.pose.y - y)
            dz = float(event.pose.z - z)
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= max_distance:
                payload = event.to_dict()
                payload['distance'] = dist
                candidates.append((dist, payload))
        if not candidates:
            return
        candidates.sort(key=lambda tup: tup[0])
        payload = {
            'query_type': 'nearest',
            'events': [c[1] for c in candidates[:max_results]],
        }
        self._node._publish_query_response(payload)
