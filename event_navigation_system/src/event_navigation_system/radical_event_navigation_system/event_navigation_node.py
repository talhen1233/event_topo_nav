#!/usr/bin/env python3
"""Smart navigation: committed graph intent plus event-based junction decisions."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

import math
import numpy as np
import json
from typing import Optional, List, Dict, Any

from std_msgs.msg import String, Float32, Bool
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

try:  # pragma: no cover - import guard for direct execution
    from .core.approach_history import ApproachHistory
    from .core.event_types import EventType, NavigationMode, JunctionType
    from .core.navigation_event import NavigationEvent, Pose3D
    from .core.route_progress import (
        CorrectionAwareDistanceTracker,
        LocalizationMode,
        RouteProgressAccumulator,
        VarianceRates,
    )
except ImportError:  # pragma: no cover
    from core.approach_history import ApproachHistory
    from core.event_types import EventType, NavigationMode, JunctionType
    from core.navigation_event import NavigationEvent, Pose3D
    from core.route_progress import (
        CorrectionAwareDistanceTracker,
        LocalizationMode,
        RouteProgressAccumulator,
        VarianceRates,
    )


class SmartNavigationNode(Node):
    """Follow the skeleton graph and decide junction arms from event semantics."""
    
    def __init__(self):
        super().__init__('smart_navigation_node')

        self.declare_parameters(
            namespace='',
            parameters=[                
                ('junction_commit_sigma', 3.0),
                ('junction_commit_min_m', 1.0),
                ('junction_commit_exit_hysteresis_m', 0.2),
                ('entry_history_max_distance_m', 3.0),
                ('entry_history_sample_distance_m', 1.5),
                ('entry_history_min_sample_spacing_m', 0.05),
                ('entry_history_max_arm_error_deg', 35.0),
                ('battery_reserve', 55.0),  # force RETURN_TO_BASE below this percentage
                ('max_mission_time', 1800.0),
                ('debug_mode', True),
                ('inertia_penalty_weight', 0.3),  # discourage oscillation between symmetric arms
            ]
        )
        self.declare_parameters(
            namespace='',
            parameters=[
                ('return_gate_sigma', 3.0),
                # Floor on the 1D search half-width even with tiny event sigma and no travel yet.
                ('return_gate_min_m', 1.0),
                # Observability uncertainty along the route, not an XY covariance.
                ('return_event_sigma_m', 0.6),
                ('return_qs_normal_m', 0.005),
                ('return_qt_normal_m2_per_s', 0.00005),
                ('return_qs_degraded_m', 0.025),
                ('return_qt_degraded_m2_per_s', 0.0005),
                ('return_qs_of_degraded_m', 0.1),
                ('return_qt_of_degraded_m2_per_s', 0.005),
                # Samples compare against the last accepted anchor so slow motion still accumulates.
                ('return_progress_min_step_m', 0.03),
                # Treat larger one-sample changes as estimator discontinuities.
                ('return_progress_max_step_m', 0.75),
                # Hold a short stamped window so correction ACK and odometry may arrive in either order.
                ('return_correction_reorder_window_s', 0.1),
                ('of_degraded_topic', '/drone/odometry/of_degraded'),
                ('of_degraded_timeout_s', 0.5),
                ('odom_sigma_degraded_threshold_m', 0.75),
            ]
        )
        
        self.return_gate_sigma = float(self.get_parameter('return_gate_sigma').value)
        self.return_gate_min_m = float(self.get_parameter('return_gate_min_m').value)
        self.return_event_sigma_m = float(self.get_parameter('return_event_sigma_m').value)
        self.return_progress_min_step_m = float(self.get_parameter('return_progress_min_step_m').value)
        self.return_progress_max_step_m = float(self.get_parameter('return_progress_max_step_m').value)
        self.return_correction_reorder_window_s = float(
            self.get_parameter('return_correction_reorder_window_s').value
        )
        self.of_degraded_timeout_s = float(self.get_parameter('of_degraded_timeout_s').value)
        self.odom_sigma_degraded_threshold_m = float(
            self.get_parameter('odom_sigma_degraded_threshold_m').value
        )
        self.junction_commit_sigma = float(self.get_parameter('junction_commit_sigma').value)
        self.junction_commit_min_m = float(self.get_parameter('junction_commit_min_m').value)
        self.junction_commit_exit_hysteresis_m = float(
            self.get_parameter('junction_commit_exit_hysteresis_m').value
        )
        self.entry_history_max_distance_m = float(
            self.get_parameter('entry_history_max_distance_m').value
        )
        self.entry_history_sample_distance_m = float(
            self.get_parameter('entry_history_sample_distance_m').value
        )
        self.entry_history_min_sample_spacing_m = float(
            self.get_parameter('entry_history_min_sample_spacing_m').value
        )
        self.entry_history_max_arm_error_deg = float(
            self.get_parameter('entry_history_max_arm_error_deg').value
        )
        
        self.battery_reserve = float(self.get_parameter('battery_reserve').value)
        self.max_mission_time = float(self.get_parameter('max_mission_time').value)
        
        self.debug_mode = self.get_parameter('debug_mode').value
        
        self.inertia_penalty_weight = float(self.get_parameter('inertia_penalty_weight').value)

        self.return_gate_sigma = float(max(0.0, self.return_gate_sigma))
        self.return_gate_min_m = float(max(0.0, self.return_gate_min_m))
        self.return_event_sigma_m = float(max(1e-3, self.return_event_sigma_m))
        self.return_progress_min_step_m = float(max(0.0, self.return_progress_min_step_m))
        self.return_progress_max_step_m = float(max(0.0, self.return_progress_max_step_m))
        if (
            self.return_progress_max_step_m > 0.0
            and self.return_progress_max_step_m <= self.return_progress_min_step_m
        ):
            self.return_progress_max_step_m = 0.0
        self._distance_tracker = CorrectionAwareDistanceTracker(
            self.return_progress_min_step_m,
            self.return_progress_max_step_m,
            max(0.0, self.return_correction_reorder_window_s),
        )
        self._approach_history = ApproachHistory(
            max_distance_m=self.entry_history_max_distance_m,
            sample_distance_m=self.entry_history_sample_distance_m,
            min_sample_spacing_m=self.entry_history_min_sample_spacing_m,
            max_arm_error_rad=math.radians(
                self.entry_history_max_arm_error_deg
            ),
        )
        self._return_progress = RouteProgressAccumulator({
            LocalizationMode.NORMAL: VarianceRates(
                float(max(0.0, self.get_parameter('return_qs_normal_m').value)),
                float(max(0.0, self.get_parameter('return_qt_normal_m2_per_s').value)),
            ),
            LocalizationMode.DEGRADED: VarianceRates(
                float(max(0.0, self.get_parameter('return_qs_degraded_m').value)),
                float(max(0.0, self.get_parameter('return_qt_degraded_m2_per_s').value)),
            ),
            LocalizationMode.OF_DEGRADED: VarianceRates(
                float(max(0.0, self.get_parameter('return_qs_of_degraded_m').value)),
                float(max(0.0, self.get_parameter('return_qt_of_degraded_m2_per_s').value)),
            ),
        })

        self.qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        # Transient-local so late-joining dashboards receive the last known value.
        self.qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        self.current_mode = NavigationMode.IDLE
        self.mission_active = False
        self._start_ready = False
        self._pending_start_event_id: Optional[str] = None
        self._pending_start_event_time_s = -1.0
        self.mission_start_time = None
        
        self.current_pose: Optional[Pose3D] = None
        self.current_event: Optional[NavigationEvent] = None
        self.last_selected_direction: Optional[float] = None
        
        self.at_junction = False
        self._active_commit_event_id: Optional[str] = None
        self._active_commit_center_xy: Optional[np.ndarray] = None
        self._active_commit_radius_m: Optional[float] = None
        self._active_commit_gate_entered = False
        self.junction_history: List[Dict[str, Any]] = []
        self.returning_to_prev_junction = False
        self.return_target_event_id: Optional[str] = None
        self.prev_junction_event_id: Optional[str] = None
        self.home_event_id: Optional[str] = None
        # Return targets use exploration-route distance differences, never XY deltas.
        self._return_anchor_route_distance_m: Optional[float] = None
        self._return_expected_distance_m: Optional[float] = None
        self._return_handover_from_event_id: Optional[str] = None
        # Localization-anchor distance resets only on mission start or a real matcher confirmation.
        self._distance_since_localization_anchor_m = 0.0
        self._localization_anchor_kind = "none"
        self._localization_anchor_event_id: Optional[str] = None
        self._last_odom_update_time_s: Optional[float] = None
        self._of_degraded = False
        self._of_degraded_update_time_s = -1.0
        self._pending_match_observation: Optional[Dict[str, Any]] = None

        self._last_expected_event_id_str: str = ""
        
        self.battery_level = 100.0
        self.total_distance = 0.0
        self.route_distance = 0.0

        
        self.create_subscription(
            Odometry, '/drone/state_estimate',
            self.odometry_callback, 
            self.qos_sensor)
        self.create_subscription(
            Bool, str(self.get_parameter('of_degraded_topic').value),
            self._of_degraded_callback, self.qos_sensor
        )
        self.create_subscription(
            PoseWithCovarianceStamped, '/localization/pose_correction_applied',
            self._pose_correction_callback, self.qos_reliable
        )
        
        self.create_subscription(
            String, '/navigation_events/detected',
            self.event_detected_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/mission_control/start_event',
            self._start_event_callback, self.qos_latched
        )
        
        self.create_subscription(
            Float32, '/battery_level',
            self.battery_callback, self.qos_sensor
        )
        
        self.create_subscription(
            Bool, '/mission_control/start',
            self.mission_start_callback, self.qos_sensor
        )
        self.create_subscription(
            Bool, '/mission_control/start_ready',
            self._start_ready_callback, self.qos_latched
        )
        
        self.create_subscription(
            Bool, '/mission_control/stop',
            self.mission_stop_callback, self.qos_sensor
        )
        self.create_subscription(
            Bool, '/mission_control/reset',
            self.mission_reset_callback, self.qos_sensor
        )
        
        self.create_subscription(
            String, '/navigation_events/match_found',
            self.match_found_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation_events/query/response',
            self.query_response_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation_events/deleted',
            self.event_deleted_callback, self.qos_sensor
        )
        
        self.graph_target_pub = self.create_publisher(
            String, '/navigation/graph_target', self.qos_sensor
        )
        
        self.mode_pub = self.create_publisher(
            String, '/navigation/mode', self.qos_latched
        )
        self.dead_end_path_pub = self.create_publisher(
            String, '/navigation_events/dead_end_path', self.qos_sensor
        )
        self.expected_event_pub = self.create_publisher(
            String, '/navigation/expected_event', self.qos_latched
        )
        self.return_viz_pub = self.create_publisher(
            String, '/navigation_events/return_viz', self.qos_sensor
        )
        self.mission_stop_pub = self.create_publisher(
            Bool, '/mission_control/stop', self.qos_sensor
        )
        self.navigation_ready_pub = self.create_publisher(
            Bool, '/mission_control/navigation_ready', self.qos_latched
        )
        self._publish_navigation_ready(False)
        self.decision_update_pub = self.create_publisher(
            String, '/navigation_events/decision/update', self.qos_sensor
        )
        
        
        self.query_event_pub = self.create_publisher(
            String, '/navigation_events/query/by_id', self.qos_sensor
        )
        
        # Route-window transitions must run faster than descriptor matching at flight speed.
        self.create_timer(0.2, self.navigation_update)
        
        
        self.get_logger().info("Smart Navigation Node initialized")
        self.attempted_paths_per_event: Dict[str, set] = {}
        self.pending_decision_event_id: Optional[str] = None
        # Pose covariance is used only to classify localization quality.
        self.current_pose_cov: Optional[np.ndarray] = None

    @staticmethod
    def _sigma_xy_from_pose_cov(cov: Any) -> Optional[float]:
        """Planar sigma from a 3x3 pose covariance, or None if unavailable."""
        if cov is None:
            return None
        try:
            cov_arr = np.asarray(cov, dtype=np.float32).reshape(3, 3)
        except (TypeError, ValueError):
            return None
        return float(np.sqrt(max(1e-6, 0.5 * (float(cov_arr[0, 0]) + float(cov_arr[1, 1])))))

    def _junction_duplicate_ignore_radius_m(self, new_event: NavigationEvent) -> float:
        """Ignore nearby duplicate junction ids so drift cannot flip a committed arm."""
        base = 0.75

        sigma = None
        if getattr(self, "current_event", None) is not None:
            sigma = self._sigma_xy_from_pose_cov(getattr(self.current_event, "pose_uncertainty", None))
        if sigma is None:
            sigma = self._sigma_xy_from_pose_cov(getattr(new_event, "pose_uncertainty", None))

        if sigma is None:
            return float(base)

        return float(min(3.0, max(base, 3.0 * float(sigma), 1.5)))

    def _junction_commit_radius(self, event: NavigationEvent) -> float:
        """Spatial radius beyond which the drone has committed to the selected arm."""
        sigma = self._sigma_xy_from_pose_cov(getattr(event, "pose_uncertainty", None))
        if sigma is None:
            return float(self.junction_commit_min_m)
        return float(max(self.junction_commit_min_m, self.junction_commit_sigma * float(sigma)))

    def _junction_commit_radius_from_cov(self, pose_cov: Optional[np.ndarray]) -> float:
        sigma = self._sigma_xy_from_pose_cov(pose_cov)
        if sigma is None:
            return float(self.junction_commit_min_m)
        return float(max(self.junction_commit_min_m, self.junction_commit_sigma * float(sigma)))

    def _find_junction_history_index(self, event_id: str) -> Optional[int]:
        """Return the latest history index of event_id, scanning from the end."""
        for i in range(len(self.junction_history) - 1, -1, -1):
            if self.junction_history[i].get("event_id") == event_id:
                return int(i)
        return None

    def _now_s(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _of_degraded_callback(self, msg: Bool) -> None:
        self._of_degraded = bool(msg.data)
        self._of_degraded_update_time_s = self._now_s()

    def _pose_correction_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """Register the timestamp of a correction accepted by the estimator."""
        stamp_s = (
            float(msg.header.stamp.sec)
            + 1e-9 * float(msg.header.stamp.nanosec)
        )
        if stamp_s > 0.0:
            self._distance_tracker.acknowledge_correction(stamp_s)

    def _localization_mode(self, now_s: Optional[float] = None) -> LocalizationMode:
        """Classify localization quality for route-variance accumulation."""
        now = self._now_s() if now_s is None else float(now_s)
        mode_fresh = (
            self._of_degraded_update_time_s >= 0.0
            and (
                self.of_degraded_timeout_s <= 0.0
                or 0.0 <= now - self._of_degraded_update_time_s <= self.of_degraded_timeout_s
            )
        )
        if not mode_fresh or self._of_degraded:
            return LocalizationMode.OF_DEGRADED

        sigma_xy = self._sigma_xy_from_pose_cov(self.current_pose_cov)
        if (
            sigma_xy is not None
            and sigma_xy >= self.odom_sigma_degraded_threshold_m
        ):
            return LocalizationMode.DEGRADED
        return LocalizationMode.NORMAL
    
    def odometry_callback(self, msg: Odometry):
        now_s = self._now_s()
        dt_s = 0.0
        if self._last_odom_update_time_s is not None:
            dt_s = max(0.0, now_s - self._last_odom_update_time_s)
        self._last_odom_update_time_s = now_s

        pos = msg.pose.pose.position
        new_pose = Pose3D(x=pos.x, y=pos.y, z=pos.z)
        
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        new_pose.yaw = yaw

        try:
            self.current_pose_cov = (
                np.asarray(msg.pose.covariance, dtype=np.float32)
                .reshape(6, 6)[:3, :3]
                .copy()
            )
        except (TypeError, ValueError):
            self.current_pose_cov = None

        # Pose corrections and implausible jumps rebase the distance anchor without adding ds.
        position_xyz = (new_pose.x, new_pose.y, new_pose.z)
        stamp_s = (
            float(msg.header.stamp.sec)
            + 1e-9 * float(msg.header.stamp.nanosec)
        )
        if stamp_s <= 0.0:
            stamp_s = now_s
        distance_update = self._distance_tracker.update(position_xyz, stamp_s)
        dist = float(distance_update.distance_m)

        if distance_update is not None and distance_update.reason == "discontinuity":
            self.get_logger().warn(
                "Ignored odometry discontinuity while accumulating route distance"
            )

        self.total_distance += dist
        if self.mission_active:
            self._distance_since_localization_anchor_m += dist
        if self.mission_active and self.current_mode == NavigationMode.EXPLORE:
            self.route_distance += dist

        if (
            self.returning_to_prev_junction
            and self.current_mode in (NavigationMode.BACKTRACK, NavigationMode.RETURN_TO_BASE)
        ):
            self._return_progress.update(
                dist,
                dt_s,
                self._localization_mode(now_s),
            )
        
        self.current_pose = new_pose
        self._approach_history.update(
            np.array([new_pose.x, new_pose.y], dtype=np.float32)
        )

    def event_deleted_callback(self, msg: String):
        """Remove deleted events from junction_history and retarget return if needed."""
        data = json.loads(msg.data)
        event_id = data.get('event_id')
        if not isinstance(event_id, str):
            return
        
        deleted_index = self._find_junction_history_index(event_id)
        predecessor = None
        if deleted_index is not None and deleted_index > 0:
            predecessor = self.junction_history[deleted_index - 1]

        original_len = len(self.junction_history)
        self.junction_history = [
            entry for entry in self.junction_history 
            if entry.get('event_id') != event_id
        ]
        removed = original_len - len(self.junction_history)
        
        if event_id in self.attempted_paths_per_event:
            del self.attempted_paths_per_event[event_id]
        
        if self.return_target_event_id == event_id:
            self._return_handover_from_event_id = None
            if self.debug_mode:
                self.get_logger().warn(
                    f"Return target {event_id} was deleted; finding alternative."
                )
            if predecessor is not None:
                # Prefer the deleted target's predecessor, not the newest history entry.
                new_target = predecessor
                self.return_target_event_id = new_target.get('event_id')
                self.prev_junction_event_id = self.return_target_event_id
                self._start_return_search_for_current_target(reset_anchor=False)
                self._update_return_progress_and_maybe_skip()
            else:
                self.returning_to_prev_junction = self.home_event_id is not None
                self.return_target_event_id = self.home_event_id
                self._start_return_search_for_current_target(reset_anchor=False)
        
        if self.prev_junction_event_id == event_id:
            if predecessor is not None:
                self.prev_junction_event_id = predecessor.get('event_id')
            else:
                self.prev_junction_event_id = None
        
        if removed > 0 and self.debug_mode:
            self.get_logger().info(
                f"Removed deleted event {event_id} from junction_history "
                f"(reason: {data.get('reason', 'unknown')})"
            )
    
    def event_detected_callback(self, msg: String):
        event_dict = json.loads(msg.data)
        event = NavigationEvent.from_dict(event_dict)

        if event.event_type == EventType.START:
            # Accept HOME even if mission_active is not set yet; topic order can race /start.
            if self.home_event_id is None or self.mission_active:
                self.home_event_id = str(event.event_id)
                if self.debug_mode:
                    self.get_logger().info(f"Captured HOME START event id={self.home_event_id}.")
            self._activate_exploration_if_start_ready()
            return
        
        if event.event_type == EventType.JUNCTION and event.junction_config:
            jt = event.junction_config.junction_type
            if jt != JunctionType.DEAD_END:
                if self.current_mode == NavigationMode.EXPLORE:
                    # Ignore new ids at the same place so merge failures cannot flip a committed arm.
                    if self.at_junction and self.current_event is not None:
                        if (
                            isinstance(getattr(event, "event_id", None), str)
                            and isinstance(getattr(self.current_event, "event_id", None), str)
                            and event.event_id != self.current_event.event_id
                            and getattr(event, "pose", None) is not None
                            and getattr(self.current_event, "pose", None) is not None
                        ):
                            try:
                                dx = float(event.pose.x) - float(self.current_event.pose.x)
                                dy = float(event.pose.y) - float(self.current_event.pose.y)
                            except (TypeError, ValueError, AttributeError):
                                dx = None
                                dy = None
                            if dx is not None and dy is not None:
                                dist = float(np.hypot(dx, dy))
                                dup_r = float(self._junction_duplicate_ignore_radius_m(event))
                                if dist <= dup_r:
                                    if self.debug_mode:
                                        self.get_logger().info(
                                            f"Ignoring near-duplicate junction id={event.event_id} "
                                            f"(~{dist:.2f}m <= {dup_r:.2f}m from current id={self.current_event.event_id}) "
                                            "while in junction context."
                                        )
                                    return
                    if self._is_junction_update_event(event):
                        self._apply_junction_update(event)
                        return
                    # Known ids after leaving junction context are updates, never a new decision.
                    if self._find_junction_history_index(str(event.event_id)) is not None:
                        self._apply_junction_update(event)
                        return
                    self._apply_authoritative_semantics_for_new_junction(event)
                    self._record_junction_event(event)
                    if self.debug_mode:
                        self.get_logger().info(
                            f"Updated previous junction: {jt.name} id={event.event_id}"
                        )
                    self.handle_junction_event(event)
                elif self.current_mode in (NavigationMode.BACKTRACK, NavigationMode.RETURN_TO_BASE):
                    # Return is driven by stored history + targeted matching, not new detections.
                    return
            else:
                self._handoff_dead_end_reverse()
                if self.prev_junction_event_id is not None:
                    if self.current_mode != NavigationMode.BACKTRACK:
                        self.set_mode(NavigationMode.BACKTRACK)
                    self.returning_to_prev_junction = True
                    self.return_target_event_id = self.prev_junction_event_id
                    self._return_handover_from_event_id = None
                    self._start_return_search_for_current_target(reset_anchor=True)
                    dead_msg = String()
                    dead_msg.data = json.dumps({'event_id': self.prev_junction_event_id})
                    self.dead_end_path_pub.publish(dead_msg)
                    self._update_return_progress_and_maybe_skip()
                    if self.debug_mode:
                        self.get_logger().info(f"Dead end → returning to event {self.return_target_event_id}")

    def _is_junction_update_event(self, event: NavigationEvent) -> bool:
        """True if this is a merge-update of the junction already being handled."""
        if self.current_mode != NavigationMode.EXPLORE:
            return False
        if not self.at_junction:
            return False
        if not isinstance(getattr(event, "event_id", None), str):
            return False

        if self.current_event is not None and getattr(self.current_event, "event_id", None) == event.event_id:
            return True

        if self.junction_history:
            last_id = self.junction_history[-1].get("event_id")
            if last_id == event.event_id:
                return True

        return False

    def _apply_junction_update(self, event: NavigationEvent) -> None:
        """Refresh measurement fields without changing entry_angle or selected_path_index."""
        event_id = str(getattr(event, "event_id", ""))
        hist_idx = self._find_junction_history_index(event_id) if event_id else None
        hist_entry: Optional[Dict[str, Any]] = None
        if hist_idx is not None and 0 <= hist_idx < len(self.junction_history):
            hist_entry = self.junction_history[hist_idx]

        old_angles: List[float] = []
        if (
            self.current_event is not None
            and getattr(self.current_event, "event_id", None) == getattr(event, "event_id", None)
            and self.current_event.junction_config is not None
        ):
            old_angles = [float(a) for a in (self.current_event.junction_config.path_angles or [])]
        elif hist_entry is not None:
            try:
                old_angles = [float(a) for a in (hist_entry.get("path_angles") or [])]
            except (TypeError, ValueError):
                old_angles = []
        new_angles: List[float] = []
        if event.junction_config is not None:
            new_angles = [float(a) for a in (event.junction_config.path_angles or [])]

        # Navigation decisions stay local; the event system must not overwrite them.
        authoritative_entry_angle: Optional[float] = None
        authoritative_selected_idx: Optional[int] = None
        authoritative_previous_event_id: Optional[str] = None

        if hist_entry is not None:
            h_entry = hist_entry.get("entry_angle", None)
            if h_entry is not None:
                try:
                    authoritative_entry_angle = float(h_entry)
                except (TypeError, ValueError):
                    pass
            h_prev = hist_entry.get("previous_event_id", None)
            if isinstance(h_prev, str) and h_prev:
                authoritative_previous_event_id = h_prev
            h_sel = hist_entry.get("selected_path_index", None)
            if isinstance(h_sel, int) and h_sel >= 0:
                authoritative_selected_idx = int(h_sel)

        if self.current_event is not None and getattr(self.current_event, "event_id", None) == getattr(event, "event_id", None):
            if authoritative_entry_angle is None:
                entry = getattr(self.current_event, "entry_angle", None)
                if entry is not None:
                    try:
                        authoritative_entry_angle = float(entry)
                    except (TypeError, ValueError):
                        pass
            if self.current_event.junction_config is not None:
                sel = getattr(self.current_event.junction_config, "selected_path_index", -1)
                if isinstance(sel, int) and sel >= 0:
                    authoritative_selected_idx = int(sel)

        if old_angles and new_angles and old_angles != new_angles:
            attempted = self.attempted_paths_per_event.get(event_id)
            if attempted:
                mapped: set[int] = set()
                for idx in attempted:
                    if not isinstance(idx, int) or idx < 0 or idx >= len(old_angles):
                        continue
                    target = float(old_angles[idx])
                    diffs = [abs(self._angle_diff(float(a), target)) for a in new_angles]
                    if diffs:
                        mapped.add(int(np.argmin(np.array(diffs, dtype=np.float32))))
                self.attempted_paths_per_event[event_id] = mapped
            
            if authoritative_selected_idx is not None and 0 <= authoritative_selected_idx < len(old_angles):
                old_selected_angle = float(old_angles[authoritative_selected_idx])
                diffs = [abs(self._angle_diff(float(a), old_selected_angle)) for a in new_angles]
                if diffs:
                    authoritative_selected_idx = int(np.argmin(np.array(diffs, dtype=np.float32)))

        if authoritative_entry_angle is not None:
            event.entry_angle = float(authoritative_entry_angle)
        if authoritative_previous_event_id is not None:
            event.previous_event_id = str(authoritative_previous_event_id)
        if event.junction_config is not None and authoritative_selected_idx is not None:
            event.junction_config.selected_path_index = int(authoritative_selected_idx)

        if hist_entry is not None:
            entry = hist_entry
            entry["pose"] = event.pose
            entry["cov"] = event.pose_uncertainty
            entry["confidence"] = float(getattr(event, "confidence", entry.get("confidence", 1.0)))
            entry["descriptor"] = getattr(event, "radial_descriptor", entry.get("descriptor", None))
            if getattr(event, "entry_angle", None) is not None:
                entry["entry_angle"] = float(getattr(event, "entry_angle"))
            if (
                event.junction_config is not None
                and isinstance(getattr(event.junction_config, "selected_path_index", None), int)
                and int(getattr(event.junction_config, "selected_path_index")) >= 0
            ):
                entry["selected_path_index"] = int(getattr(event.junction_config, "selected_path_index"))
            existing_prev = entry.get("previous_event_id", None)
            if not (isinstance(existing_prev, str) and existing_prev):
                new_prev = getattr(event, "previous_event_id", None)
                if isinstance(new_prev, str) and new_prev:
                    entry["previous_event_id"] = str(new_prev)
            jc = event.junction_config
            if jc is not None:
                entry["path_angles"] = [float(a) for a in (jc.path_angles or [])]
                entry["path_widths"] = (
                    [float(w) for w in (jc.path_widths or [])]
                    if getattr(jc, "path_widths", None)
                    else []
                )
                entry["dead_end_paths"] = (
                    [int(i) for i in (jc.dead_end_paths or [])]
                    if getattr(jc, "dead_end_paths", None)
                    else []
                )

        # Do not clobber an in-progress junction with a different event id.
        if self.current_event is None or getattr(self.current_event, "event_id", None) == getattr(event, "event_id", None):
            self.current_event = event

        if self.debug_mode:
            self.get_logger().info(
                f"Junction update received for id={event.event_id}; "
                "updated local state without changing the current goal/decision."
            )

    def _apply_authoritative_semantics_for_new_junction(self, event: NavigationEvent) -> None:
        """Stamp entry_angle and previous_event_id from local history and approach path."""
        if event.junction_config is None or not event.junction_config.path_angles:
            return

        if self.junction_history:
            prev = self.junction_history[-1].get("event_id")
            if isinstance(prev, str) and prev:
                event.previous_event_id = str(prev)

        arms = [float(a) for a in (event.junction_config.path_angles or [])]
        if not arms:
            return

        if event.pose is not None:
            historical_entry = self._approach_history.estimate_entry_arm(
                np.array([event.pose.x, event.pose.y], dtype=np.float32),
                arms,
            )
            if historical_entry is not None:
                event.entry_angle = historical_entry
                return

        # Bounded fallback for startup or discontinuous odometry.
        yaw = None
        if self.current_pose is not None:
            yaw = float(self.current_pose.yaw)
        elif event.pose is not None:
            try:
                yaw = float(getattr(event.pose, "yaw", 0.0))
            except (TypeError, ValueError):
                yaw = None
        if yaw is None:
            return
        desired = float(yaw + math.pi)
        desired = float((desired + math.pi) % (2.0 * math.pi) - math.pi)
        diffs = [abs(self._angle_diff(float(a), desired)) for a in arms]
        entry_idx = int(np.argmin(np.asarray(diffs, dtype=np.float32)))
        if diffs[entry_idx] > math.radians(
            max(0.0, self.entry_history_max_arm_error_deg)
        ):
            return
        event.entry_angle = float(arms[entry_idx])
    
    def match_found_callback(self, msg: String):
        """Handle matcher confirmation while BACKTRACK / RETURN_TO_BASE."""
        data = json.loads(msg.data)
        matched_event_id = data.get('event_id', None)
        if (
            not isinstance(matched_event_id, str)
            or not self.returning_to_prev_junction
            or matched_event_id != self.return_target_event_id
        ):
            return

        # Only a confirmed match after the id/target guards may reset the localization-anchor distance.
        self._distance_since_localization_anchor_m = 0.0
        self._localization_anchor_kind = "match"
        self._localization_anchor_event_id = matched_event_id

        if self.current_mode == NavigationMode.BACKTRACK:
            rv = {
                'target_event_id': self.return_target_event_id,
                'status': 'success',
            }
            rv_msg = String(); rv_msg.data = json.dumps(rv)
            self.return_viz_pub.publish(rv_msg)
            self._publish_expected_event_id(None)
            self.returning_to_prev_junction = False
            self.return_target_event_id = None
            self._return_handover_from_event_id = None
            matched_entry = self._find_junction_entry(matched_event_id)
            if matched_entry is not None:
                self.route_distance = float(matched_entry.get('route_distance_m', 0.0))
                matched_index = self._find_junction_history_index(matched_event_id)
                if matched_index is not None:
                    # Abandoned-branch events stay in the repository but leave the active return route.
                    self.junction_history = self.junction_history[:matched_index + 1]
                self.prev_junction_event_id = matched_event_id
            self._return_anchor_route_distance_m = None
            self._return_expected_distance_m = None
            self._return_progress.reset()
            self._clear_graph_intent()
            self.set_mode(NavigationMode.EXPLORE)
            if self.debug_mode:
                self.get_logger().info("Return to previous junction confirmed by matcher")
            observation = data.get('observed_junction')
            self._pending_match_observation = (
                observation if isinstance(observation, dict) else None
            )
            self.pending_decision_event_id = matched_event_id
            self.query_event_by_id(matched_event_id)
            return

        if self.current_mode == NavigationMode.RETURN_TO_BASE:
            self._handle_return_to_base_match(matched_event_id, data)
            return
    
    def query_response_callback(self, msg: String):
        """Handle event details response and choose new arm when pending."""
        data = json.loads(msg.data)
        if not isinstance(data, dict):
            return
        if data.get('query_type') != 'by_id' or not isinstance(data.get('events'), list):
            return
        events = data.get('events', [])
        if not events or self.pending_decision_event_id is None:
            return
        requested_id = str(self.pending_decision_event_id)
        for ev in events:
            if str(ev.get('event_id', '')) != requested_id:
                continue
            if ev.get('event_type') != 'JUNCTION' or ev.get('junction_config') is None:
                self.pending_decision_event_id = None
                return
            self._handle_reidentified_junction(ev)
            self.pending_decision_event_id = None
            self._pending_match_observation = None
            return

    def _handle_reidentified_junction(self, event_dict: Dict[str, Any]) -> None:
        """Choose a new arm after matcher success and publish the committed graph intent."""
        try:
            event = NavigationEvent.from_dict(event_dict)
        except (KeyError, TypeError, ValueError) as exc:
            if self.debug_mode:
                self.get_logger().warn(f"Failed to parse re-identified junction: {exc}")
            return

        if event.junction_config is None or not event.junction_config.path_angles:
            return

        observation = self._pending_match_observation or {}
        observed_angles = observation.get('path_angles', [])
        if not isinstance(observed_angles, list):
            observed_angles = []
        stored_angles = [float(a) for a in event.junction_config.path_angles]
        selected_idx = int(event.junction_config.selected_path_index)
        event.junction_config.path_angles = self._align_stored_angles_to_current(
            stored_angles,
            selected_idx,
            observed_angles,
        )
        center = observation.get('center_xy')
        if isinstance(center, list) and len(center) >= 2:
            event.pose.x = float(center[0])
            event.pose.y = float(center[1])
        elif self.current_pose is not None:
            event.pose.x = float(self.current_pose.x)
            event.pose.y = float(self.current_pose.y)

        self.at_junction = True
        self.current_event = event

        entry_idx, forbidden = self._reidentified_arm_constraints(event_dict)
        unexplored_exits = [
            arm_idx
            for arm_idx in range(len(event.junction_config.path_angles))
            if arm_idx not in forbidden
        ]
        if (
            entry_idx >= 0
            and not unexplored_exits
            and self._backtrack_from_exhausted_junction(event, observation)
        ):
            return

        idx = int(self._select_next_arm_after_reid(
            event_dict,
            path_angles_override=event.junction_config.path_angles,
        ))
        if idx < 0 or idx >= len(event.junction_config.path_angles):
            idx = 0

        event.junction_config.selected_path_index = int(idx)
        hist_idx = self._find_junction_history_index(str(event.event_id))
        if hist_idx is not None:
            self.junction_history[hist_idx]["selected_path_index"] = int(idx)

        upd = {'event_id': str(event.event_id), 'selected_path_index': int(idx)}
        if getattr(event, "entry_angle", None) is not None:
            upd['entry_angle'] = float(getattr(event, "entry_angle"))
        if isinstance(getattr(event, "previous_event_id", None), str) and getattr(event, "previous_event_id"):
            upd['previous_event_id'] = str(getattr(event, "previous_event_id"))
        upd_msg = String()
        upd_msg.data = json.dumps(upd)
        self.decision_update_pub.publish(upd_msg)

        tried = self.attempted_paths_per_event.setdefault(str(event.event_id), set())
        tried.add(int(idx))

        arm_angle = float(event.junction_config.path_angles[idx])
        junction_xy = np.array([event.pose.x, event.pose.y], dtype=np.float32)
        self._publish_committed_graph_intent(
            event_id=str(event.event_id),
            event_center_xy=junction_xy,
            gate_radius_m=self._junction_commit_radius(event),
            selected_arm_angle_rad=arm_angle,
            mode="explore",
        )

    @staticmethod
    def _select_preferred_arm(
        scores: np.ndarray,
        forbidden: set[int],
        entry_idx: int,
    ) -> int:
        """Choose an unexplored arm, retreating through entry when exhausted."""
        candidates = [
            idx for idx in range(int(scores.size))
            if idx not in forbidden
        ]
        if candidates:
            return int(max(candidates, key=lambda idx: scores[idx]))
        if 0 <= entry_idx < int(scores.size):
            return int(entry_idx)
        return int(np.argmax(scores)) if scores.size else 0

    def _reidentified_arm_constraints(
        self,
        event_dict: Dict[str, Any],
    ) -> tuple[int, set[int]]:
        """Return the entry index and all arms unavailable for exploration."""
        junction = event_dict.get("junction_config", {})
        if not isinstance(junction, dict):
            return -1, set()

        path_angles = [float(angle) for angle in junction.get("path_angles", [])]
        entry_idx = -1
        entry_angle = event_dict.get("entry_angle")
        if entry_angle is not None and path_angles:
            try:
                entry_angle_f = float(entry_angle)
            except (TypeError, ValueError):
                entry_angle_f = None
            if entry_angle_f is not None:
                differences = [
                    abs(self._angle_diff(angle, entry_angle_f))
                    for angle in path_angles
                ]
                entry_idx = int(np.argmin(np.asarray(differences, dtype=np.float32)))

        event_id = str(event_dict.get("event_id", ""))
        forbidden = set(self.attempted_paths_per_event.get(event_id, set()))
        forbidden.update(
            int(index)
            for index in junction.get("dead_end_paths", [])
            if isinstance(index, (int, float))
        )
        if entry_idx >= 0:
            forbidden.add(entry_idx)
        return entry_idx, forbidden

    def _backtrack_from_exhausted_junction(
        self,
        event: NavigationEvent,
        observation: Dict[str, Any],
    ) -> bool:
        """Continue through the entry arm when no unexplored exits remain."""
        event_id = str(event.event_id)
        current_entry = self._find_junction_entry(event_id)
        if current_entry is None:
            return False

        previous_entry = self._find_previous_junction_before(event_id)
        previous_id = (
            previous_entry.get("event_id")
            if previous_entry is not None
            else self.home_event_id
        )
        if not isinstance(previous_id, str) or not previous_id:
            return False

        return_mode = (
            NavigationMode.RETURN_TO_BASE
            if previous_id == self.home_event_id
            else NavigationMode.BACKTRACK
        )
        intent_mode = "return" if return_mode == NavigationMode.RETURN_TO_BASE else "backtrack"
        if not self._publish_entry_arm_commit(
            current_entry,
            mode=intent_mode,
            observation=observation,
        ):
            return False

        self.set_mode(return_mode)
        self.returning_to_prev_junction = True
        self.return_target_event_id = previous_id
        self._return_handover_from_event_id = None
        self._start_return_search_for_current_target(reset_anchor=True)
        self._update_return_progress_and_maybe_skip()
        if self.debug_mode:
            self.get_logger().info(
                "Junction exits exhausted; backtracking through entry arm "
                f"from {event_id} toward {previous_id}"
            )
        return True
    
    def _select_next_arm_after_reid(
        self,
        event_dict: Dict[str, Any],
        *,
        path_angles_override: Optional[List[float]] = None,
    ) -> int:
        """Choose the next unexplored arm after re-identification, avoiding entry and dead-ends."""
        jc = event_dict.get('junction_config', {})
        stored_path_angles = list(jc.get('path_angles', [])) if isinstance(jc, dict) else []
        if path_angles_override is not None:
            path_angles = [float(a) for a in path_angles_override]
        elif isinstance(jc, dict):
            path_angles = list(stored_path_angles)
        else:
            path_angles = []
        if not path_angles:
            return 0

        entry_idx, forbidden = self._reidentified_arm_constraints(event_dict)

        scores = np.zeros(len(path_angles), dtype=np.float32)
        if self.current_pose is not None:
            heading = np.array(
                [np.cos(self.current_pose.yaw), np.sin(self.current_pose.yaw)],
                dtype=np.float32,
            )
            for i, ang in enumerate(path_angles):
                d = np.array([np.cos(float(ang)), np.sin(float(ang))], dtype=np.float32)
                scores[i] = float(np.dot(heading, d))
        if self.last_selected_direction is not None:
            for i, ang in enumerate(path_angles):
                dtheta = self._angle_diff(float(ang), self.last_selected_direction)
                scores[i] -= float(self.inertia_penalty_weight) * abs(dtheta)

        return self._select_preferred_arm(scores, forbidden, entry_idx)
    
    def battery_callback(self, msg: Float32):
        self.battery_level = msg.data
    
    def mission_start_callback(self, msg: Bool):
        if msg.data and not self.mission_active:
            self.start_mission()

    def _start_ready_callback(self, msg: Bool) -> None:
        """Release exploration only after the event system stored stable HOME."""
        self._start_ready = bool(msg.data)
        self._activate_exploration_if_start_ready()

    def _start_event_callback(self, msg: String) -> None:
        """Receive the reliable stable-START record as one atomic payload."""
        try:
            event = NavigationEvent.from_dict(json.loads(msg.data))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        if event.event_type != EventType.START:
            return
        self._pending_start_event_id = str(event.event_id)
        self._pending_start_event_time_s = self._now_s()
        if self.mission_active:
            self.home_event_id = self._pending_start_event_id
            self._start_ready = True
            self._activate_exploration_if_start_ready()

    def _publish_navigation_ready(self, ready: bool) -> None:
        msg = Bool()
        msg.data = bool(ready)
        self.navigation_ready_pub.publish(msg)

    def _activate_exploration_if_start_ready(self) -> None:
        if not self.mission_active or not self._start_ready:
            return
        if self.home_event_id is None:
            return
        if self.current_mode == NavigationMode.WAIT_START:
            self.set_mode(NavigationMode.EXPLORE)
            self._publish_navigation_ready(True)
            self.get_logger().info(
                "Stable START captured - Smart exploration active"
            )
    
    def mission_stop_callback(self, msg: Bool):
        if msg.data and self.mission_active:
            self.stop_mission()

    def mission_reset_callback(self, msg: Bool):
        if bool(getattr(msg, "data", False)):
            self._hard_reset()

    def _hard_reset(self) -> None:
        self._publish_mission_stop()
        self._publish_navigation_ready(False)
        self._clear_graph_intent()
        self._publish_expected_event_id(None)

        self.mission_active = False
        self._start_ready = False
        self.mission_start_time = None
        self.current_pose = None
        self.current_event = None
        self.last_selected_direction = None

        self.at_junction = False
        self._active_commit_event_id = None
        self._active_commit_center_xy = None
        self._active_commit_radius_m = None
        self._active_commit_gate_entered = False
        self.junction_history.clear()

        self.returning_to_prev_junction = False
        self.return_target_event_id = None
        self.prev_junction_event_id = None

        self.home_event_id = None
        self._pending_start_event_id = None
        self._pending_start_event_time_s = -1.0

        self._return_anchor_route_distance_m = None
        self._return_expected_distance_m = None
        self._return_handover_from_event_id = None
        self._return_progress.reset()
        self._distance_since_localization_anchor_m = 0.0
        self._localization_anchor_kind = "none"
        self._localization_anchor_event_id = None
        self._distance_tracker.reset()
        self._approach_history.reset()
        self._last_odom_update_time_s = None
        self._pending_match_observation = None

        self.total_distance = 0.0
        self.route_distance = 0.0
        self.battery_level = 100.0

        self.attempted_paths_per_event.clear()
        self.pending_decision_event_id = None

        self.set_mode(NavigationMode.IDLE)
        rv_msg = String()
        rv_msg.data = json.dumps({"target_event_id": "", "status": ""})
        self.return_viz_pub.publish(rv_msg)

        self.get_logger().info("Mission hard reset: cleared navigation state")

    def _publish_mission_stop(self) -> None:
        stop_msg = Bool()
        stop_msg.data = True
        self.mission_stop_pub.publish(stop_msg)
    
    def start_mission(self):
        self.mission_active = True
        now_s = self._now_s()
        pending_is_current = (
            self._pending_start_event_id is not None
            and 0.0 <= now_s - self._pending_start_event_time_s <= 1.0
        )
        self._start_ready = bool(pending_is_current)
        self.mission_start_time = self.get_clock().now()
        self.total_distance = 0.0
        self.route_distance = 0.0
        self._approach_history.reset()
        if self.current_pose is not None:
            self._approach_history.update(
                np.array(
                    [self.current_pose.x, self.current_pose.y],
                    dtype=np.float32,
                )
            )
        self.junction_history.clear()
        self.returning_to_prev_junction = False
        self.home_event_id = (
            self._pending_start_event_id if pending_is_current else None
        )
        self._publish_navigation_ready(False)
        self.return_target_event_id = None
        self._return_anchor_route_distance_m = None
        self._return_expected_distance_m = None
        self._return_handover_from_event_id = None
        self._return_progress.reset()
        # Mission-start pose is the bootstrap localization anchor; later resets require a real match.
        self._distance_since_localization_anchor_m = 0.0
        self._localization_anchor_kind = "mission_start"
        self._localization_anchor_event_id = None
        if self.current_pose is not None:
            self._distance_tracker.rebase(
                (self.current_pose.x, self.current_pose.y, self.current_pose.z)
            )
        self._publish_expected_event_id(None)
        
        self.set_mode(NavigationMode.WAIT_START)
        self._activate_exploration_if_start_ready()

        self.get_logger().info("Mission requested - waiting for stable START")
    
    def stop_mission(self):
        self.mission_active = False
        self._start_ready = False
        self._pending_start_event_id = None
        self._pending_start_event_time_s = -1.0
        self._publish_mission_stop()
        self._publish_navigation_ready(False)
        self.set_mode(NavigationMode.IDLE)
        self._clear_graph_intent()
        self._publish_expected_event_id(None)
        self._return_anchor_route_distance_m = None
        self._return_expected_distance_m = None
        self._return_handover_from_event_id = None
        self._return_progress.reset()
        self._distance_since_localization_anchor_m = 0.0
        self._localization_anchor_kind = "none"
        self._localization_anchor_event_id = None
        self.get_logger().info(f"Mission stopped - Total distance traveled: {self.total_distance:.1f} m")
    
    def navigation_update(self):
        if not self.mission_active or not self.current_pose:
            return

        if self.current_mode == NavigationMode.WAIT_START:
            return

        # Distant detections must enter the gate; only a later exit releases the committed arm.
        if self.at_junction and self._active_commit_center_xy is not None and self._active_commit_radius_m is not None:
            dx = float(self.current_pose.x) - float(self._active_commit_center_xy[0])
            dy = float(self.current_pose.y) - float(self._active_commit_center_xy[1])
            dist = float(np.hypot(dx, dy))
            radius_m = float(self._active_commit_radius_m)
            if dist <= radius_m:
                self._active_commit_gate_entered = True
            exit_radius_m = radius_m + max(
                0.0,
                self.junction_commit_exit_hysteresis_m,
            )
            if (
                self._active_commit_gate_entered
                and dist > exit_radius_m
            ):
                self._exit_junction_context()

        if self.current_mode not in (NavigationMode.BACKTRACK, NavigationMode.RETURN_TO_BASE):
            if self.should_return():
                self._initiate_return_to_base(reason="constraints")
        
        if self.returning_to_prev_junction and self.return_target_event_id is not None:
            if self._update_return_progress_and_maybe_skip():
                return

    def _publish_expected_event_id(self, event_id: Optional[str]) -> None:
        """Publish the current return target event id (empty string clears)."""
        data = str(event_id) if event_id else ""
        if data == getattr(self, "_last_expected_event_id_str", ""):
            return
        self._last_expected_event_id_str = data
        msg = String()
        msg.data = data
        self.expected_event_pub.publish(msg)

    def _initiate_return_to_base(self, reason: str) -> None:
        """Enter RETURN_TO_BASE and hand steering to graph intent."""
        if self.current_mode == NavigationMode.RETURN_TO_BASE:
            return
        if self.current_pose is None:
            return

        self.set_mode(NavigationMode.RETURN_TO_BASE)
        self._return_handover_from_event_id = None
        if self.at_junction:
            self._exit_junction_context()
        self._send_reverse_command(mode="return")

        if self.junction_history:
            last_id_val = self.junction_history[-1].get("event_id")
            last_id = last_id_val if isinstance(last_id_val, str) and last_id_val else None
        else:
            last_id = None

        if last_id is not None:
            self.returning_to_prev_junction = True
            self.return_target_event_id = last_id
            self._start_return_search_for_current_target(reset_anchor=True)
            self._update_return_progress_and_maybe_skip()
            if self.debug_mode:
                self.get_logger().warn(
                    f"RETURN_TO_BASE initiated ({reason}); targeting last junction {self.return_target_event_id}."
                )
        else:
            if self.debug_mode:
                self.get_logger().warn(
                    f"RETURN_TO_BASE initiated ({reason}) without junction history; targeting HOME directly."
                )
            self.returning_to_prev_junction = self.home_event_id is not None
            self.return_target_event_id = self.home_event_id
            self._start_return_search_for_current_target(reset_anchor=True)

    def _handle_return_to_base_match(
        self,
        matched_event_id: str,
        match_data: Dict[str, Any],
    ) -> None:
        """Advance RETURN_TO_BASE one step after matcher confirms the current target."""
        if self.home_event_id is not None and matched_event_id == self.home_event_id:
            if self.debug_mode:
                self.get_logger().info(
                    f"HOME reached (START event matched) id={matched_event_id}. Stopping mission."
                )
            self.returning_to_prev_junction = False
            self.return_target_event_id = None
            self._return_handover_from_event_id = None
            self._publish_expected_event_id(None)
            self._return_anchor_route_distance_m = None
            self._return_expected_distance_m = None
            self._return_progress.reset()
            rv = {"target_event_id": matched_event_id, "status": "success"}
            rv_msg = String()
            rv_msg.data = json.dumps(rv)
            self.return_viz_pub.publish(rv_msg)
            self.stop_mission()
            return

        curr_entry = self._find_junction_entry(matched_event_id)
        observation = match_data.get('observed_junction')
        if not isinstance(observation, dict):
            observation = None
        if curr_entry is None or not self._publish_entry_arm_commit(
            curr_entry,
            mode="return",
            observation=observation,
        ):
            self.get_logger().error(
                f"RETURN_TO_BASE cannot continue from junction {matched_event_id}: missing entry-arm metadata."
            )
            rv_msg = String()
            rv_msg.data = json.dumps({
                "target_event_id": matched_event_id,
                "status": "transition_error",
            })
            self.return_viz_pub.publish(rv_msg)
            return

        # Mark reached only after the next route command can be formed.
        rv_msg = String()
        rv_msg.data = json.dumps({
            "target_event_id": matched_event_id,
            "status": "success",
        })
        self.return_viz_pub.publish(rv_msg)

        prev_entry = self._find_previous_junction_before(matched_event_id)
        next_id = None
        if prev_entry is not None:
            next_id_val = prev_entry.get("event_id")
            next_id = next_id_val if isinstance(next_id_val, str) and next_id_val else None
        elif self.home_event_id is not None:
            next_id = str(self.home_event_id)

        if next_id is None:
            self.get_logger().error(
                f"RETURN_TO_BASE reached {matched_event_id} but found no older target and no HOME event."
            )
            self.returning_to_prev_junction = False
            self.return_target_event_id = None
            self._publish_expected_event_id(None)
            return

        self._return_anchor_route_distance_m = float(
            curr_entry.get('route_distance_m', 0.0)
        )
        self.route_distance = self._return_anchor_route_distance_m
        self._return_handover_from_event_id = str(matched_event_id)
        self.return_target_event_id = next_id
        self.returning_to_prev_junction = True
        self._start_return_search_for_current_target(reset_anchor=True)
        self._update_return_progress_and_maybe_skip()
        if self.debug_mode:
            self.get_logger().info(f"RETURN_TO_BASE: advancing target to {self.return_target_event_id}")

    def should_return(self) -> bool:
        if self.mission_start_time:
            elapsed = (self.get_clock().now() - self.mission_start_time).nanoseconds / 1e9
            if elapsed > self.max_mission_time:
                return True
        
        if self.battery_level < self.battery_reserve:
            self.get_logger().warn(
                f"Battery low: {self.battery_level:.1f}% < reserve {self.battery_reserve:.1f}%"
            )
            return True
        
        return False
    def _publish_committed_graph_intent(
        self,
        *,
        event_id: str,
        event_center_xy: np.ndarray,
        gate_radius_m: float,
        selected_arm_angle_rad: float,
        mode: str,
    ) -> None:
        """Publish authoritative committed arm intent to the local planner."""
        payload = {
            "kind": "commit",
            "event_id": str(event_id),
            "mode": str(mode),
            "event_center_xy": [float(event_center_xy[0]), float(event_center_xy[1])],
            "selected_arm_angle_rad": float(selected_arm_angle_rad),
            "gate_radius_m": float(max(0.0, gate_radius_m)),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.graph_target_pub.publish(msg)

        self.at_junction = True
        self._active_commit_event_id = str(event_id)
        self._active_commit_center_xy = np.array(event_center_xy[:2], dtype=np.float32)
        self._active_commit_radius_m = float(max(0.0, gate_radius_m))
        if self.current_pose is None:
            self._active_commit_gate_entered = False
        else:
            current_xy = np.array(
                [self.current_pose.x, self.current_pose.y],
                dtype=np.float32,
            )
            self._active_commit_gate_entered = bool(
                np.linalg.norm(current_xy - self._active_commit_center_xy)
                <= self._active_commit_radius_m
            )
        self.last_selected_direction = float(selected_arm_angle_rad)

        if self.debug_mode:
            self.get_logger().info(
                "Committed graph intent: "
                f"id={event_id} mode={mode} arm={math.degrees(selected_arm_angle_rad):.1f}deg "
                f"gate={gate_radius_m:.2f}m"
            )

    def _clear_graph_intent(self) -> None:
        msg = String()
        msg.data = json.dumps({"kind": "clear"})
        self.graph_target_pub.publish(msg)
        self._active_commit_event_id = None
        self._active_commit_center_xy = None
        self._active_commit_radius_m = None
        self._active_commit_gate_entered = False

    def _send_reverse_command(self, *, mode: Optional[str] = None) -> None:
        """Command the local planner to reverse along the graph (dead-end / return)."""
        payload: Dict[str, Any] = {"kind": "reverse"}
        if isinstance(mode, str) and mode:
            payload["mode"] = mode
        msg = String()
        msg.data = json.dumps(payload)
        self.graph_target_pub.publish(msg)
        if self.debug_mode:
            self.get_logger().info("Sent reverse command to local planner")

    def _handoff_dead_end_reverse(self) -> None:
        """Clear junction intent before issuing the authoritative reverse."""
        if self.at_junction:
            self._exit_junction_context()
        self._send_reverse_command(mode="backtrack")

    def handle_junction_event(self, event: NavigationEvent):
        """Decide a junction arm and publish committed graph intent."""
        if event.junction_config is None or event.junction_config.num_paths <= 0:
            return
        self.at_junction = True
        self.current_event = event
        arm_idx = self._select_junction_arm(event)
        if arm_idx is None or arm_idx < 0 or arm_idx >= len(event.junction_config.path_angles):
            arm_idx = 0

        event.junction_config.selected_path_index = int(arm_idx)
        if self.junction_history and self.junction_history[-1].get("event_id") == event.event_id:
            self.junction_history[-1]["selected_path_index"] = int(arm_idx)

        tried = self.attempted_paths_per_event.setdefault(str(event.event_id), set())
        tried.add(int(arm_idx))

        arm_angle = float(event.junction_config.path_angles[arm_idx])
        junction_xy = np.array([event.pose.x, event.pose.y], dtype=np.float32)
        self._publish_committed_graph_intent(
            event_id=str(event.event_id),
            event_center_xy=junction_xy,
            gate_radius_m=self._junction_commit_radius(event),
            selected_arm_angle_rad=arm_angle,
            mode="explore",
        )

        upd = {'event_id': event.event_id, 'selected_path_index': int(arm_idx)}
        if getattr(event, "entry_angle", None) is not None:
            upd['entry_angle'] = float(getattr(event, "entry_angle"))
        if isinstance(getattr(event, "previous_event_id", None), str) and getattr(event, "previous_event_id"):
            upd['previous_event_id'] = str(getattr(event, "previous_event_id"))
        upd_msg = String(); upd_msg.data = json.dumps(upd)
        self.decision_update_pub.publish(upd_msg)
    
    def _exit_junction_context(self):
        """Leave committed-intent context and return planner control to speculation."""
        self.at_junction = False
        self.current_event = None
        self._clear_graph_intent()
    
    def _select_junction_arm(self, event: NavigationEvent) -> int:
        """Select arm index using current heading and inertia; path_angles are world-frame."""
        path_angles = event.junction_config.path_angles
        if not path_angles:
            return 0
        arm_abs_angles: List[float] = [float(ang) for ang in path_angles]
        arm_dirs = [
            np.array([np.cos(abs_ang), np.sin(abs_ang)], dtype=np.float32)
            for abs_ang in arm_abs_angles
        ]
        scores = np.zeros(len(arm_dirs), dtype=np.float32)
        if self.current_pose is not None:
            heading = np.array(
                [np.cos(float(self.current_pose.yaw)), np.sin(float(self.current_pose.yaw))],
                dtype=np.float32,
            )
            for i, d in enumerate(arm_dirs):
                scores[i] = float(np.dot(heading, d))
        else:
            scores[:] = 0.0
        if self.last_selected_direction is not None:
            for i, abs_ang in enumerate(arm_abs_angles):
                dtheta = self._angle_diff(abs_ang, self.last_selected_direction)
                scores[i] -= float(self.inertia_penalty_weight) * abs(dtheta)

        forbidden: set[int] = set()
        entry_idx = -1

        entry_angle = getattr(event, "entry_angle", None)
        if entry_angle is not None:
            try:
                entry_angle_f = float(entry_angle)
            except (TypeError, ValueError):
                entry_angle_f = None
            if entry_angle_f is not None:
                diffs = [abs(self._angle_diff(abs_ang, entry_angle_f)) for abs_ang in arm_abs_angles]
                entry_idx = int(np.argmin(diffs))
                forbidden.add(entry_idx)

        attempted = self.attempted_paths_per_event.get(str(event.event_id), set())
        for idx in attempted:
            if isinstance(idx, int) and 0 <= idx < len(arm_abs_angles):
                forbidden.add(idx)

        if event.junction_config and getattr(event.junction_config, "dead_end_paths", None):
            for idx in event.junction_config.dead_end_paths:
                if isinstance(idx, int) and 0 <= idx < len(arm_abs_angles):
                    forbidden.add(idx)

        return self._select_preferred_arm(scores, forbidden, entry_idx)
    
    def _record_junction_event(self, event: NavigationEvent) -> None:
        """Append a junction event to the local history and update previous-junction state."""
        entry_conf = float(getattr(event, "confidence", 1.0))
        jc = event.junction_config
        path_angles = [float(a) for a in (jc.path_angles or [])] if jc is not None else []
        path_widths = (
            [float(w) for w in (jc.path_widths or [])]
            if (jc is not None and getattr(jc, "path_widths", None))
            else []
        )
        dead_end_paths = (
            [int(i) for i in (jc.dead_end_paths or [])]
            if (jc is not None and getattr(jc, "dead_end_paths", None))
            else []
        )
        entry = {
            "event_id": event.event_id,
            "pose": event.pose,
            "cov": event.pose_uncertainty,
            "confidence": entry_conf,
            # Route distance resets when a dead-end branch is backtracked; total_distance does not.
            "route_distance_m": float(self.route_distance),
            "descriptor": getattr(event, "radial_descriptor", None),
            "entry_angle": getattr(event, "entry_angle", None),
            "previous_event_id": getattr(event, "previous_event_id", None),
            "selected_path_index": int(getattr(jc, "selected_path_index", -1)) if jc is not None else -1,
            "path_angles": path_angles,
            "path_widths": path_widths,
            "dead_end_paths": dead_end_paths,
        }
        self.junction_history.append(entry)
        self.prev_junction_event_id = entry["event_id"]

    def _target_route_distance_m(self, event_id: str) -> Optional[float]:
        if self.home_event_id is not None and event_id == self.home_event_id:
            return 0.0
        entry = self._find_junction_entry(event_id)
        if entry is None:
            return None
        try:
            return float(entry['route_distance_m'])
        except (KeyError, TypeError, ValueError):
            return None

    def _start_return_search_for_current_target(self, *, reset_anchor: bool) -> None:
        """Set the target's expected distance from the current route anchor."""
        self._publish_expected_event_id(None)
        if self.return_target_event_id is None:
            self._return_expected_distance_m = None
            return

        if reset_anchor or self._return_anchor_route_distance_m is None:
            self._return_anchor_route_distance_m = float(self.route_distance)
            self._return_progress.reset()

        target_distance = self._target_route_distance_m(self.return_target_event_id)
        if target_distance is None:
            self._return_expected_distance_m = None
            self.get_logger().error(
                f"Return target {self.return_target_event_id} has no route distance"
            )
            return
        self._return_expected_distance_m = max(
            0.0,
            float(self._return_anchor_route_distance_m) - target_distance,
        )

    def _update_return_progress_and_maybe_skip(self) -> bool:
        """Open matching only inside the target's 1D route-distance window."""
        if (
            self.return_target_event_id is None
            or self._return_expected_distance_m is None
        ):
            self._publish_expected_event_id(None)
            return False

        window = self._return_progress.window(
            expected_distance_m=self._return_expected_distance_m,
            event_sigma_m=self.return_event_sigma_m,
            sigma_multiplier=self.return_gate_sigma,
            min_half_width_m=self.return_gate_min_m,
        )
        progress = float(self._return_progress.progress_m)
        search_active = window.contains(progress)
        is_home = (
            self.home_event_id is not None
            and self.return_target_event_id == self.home_event_id
        )

        # HOME is terminal: keep searching past the upper bound instead of dropping the only target.
        if is_home and progress >= window.lower_m:
            search_active = True

        self._publish_return_progress_update(
            status="returning",
            search_active=search_active,
            window_start_m=window.lower_m,
            window_end_m=window.upper_m,
            sigma_m=window.sigma_m,
        )
        # Publish handover context before opening the matcher on its first active cycle.
        self._publish_expected_event_id(
            self.return_target_event_id if search_active else None
        )

        if progress > window.upper_m and not is_home:
            self._publish_expected_event_id(None)
            self._skip_return_target(reason="passed_route_window")
            return True
        return False

    def _skip_return_target(self, reason: str) -> None:
        """Skip the current return target and fall back to an older event."""
        if self.return_target_event_id is None or self.current_pose is None:
            return

        cur_id = str(self.return_target_event_id)
        self._return_handover_from_event_id = None
        prev_entry = self._find_previous_junction_before(cur_id)
        if prev_entry is None:
            if self.debug_mode:
                self.get_logger().warn(
                    f"Skip return target {cur_id} ({reason}): no earlier junction to fall back to."
                )
            if self.current_mode == NavigationMode.BACKTRACK:
                self.set_mode(NavigationMode.RETURN_TO_BASE)
                self.return_target_event_id = self.home_event_id
                self.returning_to_prev_junction = self.home_event_id is not None
                self._start_return_search_for_current_target(reset_anchor=False)
            elif self.current_mode == NavigationMode.RETURN_TO_BASE:
                self.return_target_event_id = self.home_event_id
                self.returning_to_prev_junction = self.home_event_id is not None
                self._start_return_search_for_current_target(reset_anchor=False)
            return

        next_id_val = prev_entry.get("event_id")
        next_id = str(next_id_val) if isinstance(next_id_val, str) and next_id_val else None
        if next_id is None:
            return

        # Keep the same route anchor when a target is missed; the next target is farther along.
        self.return_target_event_id = next_id
        self.returning_to_prev_junction = True
        self._start_return_search_for_current_target(reset_anchor=False)

        if self.debug_mode:
            self.get_logger().info(f"Skipped return target {cur_id} -> {next_id} (reason: {reason})")
    
    def _find_previous_junction_before(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Return the history entry that precedes event_id, or None."""
        prev: Optional[Dict[str, Any]] = None
        for entry in self.junction_history:
            if entry.get("event_id") == event_id:
                return prev
            prev = entry
        return None

    def _find_junction_entry(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Return the history entry for the given event id, or None if missing."""
        for entry in self.junction_history:
            if entry.get("event_id") == event_id:
                return entry
        return None

    def _align_stored_angles_to_current(
        self,
        stored_angles: List[float],
        selected_path_index: int,
        observed_angles: List[float],
    ) -> List[float]:
        """Rotate stored arm semantics into the current local observation frame."""
        if not stored_angles or self.current_pose is None:
            return [float(a) for a in stored_angles]
        if 0 <= int(selected_path_index) < len(stored_angles):
            stored_reference = float(stored_angles[int(selected_path_index)])
        else:
            stored_reference = float(stored_angles[0])

        approach_guess = float(self.current_pose.yaw + math.pi)
        current_reference = approach_guess
        clean_observed: List[float] = []
        try:
            clean_observed = [float(a) for a in observed_angles]
        except (TypeError, ValueError):
            clean_observed = []
        if clean_observed:
            current_reference = min(
                clean_observed,
                key=lambda angle: abs(self._angle_diff(angle, approach_guess)),
            )
        offset = self._angle_diff(current_reference, stored_reference)
        return [
            float(math.atan2(math.sin(float(a) + offset), math.cos(float(a) + offset)))
            for a in stored_angles
        ]

    def _publish_entry_arm_commit(
        self,
        entry: Dict[str, Any],
        *,
        mode: str,
        observation: Optional[Dict[str, Any]],
    ) -> bool:
        """Follow the stored entry-arm semantic in the current junction frame."""
        event_id = entry.get("event_id")
        entry_angle = entry.get("entry_angle")
        path_angles = entry.get("path_angles")
        if not isinstance(event_id, str) or entry_angle is None or not isinstance(path_angles, list):
            return False

        try:
            entry_angle_f = float(entry_angle)
            angles = [float(a) for a in path_angles]
        except (TypeError, ValueError):
            return False
        if not angles:
            return False

        selected_idx = int(entry.get('selected_path_index', -1))
        observed_angles = []
        if isinstance(observation, dict) and isinstance(observation.get('path_angles'), list):
            observed_angles = observation['path_angles']
        aligned_angles = self._align_stored_angles_to_current(
            angles,
            selected_idx,
            observed_angles,
        )
        diffs = [abs(self._angle_diff(angle, entry_angle_f)) for angle in angles]
        arm_idx = int(np.argmin(np.asarray(diffs, dtype=np.float32)))
        arm_angle = aligned_angles[arm_idx]
        center = observation.get('center_xy') if isinstance(observation, dict) else None
        if isinstance(center, list) and len(center) >= 2:
            center_xy = np.array([float(center[0]), float(center[1])], dtype=np.float32)
        elif self.current_pose is not None:
            center_xy = np.array(
                [float(self.current_pose.x), float(self.current_pose.y)],
                dtype=np.float32,
            )
        else:
            return False
        gate_radius_m = self._junction_commit_radius_from_cov(self.current_pose_cov)
        self._publish_committed_graph_intent(
            event_id=event_id,
            event_center_xy=center_xy,
            gate_radius_m=gate_radius_m,
            selected_arm_angle_rad=arm_angle,
            mode=mode,
        )
        return True
    
    def _angle_diff(self, a: float, b: float) -> float:
        """Smallest signed angle difference a-b in radians."""
        return float(math.atan2(math.sin(a - b), math.cos(a - b)))

    def _publish_return_progress_update(
        self,
        *,
        status: str,
        search_active: bool,
        window_start_m: float,
        window_end_m: float,
        sigma_m: float,
    ) -> None:
        """Publish the scalar return-progress state for UI and diagnostics."""
        if self.return_target_event_id is None:
            return
        rv: Dict[str, Any] = {
            "target_event_id": str(self.return_target_event_id),
            "status": str(status),
            "search_active": bool(search_active),
            "expected_distance_m": float(self._return_expected_distance_m or 0.0),
            "progress_m": float(self._return_progress.progress_m),
            "remaining_m": float(
                (self._return_expected_distance_m or 0.0)
                - self._return_progress.progress_m
            ),
            "window_start_m": float(window_start_m),
            "window_end_m": float(window_end_m),
            "sigma_m": float(sigma_m),
            "gate_inner_radius_m": float(sigma_m),
            "gate_outer_radius_m": float(
                max(self.return_gate_min_m, self.return_gate_sigma * sigma_m)
            ),
            "integration_variance_m2": float(
                self._return_progress.integration_variance_m2
            ),
            "distance_since_anchor_m": float(
                self._distance_since_localization_anchor_m
            ),
            "distance_anchor_kind": str(self._localization_anchor_kind),
            "distance_anchor_event_id": (
                str(self._localization_anchor_event_id)
                if self._localization_anchor_event_id
                else None
            ),
            "localization_mode": self._localization_mode().value,
            "handover_from_event_id": (
                str(self._return_handover_from_event_id)
                if self._return_handover_from_event_id
                else None
            ),
        }
        rv_msg = String()
        rv_msg.data = json.dumps(rv)
        self.return_viz_pub.publish(rv_msg)
    
    def set_mode(self, mode: NavigationMode):
        if mode != self.current_mode:
            self.current_mode = mode
            
            msg = String()
            msg.data = mode.name
            self.mode_pub.publish(msg)
            
            self.get_logger().info(f"Mode changed to: {mode.name}")
    
    def query_event_by_id(self, event_id: str):
        """Wrap the id in JSON; the repository parser expects an object, not a raw string."""
        msg = String()
        payload = {"event_id": str(event_id)}
        msg.data = json.dumps(payload)
        self.query_event_pub.publish(msg)
    
def main(args=None):
    rclpy.init(args=args)
    node = SmartNavigationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
