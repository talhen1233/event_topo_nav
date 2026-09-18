"""Sequential descriptor matching for route-progress-gated return navigation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, Optional

import numpy as np
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray

try:  # pragma: no cover - allow running module directly
    from .event_types import EventType, JunctionType, NavigationMode
    from .navigation_event import NavigationEvent, Pose3D
    from .radial_descriptor import RadialDescriptorMatcher
    from .return_matching import (
        HandoverDecision,
        HandoverRegion,
        correction_covariance_xy,
        estimate_pose_correction,
        evaluate_geometric_handover,
        event_quality_allows_correction,
        pose_correction_translation_limit,
    )
except ImportError:  # pragma: no cover
    from core.event_types import EventType, JunctionType, NavigationMode
    from core.navigation_event import NavigationEvent, Pose3D
    from core.radial_descriptor import RadialDescriptorMatcher
    from core.return_matching import (
        HandoverDecision,
        HandoverRegion,
        correction_covariance_xy,
        estimate_pose_correction,
        evaluate_geometric_handover,
        event_quality_allows_correction,
        pose_correction_translation_limit,
    )


@dataclass
class MatchCandidate:
    """Descriptor evidence for the single expected event."""

    event: NavigationEvent
    statistical_similarity: float
    spectral_similarity: float
    score: float


class EventMatcher:
    """Confirm only the event enabled by the navigator's distance window."""

    def __init__(self, node, repository, matcher_config: Dict[str, Any]):
        self._node = node
        self._repository = repository
        self.config = matcher_config
        self.current_nav_mode = NavigationMode.IDLE
        self.current_descriptor = None
        self.expected_event_id: Optional[str] = None
        self.last_matched_event_id: Optional[str] = None
        self._matched_expected_id: Optional[str] = None
        self._last_handover_log_key = None
        self.debug_mode = bool(self.config['matcher_debug_mode'])
        self.radial_matcher = RadialDescriptorMatcher(
            weight_statistical=float(self.config['weight_statistical']),
            weight_spectral=float(self.config['weight_spectral']),
        )
        self._matching_timer = None

    def initialize_timer(self) -> None:
        self._reset_matching_timer(self._get_matching_interval())

    def reset(self) -> None:
        """Reset runtime state while keeping ROS wiring intact."""
        self.current_nav_mode = NavigationMode.IDLE
        self.current_descriptor = None
        self.expected_event_id = None
        self.last_matched_event_id = None
        self._matched_expected_id = None
        self._last_handover_log_key = None

    def mode_callback(self, msg: String) -> None:
        mode = NavigationMode[msg.data]
        if mode == self.current_nav_mode:
            return
        self.current_nav_mode = mode
        self._reset_matching_timer(self._get_matching_interval())
        if self.debug_mode:
            self._node.get_logger().info(f"Navigation mode updated to {mode.name}")

    def expected_event_callback(self, msg: String) -> None:
        new_id = msg.data.strip() or None
        if new_id != self.expected_event_id:
            self._matched_expected_id = None
        self.expected_event_id = new_id
        if self.debug_mode and new_id:
            self._node.get_logger().info(f"Distance window opened for event {new_id}")

    def update_current_descriptor(self, descriptor) -> None:
        self.current_descriptor = descriptor

    def handle_pose_update(
        self,
        pose: Pose3D,
        covariance: Optional[np.ndarray],
        dist_increment: float,
    ) -> None:
        """Compatibility hook; return matching deliberately ignores stored pose."""
        del pose, covariance, dist_increment

    def perform_matching(self) -> None:
        if not self._should_perform_matching():
            return
        event = self._repository.stored_events.get(self.expected_event_id)
        if event is None:
        # Sequence is the data association; never fall back to a global scan.
            return
        candidate = self._evaluate_expected_event(event)
        if candidate is None:
            return
        if candidate.score < float(self.config['descriptor_score_threshold']):
            return
        if not self._passes_nearby_event_handover(candidate):
            return
        self._handle_match_result(candidate)

    def _should_perform_matching(self) -> bool:
        if self.current_nav_mode not in (
            NavigationMode.BACKTRACK,
            NavigationMode.RETURN_TO_BASE,
        ):
            return False
        if self.expected_event_id is None or self.current_descriptor is None:
            return False
        if self._matched_expected_id == self.expected_event_id:
            return False
        return bool(self._repository.stored_events)

    def _evaluate_expected_event(
        self,
        event: NavigationEvent,
    ) -> Optional[MatchCandidate]:
        if (
            event.junction_config is not None
            and event.junction_config.junction_type == JunctionType.DEAD_END
        ):
            return None
        if event.radial_descriptor is None:
            return None
        similarity = self.radial_matcher.compute_similarity(
            self.current_descriptor,
            event.radial_descriptor,
        )
        return MatchCandidate(
            event=event,
            statistical_similarity=float(similarity['statistical_similarity']),
            spectral_similarity=float(similarity['spectral_similarity']),
            score=float(similarity['combined_score']),
        )

    @staticmethod
    def _sigma_xy_from_covariance(covariance: Any) -> float:
        if covariance is None:
            return 0.0
        try:
            cov = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
        except (TypeError, ValueError):
            return 0.0
        variance = 0.5 * (float(cov[0, 0]) + float(cov[1, 1]))
        return float(np.sqrt(max(0.0, variance)))

    def _handover_context(
        self,
        target_event: NavigationEvent,
    ) -> tuple[Optional[NavigationEvent], Optional[HandoverDecision]]:
        """Return an active short-range A-to-B handover, if one is valid."""
        if not bool(self.config.get('return_handover_enabled', True)):
            return None, None
        state = getattr(self._repository, 'return_viz_state', None)
        if not isinstance(state, dict):
            return None, None
        if state.get('target_event_id') != target_event.event_id:
            return None, None
        if str(state.get('status') or '') != 'returning':
            return None, None
        from_id = state.get('handover_from_event_id')
        if not isinstance(from_id, str) or not from_id:
            return None, None
        from_event = self._repository.stored_events.get(from_id)
        pose = getattr(self._node, 'current_pose', None)
        if (
            from_event is None
            or from_event.pose is None
            or target_event.pose is None
            or pose is None
        ):
            return None, None
        try:
            route_gap = float(state['expected_distance_m'])
        except (KeyError, TypeError, ValueError):
            route_gap = None
        decision = evaluate_geometric_handover(
            current_xy=(pose.x, pose.y),
            from_xy=(from_event.pose.x, from_event.pose.y),
            target_xy=(target_event.pose.x, target_event.pose.y),
            expected_route_gap_m=route_gap,
            odom_sigma_xy_m=self._sigma_xy_from_covariance(
                getattr(self._node, 'pose_covariance', None)
            ),
            max_distance_m=float(self.config['return_handover_max_distance_m']),
            margin_base_m=float(self.config['return_handover_margin_base_m']),
            margin_sigma_multiplier=float(
                self.config['return_handover_margin_sigma_mult']
            ),
            margin_min_m=float(self.config['return_handover_margin_min_m']),
            margin_max_fraction=float(
                self.config['return_handover_margin_max_fraction']
            ),
        )
        if decision.active and route_gap is not None:
            try:
                progress_m = float(state['progress_m'])
            except (KeyError, TypeError, ValueError):
                progress_m = None
            if progress_m is not None:
                progress_decision = evaluate_geometric_handover(
                    current_xy=(progress_m, 0.0),
                    from_xy=(0.0, 0.0),
                    target_xy=(route_gap, 0.0),
                    expected_route_gap_m=route_gap,
                    odom_sigma_xy_m=self._sigma_xy_from_covariance(
                        getattr(self._node, 'pose_covariance', None)
                    ),
                    max_distance_m=float(
                        self.config['return_handover_max_distance_m']
                    ),
                    margin_base_m=float(
                        self.config['return_handover_margin_base_m']
                    ),
                    margin_sigma_multiplier=float(
                        self.config['return_handover_margin_sigma_mult']
                    ),
                    margin_min_m=float(
                        self.config['return_handover_margin_min_m']
                    ),
                    margin_max_fraction=float(
                        self.config['return_handover_margin_max_fraction']
                    ),
                )
                regions = {decision.region, progress_decision.region}
                if HandoverRegion.FROM_ONLY in regions:
                    combined_region = HandoverRegion.FROM_ONLY
                elif regions == {HandoverRegion.TARGET_ONLY}:
                    combined_region = HandoverRegion.TARGET_ONLY
                else:
                    combined_region = HandoverRegion.COMPETE
                decision = HandoverDecision(
                    region=combined_region,
                    pair_distance_m=decision.pair_distance_m,
                    distance_from_m=decision.distance_from_m,
                    distance_target_m=decision.distance_target_m,
                    margin_m=max(
                        decision.margin_m,
                        progress_decision.margin_m,
                    ),
                    reason='geometry_and_progress',
                )
        return from_event, decision

    def _passes_nearby_event_handover(
        self,
        target_candidate: MatchCandidate,
    ) -> bool:
        from_event, decision = self._handover_context(target_candidate.event)
        if decision is None or not decision.active:
            return True

        if self.debug_mode:
            log_key = (
                target_candidate.event.event_id,
                decision.region.value,
                round(decision.margin_m, 2),
            )
            if log_key != self._last_handover_log_key:
                self._last_handover_log_key = log_key
                self._node.get_logger().info(
                    "Return handover "
                    f"region={decision.region.value} "
                    f"pair={decision.pair_distance_m:.2f}m "
                    f"margin={decision.margin_m:.2f}m"
                )

        if decision.region == HandoverRegion.FROM_ONLY:
            return False
        if decision.region == HandoverRegion.TARGET_ONLY:
            return True
        if from_event is None:
            return False
        from_candidate = self._evaluate_expected_event(from_event)
        if from_candidate is None:
            return False
        delta = max(
            0.0,
            float(self.config['return_handover_competition_delta']),
        )
        return target_candidate.score >= from_candidate.score + delta

    def _observe_current_junction(self) -> Optional[Dict[str, Any]]:
        """Extract current junction center and arms from the rolling local graph."""
        pose = getattr(self._node, 'current_pose', None)
        processor = getattr(self._node, 'processor', None)
        graph = getattr(processor, '_last_graph', None)
        grid = getattr(processor, '_last_grid_params', None)
        if pose is None or graph is None or grid is None:
            return None
        origin_x, origin_y, resolution = grid

        def node_xy(node_id) -> Optional[np.ndarray]:
            raw = np.asarray(graph.nodes[node_id].get('o', []), dtype=np.float64).ravel()
            if raw.size < 2:
                return None
            return np.array([
                float(origin_x) + float(raw[1] - 1.0) * float(resolution),
                float(origin_y) + float(raw[0] - 1.0) * float(resolution),
            ])

        best_id = None
        best_xy = None
        best_distance = float('inf')
        for node_id in graph.nodes:
            if int(graph.degree(node_id)) < 3:
                continue
            xy = node_xy(node_id)
            if xy is None:
                continue
            distance = float(np.hypot(xy[0] - pose.x, xy[1] - pose.y))
            if distance < best_distance:
                best_id, best_xy, best_distance = node_id, xy, distance
        if best_id is None or best_xy is None:
            return None

        max_distance = float(
            getattr(self._node, 'max_junction_detection_distance_junction', 0.0)
        )
        if max_distance > 0.0 and best_distance > max_distance:
            return None

        path_angles = []
        for neighbor in graph.neighbors(best_id):
            xy = node_xy(neighbor)
            if xy is None:
                continue
            dx = float(xy[0] - best_xy[0])
            dy = float(xy[1] - best_xy[1])
            if math.hypot(dx, dy) > 1e-6:
                path_angles.append(float(math.atan2(dy, dx)))
        return {
            'center_xy': [float(best_xy[0]), float(best_xy[1])],
            'path_angles': path_angles,
        }

    def _build_pose_correction(
        self,
        event: NavigationEvent,
        score: float,
        observation: Optional[Dict[str, Any]],
    ) -> Optional[PoseWithCovarianceStamped]:
        """Build a guarded XY correction; never fall back to a direct snap."""
        if not bool(self.config['pose_correction_enabled']):
            return None
        if event.event_type != EventType.JUNCTION or observation is None:
            return None
        if score < float(self.config['pose_correction_min_descriptor_score']):
            return None
        center = observation.get('center_xy')
        pose = getattr(self._node, 'current_pose', None)
        if not isinstance(center, list) or len(center) < 2 or pose is None:
            return None
        if event.pose is None or event.radial_descriptor is None:
            return None

        event_covariance = self._repository._compute_event_cov_xy_from_descriptor(
            event.radial_descriptor
        )
        event_confidence = float(getattr(event, 'confidence', 1.0))
        if not event_quality_allows_correction(
            event_confidence=event_confidence,
            covariance_trace_xy=float(np.trace(event_covariance)),
            confidence_min=float(
                self.config['pose_correction_event_confidence_min']
            ),
            covariance_trace_max_xy=float(
                self.config['pose_correction_event_cov_trace_max_xy']
            ),
        ):
            if self.debug_mode:
                self._node.get_logger().warn(
                    f"Pose correction suppressed for weak event {event.event_id}"
                )
            return None

        state = getattr(self._repository, 'return_viz_state', None)
        distance_since_anchor_m = 0.0
        if (
            isinstance(state, dict)
            and state.get('target_event_id') == event.event_id
            and str(state.get('status') or '') == 'returning'
        ):
            try:
                distance_since_anchor_m = max(
                    0.0,
                    float(state.get('distance_since_anchor_m', 0.0)),
                )
            except (TypeError, ValueError):
                distance_since_anchor_m = 0.0
        translation_limit_m = pose_correction_translation_limit(
            distance_since_anchor_m=distance_since_anchor_m,
            base_m=float(self.config['pose_correction_translation_base_m']),
            growth_per_meter=float(
                self.config['pose_correction_translation_growth_per_meter']
            ),
            hard_cap_m=float(
                self.config['pose_correction_translation_hard_cap_m']
            ),
        )

        try:
            estimate = estimate_pose_correction(
                robot_xy=(pose.x, pose.y),
                stored_center_xy=(event.pose.x, event.pose.y),
                observed_center_xy=center,
                max_translation_m=translation_limit_m,
            )
        except (TypeError, ValueError):
            return None
        if not estimate.accepted:
            self._node.get_logger().warn(
                f"Pose correction rejected for {event.event_id}: "
                f"{estimate.reason}, delta={estimate.translation_norm_m:.2f}m, "
                f"dynamic_limit={translation_limit_m:.2f}m after "
                f"{distance_since_anchor_m:.1f}m from anchor"
            )
            return None

        covariance_xy = correction_covariance_xy(
            event_covariance_xy=event_covariance,
            detection_noise_m=float(self.config['pose_correction_detection_noise_m']),
            match_score=score,
            acceptance_threshold=float(self.config['descriptor_score_threshold']),
            covariance_scale=float(self.config['pose_correction_cov_scale']),
            minimum_variance_m2=float(
                self.config['pose_correction_min_variance_m2']
            ),
        )
        correction = PoseWithCovarianceStamped()
        correction.header.stamp = self._node.get_clock().now().to_msg()
        correction.header.frame_id = 'odom'
        correction.pose.pose.position.x = estimate.corrected_xy[0]
        correction.pose.pose.position.y = estimate.corrected_xy[1]
        correction.pose.pose.position.z = float(pose.z)
        correction.pose.pose.orientation.w = 1.0
        covariance = np.eye(6, dtype=np.float64) * 1e3
        covariance[:2, :2] = covariance_xy
        covariance[2, 2] = 1e6
        correction.pose.covariance = covariance.reshape(-1).tolist()
        return correction

    def _publish_match_debug(self, event: NavigationEvent) -> None:
        """Publish XY context as diagnostics, not as an acceptance gate."""
        publisher = getattr(self._node, 'gating_debug_pub', None)
        pose = getattr(self._node, 'current_pose', None)
        if not self.debug_mode or publisher is None or pose is None:
            return
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        now = self._node.get_clock().now().to_msg()

        arrow = Marker()
        arrow.header.frame_id = 'odom'
        arrow.header.stamp = now
        arrow.ns = 'match_context'
        arrow.id = 0
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD
        start = Point(x=float(pose.x), y=float(pose.y), z=float(pose.z + 0.05))
        end = Point(
            x=float(event.pose.x),
            y=float(event.pose.y),
            z=float(event.pose.z + 0.05),
        )
        arrow.points = [start, end]
        arrow.scale.x = 0.05
        arrow.scale.y = 0.10
        arrow.scale.z = 0.10
        arrow.color.r = 0.2
        arrow.color.g = 0.9
        arrow.color.b = 0.3
        arrow.color.a = 0.9
        markers.markers.append(arrow)

        label = Marker()
        label.header.frame_id = 'odom'
        label.header.stamp = now
        label.ns = 'match_context_distance'
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = 0.5 * (pose.x + event.pose.x)
        label.pose.position.y = 0.5 * (pose.y + event.pose.y)
        label.pose.position.z = event.pose.z + 0.35
        label.scale.z = 0.18
        label.color.r = label.color.g = label.color.b = 1.0
        label.color.a = 1.0
        label.text = f"{math.hypot(pose.x - event.pose.x, pose.y - event.pose.y):.2f}m"
        markers.markers.append(label)
        publisher.publish(markers)

    def _handle_match_result(self, candidate: MatchCandidate) -> None:
        event = candidate.event
        self._node.get_logger().info(
            f"Matched event {event.event_id} ({event.event_type.name}) "
            f"descriptor_score={candidate.score:.2f}"
        )
        self.last_matched_event_id = event.event_id
        self._matched_expected_id = event.event_id
        self._repository.update_last_matched_descriptor(event.radial_descriptor)
        payload: Dict[str, Any] = {
            'event_id': event.event_id,
            'event_type': event.event_type.name,
            'confidence': candidate.score,
            'geometric_similarity': candidate.statistical_similarity,
            'depth_similarity': candidate.spectral_similarity,
        }
        observation = self._observe_current_junction()
        if observation is not None:
            payload['observed_junction'] = observation
        correction = self._build_pose_correction(event, candidate.score, observation)
        payload['pose_correction_published'] = correction is not None
        if correction is not None:
            self._node.pose_correction_pub.publish(correction)

        desc = event.radial_descriptor
        if desc is not None:
            max_pts = 72
            payload['stored_descriptor'] = {
                'angles': [float(a) for a in list(desc.angles)[:max_pts]],
                'filtered_distances': [
                    float(d) for d in list(desc.filtered_distances)[:max_pts]
                ],
            }
        match_msg = String()
        match_msg.data = json.dumps(payload)
        self._node.match_found_pub.publish(match_msg)
        conf_msg = Float32()
        conf_msg.data = candidate.score
        self._node.match_confidence_pub.publish(conf_msg)
        self._publish_match_debug(event)

    def _get_matching_interval(self) -> float:
        if self.current_nav_mode in (
            NavigationMode.RETURN_TO_BASE,
            NavigationMode.BACKTRACK,
        ):
            return float(self.config['active_matching_interval'])
        return float(self.config['passive_matching_interval'])

    def _reset_matching_timer(self, interval: float) -> None:
        if interval <= 0.0:
            interval = 1.0
        if self._matching_timer is not None:
            self._matching_timer.cancel()
        self._matching_timer = self._node.create_timer(interval, self.perform_matching)
