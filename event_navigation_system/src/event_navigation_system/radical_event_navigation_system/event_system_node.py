#!/usr/bin/env python3
"""Radial-profile processing, event storage, and targeted event matching."""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor

import tf2_py
import tf2_ros

import numpy as np
from typing import Optional, Dict, Tuple, Any, List
import copy
import json
import math
import uuid
from datetime import datetime
from collections import deque

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, CompressedImage
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String, Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseWithCovarianceStamped

try:  # pragma: no cover - import guard for direct execution
    from .core.radial_profile import RadialProfile, SensorConfig
    from .core.navigation_event import Pose3D, NavigationEvent, TunnelGeometry
    from .core.skeleton_graph import SkeletonGraphProcessor
    from .core.event_repository import EventRepository
    from .core.radial_profile_processor import RadialProfileProcessor
    from .core.event_matcher import EventMatcher
    from .core.event_types import EventType, NavigationMode, JunctionType, Decision
    from .core.start_stability import (
        StartStabilityParams,
        StartStabilityTracker,
    )
except ImportError:  # pragma: no cover
    from core.radial_profile import RadialProfile, SensorConfig
    from core.navigation_event import Pose3D, NavigationEvent, TunnelGeometry
    from core.skeleton_graph import SkeletonGraphProcessor
    from core.event_repository import EventRepository
    from core.radial_profile_processor import RadialProfileProcessor
    from core.event_matcher import EventMatcher
    from core.event_types import EventType, NavigationMode, JunctionType, Decision
    from core.start_stability import (
        StartStabilityParams,
        StartStabilityTracker,
    )

_PARAMS_RADIAL_PROFILE = [
    ('processing_interval_s', 0.1),
    ('angular_resolution_deg', 45 / 8),
    ('sensor_max_range_m', 4.0),
    (
        'sensor_max_ranges_m',
        [4.0, 4.0, 4.0, 4.0],
        ParameterDescriptor(dynamic_typing=True),
    ),
    ('confidence_decay_rate', 1.0),
    ('confidence_rise_rate', 1.0),
    ('sensor_directions_deg', [0.0, 90.0, 180.0, 270.0]),
    ('sensor_fov_deg', 45.0),
    ('voxel_clearance_extra_voxels', 2.0),
    ('vertical_pitch_half_angle_deg', 5.0),
]

_PARAMS_TIME_ALIGNMENT = [
    ('profile_pose_delay_s', 0.3),
]

_PARAMS_START_STABILITY = [
    ('start_stability_min_frames', 10),
    ('start_stability_max_normalized_change', 0.05),
    ('start_stability_max_speed_mps', 0.05),
]

_PARAMS_ODOMETRY_MODE = [
    ('of_degraded_topic', '/drone/odometry/of_degraded'),
    ('of_degraded_timeout_s', 0.5),  # stale of_degraded is treated as degraded; <=0 disables timeout
    # OF-degraded map smear can delete/alter stored junctions during arm revalidation.
    ('disable_junction_arm_recheck_when_of_degraded', True),
]

_PARAMS_FILTERING = [
    ('median_window', 5),
    ('gaussian_sigma_samples', 1.3),
    ('filter_confidence_threshold', 0.6),
]


_PARAMS_EVENT_DETECTION = [
    ('min_event_distance_junction_m', 1.5),
    ('max_junction_detection_distance_junction_m', 3.0),
    ('dead_end_terminal_proximity_m', 0.3),
    ('dead_end_proximity_potential_max', 0.2),
    ('dead_end_turn_potential_rate', 0.15),
    ('dead_end_confirmation_threshold', 0.85),
    ('junction_arm_recheck_radius_m', 0.5),
    ('junction_arm_recheck_event_distance_m', 0.125),
    ('junction_arm_recheck_sample_step_m', 0.15),
    ('junction_arm_recheck_inflate_xy_voxels', 1),
    ('junction_arm_recheck_inflate_z_voxels', 1),
    ('junction_arm_recheck_match_max_angle_deg', 45.0),
    ('junction_arm_recheck_min_hit_voxels', 8),
]

_PARAMS_VOXEL_MAP = [
    ('voxel_size', 0.15),
]

_PARAMS_SKELETON_GRAPH = [
    ('skeleton_grid_width_m', 20.0),
    ('skeleton_grid_height_m', 20.0),
    ('skeleton_grid_resolution_m', 0.1),
    ('graph_min_edge_len_m', 0.2),
    ('graph_node_merge_radius_m', 0.35),
    ('arm_direction_fit_min_m', 0.35),
    ('arm_direction_fit_max_m', 2.0),
    ('arm_direction_fit_min_span_m', 0.8),
    ('event_potential_cell_size_m', 3.0),
]

_PARAMS_DEBUG = [
    ('debug_level', 2),  # 0=off, 1=RViz, 2=images+matcher debug
    ('profiling_enabled', False),
]

_PARAMS_EVENT_STABILIZATION = [
    ('event_potential_increment', 0.4),
    ('event_potential_decay', 0.2),
    ('event_potential_threshold', 0.8),
    ('event_potential_arm_consistency_enabled', True),
    ('event_potential_arm_worst_match_max_deg', 65.0),
    ('event_potential_deg_penalty_base', 0.8),
    ('event_potential_increment_min_scale', 0.3),
    ('junction_topology_settle_s', 0.8),
    ('junction_potential_topic', '/navigation_events/junction_potential'),
    ('junction_potential_publish_interval_s', 0.1),
    ('max_event_potential_update_interval_s', 1.0),
    ('dead_end_event_rearm_distance_m', 0.75),
    # After invalidation, suppress potential re-growth so the same junction is not immediately recreated.
    ('event_potential_invalidation_suppress_s', 2.0),
    ('min_event_potential_update_distance_m', 0.2),
    ('min_event_potential_update_yaw_deg', 45.0),
]

_PARAMS_DESCRIPTOR_STORAGE = [
    ('store_raw_profile', True),  # dashboard matching visualization needs the raw arrays
    ('fft_num_harmonics', 8),
    ('default_event_cov_xy', 0.5),
    ('default_event_cov_z', 0.2),
]

_PARAMS_EVENT_MERGE = [
    ('event_merge_score_threshold', 0.8),
    ('event_merge_mahal_threshold', 4.0),
    ('event_merge_discard_on_mismatch', True),
    # Euclidean cap: Mahalanobis alone permits long-range merges when covariances inflate.
    ('event_merge_max_euclidean_distance_m', 3.0),
    ('event_merge_min_update_interval_s', 5.0),
    ('event_merge_update_alpha_max', 0.5),
    ('event_merge_update_use_arm_consistency', True),
    ('event_merge_update_arm_worst_match_max_deg', 45.0),
    ('event_merge_update_deg_penalty_base', 0.7),
]

_PARAMS_MATCHER = [
    # Descriptor confirmation only; SmartNavigation opens this matcher via route-progress gating.
    ('descriptor_score_threshold', 0.8),
    ('active_matching_interval', 0.5),
    ('passive_matching_interval', 5.0),
    ('descriptor_weight_spectral', 0.7),  # statistical weight is 1-w
    # Short-range A/B handover; disabled automatically for long route gaps.
    ('return_handover_enabled', True),
    ('return_handover_max_distance_m', 5.0),
    ('return_handover_margin_base_m', 0.2),
    ('return_handover_margin_sigma_mult', 1.0),
    ('return_handover_margin_min_m', 0.1),
    ('return_handover_margin_max_fraction', 0.4),
    ('return_handover_competition_delta', 0.05),
    ('pose_correction_enabled', True),
    ('pose_correction_min_descriptor_score', 0.85),
    ('pose_correction_detection_noise_m', 0.3),
    ('pose_correction_cov_scale', 1.0),
    ('pose_correction_min_variance_m2', 0.0225),
    # Allowance is base + growth*travel; hard_cap is independent of route length.
    ('pose_correction_translation_base_m', 1.0),
    ('pose_correction_translation_growth_per_meter', 0.12),
    ('pose_correction_translation_hard_cap_m', 8.0),
    ('pose_correction_event_confidence_min', 0.2),
    ('pose_correction_event_cov_trace_max_xy', 0.5),
]

_EVENT_SYSTEM_NODE_PARAMETERS = (
    _PARAMS_RADIAL_PROFILE
    + _PARAMS_TIME_ALIGNMENT
    + _PARAMS_START_STABILITY
    + _PARAMS_ODOMETRY_MODE
    + _PARAMS_FILTERING
    + _PARAMS_EVENT_DETECTION
    + _PARAMS_VOXEL_MAP
    + _PARAMS_SKELETON_GRAPH
    + _PARAMS_DEBUG
    + _PARAMS_EVENT_STABILIZATION
    + _PARAMS_DESCRIPTOR_STORAGE
    + _PARAMS_EVENT_MERGE
    + _PARAMS_MATCHER
)


class _Voxel:
    """Lightweight voxel container with slots for memory efficiency."""
    __slots__ = ('occupancy', 'x', 'y', 'z')
    
    def __init__(self, ix: int, iy: int, iz: int):
        self.occupancy = True
        self.x = ix
        self.y = iy
        self.z = iz

class EventSystemNode(Node):
    """Build radial profiles, store events, and confirm return matches."""
    
    def __init__(self):
        super().__init__('event_system_node')
        
        self.qos_sensor = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.qos_navigation = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Voxel-map TF uses translation only; rotation would smear occupancy into odom.
        self._odom_frame_id = "odom"
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._last_tf_warn_time_s: float = 0.0
        
        self._declare_node_parameters()
        
        def _p(name: str) -> Any:
            return self.get_parameter(name).value

        def _pf(name: str) -> float:
            return float(_p(name))

        def _pi(name: str) -> int:
            return int(_p(name))

        def _pb(name: str) -> bool:
            return bool(_p(name))

        angular_res = _pf('angular_resolution_deg')
        sensor_max_range = _pf('sensor_max_range_m')
        confidence_decay = _pf('confidence_decay_rate')
        confidence_rise = _pf('confidence_rise_rate')
        sensor_dirs_param = _p('sensor_directions_deg')
        self.vertical_pitch_half_angle_deg = _pf('vertical_pitch_half_angle_deg')
        median_win = _pi('median_window')
        gaussian_sigma = _pf('gaussian_sigma_samples')
        sensor_directions = [float(a) for a in sensor_dirs_param]
        sensor_max_ranges_param = _p('sensor_max_ranges_m')
        sensor_max_ranges = None
        if isinstance(sensor_max_ranges_param, (list, tuple)) and len(sensor_max_ranges_param) == len(sensor_directions):
            sensor_max_ranges = [float(r) for r in sensor_max_ranges_param]
        elif sensor_max_ranges_param:
            self.get_logger().warning(
                "Parameter 'sensor_max_ranges_m' is set but its length does not "
                "match 'sensor_directions_deg'; ignoring per-sensor ranges."
            )
        
        self.min_event_distance_junction = _pf('min_event_distance_junction_m')
        self.max_junction_detection_distance_junction = _pf('max_junction_detection_distance_junction_m')
        self.junction_arm_recheck_radius = _pf('junction_arm_recheck_radius_m')
        self.junction_arm_recheck_event_distance_m = _pf('junction_arm_recheck_event_distance_m')
        self.junction_arm_recheck_sample_step_m = _pf('junction_arm_recheck_sample_step_m')
        self.junction_arm_recheck_inflate_xy_voxels = _pi('junction_arm_recheck_inflate_xy_voxels')
        self.junction_arm_recheck_inflate_z_voxels = _pi('junction_arm_recheck_inflate_z_voxels')
        self.junction_arm_recheck_match_max_angle_deg = _pf('junction_arm_recheck_match_max_angle_deg')
        self.min_event_distance_geometry = self.min_event_distance_junction
        self.voxel_size = _pf('voxel_size')
        voxel_extra = _pf('voxel_clearance_extra_voxels')
        voxel_prox = float((1.0 + voxel_extra) * self.voxel_size)
        debug_level = int(_pi('debug_level'))
        self.debug_level = int(max(0, min(2, debug_level)))
        self.debug_viz = self.debug_level >= 1
        self.debug_images_enabled = self.debug_level >= 2
        self.debug_images_show_voxels = self.debug_level >= 2
        self._profiling_enabled = bool(self.get_parameter('profiling_enabled').value)
        self.viz_skeleton_overlay = True
        self.viz_ray_scale = 0.9
        self.viz_interval = 1.0
        self.processing_interval = _pf('processing_interval_s')
        self.viz_min_confidence = 0.5
        self.filter_confidence_threshold = _pf('filter_confidence_threshold')
        self.profile_pose_delay_s = float(max(0.0, _pf('profile_pose_delay_s')))
        self.viz_curve_use_gradient = True
        self.viz_curve_low = (1.0, 0.3, 0.3)
        self.viz_curve_high = (0.2, 0.8, 1.0)
        self.viz_curve_alpha = 0.9
        self.viz_event_uncertainty = True
        self.event_potential_increment = _pf('event_potential_increment')
        self.event_potential_decay = _pf('event_potential_decay')
        self.event_potential_threshold = _pf('event_potential_threshold')
        self.event_potential_arm_consistency_enabled = _pb('event_potential_arm_consistency_enabled')
        self.event_potential_arm_worst_match_max_deg = _pf('event_potential_arm_worst_match_max_deg')
        self.event_potential_deg_penalty_base = _pf('event_potential_deg_penalty_base')
        self.event_potential_increment_min_scale = _pf('event_potential_increment_min_scale')
        self.junction_topology_settle_s = _pf('junction_topology_settle_s')
        self.junction_potential_topic = str(_p('junction_potential_topic'))
        self.junction_potential_publish_interval_s = _pf(
            'junction_potential_publish_interval_s'
        )
        self.max_potential_update_interval_s = _pf(
            'max_event_potential_update_interval_s'
        )
        self.dead_end_event_rearm_distance_m = _pf(
            'dead_end_event_rearm_distance_m'
        )
        self.event_potential_invalidation_suppress_s = _pf('event_potential_invalidation_suppress_s')
        self.min_potential_update_distance = _pf('min_event_potential_update_distance_m')
        self.min_potential_update_yaw_deg = _pf('min_event_potential_update_yaw_deg')
        self.store_raw_profile = _pb('store_raw_profile')
        self.fft_num_harmonics = _pi('fft_num_harmonics')
        self.default_event_cov_xy = _pf('default_event_cov_xy')
        self.default_event_cov_z = _pf('default_event_cov_z')
        self.viz_entry_from = True
        self.event_merge_score_threshold = _pf('event_merge_score_threshold')
        self.event_merge_mahal_threshold = _pf('event_merge_mahal_threshold')
        self.event_merge_discard_on_mismatch = _pb('event_merge_discard_on_mismatch')
        self.event_merge_max_euclidean_distance_m = _pf('event_merge_max_euclidean_distance_m')
        self.event_merge_min_update_interval_s = _pf('event_merge_min_update_interval_s')
        self.event_merge_update_alpha_max = _pf('event_merge_update_alpha_max')
        self.event_merge_update_use_arm_consistency = _pb('event_merge_update_use_arm_consistency')
        self.event_merge_update_arm_worst_match_max_deg = _pf('event_merge_update_arm_worst_match_max_deg')
        self.event_merge_update_deg_penalty_base = _pf('event_merge_update_deg_penalty_base')
        
        descriptor_weight_spectral = float(_pf('descriptor_weight_spectral'))
        descriptor_weight_spectral = float(np.clip(descriptor_weight_spectral, 0.0, 1.0))
        self.descriptor_weight_spectral = descriptor_weight_spectral
        descriptor_weight_statistical = float(1.0 - descriptor_weight_spectral)

        self.matcher_config = {
            'descriptor_score_threshold': _pf('descriptor_score_threshold'),
            'active_matching_interval': _pf('active_matching_interval'),
            'passive_matching_interval': _pf('passive_matching_interval'),
            'weight_statistical': descriptor_weight_statistical,
            'weight_spectral': descriptor_weight_spectral,
            'return_handover_enabled': _pb('return_handover_enabled'),
            'return_handover_max_distance_m': _pf('return_handover_max_distance_m'),
            'return_handover_margin_base_m': _pf('return_handover_margin_base_m'),
            'return_handover_margin_sigma_mult': _pf('return_handover_margin_sigma_mult'),
            'return_handover_margin_min_m': _pf('return_handover_margin_min_m'),
            'return_handover_margin_max_fraction': _pf('return_handover_margin_max_fraction'),
            'return_handover_competition_delta': _pf('return_handover_competition_delta'),
            'pose_correction_enabled': _pb('pose_correction_enabled'),
            'pose_correction_min_descriptor_score': _pf('pose_correction_min_descriptor_score'),
            'pose_correction_detection_noise_m': _pf('pose_correction_detection_noise_m'),
            'pose_correction_cov_scale': _pf('pose_correction_cov_scale'),
            'pose_correction_min_variance_m2': _pf('pose_correction_min_variance_m2'),
            'pose_correction_translation_base_m': _pf('pose_correction_translation_base_m'),
            'pose_correction_translation_growth_per_meter': _pf(
                'pose_correction_translation_growth_per_meter'
            ),
            'pose_correction_translation_hard_cap_m': _pf(
                'pose_correction_translation_hard_cap_m'
            ),
            'pose_correction_event_confidence_min': _pf('pose_correction_event_confidence_min'),
            'pose_correction_event_cov_trace_max_xy': _pf('pose_correction_event_cov_trace_max_xy'),
            'matcher_debug_mode': bool(self.debug_level >= 1),
        }
        
        self.skel_grid_width_m = _pf('skeleton_grid_width_m')
        self.skel_grid_height_m = _pf('skeleton_grid_height_m')
        self.skel_grid_resolution_m = _pf('skeleton_grid_resolution_m')
        self.graph_min_edge_len_m = _pf('graph_min_edge_len_m')
        self.graph_node_merge_radius_m = _pf('graph_node_merge_radius_m')
        self.arm_direction_fit_min_m = _pf('arm_direction_fit_min_m')
        self.arm_direction_fit_max_m = _pf('arm_direction_fit_max_m')
        self.arm_direction_fit_min_span_m = _pf('arm_direction_fit_min_span_m')
        self.event_potential_cell_size_m = _pf('event_potential_cell_size_m')

        sensor_config = SensorConfig(
            directions=sensor_directions,
            fov_deg=_pf('sensor_fov_deg'),
            max_range=sensor_max_range,
            max_ranges=sensor_max_ranges,
        )
        
        self.profile = RadialProfile(
            angular_resolution_deg=angular_res,
            sensor_config=sensor_config,
            confidence_decay_rate=confidence_decay,
            confidence_rise_rate=confidence_rise,
            voxel_proximity_stop=voxel_prox,
            median_window=median_win,
            gaussian_sigma_samples=gaussian_sigma,
            vertical_pitch_half_angle_deg=self.vertical_pitch_half_angle_deg,
        )
        self.current_pose: Optional[Pose3D] = None
        self.pose_covariance: Optional[np.ndarray] = None
        self.distance_traveled = 0.0
        self.last_event_pose: Optional[Pose3D] = None
        # DEAD_END is planner-driven; the graph only assists confirmation.
        self.local_planner_state: str = "UNKNOWN"
        
        self.voxel_map: Dict[Tuple[int, int, int], any] = {}
        self._have_voxel_map_message = False
        self._last_voxel_map_update_s = -1.0
        self._last_odometry_update_s = -1.0
        self._current_linear_speed_mps = float("inf")
        self._start_stability = StartStabilityTracker(StartStabilityParams(
            min_frames=max(1, _pi('start_stability_min_frames')),
            max_normalized_change=max(0.0, _pf('start_stability_max_normalized_change')),
            max_speed_mps=max(0.0, _pf('start_stability_max_speed_mps')),
        ))
        self._last_start_wait_log_s = -1.0

        self._odom_history: deque = deque()
        self._odom_history_max_age_s = float(max(1.0, self.profile_pose_delay_s + 1.0))
        self._last_profile_pose_used: Optional[Pose3D] = None
        
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        # Decay follows covariance *changes*, not absolute size.
        self._base_confidence_decay_rate = float(confidence_decay)
        self._decay_scale = 1.0
        self._decay_scale_min = 0.5
        self._decay_scale_max = 3.0
        self._prev_pose_cov_trace_xy: Optional[float] = None
        
        self.skg = SkeletonGraphProcessor(
            grid_width_m=self.skel_grid_width_m,
            grid_height_m=self.skel_grid_height_m,
            grid_resolution_m=self.skel_grid_resolution_m,
            graph_min_edge_len_m=self.graph_min_edge_len_m,
            graph_node_merge_radius_m=self.graph_node_merge_radius_m,
            arm_direction_fit_min_m=self.arm_direction_fit_min_m,
            arm_direction_fit_max_m=self.arm_direction_fit_max_m,
            arm_direction_fit_min_span_m=self.arm_direction_fit_min_span_m,
            event_potential_cell_size_m=self.event_potential_cell_size_m,
            min_event_distance_junction_m=self.min_event_distance_junction,
            event_potential_increment=self.event_potential_increment,
            event_potential_decay=self.event_potential_decay,
            event_potential_threshold=self.event_potential_threshold,
            max_junction_detection_distance_junction_m=self.max_junction_detection_distance_junction,
            event_potential_arm_consistency_enabled=self.event_potential_arm_consistency_enabled,
            event_potential_arm_worst_match_max_deg=self.event_potential_arm_worst_match_max_deg,
            event_potential_deg_penalty_base=self.event_potential_deg_penalty_base,
            event_potential_increment_min_scale=self.event_potential_increment_min_scale,
            junction_topology_settle_s=self.junction_topology_settle_s,
            event_potential_invalidation_suppress_s=self.event_potential_invalidation_suppress_s,
            clock_s=lambda: self.get_clock().now().nanoseconds * 1e-9,
            logger=self.get_logger(),
        )
        self.event_pub = self.create_publisher(
            String, '/navigation_events/detected', self.qos_sensor
        )
        
        self.event_marker_pub = self.create_publisher(
            MarkerArray, '/navigation_events/markers', self.qos_sensor
        )
        
        self.radial_viz_pub = self.create_publisher(
            MarkerArray, '/radial_profile/visualization', self.qos_sensor
        )
        self.junction_profile_viz_pub = self.create_publisher(
            MarkerArray, '/navigation_events/junction_profile', self.qos_sensor
        )
        
        self.skeleton_graph_pub = self.create_publisher(
            MarkerArray, '/skeleton_graph/visualization', self.qos_sensor
        )
        
        self.raw_profile_image_pub = self.create_publisher(
            CompressedImage, '/radial_profile/raw_debug/compressed', self.qos_sensor
        )
        self.skeleton_overlay_image_pub = self.create_publisher(
            CompressedImage, '/skeleton_overlay/compressed', self.qos_sensor
        )
        
        self.matched_profile_pub = self.create_publisher(
            MarkerArray, '/radial_profile/matched_overlay', self.qos_sensor
        )
        self.match_found_pub = self.create_publisher(
            String, '/navigation_events/match_found', self.qos_sensor
        )
        self.match_confidence_pub = self.create_publisher(
            Float32, '/navigation_events/match_confidence', self.qos_sensor
        )
        self.pose_correction_pub = self.create_publisher(
            PoseWithCovarianceStamped,
            '/localization/pose_correction',
            self.qos_sensor,
        )
        # Transient-local so late-joining dashboards receive the live correction-limit policy.
        self.pose_correction_policy_pub = self.create_publisher(
            String,
            '/navigation_events/pose_correction_policy',
            self.qos_latched,
        )
        correction_policy_msg = String()
        correction_policy_msg.data = json.dumps({
            'enabled': bool(self.matcher_config['pose_correction_enabled']),
            'base_m': float(
                self.matcher_config['pose_correction_translation_base_m']
            ),
            'growth_per_meter': float(
                self.matcher_config[
                    'pose_correction_translation_growth_per_meter'
                ]
            ),
            'hard_cap_m': float(
                self.matcher_config['pose_correction_translation_hard_cap_m']
            ),
        })
        self.pose_correction_policy_pub.publish(correction_policy_msg)
        self.gating_debug_pub = self.create_publisher(
            MarkerArray, '/navigation_events/gating_debug', self.qos_sensor
        )
        self.event_potential_debug_pub = self.create_publisher(
            MarkerArray, '/navigation_events/event_potential_debug', self.qos_sensor
        )
        self.junction_potential_pub = self.create_publisher(
            Float32,
            self.junction_potential_topic,
            self.qos_sensor,
        )
        self.create_timer(
            max(0.02, self.junction_potential_publish_interval_s),
            self._publish_junction_potential_heartbeat,
        )
        self.event_stored_pub = self.create_publisher(
            String, '/navigation_events/stored', self.qos_sensor
        )
        self.event_deleted_pub = self.create_publisher(
            String, '/navigation_events/deleted', self.qos_sensor
        )
        self.event_query_response_pub = self.create_publisher(
            String, '/navigation_events/query/response', self.qos_sensor
        )
        
        self.current_descriptor_pub = self.create_publisher(
            String, '/radial_descriptor/current', self.qos_sensor
        )
        self.start_ready_pub = self.create_publisher(
            Bool, '/mission_control/start_ready', self.qos_latched
        )
        self.start_event_pub = self.create_publisher(
            String, '/mission_control/start_event', self.qos_latched
        )
        self._publish_start_ready(False)
        
        self.skeleton_graph_json_pub = self.create_publisher(
            String, '/skeleton_graph/json', self.qos_sensor
        )
        self.repository = EventRepository(
            node=self,
            default_event_cov_xy=self.default_event_cov_xy,
            default_event_cov_z=self.default_event_cov_z,
        )
        self.matcher = EventMatcher(self, self.repository, self.matcher_config)
        self.processor = RadialProfileProcessor(self, self.repository)
        
        # Subscribers after helpers so callbacks never hit missing attributes.
        self.create_subscription(
            Odometry, '/drone/state_estimate',
            self.odometry_callback, self.qos_sensor
        )
        self.create_subscription(
            PointCloud2, '/map_pointcloud',
            self.voxel_map_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation/mode',
            self.matcher.mode_callback, self.qos_navigation
        )
        self.create_subscription(
            String, '/navigation/expected_event',
            self.matcher.expected_event_callback, self.qos_navigation
        )
        self.create_subscription(
            String, '/navigation_events/query/by_id',
            self.repository.query_by_id_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation_events/query/nearest',
            self.repository.query_nearest_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation_events/match_found',
            self.repository.match_found_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation_events/return_viz',
            self.repository.return_viz_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation_events/decision/update',
            self.repository.decision_update_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/navigation_events/dead_end_path',
            self.repository.dead_end_path_callback, self.qos_sensor
        )
        self.create_subscription(
            String, '/local_planner/state',
            self.local_planner_state_callback, self.qos_sensor
        )
        self.create_subscription(
            Bool, '/mission_control/start',
            self._mission_start_callback, self.qos_sensor
        )
        self.create_subscription(
            Bool, '/mission_control/stop',
            self._mission_stop_callback, self.qos_sensor
        )
        self.create_subscription(
            Bool, '/mission_control/reset',
            self._mission_reset_callback, self.qos_sensor
        )
        self._mission_active: bool = False
        self._start_event_published: bool = False
        
        self.create_timer(self.processing_interval, self.processor.process)
        if self.debug_viz or self.debug_images_enabled:
            self.create_timer(self.viz_interval, self.processor.publish_visualizations)
        self.matcher.initialize_timer()
        
        self.get_logger().info(
            f"Event System initialized with {self.profile.num_rays} rays "
            f"at {angular_res}° resolution (processing @ {1.0/self.processing_interval:.1f}Hz, "
            f"viz @ {1.0/self.viz_interval:.1f}Hz)"
        )

        self._of_degraded: bool = False
        self._of_degraded_last_update_s: float = -1.0
        self._of_degraded_timeout_s: float = float(_pf('of_degraded_timeout_s'))
        self._disable_junction_arm_recheck_when_of_degraded: bool = bool(
            _pb('disable_junction_arm_recheck_when_of_degraded')
        )
        self._of_degraded_topic: str = str(_p('of_degraded_topic') or "").strip()
        if self._of_degraded_topic:
            self.create_subscription(
                Bool,
                self._of_degraded_topic,
                self._of_degraded_callback,
                self.qos_sensor,
            )

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        """Convert ROS builtin_interfaces/Time to float seconds."""
        return float(getattr(stamp, "sec", 0)) + 1e-9 * float(getattr(stamp, "nanosec", 0))

    def _push_odometry_history(self, stamp_s: float, pose: Pose3D, cov_xyzh: np.ndarray) -> None:
        """Append an odometry sample and prune old history."""
        self._odom_history.append((float(stamp_s), pose, cov_xyzh))
        cutoff = float(stamp_s) - float(self._odom_history_max_age_s)
        while self._odom_history and float(self._odom_history[0][0]) < cutoff:
            self._odom_history.popleft()

    def get_profile_pose(self) -> Optional[Pose3D]:
        """Return a delayed odometry sample for radial-profile generation."""
        if self.current_pose is None:
            return None
        delay = float(getattr(self, "profile_pose_delay_s", 0.0))
        if delay <= 1e-6:
            return self.current_pose
        if not self._odom_history:
            return self.current_pose
        base_t = float(self._odom_history[-1][0])
        target_t = base_t - delay
        for t_s, pose, _cov in reversed(self._odom_history):
            if float(t_s) <= target_t:
                return pose
        return self._odom_history[0][1] if self._odom_history else self.current_pose
    
    def _declare_node_parameters(self) -> None:
        """Declare node parameters from the grouped defaults above."""
        self.declare_parameters(namespace='', parameters=_EVENT_SYSTEM_NODE_PARAMETERS)

    def _of_degraded_callback(self, msg: Bool) -> None:
        self._of_degraded = bool(getattr(msg, "data", False))
        self._of_degraded_last_update_s = float(self.get_clock().now().nanoseconds) * 1e-9

    def _is_of_degraded(self) -> bool:
        if self._of_degraded_last_update_s < 0.0:
            return True
        timeout_s = float(self._of_degraded_timeout_s)
        if timeout_s <= 0.0:
            return bool(self._of_degraded)
        now_s = float(self.get_clock().now().nanoseconds) * 1e-9
        if now_s - float(self._of_degraded_last_update_s) > timeout_s:
            return True
        return bool(self._of_degraded)

    def odometry_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        new_pose = Pose3D(x=pos.x, y=pos.y, z=pos.z)
        
        q = msg.pose.pose.orientation
        yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        new_pose.yaw = yaw
        
        dist_increment = 0.0
        if self.current_pose:
            dist_increment = new_pose.distance_to(self.current_pose)
            self.distance_traveled += dist_increment
        else:
            self.distance_traveled = 0.0
        
        self.current_pose = new_pose
        self._last_odometry_update_s = self.get_clock().now().nanoseconds * 1e-9
        self.pose_covariance = np.array(msg.pose.covariance).reshape(6, 6)[:3, :3]
        
        linear = msg.twist.twist.linear
        self._current_linear_speed_mps = float(
            np.linalg.norm([float(linear.x), float(linear.y), float(linear.z)])
        )
        
        stamp_s = self._stamp_to_sec(msg.header.stamp)
        self._push_odometry_history(
            stamp_s=stamp_s,
            pose=Pose3D(x=float(new_pose.x), y=float(new_pose.y), z=float(new_pose.z), yaw=float(new_pose.yaw)),
            cov_xyzh=np.asarray(self.pose_covariance, dtype=np.float32).copy(),
        )
        self._update_profile_confidence_decay_rate()
        self.matcher.handle_pose_update(new_pose, self.pose_covariance, dist_increment)
    
    def voxel_map_callback(self, msg: PointCloud2):
        """Rebuild the odom-frame voxel map using translation-only TF."""
        cloud_frame = (getattr(msg.header, "frame_id", "") or "").strip().lstrip("/")
        translation = self._lookup_translation_to_odom(cloud_frame)
        if translation is None:
            # Keep the previous map: wiping it on TF loss produces inconsistent graphs.
            return

        self.voxel_map.clear()

        dx = float(translation[0])
        dy = float(translation[1])
        dz = float(translation[2])

        for p in point_cloud2.read_points(
            msg, field_names=('x', 'y', 'z'), skip_nans=True
        ):
            x = float(p[0]) + dx
            y = float(p[1]) + dy
            z = float(p[2]) + dz

            ix = int(np.floor(x / self.voxel_size))
            iy = int(np.floor(y / self.voxel_size))
            iz = int(np.floor(z / self.voxel_size))

            self.voxel_map[(ix, iy, iz)] = _Voxel(ix, iy, iz)

        self._have_voxel_map_message = True
        self._last_voxel_map_update_s = self.get_clock().now().nanoseconds * 1e-9

    def _lookup_translation_to_odom(self, source_frame: str) -> Optional[np.ndarray]:
        """Look up source->odom TF and return translation only (rotation ignored)."""
        src = (source_frame or "").strip().lstrip("/")
        if not src or src == self._odom_frame_id:
            return np.zeros(3, dtype=np.float32)

        try:
            tf = self._tf_buffer.lookup_transform(self._odom_frame_id, src, Time())
        except (tf2_py.LookupException, tf2_py.ExtrapolationException) as exc:
            now_s = self.get_clock().now().nanoseconds / 1e9
            if now_s - float(self._last_tf_warn_time_s) >= 2.0:
                self.get_logger().warning(
                    f"TF lookup failed for {src} -> {self._odom_frame_id}; "
                    "skipping voxel map update (keeping previous map). "
                    f"Error: {exc}"
                )
                self._last_tf_warn_time_s = now_s
            return None

        t = tf.transform.translation
        return np.array([t.x, t.y, t.z], dtype=np.float32)
    
    def local_planner_state_callback(self, msg: String) -> None:
        """Cache planner state used to gate DEAD_END events."""
        state = (msg.data or "").strip().upper()
        if not state:
            state = "UNKNOWN"
        self.local_planner_state = state

    def publish_junction_potential(self, potential: float) -> None:
        """Publish normalized nearby-junction potential for motion control."""
        msg = Float32()
        msg.data = float(np.clip(potential, 0.0, 1.0))
        self.junction_potential_pub.publish(msg)

    def _publish_junction_potential_heartbeat(self) -> None:
        """Repeat the latest potential between expensive graph evaluations."""
        self.publish_junction_potential(
            self.skg.current_junction_potential
        )

    def _mission_start_callback(self, msg: Bool) -> None:
        if msg.data:
            if self._mission_active:
                return
            self._mission_active = True
            self._start_event_published = False
            self._start_stability.reset()
            self._publish_start_ready(False)
            self.get_logger().info("Mission started - waiting for a stable START profile")

    def _mission_stop_callback(self, msg: Bool) -> None:
        if msg.data:
            self._mission_active = False
            self._start_event_published = False
            self._start_stability.reset()
            self._publish_start_ready(False)
            self.publish_junction_potential(0.0)
            self.get_logger().info("Mission stopped")

    def _mission_reset_callback(self, msg: Bool) -> None:
        if not bool(getattr(msg, "data", False)):
            return
        self._hard_reset()

    def _publish_deleteall_markerarray(self, pub) -> None:
        ma = MarkerArray()
        m = Marker()
        m.action = Marker.DELETEALL
        ma.markers.append(m)
        pub.publish(ma)

    def _hard_reset(self) -> None:
        event_ids = list(getattr(self.repository, "stored_events", {}).keys())
        for event_id in event_ids:
            del_msg = String()
            del_msg.data = json.dumps({'event_id': event_id, 'reason': 'reset', 'source': 'mission_reset'})
            self.event_deleted_pub.publish(del_msg)

        # RViz retains MarkerArray content across runs unless DELETEALL is published.
        self._publish_deleteall_markerarray(self.event_marker_pub)
        self._publish_deleteall_markerarray(self.radial_viz_pub)
        self._publish_deleteall_markerarray(self.junction_profile_viz_pub)
        self._publish_deleteall_markerarray(self.skeleton_graph_pub)
        self._publish_deleteall_markerarray(self.matched_profile_pub)
        self._publish_deleteall_markerarray(self.gating_debug_pub)
        self._publish_deleteall_markerarray(self.event_potential_debug_pub)

        self.repository.reset()
        self.repository.publish_event_markers()
        self.matcher.reset()
        self.processor.reset()
        self.skg.reset()
        self.publish_junction_potential(0.0)

        self._mission_active = False
        self._start_event_published = False
        self._start_stability.reset()
        self._publish_start_ready(False)
        self.current_pose = None
        self.pose_covariance = None
        self.distance_traveled = 0.0
        self.last_event_pose = None
        self.local_planner_state = "UNKNOWN"
        self.voxel_map.clear()
        self._have_voxel_map_message = False
        self._last_voxel_map_update_s = -1.0
        self._last_odometry_update_s = -1.0
        self._current_linear_speed_mps = float("inf")
        self._odom_history.clear()
        self._last_profile_pose_used = None
        self._prev_pose_cov_trace_xy = None
        self._decay_scale = 1.0
        self.start_time = self.get_clock().now().nanoseconds / 1e9

        try:
            self.profile.reset()
        except AttributeError:
            pass

        empty = String()
        empty.data = ""
        self.current_descriptor_pub.publish(empty)
        sk = String()
        sk.data = "{}"
        self.skeleton_graph_json_pub.publish(sk)

        self.get_logger().info("Mission hard reset: cleared event system state")

    def _publish_start_ready(self, ready: bool) -> None:
        msg = Bool()
        msg.data = bool(ready)
        self.start_ready_pub.publish(msg)

    def update_start_stability(self) -> bool:
        """Publish START only after a stationary, persistent radial profile."""
        if not self._mission_active or self._start_event_published:
            return bool(self._start_event_published)
        if self.current_pose is None:
            return False
        if not self._have_voxel_map_message:
            return False

        now_s = self.get_clock().now().nanoseconds * 1e-9
        status = self._start_stability.update(
            self.profile.filtered_distances,
            now_s,
            self._current_linear_speed_mps,
        )
        if not status.stable or status.representative_profile is None:
            if now_s - self._last_start_wait_log_s >= 1.0:
                change = (
                    "n/a" if status.normalized_change is None
                    else f"{100.0 * status.normalized_change:.1f}%"
                )
                self.get_logger().info(
                    "Waiting for stable START: "
                    f"frames={status.consecutive_frames}, "
                    f"duration={status.elapsed_s:.1f}s, change={change}, "
                    f"speed={self._current_linear_speed_mps:.3f}m/s"
                )
                self._last_start_wait_log_s = now_s
            return False

        stable_profile = copy.copy(self.profile)
        stable_profile.raw_distances = status.representative_profile.copy()
        stable_profile.filtered_distances = status.representative_profile.copy()

        try:
            from .core.radial_descriptor import create_descriptor_from_profile
        except ImportError:
            from core.radial_descriptor import create_descriptor_from_profile

        descriptor = create_descriptor_from_profile(
            stable_profile,
            num_harmonics=self.fft_num_harmonics,
            store_raw_profile=self.store_raw_profile,
        )
        
        event = NavigationEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.START,
            timestamp=datetime.now(),
            decision=Decision.HOVER,
            pose=Pose3D(
                x=float(self.current_pose.x),
                y=float(self.current_pose.y),
                z=float(self.current_pose.z),
                yaw=float(self.current_pose.yaw),
            ),
            pose_uncertainty=self.pose_covariance.copy() if self.pose_covariance is not None else np.eye(3),
            odometry_distance=0.0,
            tunnel_geometry=TunnelGeometry(width=0.0, height=0.0, shape_descriptor="start"),
        )
        event.radial_descriptor = descriptor
        event.confidence = 1.0
        
        self.repository.publish_event(event)
        
        detected_msg = String()
        detected_msg.data = json.dumps(event.to_dict())
        self.event_pub.publish(detected_msg)
        # Transient-local copy used for the atomic START handshake.
        self.start_event_pub.publish(detected_msg)
        
        self._start_event_published = True
        self._publish_start_ready(True)
        self.get_logger().info(
            f"Published START event id={event.event_id[:8]}... at "
            f"({float(self.current_pose.x):.2f}, {float(self.current_pose.y):.2f})"
        )
        return True

    def _update_profile_confidence_decay_rate(self) -> None:
        """Speed up ray decay when XY covariance grows; slow it when covariance shrinks."""
        profile = getattr(self, "profile", None)
        if profile is None:
            return
        if self.pose_covariance is None:
            self._decay_scale = 1.0
            profile.confidence_decay_rate = float(self._base_confidence_decay_rate)
            return

        cov = np.asarray(self.pose_covariance, dtype=np.float32)
        trace_xy = float(cov[0, 0] + cov[1, 1])
        if not np.isfinite(trace_xy) or trace_xy <= 0.0:
            self._decay_scale = 1.0
            profile.confidence_decay_rate = float(self._base_confidence_decay_rate)
            return

        if self._prev_pose_cov_trace_xy is None:
            self._prev_pose_cov_trace_xy = trace_xy
            self._decay_scale = 1.0
            profile.confidence_decay_rate = float(self._base_confidence_decay_rate)
            return

        denom = max(abs(self._prev_pose_cov_trace_xy), 1e-6)
        rel_change = (trace_xy - self._prev_pose_cov_trace_xy) / denom

        if abs(rel_change) < 0.02:
            self._prev_pose_cov_trace_xy = trace_xy
            profile.confidence_decay_rate = float(self._base_confidence_decay_rate * self._decay_scale)
            return

        k = 0.3
        delta_scale = 1.0 + k * rel_change
        if delta_scale <= 0.1:
            delta_scale = 0.1

        self._decay_scale *= delta_scale
        if self._decay_scale < self._decay_scale_min:
            self._decay_scale = self._decay_scale_min
        elif self._decay_scale > self._decay_scale_max:
            self._decay_scale = self._decay_scale_max

        self._prev_pose_cov_trace_xy = trace_xy
        profile.confidence_decay_rate = float(self._base_confidence_decay_rate * self._decay_scale)

    @staticmethod
    def _angle_diff_rad(a: float, b: float) -> float:
        """Smallest signed angle difference a-b in radians."""
        return float(math.atan2(math.sin(a - b), math.cos(a - b)))

    @staticmethod
    def _build_graph_node_xy_cache(
        graph,
        origin_x: float,
        origin_y: float,
        resolution: float,
    ) -> Tuple[list[int], np.ndarray, Dict[int, np.ndarray]]:
        """Convert skeleton-graph node centers from raster coordinates into world XY."""
        node_ids: list[int] = []
        node_xy: list[Tuple[float, float]] = []
        for node_id, attr in graph.nodes(data=True):
            o = np.asarray(attr.get('o', [0, 0]), dtype=np.float32)
            r = float(o[0] - 1.0)
            c = float(o[1] - 1.0)
            x = float(origin_x + c * resolution)
            y = float(origin_y + r * resolution)
            node_ids.append(int(node_id))
            node_xy.append((x, y))
        node_xy_arr = np.asarray(node_xy, dtype=np.float32) if node_xy else np.empty((0, 2), dtype=np.float32)
        node_xy_by_id = {nid: node_xy_arr[i] for i, nid in enumerate(node_ids)} if node_ids else {}
        return node_ids, node_xy_arr, node_xy_by_id

    @staticmethod
    def _nearest_graph_node_id(
        node_ids: list[int],
        node_xy_arr: np.ndarray,
        query_xy: np.ndarray,
    ) -> Tuple[int, np.ndarray]:
        diffs = node_xy_arr - query_xy[None, :]
        d2 = np.sum(diffs * diffs, axis=1)
        idx = int(np.argmin(d2))
        nid = int(node_ids[idx])
        return nid, node_xy_arr[idx]

    @staticmethod
    def _edge_polyline_xy(
        graph,
        u: int,
        v: int,
        origin_x: float,
        origin_y: float,
        resolution: float,
        node_xy_by_id: Dict[int, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Return a world-frame (N,2) polyline for the edge u-v, oriented from u to v."""
        attr = graph.get_edge_data(u, v)
        if not isinstance(attr, dict):
            return None
        pts = attr.get("pts", None)
        if pts is None:
            return None
        rc = np.asarray(pts, dtype=np.float32)
        if rc.ndim != 2 or rc.shape[0] < 2:
            return None
        r = rc[:, 0] - 1.0
        c = rc[:, 1] - 1.0
        xs = origin_x + c * resolution
        ys = origin_y + r * resolution
        poly = np.stack([xs, ys], axis=1).astype(np.float32)
        u_xy = node_xy_by_id.get(int(u))
        if u_xy is None:
            return poly
        d0 = float(np.sum((poly[0] - u_xy) ** 2))
        d1 = float(np.sum((poly[-1] - u_xy) ** 2))
        return poly if d0 <= d1 else poly[::-1].copy()

    @classmethod
    def _branch_polyline_from_neighbor(
        cls,
        graph,
        start_id: int,
        neighbor_id: int,
        origin_x: float,
        origin_y: float,
        resolution: float,
        node_xy_by_id: Dict[int, np.ndarray],
        extension_m: float,
        max_hops: int = 256,
    ) -> Optional[np.ndarray]:
        """Walk along a branch starting at start_id -> neighbor_id through deg-2 nodes."""
        segments = []
        visited: set[int] = set()
        prev = int(start_id)
        curr = int(neighbor_id)
        hop = 0
        while hop < int(max_hops):
            hop += 1
            edge_poly = cls._edge_polyline_xy(
                graph, prev, curr, origin_x, origin_y, resolution, node_xy_by_id
            )
            if edge_poly is None or edge_poly.size == 0:
                break
            segments.append(edge_poly if not segments else edge_poly[1:])
            visited.add(prev)
            deg = int(graph.degree(curr))
            if deg != 2:
                break
            nbrs = list(graph.neighbors(curr))
            if len(nbrs) != 2:
                break
            nxt = int(nbrs[0]) if int(nbrs[1]) == prev else int(nbrs[1])
            if nxt == start_id or nxt in visited:
                break
            prev, curr = curr, nxt

        if not segments:
            return None
        poly = np.vstack(segments).astype(np.float32)
        if poly.shape[0] < 2:
            return None
        if extension_m > 0.0:
            d = poly[-1] - poly[-2]
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                u = d / n
                end = (poly[-1] + u * float(extension_m)).astype(np.float32)
                poly = np.vstack([poly, end])
        return poly

    @staticmethod
    def _polyline_hits_voxels(
        poly_xy: np.ndarray,
        z_m: float,
        voxel_map: Dict[Tuple[int, int, int], any],
        voxel_size: float,
        step_m: float,
        neighbor_offsets: List[Tuple[int, int, int]],
        min_hit_voxels: int = 1,
    ) -> int:
        """Return number of distinct occupied voxels hit by samples along the polyline."""
        if poly_xy.ndim != 2 or poly_xy.shape[0] < 2:
            return 0
        if voxel_size <= 0.0:
            return 0
        step = float(max(1e-6, step_m))
        iz0 = int(np.floor(float(z_m) / voxel_size))
        min_hits = int(max(1, int(min_hit_voxels)))
        hit_voxels: set[Tuple[int, int, int]] = set()
        for i in range(poly_xy.shape[0] - 1):
            p0 = poly_xy[i]
            p1 = poly_xy[i + 1]
            seg = p1 - p0
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            steps = max(1, int(math.ceil(seg_len / step)))
            for k in range(1, steps + 1):
                t = float(k) / float(steps)
                pt = p0 + t * seg
                ix = int(np.floor(float(pt[0]) / voxel_size))
                iy = int(np.floor(float(pt[1]) / voxel_size))
                for dx, dy, dz in neighbor_offsets:
                    key = (ix + dx, iy + dy, iz0 + dz)
                    if key in voxel_map:
                        hit_voxels.add(key)
                        if len(hit_voxels) >= min_hits:
                            return int(len(hit_voxels))
                        break
        return int(len(hit_voxels))

    def revalidate_nearby_junction_arms(self) -> None:
        """Prune stored junction arms that collide with occupied voxels during exploration."""
        if self.current_pose is None or not self.voxel_map:
            return
        if not hasattr(self, "matcher") or self.matcher is None:
            return
        if float(getattr(self, "junction_arm_recheck_event_distance_m", 0.0)) <= 0.0:
            return
        if self.matcher.current_nav_mode != NavigationMode.EXPLORE:
            return

        # OF-degraded map smear produces false blocked arms and can delete stored events.
        if self._disable_junction_arm_recheck_when_of_degraded and self._is_of_degraded():
            return

        graph = getattr(getattr(self, "processor", None), "_last_graph", None)
        grid_params = getattr(getattr(self, "processor", None), "_last_grid_params", None)
        if graph is None or grid_params is None:
            return
        origin_x, origin_y, resolution = grid_params
        resolution = float(resolution)
        if resolution <= 0.0:
            return

        voxel_size = float(self.voxel_size)
        if voxel_size <= 0.0:
            return

        event_dist = float(getattr(self, "junction_arm_recheck_event_distance_m", 0.0))
        extension_m = float(getattr(self, "junction_arm_recheck_radius", 0.0))
        step_m = float(getattr(self, "junction_arm_recheck_sample_step_m", voxel_size))
        if step_m <= 0.0:
            step_m = float(voxel_size)
        inflate_xy = max(0, int(getattr(self, "junction_arm_recheck_inflate_xy_voxels", 1)))
        inflate_z = max(0, int(getattr(self, "junction_arm_recheck_inflate_z_voxels", 1)))
        match_max_deg = float(getattr(self, "junction_arm_recheck_match_max_angle_deg", 45.0))
        match_max_rad = float(np.deg2rad(max(0.0, match_max_deg)))
        min_hit_voxels = max(1, int(getattr(self, "junction_arm_recheck_min_hit_voxels", 1)))

        neighbor_offsets: List[Tuple[int, int, int]] = []
        for dx in range(-inflate_xy, inflate_xy + 1):
            for dy in range(-inflate_xy, inflate_xy + 1):
                for dz in range(-inflate_z, inflate_z + 1):
                    neighbor_offsets.append((dx, dy, dz))

        node_ids, node_xy_arr, node_xy_by_id = self._build_graph_node_xy_cache(
            graph=graph,
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            resolution=float(resolution),
        )
        if not node_ids or node_xy_arr.size == 0:
            return

        events_items = list(self.repository.stored_events.items())
        to_delete = []
        changed_any = False

        for event_id, event in events_items:
            if event.event_type != EventType.JUNCTION:
                continue
            jc = getattr(event, "junction_config", None)
            if jc is None or not getattr(jc, "path_angles", None):
                continue
            old_angles = [float(a) for a in (jc.path_angles or [])]
            old_selected_idx = getattr(jc, "selected_path_index", -1)
            old_entry_angle = getattr(event, "entry_angle", None)
            dx_e = float(event.pose.x - self.current_pose.x)
            dy_e = float(event.pose.y - self.current_pose.y)
            dist_xy = float(math.hypot(dx_e, dy_e))
            if dist_xy > float(event_dist):
                continue

            ev_xy = np.array([float(event.pose.x), float(event.pose.y)], dtype=np.float32)
            junction_node_id, junction_node_xy = self._nearest_graph_node_id(node_ids, node_xy_arr, ev_xy)

            neighbor_ids = list(graph.neighbors(junction_node_id))
            if not neighbor_ids:
                continue
            neighbor_dirs = []
            for nid in neighbor_ids:
                n_xy = node_xy_by_id.get(int(nid))
                if n_xy is None:
                    continue
                v = n_xy - junction_node_xy
                if float(np.linalg.norm(v)) < 1e-6:
                    continue
                neighbor_dirs.append((int(nid), float(math.atan2(float(v[1]), float(v[0])))))

            keep_angles = []
            index_map = {}
            for idx, ang in enumerate(jc.path_angles):
                angle = float(ang)
                best_n = None
                best_d = float("inf")
                for nid, n_ang in neighbor_dirs:
                    dtheta = abs(self._angle_diff_rad(n_ang, angle))
                    if dtheta < best_d:
                        best_d = dtheta
                        best_n = int(nid)
                # Unmatched arms are kept rather than dropped.
                if best_n is None or best_d > match_max_rad:
                    new_idx = len(keep_angles)
                    keep_angles.append(angle)
                    index_map[idx] = new_idx
                    continue

                branch_poly = self._branch_polyline_from_neighbor(
                    graph=graph,
                    start_id=junction_node_id,
                    neighbor_id=best_n,
                    origin_x=float(origin_x),
                    origin_y=float(origin_y),
                    resolution=float(resolution),
                    node_xy_by_id=node_xy_by_id,
                    extension_m=float(extension_m),
                )
                blocked = False
                if branch_poly is not None:
                    hit_count = self._polyline_hits_voxels(
                        poly_xy=branch_poly,
                        z_m=float(event.pose.z),
                        voxel_map=self.voxel_map,
                        voxel_size=float(voxel_size),
                        step_m=float(step_m),
                        neighbor_offsets=neighbor_offsets,
                        min_hit_voxels=int(min_hit_voxels),
                    )
                    blocked = int(hit_count) >= int(min_hit_voxels)

                if not blocked:
                    new_idx = len(keep_angles)
                    keep_angles.append(angle)
                    index_map[idx] = new_idx

            new_degree = len(keep_angles)
            old_degree = int(getattr(jc, "num_paths", len(jc.path_angles)))
            if new_degree == old_degree:
                continue

            if new_degree < 2:
                to_delete.append(event_id)
                continue

            jc.path_angles = keep_angles
            jc.num_paths = new_degree

            # Remap entry/selected semantics onto surviving arms instead of resetting them.
            def _closest_idx(ref_ang: float, candidates: list[float]) -> int:
                if not candidates:
                    return -1
                diffs = [abs(self._angle_diff_rad(float(a), float(ref_ang))) for a in candidates]
                return int(np.argmin(np.asarray(diffs, dtype=np.float32)))

            mapped_selected = -1
            if isinstance(old_selected_idx, int) and old_selected_idx >= 0 and old_angles:
                if old_selected_idx in index_map:
                    mapped_selected = int(index_map[old_selected_idx])
                elif 0 <= int(old_selected_idx) < len(old_angles):
                    mapped_selected = _closest_idx(float(old_angles[int(old_selected_idx)]), keep_angles)
            jc.selected_path_index = int(mapped_selected)

            if old_entry_angle is not None and old_angles:
                try:
                    entry_ang = float(old_entry_angle)
                except (TypeError, ValueError):
                    entry_ang = None
                if entry_ang is not None:
                    old_entry_idx = _closest_idx(entry_ang, old_angles)
                    if old_entry_idx in index_map:
                        event.entry_angle = float(keep_angles[int(index_map[old_entry_idx])])
                    elif 0 <= old_entry_idx < len(old_angles):
                        event.entry_angle = float(
                            keep_angles[_closest_idx(float(old_angles[old_entry_idx]), keep_angles)]
                        )

            if getattr(jc, "dead_end_paths", None):
                jc.dead_end_paths = [
                    int(index_map[i]) for i in jc.dead_end_paths if i in index_map
                ]

            attempts = self.repository._junction_attempts.get(event_id)
            if attempts:
                mapped_attempts: set[int] = set()
                for idx in attempts:
                    if not isinstance(idx, int):
                        continue
                    if idx in index_map:
                        mapped_attempts.add(int(index_map[idx]))
                        continue
                    if 0 <= idx < len(old_angles) and keep_angles:
                        mapped_attempts.add(_closest_idx(float(old_angles[idx]), keep_angles))
                self.repository._junction_attempts[event_id] = mapped_attempts

            if new_degree >= 4:
                jc.junction_type = JunctionType.CROSS_JUNCTION
            elif new_degree == 3:
                jc.junction_type = JunctionType.T_JUNCTION
            else:
                to_delete.append(event_id)
                continue

            changed_any = True
            self.get_logger().info(
                f"Revalidated junction {event_id}: DEG {old_degree} -> {new_degree}, "
                f"type={jc.junction_type.name}"
            )

        for event_id in to_delete:
            event = self.repository.stored_events.get(event_id)
            if event is not None and event.pose is not None:
                self.skg.suppress_event_potential_at(float(event.pose.x), float(event.pose.y))
            if event_id in self.repository.stored_events:
                del self.repository.stored_events[event_id]
            if event_id in self.repository._junction_attempts:
                del self.repository._junction_attempts[event_id]
            del_msg = String()
            del_msg.data = json.dumps(
                {'event_id': event_id, 'reason': 'invalid_junction', 'source': 'junction_arm_recheck'}
            )
            self.event_deleted_pub.publish(del_msg)
            changed_any = True

        if changed_any and self.debug_viz:
            self.repository.publish_event_markers()

    def _publish_query_response(self, payload: Dict[str, Any]):
        msg = String()
        msg.data = json.dumps(payload)
        self.event_query_response_pub.publish(msg)

    def _publish_event_stored_notification(self, event_id: str, event: Optional[Any] = None):
        """Publish a stored-event notice with pose and navigation fields for the UI map."""
        payload: Dict[str, Any] = {'status': 'stored', 'event_id': event_id}
        if event is not None:
            payload['x'] = float(event.pose.x)
            payload['y'] = float(event.pose.y)
            payload['yaw'] = float(event.pose.yaw)
            if hasattr(event, 'event_type') and event.event_type is not None:
                payload['event_type'] = event.event_type.name if hasattr(event.event_type, 'name') else str(event.event_type)
            if event.junction_config is not None:
                jc = event.junction_config
                payload['junction_type'] = jc.junction_type.name if hasattr(jc.junction_type, 'name') else str(jc.junction_type)
                payload['num_paths'] = len(jc.path_angles) if jc.path_angles else 0
                if jc.path_angles:
                    payload['path_angles'] = [float(a) for a in jc.path_angles]
                if hasattr(jc, 'selected_path_index') and jc.selected_path_index is not None:
                    payload['selected_path_index'] = int(jc.selected_path_index)
                payload['dead_end_paths'] = [
                    int(index)
                    for index in (
                        getattr(jc, 'dead_end_paths', None)
                        or []
                    )
                ]
            if hasattr(event, 'entry_angle') and event.entry_angle is not None:
                payload['entry_angle'] = float(event.entry_angle)
        msg = String()
        msg.data = json.dumps(payload)
        self.event_stored_pub.publish(msg)
        self.get_logger().info(
            f"Published event_stored: {event_id[:8]}... sel_idx={payload.get('selected_path_index')}, "
            f"entry={payload.get('entry_angle')}, paths={len(payload.get('path_angles', []))}"
        )
    
def main(args=None):
    rclpy.init(args=args)
    node = EventSystemNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
