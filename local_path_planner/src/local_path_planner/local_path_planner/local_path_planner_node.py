#!/usr/bin/env python3
"""Fly the skeleton graph; navigation owns where to go."""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool, Float32
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker

try:
    from local_path_planner.scripts.arm_commit_tracker import ArmCommitTracker, CommittedArmIntent
    from local_path_planner.scripts.committed_arm_lock import (
        CommittedArmLock,
        CommittedArmLockParams,
    )
    from local_path_planner.scripts.graph_centering_controller import (
        GraphCenteringController,
        GraphCenteringOutput,
        GraphCenteringParams,
    )
    from local_path_planner.scripts.graph_tracker import (
        GraphArmCandidate,
        GraphTracker,
        GraphTrackerParams,
        GraphTrackingState,
    )
    from local_path_planner.scripts.height_controller import HeightController, HeightControllerParams
    from local_path_planner.scripts.leaf_progress import LeafProgressTracker
    from local_path_planner.scripts.junction_potential_speed import (
        JunctionPotentialSpeed,
        JunctionPotentialSpeedParams,
    )
    from local_path_planner.scripts.markers import CmdVelMarkers, DebugMarkers
    from local_path_planner.scripts.ray_sectors import RaySectors, RaySectorsParams
    from local_path_planner.scripts.safety_controller import SafetyController, SafetyControllerParams, SafetyOutput
    from local_path_planner.scripts.yaw_intent_filter import YawIntentFilter, YawIntentFilterParams, YawIntentOutput
except ImportError:
    from scripts.arm_commit_tracker import ArmCommitTracker, CommittedArmIntent
    from scripts.committed_arm_lock import CommittedArmLock, CommittedArmLockParams
    from scripts.graph_centering_controller import (
        GraphCenteringController,
        GraphCenteringOutput,
        GraphCenteringParams,
    )
    from scripts.graph_tracker import GraphArmCandidate, GraphTracker, GraphTrackerParams, GraphTrackingState
    from scripts.height_controller import HeightController, HeightControllerParams
    from scripts.leaf_progress import LeafProgressTracker
    from scripts.junction_potential_speed import (
        JunctionPotentialSpeed,
        JunctionPotentialSpeedParams,
    )
    from scripts.markers import CmdVelMarkers, DebugMarkers
    from scripts.ray_sectors import RaySectors, RaySectorsParams
    from scripts.safety_controller import SafetyController, SafetyControllerParams, SafetyOutput
    from scripts.yaw_intent_filter import YawIntentFilter, YawIntentFilterParams, YawIntentOutput


class LocalPathPlanner(Node):
    def __init__(self) -> None:
        super().__init__("local_path_planner")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._declare_parameters()
        p = self._fetch_parameters()

        self.create_subscription(PointCloud2, p["map_topic"], self._on_cloud, qos)
        self.create_subscription(Odometry, p["odom_topic"], self._on_odom, qos)
        self.create_subscription(String, p["graph_topic"], self._on_graph_json, qos)
        self.create_subscription(String, p["graph_target_topic"], self._on_graph_target, qos)
        self.create_subscription(
            Float32,
            p["junction_potential_topic"],
            self._on_junction_potential,
            qos,
        )
        self.create_subscription(Bool, "/mission_control/start", self._on_mission_start, qos)
        self.create_subscription(
            Bool,
            "/mission_control/navigation_ready",
            self._on_start_ready,
            qos_latched,
        )
        self.create_subscription(Bool, "/mission_control/stop", self._on_mission_stop, qos)

        self.cmd_pub = self.create_publisher(Twist, p["cmd_vel_topic"], 1)
        self.state_pub = self.create_publisher(String, p["state_topic"], qos)
        self.debug_state_pub = self.create_publisher(String, p["debug_state_topic"], qos)
        self.cmd_marker_pub = self.create_publisher(Marker, "/cmd_vel_markers", qos)
        self.debug_marker_pub = self.create_publisher(Marker, "/local_planner/debug_markers", qos)

        self._graph_tracker = GraphTracker(
            GraphTrackerParams(
                junction_approach_distance_m=p["junction_approach_distance_m"],
                edge_hysteresis_m=p["edge_hysteresis_m"],
                graph_stale_timeout_s=p["graph_stale_timeout_s"],
                lookahead_distance_m=p["graph_lookahead_m"],
                heading_fit_distance_m=p["graph_heading_fit_distance_m"],
                arm_fit_min_distance_m=p["arm_direction_fit_min_m"],
                arm_fit_max_distance_m=p["arm_direction_fit_max_m"],
                arm_fit_min_span_m=p["arm_direction_fit_min_span_m"],
            )
        )
        self._graph_center = GraphCenteringController(
            GraphCenteringParams(
                enabled=p["graph_centering_enabled"],
                k_y=p["graph_center_k_y"],
                deadband_m=p["graph_center_deadband_m"],
                vy_max=p["graph_center_vy_max"],
                filter_tau_s=p["graph_center_filter_tau_s"],
            )
        )
        self._arm_commit = ArmCommitTracker()
        self._committed_arm_lock = CommittedArmLock(
            CommittedArmLockParams(
                entry_distance_m=p["commit_entry_distance_m"],
                entry_cross_track_m=p["commit_entry_cross_track_m"],
                heading_tolerance_rad=math.radians(
                    p["commit_support_angle_deg"]
                ),
                max_distance_m=p["commit_max_distance_m"],
                direction_smoothing_tau_s=p[
                    "arm_direction_smoothing_tau_s"
                ],
                direction_max_rate_rad_s=math.radians(
                    p["arm_direction_max_rate_deg_s"]
                ),
                direction_min_confidence=p[
                    "arm_direction_min_confidence"
                ],
                acquisition_timeout_s=p[
                    "commit_lock_acquisition_timeout_s"
                ],
            )
        )
        self._junction_potential_speed = JunctionPotentialSpeed(
            JunctionPotentialSpeedParams(
                slowdown_start=p["junction_potential_slowdown_start"],
                full_slowdown=p["junction_potential_full_slowdown"],
                min_speed_scale=p[
                    "junction_potential_min_speed_scale"
                ],
                timeout_s=p["junction_potential_timeout_s"],
            )
        )
        self._leaf_progress = LeafProgressTracker(
            near_m=p["leaf_stall_near_m"],
            timeout_s=p["leaf_stall_timeout_s"],
            progress_m=p["leaf_stall_progress_m"],
        )
        self._yaw_intent = YawIntentFilter(
            YawIntentFilterParams(
                num_bins=p["yaw_intent_num_bins"],
                potential_tau_s=p["yaw_intent_potential_tau_s"],
                candidate_sigma_rad=math.radians(p["yaw_intent_candidate_sigma_deg"]),
                switch_margin=p["yaw_intent_switch_margin"],
                slew_rate_rps=p["yaw_intent_slew_rate_rps"],
                slowdown_distance_m=p["junction_slowdown_distance_m"],
                uncommitted_hold_distance_m=p[
                    "uncommitted_junction_hold_distance_m"
                ],
                uncommitted_hold_timeout_s=p[
                    "uncommitted_junction_hold_timeout_s"
                ],
                uncommitted_creep_speed_scale=p[
                    "uncommitted_junction_creep_speed_scale"
                ],
                uncommitted_hold_reset_s=p[
                    "uncommitted_junction_hold_reset_s"
                ],
                min_arm_length_m=p["min_arm_length_m"],
                commit_support_angle_rad=math.radians(p["commit_support_angle_deg"]),
            )
        )
        self._height_ctrl = HeightController(HeightControllerParams(
            xy_max_range=p["xy_max_range"],
            up_pitch_min=math.radians(p["up_pitch_min_deg"]),
            down_pitch_min=math.radians(p["down_pitch_min_deg"]),
            d0_ceiling=p["d0_ceiling_m"],
            d0_floor=p["d0_floor_m"],
            k_ceiling=p["k_ceiling"],
            k_floor=p["k_floor"],
            vz_max=p["vz_max"],
            vertical_dist_percentile=p["vertical_dist_percentile"],
            vz_filter_tau=p["vz_filter_tau_s"],
            vz_deadband=p["vz_deadband_mps"],
            control_rate=p["control_rate"],
        ))
        self._safety_ctrl = SafetyController(SafetyControllerParams(
            front_block_dist_m=p["front_block_dist_m"],
            front_clear_dist_m=p["front_clear_dist_m"],
            front_block_hysteresis_m=p["front_block_hysteresis_m"],
            front_block_width_margin_m=p["front_block_width_margin_m"],
            front_block_dynamic_max_m=p["front_block_dynamic_max_m"],
            front_width_filter_tau_s=p["front_width_filter_tau_s"],
            front_stop_dist_m=p["front_stop_dist_m"],
            front_slow_dist_m=p["front_slow_dist_m"],
            max_decel_mps2=p["safety_max_decel_mps2"],
            reaction_time_s=p["safety_reaction_time_s"],
            vehicle_radius_m=p["vehicle_radius_m"],
            stop_margin_m=p["safety_stop_margin_m"],
            slow_margin_m=p["safety_slow_margin_m"],
            open_dist_m=p["open_dist_m"],
            block_debounce_s=p["block_debounce_s"],
            dead_end_time_s=p["dead_end_time_s"],
            turn_search_max_yaw_rad=math.radians(p["turn_search_max_yaw_deg"]),
            yaw_rate_max=p["yaw_rate_max_rps"],
            xy_max_range=p["xy_max_range"],
        ))

        self._sectors = RaySectors(RaySectorsParams(
            xy_max_range=p["xy_max_range"],
            horiz_pitch_half=math.radians(p["horiz_pitch_half_deg"]),
            front_yaw_half=math.radians(p["front_yaw_half_deg"]),
            back_yaw_half=math.radians(p["back_yaw_half_deg"]),
            side_yaw_half=math.radians(p["side_yaw_half_deg"]),
            up_pitch_min=math.radians(p["up_pitch_min_deg"]),
            down_pitch_min=math.radians(p["down_pitch_min_deg"]),
            d0_side=p["d0_side_m"],
            k_y=p["k_y"],
            center_deadband=p["center_deadband_m"],
            side_dist_percentile=p["side_dist_percentile"],
            min_points_per_dir=p["min_points_per_dir"],
            dist_filter_tau=p["dist_filter_tau_s"],
            vy_filter_tau=p["vy_filter_tau_s"],
            control_rate=p["control_rate"],
            steer_enable=False,
            steer_yaw_max=0.0,
            steer_yaw_step=0.0,
            steer_yaw_half=0.0,
            steer_dist_percentile=0.0,
            steer_min_points=1,
        ))
        self._cmd_markers = CmdVelMarkers(self.cmd_marker_pub)
        self._debug_markers_pub = DebugMarkers(self.debug_marker_pub)

        self._mission_active: bool = False
        self._start_ready: bool = False
        self._last_start_ready_time_s: Optional[float] = None
        self._obstacles: Optional[np.ndarray] = None
        self._last_cloud_time: Optional[float] = None
        self._odom: Optional[Odometry] = None
        self._mode: str = "IDLE"
        self._last_cmd_body = np.zeros(3, dtype=np.float32)
        self._last_time = self.get_clock().now()
        self._last_log_time: Optional[float] = None
        self._dead_end_reverse_yaw: Optional[float] = None
        self._last_graph_reverse_yaw: Optional[float] = None
        self._reverse_turn_yaw: Optional[float] = None
        self._reverse_heading_hint: Optional[float] = None
        self._reverse_active_prev: bool = False

        self._p = p

        timer_period = 1.0 / max(p["control_rate"], 1e-3)
        self.create_timer(timer_period, self._on_timer)

        self.get_logger().info("LocalPathPlanner started (graph-following mode).")

    def _declare_parameters(self) -> None:
        decl = self.declare_parameter

        decl("control_rate", 20.0)
        decl("map_topic", "/map_pointcloud")
        decl("odom_topic", "/drone/state_estimate")
        decl("cmd_vel_topic", "/crazyflie/cmd_vel_user")
        decl("state_topic", "/local_planner/state")
        decl("debug_state_topic", "/local_planner/debug_state")
        decl("verbose", True)
        decl("debug_markers", True)

        decl("graph_topic", "/skeleton_graph/json")
        decl("graph_target_topic", "/navigation/graph_target")
        decl(
            "junction_potential_topic",
            "/navigation_events/junction_potential",
        )
        decl("graph_stale_timeout_s", 1.0)
        decl("edge_hysteresis_m", 0.3)
        decl("graph_lookahead_m", 0.8)
        decl("graph_heading_fit_distance_m", 1.5)
        decl("arm_direction_fit_min_m", 0.35)
        decl("arm_direction_fit_max_m", 2.0)
        decl("arm_direction_fit_min_span_m", 0.8)
        decl("graph_centering_enabled", True)
        decl("graph_center_k_y", 1.0)
        decl("graph_center_deadband_m", 0.03)
        decl("graph_center_vy_max", 0.15)
        decl("graph_center_filter_tau_s", 0.40)

        decl("junction_approach_distance_m", 1.5)
        decl("junction_slowdown_distance_m", 1.5)
        decl("uncommitted_junction_hold_distance_m", 0.4)
        decl("uncommitted_junction_hold_timeout_s", 3.0)
        decl("uncommitted_junction_creep_speed_scale", 0.15)
        decl("uncommitted_junction_hold_reset_s", 1.0)
        decl("junction_potential_slowdown_start", 0.1)
        decl("junction_potential_full_slowdown", 0.8)
        decl("junction_potential_min_speed_scale", 0.35)
        decl("junction_potential_timeout_s", 1.5)
        decl("leaf_stall_near_m", 0.30)
        decl("leaf_stall_timeout_s", 2.0)
        decl("leaf_stall_progress_m", 0.10)
        decl("min_arm_length_m", 0.3)
        decl("yaw_intent_num_bins", 36)
        decl("yaw_intent_potential_tau_s", 1.5)
        decl("yaw_intent_candidate_sigma_deg", 18.0)
        decl("yaw_intent_switch_margin", 0.35)
        decl("yaw_intent_slew_rate_rps", 0.9)
        decl("commit_support_angle_deg", 40.0)
        decl("commit_entry_distance_m", 0.8)
        decl("commit_entry_cross_track_m", 0.6)
        decl("commit_max_distance_m", 4.0)
        decl("commit_lock_acquisition_timeout_s", 3.0)
        decl("arm_direction_smoothing_tau_s", 0.35)
        decl("arm_direction_max_rate_deg_s", 25.0)
        decl("arm_direction_min_confidence", 0.65)

        decl("xy_max_range", 2.0)
        decl("horiz_pitch_half_deg", 30.0)
        decl("sensor_timeout_s", 0.8)
        decl("front_yaw_half_deg", 15.0)
        decl("back_yaw_half_deg", 30.0)
        decl("side_yaw_half_deg", 30.0)
        decl("up_pitch_min_deg", 30.0)
        decl("down_pitch_min_deg", 30.0)
        decl("side_dist_percentile", 10.0)
        decl("min_points_per_dir", 10)
        decl("dist_filter_tau_s", 0.30)
        decl("vy_filter_tau_s", 0.20)

        decl("d0_side_m", 1.0)
        decl("k_y", 1.0)
        decl("center_deadband_m", 0.10)
        decl("vy_max", 0.3)

        decl("d0_ceiling_m", 0.8)
        decl("d0_floor_m", 1.0)
        decl("k_floor", 1.0)
        decl("k_ceiling", 1.0)
        decl("vz_max", 0.5)
        decl("vertical_dist_percentile", 15.0)
        decl("vz_filter_tau_s", 0.30)
        decl("vz_deadband_mps", 0.02)

        decl("front_block_dist_m", 0.35)
        decl("front_clear_dist_m", 0.75)
        decl("front_block_hysteresis_m", 0.20)
        decl("front_block_width_margin_m", 0.075)
        decl("front_block_dynamic_max_m", 1.2)
        decl("front_width_filter_tau_s", 0.25)
        decl("front_stop_dist_m", 0.15)
        decl("front_slow_dist_m", 0.5)
        decl("safety_max_decel_mps2", 2.0)
        decl("safety_reaction_time_s", 0.25)
        decl("vehicle_radius_m", 0.15)
        decl("safety_stop_margin_m", 0.10)
        decl("safety_slow_margin_m", 0.40)
        decl("open_dist_m", 1.5)
        decl("block_debounce_s", 0.25)
        decl("dead_end_time_s", 1.0)
        decl("turn_search_max_yaw_deg", 70.0)

        decl("v_nom", 0.6)
        decl("vx_max", 0.6)

        decl("yaw_kp", 0.55)
        decl("yaw_rate_max_rps", 0.4)
        decl("yaw_align_gate_rad", 0.4)

        decl("max_forward_accel_mps2", 0.2)
        decl("max_lateral_accel_mps2", 0.25)
        decl("max_lin_acc_mps2", 0.5)

    def _fetch_parameters(self) -> dict:
        g = lambda name: self.get_parameter(name).value  # noqa: E731
        return {
            "control_rate": float(g("control_rate")),
            "map_topic": str(g("map_topic")),
            "odom_topic": str(g("odom_topic")),
            "cmd_vel_topic": str(g("cmd_vel_topic")),
            "state_topic": str(g("state_topic")),
            "debug_state_topic": str(g("debug_state_topic")),
            "verbose": bool(g("verbose")),
            "debug_markers": bool(g("debug_markers")),
            "graph_topic": str(g("graph_topic")),
            "graph_target_topic": str(g("graph_target_topic")),
            "junction_potential_topic": str(g("junction_potential_topic")),
            "graph_stale_timeout_s": float(g("graph_stale_timeout_s")),
            "edge_hysteresis_m": float(g("edge_hysteresis_m")),
            "graph_lookahead_m": float(g("graph_lookahead_m")),
            "graph_heading_fit_distance_m": float(
                g("graph_heading_fit_distance_m")
            ),
            "arm_direction_fit_min_m": float(g("arm_direction_fit_min_m")),
            "arm_direction_fit_max_m": float(g("arm_direction_fit_max_m")),
            "arm_direction_fit_min_span_m": float(
                g("arm_direction_fit_min_span_m")
            ),
            "graph_centering_enabled": bool(g("graph_centering_enabled")),
            "graph_center_k_y": float(g("graph_center_k_y")),
            "graph_center_deadband_m": float(g("graph_center_deadband_m")),
            "graph_center_vy_max": float(g("graph_center_vy_max")),
            "graph_center_filter_tau_s": float(g("graph_center_filter_tau_s")),
            "junction_approach_distance_m": float(g("junction_approach_distance_m")),
            "junction_slowdown_distance_m": float(g("junction_slowdown_distance_m")),
            "uncommitted_junction_hold_distance_m": float(
                g("uncommitted_junction_hold_distance_m")
            ),
            "uncommitted_junction_hold_timeout_s": float(
                g("uncommitted_junction_hold_timeout_s")
            ),
            "uncommitted_junction_creep_speed_scale": float(
                g("uncommitted_junction_creep_speed_scale")
            ),
            "uncommitted_junction_hold_reset_s": float(
                g("uncommitted_junction_hold_reset_s")
            ),
            "junction_potential_slowdown_start": float(
                g("junction_potential_slowdown_start")
            ),
            "junction_potential_full_slowdown": float(
                g("junction_potential_full_slowdown")
            ),
            "junction_potential_min_speed_scale": float(
                g("junction_potential_min_speed_scale")
            ),
            "junction_potential_timeout_s": float(
                g("junction_potential_timeout_s")
            ),
            "leaf_stall_near_m": float(g("leaf_stall_near_m")),
            "leaf_stall_timeout_s": float(g("leaf_stall_timeout_s")),
            "leaf_stall_progress_m": float(g("leaf_stall_progress_m")),
            "min_arm_length_m": float(g("min_arm_length_m")),
            "yaw_intent_num_bins": int(g("yaw_intent_num_bins")),
            "yaw_intent_potential_tau_s": float(g("yaw_intent_potential_tau_s")),
            "yaw_intent_candidate_sigma_deg": float(g("yaw_intent_candidate_sigma_deg")),
            "yaw_intent_switch_margin": float(g("yaw_intent_switch_margin")),
            "yaw_intent_slew_rate_rps": float(g("yaw_intent_slew_rate_rps")),
            "commit_support_angle_deg": float(g("commit_support_angle_deg")),
            "commit_entry_distance_m": float(g("commit_entry_distance_m")),
            "commit_entry_cross_track_m": float(
                g("commit_entry_cross_track_m")
            ),
            "commit_max_distance_m": float(g("commit_max_distance_m")),
            "commit_lock_acquisition_timeout_s": float(
                g("commit_lock_acquisition_timeout_s")
            ),
            "arm_direction_smoothing_tau_s": float(
                g("arm_direction_smoothing_tau_s")
            ),
            "arm_direction_max_rate_deg_s": float(
                g("arm_direction_max_rate_deg_s")
            ),
            "arm_direction_min_confidence": float(
                g("arm_direction_min_confidence")
            ),
            "xy_max_range": float(g("xy_max_range")),
            "horiz_pitch_half_deg": float(g("horiz_pitch_half_deg")),
            "sensor_timeout_s": float(g("sensor_timeout_s")),
            "front_yaw_half_deg": float(g("front_yaw_half_deg")),
            "back_yaw_half_deg": float(g("back_yaw_half_deg")),
            "side_yaw_half_deg": float(g("side_yaw_half_deg")),
            "up_pitch_min_deg": float(g("up_pitch_min_deg")),
            "down_pitch_min_deg": float(g("down_pitch_min_deg")),
            "side_dist_percentile": float(g("side_dist_percentile")),
            "min_points_per_dir": int(g("min_points_per_dir")),
            "dist_filter_tau_s": float(g("dist_filter_tau_s")),
            "vy_filter_tau_s": float(g("vy_filter_tau_s")),
            "d0_side_m": float(g("d0_side_m")),
            "k_y": float(g("k_y")),
            "center_deadband_m": float(g("center_deadband_m")),
            "vy_max": float(g("vy_max")),
            "d0_ceiling_m": float(g("d0_ceiling_m")),
            "d0_floor_m": float(g("d0_floor_m")),
            "k_floor": float(g("k_floor")),
            "k_ceiling": float(g("k_ceiling")),
            "vz_max": float(g("vz_max")),
            "vertical_dist_percentile": float(g("vertical_dist_percentile")),
            "vz_filter_tau_s": float(g("vz_filter_tau_s")),
            "vz_deadband_mps": float(g("vz_deadband_mps")),
            "front_block_dist_m": float(g("front_block_dist_m")),
            "front_clear_dist_m": float(g("front_clear_dist_m")),
            "front_block_hysteresis_m": float(g("front_block_hysteresis_m")),
            "front_block_width_margin_m": float(g("front_block_width_margin_m")),
            "front_block_dynamic_max_m": float(g("front_block_dynamic_max_m")),
            "front_width_filter_tau_s": float(g("front_width_filter_tau_s")),
            "front_stop_dist_m": float(g("front_stop_dist_m")),
            "front_slow_dist_m": float(g("front_slow_dist_m")),
            "safety_max_decel_mps2": float(g("safety_max_decel_mps2")),
            "safety_reaction_time_s": float(g("safety_reaction_time_s")),
            "vehicle_radius_m": float(g("vehicle_radius_m")),
            "safety_stop_margin_m": float(g("safety_stop_margin_m")),
            "safety_slow_margin_m": float(g("safety_slow_margin_m")),
            "open_dist_m": float(g("open_dist_m")),
            "block_debounce_s": float(g("block_debounce_s")),
            "dead_end_time_s": float(g("dead_end_time_s")),
            "turn_search_max_yaw_deg": float(g("turn_search_max_yaw_deg")),
            "v_nom": float(g("v_nom")),
            "vx_max": float(g("vx_max")),
            "yaw_kp": float(g("yaw_kp")),
            "yaw_rate_max_rps": float(g("yaw_rate_max_rps")),
            "yaw_align_gate_rad": float(g("yaw_align_gate_rad")),
            "max_forward_accel_mps2": float(g("max_forward_accel_mps2")),
            "max_lateral_accel_mps2": float(g("max_lateral_accel_mps2")),
            "max_lin_acc_mps2": float(g("max_lin_acc_mps2")),
        }

    def _on_cloud(self, msg: PointCloud2) -> None:
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        arr = np.array(list(gen))
        if arr.size == 0:
            self._obstacles = None
        elif arr.dtype.fields is not None and {"x", "y", "z"}.issubset(arr.dtype.fields.keys()):
            self._obstacles = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)
        elif arr.ndim == 2 and arr.shape[1] >= 3:
            self._obstacles = arr[:, :3].astype(np.float32, copy=False)
        else:
            self._obstacles = None
        self._last_cloud_time = self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_graph_json(self, msg: String) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._graph_tracker.on_graph_json(msg.data, now_s)

    def _on_junction_potential(self, msg: Float32) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        self._junction_potential_speed.update(float(msg.data), now_s)

    def _on_graph_target(self, msg: String) -> None:
        old = self._arm_commit.active_intent
        old_key = None if old is None else (old.event_id, old.mode, old.selected_arm_angle_rad)
        old_reverse = self._arm_commit.reverse_requested
        action = self._arm_commit.on_graph_target(msg.data)
        new = self._arm_commit.active_intent
        new_key = None if new is None else (new.event_id, new.mode, new.selected_arm_angle_rad)
        if action == "commit" and new is not None:
            self._committed_arm_lock.activate(new)
        elif action == "clear":
            self._committed_arm_lock.request_release()
        elif action == "reverse":
            self._committed_arm_lock.reset()
            reverse_heading = self._reverse_heading_for_handover()
            self._reverse_turn_yaw = reverse_heading
            self._reverse_heading_hint = reverse_heading
            self._dead_end_reverse_yaw = reverse_heading
            self._reverse_active_prev = True
            self._yaw_intent.reset()
        if new_key != old_key or old_reverse != self._arm_commit.reverse_requested:
            self._leaf_progress.reset()
            if self._safety_ctrl.graph_recovery:
                self._safety_ctrl.reset()

    def _on_mission_start(self, msg: Bool) -> None:
        if not msg.data:
            return
        if self._mission_active:
            return
        self._mission_active = True
        self._leaf_progress.reset()
        now_s = self.get_clock().now().nanoseconds * 1e-9
        ready_is_current = (
            self._start_ready
            and self._last_start_ready_time_s is not None
            and 0.0 <= now_s - self._last_start_ready_time_s <= 1.0
        )
        self._start_ready = bool(ready_is_current)
        self._mode = "IDLE" if self._start_ready else "WAIT_START"
        self._safety_ctrl.reset()
        self._height_ctrl.reset()
        self._graph_center.reset()
        self._sectors.reset_filters()
        self._yaw_intent.reset()
        self._arm_commit.reset()
        self._committed_arm_lock.reset()
        self._junction_potential_speed.reset()
        self._last_cmd_body[:] = 0.0
        self._dead_end_reverse_yaw = None
        self._last_graph_reverse_yaw = None
        self._reverse_turn_yaw = None
        self._reverse_heading_hint = None
        self._reverse_active_prev = False
        self._publish_state()
        self._publish_zero_cmd()
        self._publish_debug_state(self._idle_debug_payload(vx_gate_reason="waiting_start"))

    def _on_start_ready(self, msg: Bool) -> None:
        self._start_ready = bool(msg.data)
        self._last_start_ready_time_s = self.get_clock().now().nanoseconds * 1e-9
        if self._mission_active and self._start_ready and self._mode == "WAIT_START":
            self._mode = "IDLE"
            self._publish_state()

    def _on_mission_stop(self, msg: Bool) -> None:
        if not msg.data:
            return
        self._mission_active = False
        self._leaf_progress.reset()
        self._start_ready = False
        self._last_start_ready_time_s = None
        self._mode = "IDLE"
        self._safety_ctrl.reset()
        self._height_ctrl.reset()
        self._graph_center.reset()
        self._sectors.reset_filters()
        self._yaw_intent.reset()
        self._arm_commit.reset()
        self._committed_arm_lock.reset()
        self._junction_potential_speed.reset()
        self._last_cmd_body[:] = 0.0
        self._dead_end_reverse_yaw = None
        self._last_graph_reverse_yaw = None
        self._reverse_turn_yaw = None
        self._reverse_heading_hint = None
        self._reverse_active_prev = False
        self._publish_zero_cmd()
        self._publish_state()
        self._publish_debug_state(self._idle_debug_payload(vx_gate_reason="inactive"))

    def _on_timer(self) -> None:
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        dt = max(
            (now - self._last_time).nanoseconds * 1e-9,
            1.0 / (10.0 * max(self._p["control_rate"], 1e-3)),
        )
        self._last_time = now
        p = self._p

        if not self._mission_active:
            return

        if not self._start_ready:
            self._leaf_progress.reset()
            self._mode = "WAIT_START"
            self._last_cmd_body[:] = 0.0
            self._publish_zero_cmd()
            self._publish_state()
            self._publish_debug_state(
                self._idle_debug_payload(vx_gate_reason="waiting_start")
            )
            return

        if self._odom is None:
            self._leaf_progress.reset()
            self._publish_zero_cmd()
            self._publish_state()
            self._publish_debug_state(self._idle_debug_payload(vx_gate_reason="waiting_odom"))
            return

        pose = self._odom.pose.pose
        pos_w = np.array(
            [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
            dtype=np.float32,
        )
        q = pose.orientation
        yaw = _yaw_from_quaternion(float(q.x), float(q.y), float(q.z), float(q.w))
        R_bw = _quaternion_to_rotation_matrix(float(q.x), float(q.y), float(q.z), float(q.w))
        drone_xy = pos_w[:2]

        cloud_stale = (
            self._obstacles is None
            or self._last_cloud_time is None
            or (now_s - self._last_cloud_time) > p["sensor_timeout_s"]
        )

        if cloud_stale:
            d_front = d_back = d_left = d_right = d_up = d_down = None
            vy_rep = 0.0
        else:
            sector_out = self._sectors.compute(pos_w, R_bw, self._obstacles, debug=p["debug_markers"])
            d_front = sector_out.distances["front"]
            d_back = sector_out.distances["back"]
            d_left = sector_out.distances["left"]
            d_right = sector_out.distances["right"]
            d_up = sector_out.distances["up"]
            d_down = sector_out.distances["down"]
            vy_rep = sector_out.vy_rep
        obstacle_clearances_m = {
            "front": d_front,
            "back": d_back,
            "left": d_left,
            "right": d_right,
            "up": d_up,
            "down": d_down,
        }
        # Missing cloud is not open space; hold until a fresh scan arrives.
        if cloud_stale:
            self._mode = "SENSOR_HOLD"
            self._leaf_progress.reset()
            self._last_cmd_body[:] = 0.0
            tw = Twist()
            self._publish_command(tw)
            self._cmd_markers.publish(self._odom, tw, yaw_rate_max=p["yaw_rate_max_rps"])
            self._publish_state()
            self._publish_debug_state(self._idle_debug_payload(vx_gate_reason="sensor_stale"))
            return

        if not cloud_stale:
            vz_cmd = self._height_ctrl.compute(pos_w, R_bw, self._obstacles)
        else:
            vz_cmd = 0.0

        reverse_active = self._arm_commit.reverse_requested
        if reverse_active != self._reverse_active_prev:
            self._yaw_intent.reset()
            self._reverse_heading_hint = None
            if reverse_active:
                self._reverse_turn_yaw = (
                    self._last_graph_reverse_yaw
                    if self._last_graph_reverse_yaw is not None
                    else _wrap(yaw + math.pi)
                )
            else:
                self._reverse_turn_yaw = None
            self._reverse_active_prev = reverse_active
        if not reverse_active:
            self._reverse_heading_hint = None
            self._reverse_turn_yaw = None

        if self._dead_end_reverse_yaw is not None and abs(
            _wrap(self._dead_end_reverse_yaw - yaw)
        ) < 0.3:
            self._dead_end_reverse_yaw = None
            self._safety_ctrl.reset()
            self._leaf_progress.reset()
            self._yaw_intent.reset()
            if reverse_active:
                self._reverse_heading_hint = yaw
                self._reverse_turn_yaw = None
        tracking_heading = self._yaw_intent.reference_heading
        if reverse_active:
            if self._reverse_turn_yaw is not None:
                tracking_heading = self._reverse_turn_yaw
            elif self._reverse_heading_hint is not None:
                tracking_heading = self._reverse_heading_hint
            else:
                base_heading = yaw if tracking_heading is None else tracking_heading
                tracking_heading = _wrap(base_heading + math.pi)
        if self._dead_end_reverse_yaw is not None:
            tracking_heading = self._dead_end_reverse_yaw
        gs = self._graph_tracker.update(
            drone_xy,
            yaw if tracking_heading is None else tracking_heading,
            now_s,
        )

        if not gs.valid:
            self._leaf_progress.reset()
            self._mode = "IDLE"
            self._last_cmd_body[:] = 0.0
            tw = Twist()
            tw.linear.z = float(vz_cmd)
            self._publish_command(tw)
            self._cmd_markers.publish(self._odom, tw, yaw_rate_max=p["yaw_rate_max_rps"])
            self._publish_state()
            self._publish_debug_state(self._idle_debug_payload(vx_gate_reason="no_graph"))
            self._publish_debug(gs)
            return

        if not reverse_active:
            self._last_graph_reverse_yaw = self._reverse_yaw_from_graph(gs, drone_xy)
            entry_confirmed = self._committed_arm_lock.update(
                drone_xy,
                gs,
                dt,
            )
            if entry_confirmed:
                self._junction_potential_speed.suppress_until_clear()

        committed_intent = (
            None
            if reverse_active
            else self._committed_arm_lock.active_intent
        )
        if reverse_active:
            locked_reverse_yaw = self._reverse_turn_yaw
            if locked_reverse_yaw is not None:
                self._reverse_heading_hint = locked_reverse_yaw
                intent = YawIntentOutput(
                    yaw_ref_rad=locked_reverse_yaw,
                    gate_heading_rad=locked_reverse_yaw,
                    selected_arm_angle_rad=None,
                    speed_scale=1.0,
                    committed=False,
                )
                unlock_turn_rad = min(0.3, max(0.15, float(p["yaw_align_gate_rad"])))
                if abs(_wrap(locked_reverse_yaw - yaw)) <= unlock_turn_rad:
                    self._reverse_turn_yaw = None
            else:
                self._reverse_heading_hint = gs.current_heading_rad
                intent = YawIntentOutput(
                    yaw_ref_rad=gs.current_heading_rad,
                    gate_heading_rad=gs.current_heading_rad,
                    selected_arm_angle_rad=None,
                    speed_scale=1.0,
                    committed=False,
                )
        else:
            intent = self._yaw_intent.update(gs, committed_intent, dt)

        graph_has_continuation = self._graph_has_continuation(gs)
        dead_end_candidate = bool(gs.approaching_leaf or not graph_has_continuation)
        gate_heading = intent.yaw_ref_rad if intent.gate_heading_rad is None else intent.gate_heading_rad
        aligning = (
            p["yaw_align_gate_rad"] > 0.0
            and abs(_wrap(gate_heading - yaw)) > p["yaw_align_gate_rad"]
        )
        graph_stalled = self._leaf_progress.update(
            eligible=(
                gs.approaching_leaf and not graph_has_continuation
                and not reverse_active
                and not aligning and self._reverse_turn_yaw is None
                and self._dead_end_reverse_yaw is None
                and self._safety_ctrl.mode in ("NONE", "BLOCKED")
                and p["v_nom"] > 0.0 and p["vx_max"] > 0.0
            ),
            remaining_m=gs.ahead_path_length_m,
            position_xy=drone_xy, heading_rad=gs.tangent_rad, now_s=now_s,
        )
        safety = self._safety_ctrl.update(
            d_front,
            d_left,
            d_right,
            graph_has_continuation,
            dead_end_candidate,
            yaw,
            now_s,
            forward_speed_mps=float(max(0.0, self._last_cmd_body[0])),
            graph_stalled=graph_stalled,
        )

        prev_mode = self._mode

        # Hold DEAD_END until the reverse heading is reached.
        if self._dead_end_reverse_yaw is not None:
            self._mode = "DEAD_END"

        if self._dead_end_reverse_yaw is None:
            if reverse_active:
                self._mode = "REVERSE_FOLLOW"
            elif self._safety_ctrl.graph_recovery and safety.mode != "NONE":
                self._mode = safety.mode
            elif safety.mode == "DEAD_END":
                self._mode = "DEAD_END"
            elif safety.mode == "TURN_TO_OPENING":
                self._mode = "TURN_TO_OPENING"
            elif safety.mode == "BLOCKED":
                self._mode = "BLOCKED"
            elif intent.committed:
                self._mode = "COMMITTED_ARM"
            else:
                self._mode = "FOLLOW_INTENT"

        if (
            self._mode == "DEAD_END"
            and prev_mode != "DEAD_END"
            and self._dead_end_reverse_yaw is None
        ):
            behind_id = gs.edge_id[0] if gs.ahead_node_id == gs.edge_id[1] else gs.edge_id[1]
            behind_xy = self._graph_tracker.get_node_xy(behind_id)
            if behind_xy is not None:
                d = behind_xy - drone_xy
                self._dead_end_reverse_yaw = float(math.atan2(float(d[1]), float(d[0])))
            else:
                self._dead_end_reverse_yaw = _wrap(gs.tangent_rad + math.pi)

        vx_cmd = 0.0
        vy_cmd = 0.0
        yaw_rate_cmd = 0.0
        yaw_err: Optional[float] = None
        graph_body_y_error_m = _body_y_error_to_target(gs.projection_xy, drone_xy, yaw)
        graph_center = GraphCenteringOutput(body_y_error_m=graph_body_y_error_m)
        junction_potential_speed_scale = (
            self._junction_potential_speed.speed_scale(now_s)
        )
        vx_blocked_by_yaw = False
        vx_blocked_by_obstacle = self._mode in ("DEAD_END", "BLOCKED", "TURN_TO_OPENING")

        if self._mode == "DEAD_END":
            reverse = self._dead_end_reverse_yaw if self._dead_end_reverse_yaw is not None else yaw
            yaw_rate_cmd = _yaw_rate(reverse, yaw, p["yaw_kp"], p["yaw_rate_max_rps"])
            self._graph_center.reset()

        elif self._mode == "BLOCKED":
            self._graph_center.reset()

        elif self._mode == "TURN_TO_OPENING":
            yaw_rate_cmd = safety.yaw_rate_override if safety.yaw_rate_override is not None else 0.0
            self._graph_center.reset()

        elif self._mode == "REVERSE_FOLLOW":
            vx_cmd = p["v_nom"] * safety.vx_scale
            if gs.approaching_leaf:
                leaf_scale = min(
                    1.0,
                    gs.dist_to_ahead_node_m
                    / max(p["junction_approach_distance_m"], 0.01),
                )
                vx_cmd *= leaf_scale

            yaw_gate_heading = (
                intent.yaw_ref_rad
                if intent.gate_heading_rad is None
                else intent.gate_heading_rad
            )
            yaw_err = abs(_wrap(yaw_gate_heading - yaw))
            gate_rad = p["yaw_align_gate_rad"]
            if gate_rad > 0.0 and yaw_err > gate_rad:
                vx_cmd = 0.0
                vx_blocked_by_yaw = True

            if safety.collision_stop:
                vx_cmd = 0.0
                vx_blocked_by_obstacle = True
            graph_center = self._graph_center.compute(
                body_y_error_m=graph_body_y_error_m,
                dt=dt,
                active=(
                    not self._committed_arm_lock.branch_locked
                    and not gs.approaching_junction
                    and not vx_blocked_by_yaw
                    and not vx_blocked_by_obstacle
                ),
            )
            vy_cmd = vy_rep + graph_center.vy_correction
            yaw_rate_cmd = _yaw_rate(intent.yaw_ref_rad, yaw, p["yaw_kp"], p["yaw_rate_max_rps"])

        else:
            vx_cmd = (
                p["v_nom"]
                * intent.speed_scale
                * junction_potential_speed_scale
                * safety.vx_scale
            )
            if gs.approaching_leaf:
                leaf_scale = min(
                    1.0,
                    gs.dist_to_ahead_node_m
                    / max(p["junction_approach_distance_m"], 0.01),
                )
                vx_cmd *= leaf_scale

            yaw_gate_heading = (
                intent.yaw_ref_rad
                if intent.gate_heading_rad is None
                else intent.gate_heading_rad
            )
            yaw_err = abs(_wrap(yaw_gate_heading - yaw))
            gate_rad = p["yaw_align_gate_rad"]
            if gate_rad > 0.0 and yaw_err > gate_rad:
                vx_cmd = 0.0
                vx_blocked_by_yaw = True

            if safety.collision_stop:
                vx_cmd = 0.0
                vx_blocked_by_obstacle = True
            graph_center = self._graph_center.compute(
                body_y_error_m=graph_body_y_error_m,
                dt=dt,
                active=(
                    not self._committed_arm_lock.branch_locked
                    and not gs.approaching_junction
                    and not vx_blocked_by_yaw
                    and not vx_blocked_by_obstacle
                ),
            )
            vy_cmd = vy_rep + graph_center.vy_correction
            yaw_rate_cmd = _yaw_rate(intent.yaw_ref_rad, yaw, p["yaw_kp"], p["yaw_rate_max_rps"])

        vx_cmd = float(np.clip(vx_cmd, 0.0, p["vx_max"]))
        vy_cmd = float(np.clip(vy_cmd, -p["vy_max"], p["vy_max"]))
        vz_cmd = float(np.clip(vz_cmd, -p["vz_max"], p["vz_max"]))

        v_des = np.array([vx_cmd, vy_cmd, vz_cmd], dtype=np.float32)
        max_dv = p["max_lin_acc_mps2"] * dt
        dv = np.clip(v_des - self._last_cmd_body, -max_dv, max_dv)
        # Cap only forward rise; braking and hard gates stay immediate.
        max_forward_rise = max(0.0, p["max_forward_accel_mps2"]) * dt
        dv[0] = min(float(dv[0]), max_forward_rise)

        # Unwind a vy reversal to zero first, then rise with the gentler lateral limit.
        vy_prev = float(self._last_cmd_body[1])
        vy_des = float(v_des[1])
        if vy_prev * vy_des < 0.0 and abs(float(dv[1])) >= abs(vy_prev):
            dv[1] = -vy_prev
        elif abs(vy_des) > abs(vy_prev):
            max_lateral_rise = max(0.0, p["max_lateral_accel_mps2"]) * dt
            dv[1] = float(np.clip(dv[1], -max_lateral_rise, max_lateral_rise))
        self._last_cmd_body += dv
        if vx_blocked_by_yaw or vx_blocked_by_obstacle:
            self._last_cmd_body[0] = 0.0
        vx_cmd, vy_cmd, vz_cmd = float(self._last_cmd_body[0]), float(self._last_cmd_body[1]), float(self._last_cmd_body[2])

        tw = Twist()
        tw.linear.x = vx_cmd
        tw.linear.y = vy_cmd
        tw.linear.z = vz_cmd
        tw.angular.z = float(yaw_rate_cmd)
        self._publish_command(tw)
        self._cmd_markers.publish(self._odom, tw, yaw_rate_max=p["yaw_rate_max_rps"])
        debug_payload = self._build_debug_payload(
            gs=gs,
            intent=intent,
            committed_intent=committed_intent,
            safety=safety,
            clearances_m=obstacle_clearances_m,
            d_front=d_front,
            yaw_error_rad=yaw_err,
            graph_body_y_error_m=graph_center.body_y_error_m,
            vy_graph_mps=graph_center.vy_correction,
            vy_rep_mps=vy_rep,
            vx_blocked_by_yaw=vx_blocked_by_yaw,
            vx_blocked_by_obstacle=vx_blocked_by_obstacle,
            cmd_vx_mps=tw.linear.x,
            cmd_vy_mps=tw.linear.y,
            cmd_vz_mps=tw.linear.z,
            cmd_yaw_rate_rps=tw.angular.z,
            junction_potential_speed_scale=junction_potential_speed_scale,
        )
        self._publish_state()
        self._publish_debug_state(debug_payload)
        self._publish_debug(
            gs,
            speculative_arm_angle=debug_payload["speculative_arm_angle_rad"],
        )

        if p["verbose"]:
            self._maybe_log(now_s, d_front, vx_cmd, vy_cmd, vz_cmd)

    def _publish_zero_cmd(self) -> None:
        tw = Twist()
        self._publish_command(tw)
        self._cmd_markers.publish(self._odom, tw, yaw_rate_max=self._p["yaw_rate_max_rps"])

    def _publish_command(self, msg: Twist) -> None:
        self.cmd_pub.publish(msg)

    def _publish_state(self) -> None:
        msg = String()
        msg.data = self._mode
        self.state_pub.publish(msg)

    @staticmethod
    def _empty_clearance_payload() -> dict[str, Optional[float]]:
        return {
            "front": None,
            "back": None,
            "left": None,
            "right": None,
            "up": None,
            "down": None,
        }

    def _clearance_payload(
        self, clearances: Optional[dict[str, Optional[float]]] = None
    ) -> dict[str, Optional[float]]:
        payload = self._empty_clearance_payload()
        if clearances is None:
            return payload
        for axis in payload:
            value = clearances.get(axis)
            payload[axis] = None if value is None else float(value)
        return payload

    def _publish_debug_state(self, payload: dict) -> None:
        msg = String()
        msg.data = json.dumps(payload)
        self.debug_state_pub.publish(msg)

    def _idle_debug_payload(self, *, vx_gate_reason: str) -> dict:
        return {
            "planner_mode": self._mode,
            "mission_active": bool(self._mission_active),
            "graph_valid": False,
            "graph_continuation_available": False,
            "dead_end_candidate": False,
            "graph_ahead_path_length_m": None,
            "graph_viable_arm_count": 0,
            "front_block_threshold_m": float(self._p["front_block_dist_m"]),
            "front_clear_threshold_m": float(self._p["front_clear_dist_m"]),
            "front_stop_threshold_m": float(self._p["front_stop_dist_m"]),
            "front_slow_threshold_m": float(self._p["front_slow_dist_m"]),
            "passage_width_estimate_m": None,
            "safety_mode": "NONE",
            "intent_source": "reverse_graph" if self._arm_commit.reverse_requested else "none",
            "yaw_ref_rad": None,
            "yaw_gate_target_rad": None,
            "cmd_vx_mps": 0.0,
            "cmd_vy_mps": 0.0,
            "cmd_vz_mps": 0.0,
            "cmd_yaw_rate_rps": 0.0,
            "selected_arm_angle_rad": None,
            "speculative_arm_angle_rad": None,
            "speculative_arm_length_m": None,
            "speculative_anchor_xy": None,
            "obstacle_clearance_m": self._empty_clearance_payload(),
            "obstacle_speed_limit_mps": None,
            "front_distance_m": None,
            "yaw_error_rad": None,
            "graph_lateral_offset_m": None,
            "graph_body_y_error_m": None,
            "vy_graph_mps": 0.0,
            "vy_rep_mps": 0.0,
            "vx_blocked_by_yaw": False,
            "vx_blocked_by_obstacle": False,
            "vx_gate_reason": vx_gate_reason,
            "junction_potential": self._junction_potential_speed.potential,
            "junction_potential_speed_scale": 1.0,
            "committed": None,
            "commit_branch_locked": self._committed_arm_lock.branch_locked,
            "commit_entry_confirmed": self._committed_arm_lock.entry_confirmed,
            "commit_release_requested": self._committed_arm_lock.release_requested,
            "reverse_requested": bool(self._arm_commit.reverse_requested),
            "reverse_mode": self._arm_commit.reverse_mode,
        }

    def _build_debug_payload(
        self,
        *,
        gs: Optional[GraphTrackingState],
        intent: Optional[YawIntentOutput],
        committed_intent: Optional[CommittedArmIntent],
        safety: Optional[SafetyOutput],
        clearances_m: Optional[dict[str, Optional[float]]],
        d_front: Optional[float],
        yaw_error_rad: Optional[float],
        graph_body_y_error_m: Optional[float],
        vy_graph_mps: float,
        vy_rep_mps: float,
        vx_blocked_by_yaw: bool,
        vx_blocked_by_obstacle: bool,
        cmd_vx_mps: float,
        cmd_vy_mps: float,
        cmd_vz_mps: float,
        cmd_yaw_rate_rps: float,
        junction_potential_speed_scale: float,
    ) -> dict:
        graph_valid = gs is not None and gs.valid
        speculative_arm_angle: Optional[float] = None
        speculative_arm_length_m: Optional[float] = None
        speculative_anchor_xy: Optional[list[float]] = None
        obstacle_clearance_m = self._clearance_payload(clearances_m)
        obstacle_speed_limit_mps = (
            float(self._p["v_nom"] * safety.vx_scale) if safety is not None else None
        )

        if (
            graph_valid
            and intent is not None
            and committed_intent is None
            and intent.selected_arm_angle_rad is not None
        ):
            candidate = self._selected_candidate(gs, intent.selected_arm_angle_rad)
            speculative_arm_angle = float(intent.selected_arm_angle_rad)
            if candidate is not None:
                speculative_arm_length_m = float(candidate.edge_length_m)
            if gs.approaching_junction:
                anchor_xy = self._graph_tracker.get_node_xy(gs.ahead_node_id)
                if anchor_xy is not None:
                    speculative_anchor_xy = [float(anchor_xy[0]), float(anchor_xy[1])]

        if committed_intent is not None:
            intent_source = "committed"
        elif self._arm_commit.reverse_requested:
            intent_source = "reverse_graph"
        elif speculative_arm_angle is not None:
            intent_source = "speculative"
        elif graph_valid:
            intent_source = "track_edge"
        else:
            intent_source = "none"

        committed_payload = None
        if committed_intent is not None:
            committed_payload = {
                "event_id": committed_intent.event_id,
                "mode": committed_intent.mode,
                "event_center_xy": [
                    float(committed_intent.event_center_xy[0]),
                    float(committed_intent.event_center_xy[1]),
                ],
                "gate_radius_m": float(committed_intent.gate_radius_m),
                "selected_arm_angle_rad": float(committed_intent.selected_arm_angle_rad),
                "branch_locked": bool(committed_intent.branch_locked),
                "arm_identity_angle_rad": (
                    None
                    if committed_intent.arm_identity_angle_rad is None
                    else float(committed_intent.arm_identity_angle_rad)
                ),
            }

        return {
            "planner_mode": self._mode,
            "mission_active": bool(self._mission_active),
            "graph_valid": bool(graph_valid),
            "graph_continuation_available": (
                bool(self._graph_has_continuation(gs)) if graph_valid else False
            ),
            "dead_end_candidate": (
                bool(gs.approaching_leaf or not self._graph_has_continuation(gs))
                if graph_valid
                else False
            ),
            "graph_ahead_path_length_m": (
                float(gs.ahead_path_length_m) if gs is not None and gs.valid else None
            ),
            "graph_viable_arm_count": (
                int(self._graph_viable_arm_count(gs)) if graph_valid else 0
            ),
            "front_block_threshold_m": float(self._safety_ctrl.active_front_block_dist_m),
            "front_clear_threshold_m": float(self._safety_ctrl.active_front_clear_dist_m),
            "front_stop_threshold_m": float(self._safety_ctrl.active_stop_dist_m),
            "front_slow_threshold_m": float(self._safety_ctrl.active_slow_dist_m),
            "passage_width_estimate_m": (
                None
                if self._safety_ctrl.passage_width_estimate_m is None
                else float(self._safety_ctrl.passage_width_estimate_m)
            ),
            "safety_mode": safety.mode if safety is not None else "NONE",
            "graph_recovery_active": self._safety_ctrl.graph_recovery,
            "leaf_stall_wait_s": self._leaf_progress.wait_s,
            "intent_source": intent_source,
            "yaw_ref_rad": float(intent.yaw_ref_rad) if intent is not None else None,
            "yaw_gate_target_rad": (
                None
                if intent is None
                else float(
                    intent.yaw_ref_rad
                    if intent.gate_heading_rad is None
                    else intent.gate_heading_rad
                )
            ),
            "cmd_vx_mps": float(cmd_vx_mps),
            "cmd_vy_mps": float(cmd_vy_mps),
            "cmd_vz_mps": float(cmd_vz_mps),
            "cmd_yaw_rate_rps": float(cmd_yaw_rate_rps),
            "selected_arm_angle_rad": (
                float(intent.selected_arm_angle_rad)
                if intent is not None and intent.selected_arm_angle_rad is not None
                else None
            ),
            "speculative_arm_angle_rad": speculative_arm_angle,
            "speculative_arm_length_m": speculative_arm_length_m,
            "speculative_anchor_xy": speculative_anchor_xy,
            "obstacle_clearance_m": obstacle_clearance_m,
            "obstacle_speed_limit_mps": obstacle_speed_limit_mps,
            "front_distance_m": None if d_front is None else float(d_front),
            "yaw_error_rad": None if yaw_error_rad is None else float(yaw_error_rad),
            "graph_lateral_offset_m": (
                float(gs.lateral_offset_m) if gs is not None and gs.valid else None
            ),
            "graph_body_y_error_m": (
                None if graph_body_y_error_m is None else float(graph_body_y_error_m)
            ),
            "vy_graph_mps": float(vy_graph_mps),
            "vy_rep_mps": float(vy_rep_mps),
            "vx_blocked_by_yaw": bool(vx_blocked_by_yaw),
            "vx_blocked_by_obstacle": bool(vx_blocked_by_obstacle),
            "vx_gate_reason": self._vx_gate_reason(vx_blocked_by_yaw, vx_blocked_by_obstacle),
            "junction_potential": self._junction_potential_speed.potential,
            "junction_potential_speed_scale": float(
                junction_potential_speed_scale
            ),
            "committed": committed_payload,
            "commit_branch_locked": self._committed_arm_lock.branch_locked,
            "commit_entry_confirmed": self._committed_arm_lock.entry_confirmed,
            "commit_release_requested": self._committed_arm_lock.release_requested,
            "reverse_requested": bool(self._arm_commit.reverse_requested),
            "reverse_mode": self._arm_commit.reverse_mode,
        }

    def _publish_debug(
        self,
        gs: object,
        *,
        speculative_arm_angle: Optional[float] = None,
    ) -> None:
        if not self._p["debug_markers"] or self._odom is None:
            return

        if not isinstance(gs, GraphTrackingState) or not gs.valid:
            self._debug_markers_pub.publish_graph_overlay(self._odom)
            return

        tracked_pts = self._graph_tracker.get_edge_points(gs.edge_id)

        junction_xy = None
        if gs.approaching_junction:
            junction_xy = self._graph_tracker.get_node_xy(gs.ahead_node_id)

        self._debug_markers_pub.publish_graph_overlay(
            self._odom,
            tracked_edge_pts=tracked_pts,
            junction_xy=junction_xy,
            speculative_arm_angle=speculative_arm_angle,
        )

    def _selected_candidate(
        self,
        gs: GraphTrackingState,
        selected_arm_angle_rad: float,
    ) -> Optional[GraphArmCandidate]:
        best: Optional[GraphArmCandidate] = None
        best_diff = float("inf")
        for candidate in gs.arm_candidates:
            diff = abs(_wrap(candidate.angle_rad - selected_arm_angle_rad))
            if diff < best_diff:
                best = candidate
                best_diff = diff
        return best

    def _graph_has_continuation(self, gs: GraphTrackingState) -> bool:
        min_length_m = max(0.05, float(self._p["min_arm_length_m"]))
        forward_continuation = (
            gs.ahead_path_length_m >= min_length_m and not gs.approaching_leaf
        )
        return bool(forward_continuation or self._graph_viable_arm_count(gs) > 0)

    def _graph_viable_arm_count(self, gs: GraphTrackingState) -> int:
        min_length_m = max(0.05, float(self._p["min_arm_length_m"]))
        return sum(
            1
            for arm in gs.arm_candidates
            if (not arm.is_backtrack) and arm.edge_length_m >= min_length_m
        )

    def _reverse_yaw_from_graph(self, gs: GraphTrackingState, drone_xy: np.ndarray) -> float:
        behind_id = gs.edge_id[0] if gs.ahead_node_id == gs.edge_id[1] else gs.edge_id[1]
        behind_xy = self._graph_tracker.get_node_xy(behind_id)
        if behind_xy is not None:
            delta = behind_xy - drone_xy
            if float(np.linalg.norm(delta)) > 1e-6:
                return float(math.atan2(float(delta[1]), float(delta[0])))
        return _wrap(gs.tangent_rad + math.pi)

    def _reverse_heading_for_handover(self) -> float:
        """Freeze one reverse heading before resetting recovery state."""
        if self._dead_end_reverse_yaw is not None:
            return self._dead_end_reverse_yaw
        if self._last_graph_reverse_yaw is not None:
            return self._last_graph_reverse_yaw
        if self._odom is not None:
            q = self._odom.pose.pose.orientation
            return _wrap(
                _yaw_from_quaternion(q.x, q.y, q.z, q.w) + math.pi
            )
        reference = self._yaw_intent.reference_heading
        return _wrap((0.0 if reference is None else reference) + math.pi)

    @staticmethod
    def _vx_gate_reason(vx_blocked_by_yaw: bool, vx_blocked_by_obstacle: bool) -> str:
        if vx_blocked_by_obstacle and vx_blocked_by_yaw:
            return "obstacle+yaw"
        if vx_blocked_by_obstacle:
            return "obstacle"
        if vx_blocked_by_yaw:
            return "yaw"
        return "free"

    def _maybe_log(self, now_s: float, d_front: Optional[float], vx: float, vy: float, vz: float) -> None:
        if self._last_log_time is not None and (now_s - self._last_log_time) < 0.5:
            return
        self._last_log_time = now_s
        df = f"{d_front:.2f}" if d_front is not None else "--"
        self.get_logger().info(
            f"[{self._mode}] dF={df} v=[{vx:+.2f},{vy:+.2f},{vz:+.2f}]"
        )


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 0.0 else 0.0
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _wrap(a: float) -> float:
    return float(math.atan2(math.sin(a), math.cos(a)))


def _body_y_error_to_target(target_xy: np.ndarray, drone_xy: np.ndarray, yaw_rad: float) -> float:
    delta = target_xy - drone_xy
    body_y_world = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float32)
    return float(np.dot(delta, body_y_world))


def _yaw_rate(target: float, current: float, kp: float, max_rate: float) -> float:
    err = math.atan2(math.sin(target - current), math.cos(target - current))
    return float(np.clip(kp * err, -max_rate, max_rate))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalPathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
