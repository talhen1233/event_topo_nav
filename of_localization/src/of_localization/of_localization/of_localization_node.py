#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, LaserScan, PointCloud2
from std_msgs.msg import Float32

try:
    from .depth_grid import BottomDepthGrid, DepthGridConfig, point_cloud_to_xyz, stamp_to_seconds
    from .estimator_mode import request_local_depth
    from .flow_stats import (
        compensate_planar_motion,
        compute_flow_inlier_mask,
        covariance2x2,
        lk_error_weights,
        robust_velocity_mask,
    )
    from .orientation_utils import (
        axial_depths_from_heights,
        gravity_direction_sensor_frame,
        heights_from_sensor_points,
        quaternion_to_euler_xyz,
    )
except ImportError:
    from depth_grid import BottomDepthGrid, DepthGridConfig, point_cloud_to_xyz, stamp_to_seconds
    from estimator_mode import request_local_depth
    from flow_stats import (
        compensate_planar_motion,
        compute_flow_inlier_mask,
        covariance2x2,
        lk_error_weights,
        robust_velocity_mask,
    )
    from orientation_utils import (
        axial_depths_from_heights,
        gravity_direction_sensor_frame,
        heights_from_sensor_points,
        quaternion_to_euler_xyz,
    )


@dataclass(frozen=True)
class VelocityEstimate:
    """Estimated horizontal velocity and its confidence."""

    body_velocity: np.ndarray
    covariance: np.ndarray
    annotated_prev: np.ndarray
    annotated_next: np.ndarray
    mode: str


class FisheyeOpticalFlowNode(Node):
    """Estimate planar velocity from bottom optical flow and local depth."""

    def __init__(self) -> None:
        super().__init__("of_localization_node")
        self._declare_parameters()

        gp = self.get_parameter
        self.image_topic = gp("image_topic").value
        self.lidar_topic = gp("lidar_topic").value
        self.bottom_depth_points_topic = gp("bottom_depth_points_topic").value
        self.imu_topic = gp("imu_topic").value
        self.speed_topic = gp("speed_topic").value
        self.annotated_image_topic = gp("annotated_image_topic").value
        self.odometry_topic = gp("odometry_topic").value
        self.use_orientation_compensation = bool(gp("use_orientation_compensation").value)
        self.min_orientation_ray_projection = float(gp("min_orientation_ray_projection").value)
        self.scale_factor = float(gp("scale_factor").value)
        self.verbose = bool(gp("verbose").value)
        self.use_camera_info = bool(gp("use_camera_info").value)
        self.camera_info_topic = gp("camera_info_topic").value
        self.distortion_model = str(gp("distortion_model").value).strip().lower() or "equidistant"
        self.use_clahe = bool(gp("use_clahe").value)
        self.clahe_clip_limit = float(gp("clahe_clip_limit").value)
        self.clahe_tile_grid = int(gp("clahe_tile_grid_size").value)
        self.altitude_filter_alpha = float(gp("altitude_filter_alpha").value)
        self.min_altitude = float(gp("min_altitude").value)
        self.min_features = int(gp("min_features").value)
        self.redetect_interval = int(gp("redetect_interval").value)
        self.flow_distance_threshold = float(gp("flow_distance_threshold").value)
        self.direction_threshold_deg = float(gp("direction_threshold_deg").value)
        self.time_factor = float(gp("time_factor").value)
        self.velocity_scale = float(gp("velocity_scale").value)
        self.plane_residual_threshold = float(gp("plane_residual_threshold").value)
        self.residual_inflation_k = float(gp("residual_inflation_k").value)
        self.min_inliers_for_low_cov = int(gp("min_inliers_for_low_cov").value)
        self.alt_inflation_k = float(gp("alt_inflation_k").value)
        self.div_inflation_k = float(gp("div_inflation_k").value)
        self.min_covariance = float(gp("min_covariance").value)
        self.max_covariance = float(gp("max_covariance").value)
        self.depth_fusion_max_altitude = float(gp("depth_fusion_max_altitude").value)
        self.depth_fusion_altitude_hysteresis = float(
            gp("depth_fusion_altitude_hysteresis").value
        )
        self.depth_min_valid_features = int(gp("depth_min_valid_features").value)
        self.depth_min_valid_ratio = float(gp("depth_min_valid_ratio").value)
        self.depth_coverage_inflation_k = float(gp("depth_coverage_inflation_k").value)
        self.depth_spread_inflation_k = float(gp("depth_spread_inflation_k").value)
        self.depth_staleness_inflation_k = float(gp("depth_staleness_inflation_k").value)
        self.use_depth_overlap_roi = bool(gp("use_depth_overlap_roi").value)

        self.feature_params = dict(
            maxCorners=int(gp("max_corners").value),
            qualityLevel=float(gp("quality_level").value),
            minDistance=float(gp("min_distance").value),
            blockSize=int(gp("block_size").value),
        )
        win_size = int(gp("win_size").value)
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=int(gp("max_level").value),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(gp("criteria_count").value),
                float(gp("criteria_epsilon").value),
            ),
        )
        self.vel_tr = np.array(
            gp("velocity_transform_matrix").value,
            dtype=np.float32,
        ).reshape(2, 2)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            CompressedImage,
            self.image_topic,
            self.image_callback,
            qos,
        )
        self.create_subscription(
            LaserScan,
            self.lidar_topic,
            self.lidar_callback,
            qos,
        )
        self.create_subscription(
            PointCloud2,
            self.bottom_depth_points_topic,
            self.depth_callback,
            qos,
        )
        self.create_subscription(
            Imu,
            self.imu_topic,
            self.imu_callback,
            qos,
        )
        if self.use_camera_info:
            self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                qos,
            )

        self.speed_pub = self.create_publisher(Float32, self.speed_topic, qos)
        self.odom_pub = self.create_publisher(Odometry, self.odometry_topic, qos)
        self.annotated_pub = self.create_publisher(
            CompressedImage,
            self.annotated_image_topic,
            qos,
        )
        self.depth_grid = BottomDepthGrid(
            DepthGridConfig(timeout_sec=float(gp("depth_cloud_timeout_sec").value))
        )
        self.height_grid = BottomDepthGrid(
            DepthGridConfig(timeout_sec=float(gp("depth_cloud_timeout_sec").value))
        )

        self._gray_bufs = [None, None]
        self._buf_idx = 0
        self.prev_gray = None
        self.curr_pts = None
        self.init_camera_info = False
        self.frame_counter = 0
        self.initialised = False
        self.last_img_t = None
        self._last_img_stamp_s = None
        self._last_estimator_mode = None
        self._local_depth_requested: bool | None = None

        self.altitude = 0.0
        self.last_lidar_alt = None
        self.latest_roll = 0.0
        self.latest_pitch = 0.0
        self.latest_yaw = 0.0
        self._have_imu_orientation = False
        self._gravity_dir_sensor = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        self._sensor_from_body = np.asarray(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        self._latest_depth_points_xyz = np.empty((0, 3), dtype=np.float32)
        self._latest_depth_stamp_s: float | None = None

        self.position = np.zeros(3, dtype=np.float32)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        self._cov6 = np.zeros(36, dtype=np.float32)
        self._pose_cov6 = np.zeros(36, dtype=np.float32)
        self._base_roi_mask = None
        self._depth_roi_mask = None
        self._cached_mask_shape = None
        self._image_shape = None
        self._cxcy = None
        self._flow_dist_thresh_sq = self.flow_distance_threshold ** 2
        self._cos_dir_thresh = math.cos(math.radians(self.direction_threshold_deg))

        self.orig_K = None
        self.orig_D = None
        self._warned_no_cam_info = False
        self._warned_invalid_cam_info = False
        if not self.use_camera_info:
            self._load_calibration_data(gp("calibration_file").value)
        self.scaled_K = None
        self.map1 = None
        self.map2 = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self._clahe = None
        if self.use_clahe:
            tile = max(1, int(self.clahe_tile_grid))
            self._clahe = cv2.createCLAHE(
                clipLimit=float(self.clahe_clip_limit),
                tileGridSize=(tile, tile),
            )

        cv2.setUseOptimized(True)
        cv2.ocl.setUseOpenCL(True)
        self.get_logger().info("OpticalFlowNode initialised.")

    def _declare_parameters(self) -> None:
        """Declare configurable ROS parameters."""
        self.declare_parameter(
            "calibration_file",
            "/home/talhen/docker_ws/tiny_drone/tiny_drone/config/of/fisheye_calibration_data.xml",
        )
        self.declare_parameter("use_camera_info", True)
        self.declare_parameter("camera_info_topic", "/crazyflie/camera/bottom/camera_info")
        self.declare_parameter("distortion_model", "equidistant")
        self.declare_parameter("scale_factor", 1.0)
        self.declare_parameter("use_clahe", True)
        self.declare_parameter("clahe_clip_limit", 3.0)
        self.declare_parameter("clahe_tile_grid_size", 8)
        self.declare_parameter("max_corners", 45)
        self.declare_parameter("quality_level", 0.01)
        self.declare_parameter("min_distance", 5.0)
        self.declare_parameter("block_size", 7)
        self.declare_parameter("win_size", 19)
        self.declare_parameter("max_level", 2)
        self.declare_parameter("criteria_epsilon", 0.03)
        self.declare_parameter("criteria_count", 7)
        self.declare_parameter("redetect_interval", 3)
        self.declare_parameter("min_features", 10)
        self.declare_parameter("verbose", True)
        self.declare_parameter("time_factor", 1.0)
        self.declare_parameter("velocity_scale", 1.0)
        self.declare_parameter("min_altitude", 0.05)
        self.declare_parameter("altitude_filter_alpha", 0.3)
        self.declare_parameter("flow_distance_threshold", 5.0)
        self.declare_parameter("direction_threshold_deg", 20.0)
        self.declare_parameter("plane_residual_threshold", 2.0)
        self.declare_parameter("residual_inflation_k", 0.5)
        self.declare_parameter("alt_inflation_k", 1.0)
        self.declare_parameter("max_covariance", 0.5)
        self.declare_parameter("min_covariance", 0.0001)
        self.declare_parameter("min_inliers_for_low_cov", 5)
        self.declare_parameter("div_inflation_k", 5.0)
        self.declare_parameter("velocity_transform_matrix", [1.0, 0.0, 0.0, 1.0])
        self.declare_parameter("image_topic", "/crazyflie/camera/bottom/compressed")
        self.declare_parameter("lidar_topic", "/crazyflie/lidar/height")
        self.declare_parameter(
            "bottom_depth_points_topic",
            "/crazyflie/depth_camera/bottom/points",
        )
        self.declare_parameter("imu_topic", "/crazyflie/imu/acc_derived")
        self.declare_parameter("speed_topic", "/drone/speed")
        self.declare_parameter(
            "annotated_image_topic",
            "/camera/bottom/annotated/compressed",
        )
        self.declare_parameter("use_orientation_compensation", True)
        self.declare_parameter("min_orientation_ray_projection", 0.15)
        self.declare_parameter("odometry_topic", "/drone/of_odometry")
        self.declare_parameter("depth_fusion_max_altitude", 1.0)
        self.declare_parameter("depth_fusion_altitude_hysteresis", 0.15)
        self.declare_parameter("depth_cloud_timeout_sec", 0.15)
        self.declare_parameter("depth_min_valid_features", 6)
        self.declare_parameter("depth_min_valid_ratio", 0.35)
        self.declare_parameter("depth_coverage_inflation_k", 2.0)
        self.declare_parameter("depth_spread_inflation_k", 2.0)
        self.declare_parameter("depth_staleness_inflation_k", 3.0)
        self.declare_parameter("use_depth_overlap_roi", True)

    def lidar_callback(self, msg: LaserScan) -> None:
        """Low-pass filter the scalar height measurement."""
        if not msg.ranges:
            return
        raw_range_m = float(msg.ranges[0])
        if not math.isfinite(raw_range_m) or raw_range_m < 0.0:
            return
        height_m = self._scalar_height_from_range(raw_range_m)
        if height_m is None:
            return
        alpha = self.altitude_filter_alpha
        self.altitude = alpha * height_m + (1.0 - alpha) * self.altitude

    def depth_callback(self, msg: PointCloud2) -> None:
        """Cache the latest bottom depth cloud."""
        stamp_s = stamp_to_seconds(msg.header.stamp)
        points_xyz = point_cloud_to_xyz(msg)
        self._latest_depth_points_xyz = points_xyz
        self._latest_depth_stamp_s = stamp_s
        self.depth_grid.update_from_points(points_xyz, stamp_s)
        if self.use_orientation_compensation and self._have_imu_orientation and points_xyz.size:
            heights_m = heights_from_sensor_points(points_xyz, self._gravity_dir_sensor)
            self.height_grid.update_from_scalar_values(points_xyz, heights_m, stamp_s)
        else:
            self.height_grid.update_from_points(points_xyz, stamp_s)

    def imu_callback(self, msg: Imu) -> None:
        """Update the latest IMU orientation used for OF compensation."""
        qx = float(msg.orientation.x)
        qy = float(msg.orientation.y)
        qz = float(msg.orientation.z)
        qw = float(msg.orientation.w)
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-6:
            return
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
        self.latest_roll, self.latest_pitch, self.latest_yaw = quaternion_to_euler_xyz(
            qx,
            qy,
            qz,
            qw,
        )
        self.orientation = np.asarray([qx, qy, qz, qw], dtype=np.float32)
        self._gravity_dir_sensor = gravity_direction_sensor_frame(
            self.latest_roll,
            self.latest_pitch,
            self.latest_yaw,
            self._sensor_from_body,
        )
        self._have_imu_orientation = True
        if self.use_orientation_compensation and self._latest_depth_points_xyz.size:
            heights_m = heights_from_sensor_points(
                self._latest_depth_points_xyz,
                self._gravity_dir_sensor,
            )
            self.height_grid.update_from_scalar_values(
                self._latest_depth_points_xyz,
                heights_m,
                self._latest_depth_stamp_s,
            )

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """Update intrinsics from a CameraInfo message."""
        k = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
            if not self._warned_invalid_cam_info:
                self.get_logger().warn(
                    "CameraInfo focal lengths are invalid; waiting for calibrated intrinsics."
                )
                self._warned_invalid_cam_info = True
            return

        if not self.init_camera_info:
            self.get_logger().info(
                f"CameraInfo received ({msg.distortion_model or self.distortion_model}) – "
                "system initialised."
            )
            self.init_camera_info = True

        model = (msg.distortion_model or self.distortion_model).strip().lower()
        if model:
            self.distortion_model = model
        d = np.array(msg.d, dtype=np.float32).reshape(-1, 1)
        if self._uses_fisheye_model():
            if d.shape[0] < 4:
                d = np.pad(d, ((0, 4 - d.shape[0]), (0, 0)), mode="constant")
            elif d.shape[0] > 4:
                d = d[:4]
        self.orig_K = k
        self.orig_D = d
        self._warned_no_cam_info = False
        self._warned_invalid_cam_info = False

    def image_callback(self, msg: CompressedImage) -> None:
        """Track features and publish a depth-aware OF odometry update."""
        if self.use_camera_info and self.orig_K is None:
            if not self._warned_no_cam_info:
                self.get_logger().warn(
                    "use_camera_info is True but no CameraInfo has been received yet; "
                    "dropping images until intrinsics are available."
                )
                self._warned_no_cam_info = True
            return

        frame_small = self._decode_and_downsample_image(msg)
        if frame_small is None:
            return

        if not self.initialised:
            height, width = frame_small.shape
            self._init_scaled_camera_matrix()
            self._init_undistort_maps(width, height)
            undistorted = self._undistort_small(frame_small)
            self._gray_bufs[0] = np.empty_like(undistorted)
            self._gray_bufs[1] = np.empty_like(undistorted)
            np.copyto(self._gray_bufs[0], undistorted)
            self.prev_gray = self._gray_bufs[0]
            self._buf_idx = 0
            self.last_img_t = time.time()
            stamp_s = self._message_stamp_seconds(msg)
            if stamp_s is not None:
                self._last_img_stamp_s = stamp_s
            self.initialised = True
            self.get_logger().info("First image received – system initialised.")
            return

        dt, image_stamp_s = self._compute_dt(msg)
        if dt <= 0.0:
            return

        if self.altitude < self.min_altitude:
            self._publish_zero_velocity_uncertain()
            return

        undistorted = self._undistort_small(frame_small)
        self._refresh_image_geometry_cache(undistorted.shape)
        local_depth_requested = request_local_depth(
            self.altitude,
            self.depth_fusion_max_altitude,
            self.depth_fusion_altitude_hysteresis,
            self._local_depth_requested,
        )
        self._local_depth_requested = local_depth_requested
        needs_redetect = (
            self.curr_pts is None
            or len(self.curr_pts) < self.min_features
            or (self.frame_counter % self.redetect_interval) == 0
        )

        if needs_redetect:
            detect_mask = self._get_detection_mask(undistorted.shape, local_depth_requested)
            gray_for_detect = self._apply_clahe(undistorted) if self._clahe else undistorted
            self.curr_pts = cv2.goodFeaturesToTrack(
                gray_for_detect,
                mask=detect_mask,
                **self.feature_params,
            )
            if self.curr_pts is None:
                self._store_tracking_frame(undistorted)
                self.frame_counter += 1
                self._publish_zero_velocity_uncertain()
                return

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            undistorted,
            self.curr_pts,
            None,
            **self.lk_params,
        )
        self._store_tracking_frame(undistorted)
        self.frame_counter += 1

        if next_pts is None or status is None:
            self.curr_pts = None
            return

        status_mask = status.ravel() == 1
        if not np.any(status_mask):
            self.curr_pts = None
            self._publish_zero_velocity_uncertain()
            return

        good_prev = self.curr_pts[status_mask].reshape(-1, 2).astype(np.float32)
        good_next = next_pts[status_mask].reshape(-1, 2).astype(np.float32)
        lk_error = None
        if err is not None:
            lk_error = err.ravel()[status_mask].astype(np.float32)

        if good_prev.shape[0] < 2:
            self.curr_pts = None
            self._publish_zero_velocity_uncertain()
            return

        flow = (good_next - good_prev).astype(np.float32)
        inlier_mask, median_flow = compute_flow_inlier_mask(
            flow,
            self._flow_dist_thresh_sq,
            self._cos_dir_thresh,
        )
        if not np.any(inlier_mask):
            self.curr_pts = None
            if self.verbose and np.linalg.norm(median_flow) > 0.1:
                self.get_logger().info(
                    "No reliable flow inliers – publish zero velocity with high covariance"
                )
            self._publish_zero_velocity_uncertain()
            return

        flow = flow[inlier_mask]
        good_prev = good_prev[inlier_mask]
        good_next = good_next[inlier_mask]
        if lk_error is not None:
            lk_error = lk_error[inlier_mask]
        self.curr_pts = good_next.reshape(-1, 1, 2)

        if flow.shape[0] < 2:
            self.curr_pts = None
            self._publish_zero_velocity_uncertain()
            return

        flow_result = compensate_planar_motion(good_prev, flow, self._cxcy)
        corrected_flow = flow_result.corrected_flow
        if corrected_flow.shape[0] < 2:
            self.curr_pts = None
            self._publish_zero_velocity_uncertain()
            return

        previous_alt = self.last_lidar_alt
        local_estimate = None
        if local_depth_requested:
            local_estimate = self._estimate_local_depth_velocity(
                corrected_flow,
                good_prev,
                good_next,
                lk_error,
                dt,
                image_stamp_s,
                flow_result.divergence_rate,
            )
        legacy_estimate = None
        if local_estimate is None or not local_depth_requested:
            legacy_estimate = self._estimate_legacy_velocity(
                corrected_flow,
                good_prev,
                good_next,
                lk_error,
                dt,
                previous_alt,
                flow_result.divergence_rate,
            )
        estimate = local_estimate if (local_depth_requested and local_estimate is not None) else legacy_estimate

        if estimate is None:
            self.curr_pts = None
            self._publish_zero_velocity_uncertain()
            return

        velocity = (self.vel_tr @ estimate.body_velocity.astype(np.float32)).astype(np.float32)
        vel_cov = (self.vel_tr @ estimate.covariance @ self.vel_tr.T).astype(np.float32)
        vx = float(velocity[0])
        vy = float(velocity[1])

        alt = max(self.altitude, self.min_altitude)
        self.last_lidar_alt = alt
        self.position[:2] += np.array([vx, vy], dtype=np.float32) * float(dt)
        self.position[2] = alt

        self._log_mode_change(estimate.mode)
        self._publish_odometry(vx, vy, vel_cov)
        self._publish_speed(math.hypot(vx, vy))
        if self.verbose and self.annotated_pub.get_subscription_count():
            self._publish_annotated(
                undistorted,
                estimate.annotated_prev,
                estimate.annotated_next,
            )

    def _estimate_local_depth_velocity(
        self,
        corrected_flow: np.ndarray,
        prev_points_xy: np.ndarray,
        next_points_xy: np.ndarray,
        lk_error: np.ndarray | None,
        dt: float,
        image_stamp_s: float | None,
        divergence_rate: float,
    ) -> VelocityEstimate | None:
        """Estimate velocity from per-feature local depth samples."""
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None

        reference_stamp_s = image_stamp_s if image_stamp_s is not None else self._current_time_seconds()
        scalar_grid = self._scalar_grid_for_metric_support()
        if not scalar_grid.is_fresh(reference_stamp_s):
            return None

        height_samples = scalar_grid.sample_values(
            next_points_xy,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            reference_stamp_s,
        )
        axial_depths, axial_valid_mask = self._axial_depths_for_points(
            height_samples.depths_m,
            next_points_xy,
        )
        valid_mask = height_samples.valid_mask & axial_valid_mask
        valid_count = int(valid_mask.sum())
        total_count = corrected_flow.shape[0]
        valid_ratio = valid_count / float(total_count)
        if valid_count < self.depth_min_valid_features or valid_ratio < self.depth_min_valid_ratio:
            return None

        local_velocities = self._velocity_samples_from_depths(
            corrected_flow[valid_mask],
            axial_depths[valid_mask],
            dt,
        )
        if local_velocities.shape[0] < 2:
            return None

        robust_mask = robust_velocity_mask(local_velocities)
        if int(robust_mask.sum()) < 2:
            return None

        weights = np.ones(local_velocities.shape[0], dtype=np.float32)
        if lk_error is not None:
            weights = lk_error_weights(lk_error[valid_mask])
            if weights.size == 0:
                weights = np.ones(local_velocities.shape[0], dtype=np.float32)

        kept_velocities = local_velocities[robust_mask]
        kept_weights = weights[robust_mask]
        body_velocity, vel_cov = covariance2x2(kept_velocities, kept_weights)
        kept_heights = height_samples.depths_m[valid_mask][robust_mask]
        kept_flow = corrected_flow[valid_mask][robust_mask]
        _, kept_flow_cov = covariance2x2(kept_flow)

        coverage_factor = 1.0 + self.depth_coverage_inflation_k * max(0.0, 1.0 - valid_ratio)
        mean_height = max(float(np.mean(kept_heights)), self.min_altitude)
        depth_spread_ratio = float(np.std(kept_heights)) / mean_height
        depth_spread_factor = 1.0 + self.depth_spread_inflation_k * depth_spread_ratio
        depth_staleness_factor = 1.0 + self.depth_staleness_inflation_k * height_samples.age_sec
        plane_res = float(np.trace(kept_flow_cov))
        residual_factor = 1.0 + self.residual_inflation_k * max(
            0.0,
            plane_res - self.plane_residual_threshold,
        )
        inlier_factor = float(self.min_inliers_for_low_cov) / float(kept_velocities.shape[0])
        inlier_factor = max(1.0, min(inlier_factor, 3.0))
        spike_factor = 1.0 + self.div_inflation_k * abs(divergence_rate)
        vel_cov *= (
            coverage_factor
            * depth_spread_factor
            * depth_staleness_factor
            * residual_factor
            * inlier_factor
            * spike_factor
        )
        vel_cov = self._clamp_covariance(vel_cov)
        kept_prev = prev_points_xy[valid_mask][robust_mask]
        kept_next = next_points_xy[valid_mask][robust_mask]
        return VelocityEstimate(
            body_velocity=body_velocity.astype(np.float32),
            covariance=vel_cov,
            annotated_prev=kept_prev,
            annotated_next=kept_next,
            mode="local_depth",
        )

    def _estimate_legacy_velocity(
        self,
        corrected_flow: np.ndarray,
        prev_points_xy: np.ndarray,
        next_points_xy: np.ndarray,
        lk_error: np.ndarray | None,
        dt: float,
        previous_alt: float | None,
        divergence_rate: float,
    ) -> VelocityEstimate | None:
        """Estimate velocity using one scalar height measurement."""
        if self.fx is None or self.fy is None:
            return None
        if corrected_flow.shape[0] < 2:
            return None

        alt = max(self.altitude, self.min_altitude)
        height_samples = np.full(corrected_flow.shape[0], alt, dtype=np.float32)
        axial_depths, axial_valid_mask = self._axial_depths_for_points(
            height_samples,
            next_points_xy,
        )
        if int(axial_valid_mask.sum()) < 2:
            return None

        velocity_samples = self._velocity_samples_from_depths(
            corrected_flow[axial_valid_mask],
            axial_depths[axial_valid_mask],
            dt,
        )
        if velocity_samples.shape[0] < 2:
            return None

        robust_mask = robust_velocity_mask(velocity_samples)
        if int(robust_mask.sum()) < 2:
            return None

        weights = np.ones(velocity_samples.shape[0], dtype=np.float32)
        if lk_error is not None:
            weights = lk_error_weights(lk_error[axial_valid_mask])
            if weights.size == 0:
                weights = np.ones(velocity_samples.shape[0], dtype=np.float32)

        kept_velocities = velocity_samples[robust_mask]
        kept_weights = weights[robust_mask]
        body_velocity, vel_cov = covariance2x2(kept_velocities, kept_weights)
        kept_flow = corrected_flow[axial_valid_mask][robust_mask]
        _, kept_flow_cov = covariance2x2(kept_flow)

        plane_res = float(np.trace(kept_flow_cov))
        residual_factor = 1.0 + self.residual_inflation_k * max(
            0.0,
            plane_res - self.plane_residual_threshold,
        )
        inlier_factor = float(self.min_inliers_for_low_cov) / float(kept_velocities.shape[0])
        inlier_factor = max(1.0, min(inlier_factor, 3.0))
        if previous_alt is None:
            alt_factor = 1.0
        else:
            alt_factor = 1.0 + self.alt_inflation_k * abs(alt - previous_alt)
        spike_factor = 1.0 + self.div_inflation_k * abs(divergence_rate)
        vel_cov *= residual_factor * inlier_factor * alt_factor * spike_factor
        vel_cov = self._clamp_covariance(vel_cov)
        return VelocityEstimate(
            body_velocity=body_velocity.astype(np.float32),
            covariance=vel_cov,
            annotated_prev=prev_points_xy[axial_valid_mask][robust_mask],
            annotated_next=next_points_xy[axial_valid_mask][robust_mask],
            mode="legacy_height",
        )

    def _velocity_samples_from_depths(
        self,
        corrected_flow: np.ndarray,
        depths_m: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Convert corrected per-feature flow into body-frame velocity samples."""
        effective_dt = max(float(dt) * max(self.time_factor, 1e-6), 1e-6)
        depth = np.asarray(depths_m, dtype=np.float32).reshape(-1)
        flow_xy = np.asarray(corrected_flow, dtype=np.float32).reshape(-1, 2)
        sx = (depth / self.fx) / effective_dt * self.velocity_scale
        sy = (depth / self.fy) / effective_dt * self.velocity_scale
        return np.column_stack((flow_xy[:, 1] * sx, flow_xy[:, 0] * sy)).astype(np.float32)

    def _orientation_compensation_ready(self) -> bool:
        """Return whether tilt compensation is currently usable."""
        return bool(self.use_orientation_compensation and self._have_imu_orientation)

    def _scalar_grid_for_metric_support(self) -> BottomDepthGrid:
        """Return the scalar grid used for metric depth support."""
        if self._orientation_compensation_ready():
            return self.height_grid
        return self.depth_grid

    def _axial_depths_for_points(
        self,
        scalar_heights_m: np.ndarray,
        image_points_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert gravity-aligned heights to optical-axis depths for the given rays."""
        if not self._orientation_compensation_ready():
            depths = np.asarray(scalar_heights_m, dtype=np.float32).reshape(-1)
            valid_mask = np.isfinite(depths) & (depths > 0.0)
            return depths, valid_mask

        depths, valid_mask, _ = axial_depths_from_heights(
            scalar_heights_m,
            image_points_xy,
            self._gravity_dir_sensor,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            min_projection=self.min_orientation_ray_projection,
        )
        return depths, valid_mask

    def _scalar_height_from_range(self, raw_range_m: float) -> float | None:
        """Convert the center range measurement into a gravity-aligned height."""
        if not self._orientation_compensation_ready():
            return float(raw_range_m)

        center_projection = float(self._gravity_dir_sensor[0])
        if center_projection <= self.min_orientation_ray_projection:
            return None
        return float(raw_range_m) * center_projection

    def _compute_dt(self, msg: CompressedImage) -> tuple[float, float | None]:
        """Compute image delta time from stamps with a wall-time fallback."""
        stamp_s = self._message_stamp_seconds(msg)
        dt = None
        if stamp_s is not None:
            if self._last_img_stamp_s is not None:
                dt = stamp_s - self._last_img_stamp_s
            self._last_img_stamp_s = stamp_s

        if dt is None:
            now = time.time()
            if self.last_img_t is None:
                self.last_img_t = now
                return 0.0, stamp_s
            dt = now - self.last_img_t
            self.last_img_t = now
        return float(dt), stamp_s

    def _message_stamp_seconds(self, msg) -> float | None:
        """Extract a ROS message header timestamp in seconds."""
        header = getattr(msg, "header", None)
        if header is None:
            return None
        return stamp_to_seconds(getattr(header, "stamp", None))

    def _current_time_seconds(self) -> float:
        """Return the current ROS clock time in seconds."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _store_tracking_frame(self, undistorted: np.ndarray) -> None:
        """Swap the tracking buffer for the next frame."""
        self._buf_idx = 1 - self._buf_idx
        np.copyto(self._gray_bufs[self._buf_idx], undistorted)
        self.prev_gray = self._gray_bufs[self._buf_idx]

    def _refresh_image_geometry_cache(self, image_shape: tuple[int, int]) -> None:
        """Cache image geometry-dependent state."""
        if self._image_shape == image_shape:
            return
        height, width = image_shape
        self._image_shape = image_shape
        self._cxcy = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        self._cached_mask_shape = None
        self._base_roi_mask = None
        self._depth_roi_mask = None

    def _get_detection_mask(
        self,
        image_shape: tuple[int, int],
        local_depth_requested: bool,
    ) -> np.ndarray:
        """Return the feature-detection ROI for the current mode."""
        if self._cached_mask_shape != image_shape:
            height, width = image_shape
            base_mask = np.zeros((height, width), dtype=np.uint8)
            base_mask[
                int(0.17 * height):int(0.83 * height),
                int(0.17 * width):int(0.83 * width),
            ] = 255
            self._base_roi_mask = base_mask
            self._depth_roi_mask = None
            self._cached_mask_shape = image_shape

        if not local_depth_requested or not self.use_depth_overlap_roi:
            return self._base_roi_mask
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return self._base_roi_mask

        if self._depth_roi_mask is None:
            height, width = image_shape
            u_min, u_max, v_min, v_max = self.depth_grid.overlap_bounds(
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                width,
                height,
            )
            depth_mask = np.zeros((height, width), dtype=np.uint8)
            depth_mask[v_min:v_max, u_min:u_max] = 255
            self._depth_roi_mask = cv2.bitwise_and(self._base_roi_mask, depth_mask)
        return self._depth_roi_mask

    def _decode_and_downsample_image(self, msg: CompressedImage) -> np.ndarray | None:
        """Decode a compressed grayscale image at the configured scale."""
        scale = self.scale_factor
        flags = cv2.IMREAD_GRAYSCALE
        if abs(scale - 0.5) < 1e-3:
            flags |= cv2.IMREAD_REDUCED_GRAYSCALE_2
        elif abs(scale - 0.25) < 1e-3:
            flags |= cv2.IMREAD_REDUCED_GRAYSCALE_4
        buffer = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(buffer, flags)
        if frame is None:
            return None
        if scale not in (0.5, 0.25):
            height, width = frame.shape
            frame = cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return frame

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to boost contrast on low-texture surfaces."""
        return self._clahe.apply(image)

    def _init_scaled_camera_matrix(self) -> None:
        """Scale the calibration matrix to the current image resolution."""
        camera_matrix = self.orig_K.copy().astype(np.float32)
        scale = np.float32(self.scale_factor)
        camera_matrix[0, 0] *= scale
        camera_matrix[1, 1] *= scale
        camera_matrix[0, 2] *= scale
        camera_matrix[1, 2] *= scale
        self.scaled_K = camera_matrix
        self.fx = float(camera_matrix[0, 0])
        self.fy = float(camera_matrix[1, 1])
        self.cx = float(camera_matrix[0, 2])
        self.cy = float(camera_matrix[1, 2])

    def _uses_fisheye_model(self) -> bool:
        """Return whether the active distortion model uses fisheye calibration."""
        return self.distortion_model in {"equidistant", "fisheye"}

    def _init_undistort_maps(self, width: int, height: int) -> None:
        """Precompute image undistortion maps."""
        if self._uses_fisheye_model():
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                self.scaled_K,
                self.orig_D[:4],
                np.eye(3, dtype=np.float32),
                self.scaled_K,
                (width, height),
                cv2.CV_16SC2,
            )
            return

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.scaled_K,
            self.orig_D,
            np.eye(3, dtype=np.float32),
            self.scaled_K,
            (width, height),
            cv2.CV_16SC2,
        )

    def _undistort_small(self, frame: np.ndarray) -> np.ndarray:
        """Apply the precomputed undistortion map."""
        if self.map1 is None:
            return frame
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def _load_calibration_data(self, path: str) -> None:
        """Load camera intrinsics from the XML calibration file."""
        if not os.path.exists(path):
            self.get_logger().fatal(f"Calibration file not found: {path}")
            raise SystemExit

        file_storage = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        self.orig_K = file_storage.getNode("K").mat().astype(np.float32)
        self.orig_D = file_storage.getNode("D").mat().astype(np.float32).reshape(-1, 1)
        distortion_node = file_storage.getNode("distortion_model")
        if not distortion_node.empty():
            model = distortion_node.string().strip().lower()
            if model:
                self.distortion_model = model
        if self._uses_fisheye_model():
            if self.orig_D.shape[0] < 4:
                self.orig_D = np.pad(
                    self.orig_D,
                    ((0, 4 - self.orig_D.shape[0]), (0, 0)),
                    mode="constant",
                )
            elif self.orig_D.shape[0] > 4:
                self.orig_D = self.orig_D[:4]
        file_storage.release()

    def _clamp_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Clamp the covariance to configured min/max values."""
        cov = np.asarray(covariance, dtype=np.float32).copy()
        diag_max = float(np.max(np.diag(cov)))
        if diag_max > self.max_covariance:
            cov *= self.max_covariance / diag_max
        diag = np.diag(cov).copy()
        diag = np.maximum(diag, self.min_covariance)
        np.fill_diagonal(cov, diag)
        return cov.astype(np.float32)

    def _log_mode_change(self, mode: str) -> None:
        """Log estimator mode transitions once."""
        if not self.verbose or mode == self._last_estimator_mode:
            return
        self._last_estimator_mode = mode
        self.get_logger().info(f"OF estimator mode: {mode}")

    def _publish_zero_velocity_uncertain(self) -> None:
        """Publish zero velocity with high covariance."""
        self.position[2] = max(self.altitude, 0.0)
        cov = np.eye(2, dtype=np.float32) * float(self.max_covariance)
        self._publish_odometry(0.0, 0.0, cov)
        self._publish_speed(0.0)

    def _publish_odometry(self, vx: float, vy: float, cov2: np.ndarray) -> None:
        """Publish the optical-flow odometry estimate."""
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "fisheye_bottom"

        odom.pose.pose.position.x = float(self.position[0])
        odom.pose.pose.position.y = float(self.position[1])
        odom.pose.pose.position.z = float(self.position[2])
        odom.pose.pose.orientation.x = float(self.orientation[0])
        odom.pose.pose.orientation.y = float(self.orientation[1])
        odom.pose.pose.orientation.z = float(self.orientation[2])
        odom.pose.pose.orientation.w = float(self.orientation[3])

        cov6 = self._cov6
        cov6.fill(0.0)
        cov6[0] = cov2[0, 0]
        cov6[1] = cov2[0, 1]
        cov6[6] = cov2[0, 1]
        cov6[7] = cov2[1, 1]
        odom.twist.covariance = cov6.tolist()

        pose_cov6 = self._pose_cov6
        pose_cov6.fill(0.0)
        pose_cov6[0] = cov2[0, 0]
        pose_cov6[1] = cov2[0, 1]
        pose_cov6[6] = cov2[0, 1]
        pose_cov6[7] = cov2[1, 1]
        odom.pose.covariance = pose_cov6.tolist()

        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        self.odom_pub.publish(odom)

    def _publish_speed(self, speed: float) -> None:
        """Publish horizontal speed magnitude."""
        msg = Float32()
        msg.data = float(speed)
        self.speed_pub.publish(msg)
        if self.verbose:
            self.get_logger().info(f"|v|={speed:.2f} m/s")

    def _publish_annotated(
        self,
        gray: np.ndarray,
        prev_points_xy: np.ndarray,
        next_points_xy: np.ndarray,
    ) -> None:
        """Publish a debug image with the contributing flow vectors."""
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for p0, p1 in zip(prev_points_xy, next_points_xy):
            pt0 = (int(p0[0]), int(p0[1]))
            pt1 = (int(p1[0]), int(p1[1]))
            cv2.arrowedLine(annotated, pt0, pt1, (0, 255, 0), 1)
        ok, buffer = cv2.imencode(".jpg", annotated)
        if not ok:
            return
        msg = CompressedImage()
        msg.format = "jpeg"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = buffer.tobytes()
        self.annotated_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FisheyeOpticalFlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
