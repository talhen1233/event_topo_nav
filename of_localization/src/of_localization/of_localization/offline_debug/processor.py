from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, LaserScan, PointCloud2

from of_localization.debug_visualization import (
    FeatureDepthDebug,
    VelocityComparisonDebug,
    render_depth_debug_canvas,
)
from of_localization.depth_grid import (
    BottomDepthGrid,
    DepthGridConfig,
    point_cloud_to_xyz,
    stamp_to_seconds,
)
from of_localization.estimator_mode import request_local_depth
from of_localization.flow_stats import (
    compensate_planar_motion,
    compute_flow_inlier_mask,
    covariance2x2,
    lk_error_weights,
    robust_velocity_mask,
)
from of_localization.orientation_utils import (
    axial_depths_from_heights,
    gravity_direction_sensor_frame,
    heights_from_sensor_points,
    quaternion_to_euler_xyz,
)


@dataclass(frozen=True)
class OfflineVelocityEstimate:
    """Estimated body-frame velocity used for offline visualization."""

    body_velocity: np.ndarray
    covariance: np.ndarray
    annotated_prev: np.ndarray
    annotated_next: np.ndarray
    mode: str
    feature_debug: FeatureDepthDebug | None = None


@dataclass(frozen=True)
class OfflineDebugFrame:
    """Rendered offline debug frame and its metadata."""

    image_bgr: np.ndarray
    stamp_s: float
    speed_mps: float
    mode: str


class OfflineDebugProcessor:
    """Rebuild OF debug frames offline from recorded topics."""

    def __init__(self, params: dict[str, object]) -> None:
        self.params = params
        self.scale_factor = float(params.get("scale_factor", 1.0))
        self.use_camera_info = bool(params.get("use_camera_info", True))
        self.distortion_model = str(params.get("distortion_model", "equidistant")).strip().lower() or "equidistant"
        self.use_clahe = bool(params.get("use_clahe", True))
        self.clahe_clip_limit = float(params.get("clahe_clip_limit", 3.0))
        self.clahe_tile_grid = int(params.get("clahe_tile_grid_size", 8))
        self.min_altitude = float(params.get("min_altitude", 0.05))
        self.altitude_filter_alpha = float(params.get("altitude_filter_alpha", 0.3))
        self.min_features = int(params.get("min_features", 10))
        self.redetect_interval = int(params.get("redetect_interval", 3))
        self.flow_distance_threshold = float(params.get("flow_distance_threshold", 5.0))
        self.direction_threshold_deg = float(params.get("direction_threshold_deg", 20.0))
        self.time_factor = float(params.get("time_factor", 1.0))
        self.velocity_scale = float(params.get("velocity_scale", 1.0))
        self.plane_residual_threshold = float(params.get("plane_residual_threshold", 2.0))
        self.residual_inflation_k = float(params.get("residual_inflation_k", 0.5))
        self.min_inliers_for_low_cov = int(params.get("min_inliers_for_low_cov", 5))
        self.alt_inflation_k = float(params.get("alt_inflation_k", 1.0))
        self.div_inflation_k = float(params.get("div_inflation_k", 5.0))
        self.min_covariance = float(params.get("min_covariance", 1e-4))
        self.max_covariance = float(params.get("max_covariance", 0.5))
        self.depth_fusion_max_altitude = float(params.get("depth_fusion_max_altitude", 1.0))
        self.depth_fusion_altitude_hysteresis = float(
            params.get("depth_fusion_altitude_hysteresis", 0.15)
        )
        self.depth_min_valid_features = int(params.get("depth_min_valid_features", 6))
        self.depth_min_valid_ratio = float(params.get("depth_min_valid_ratio", 0.35))
        self.depth_coverage_inflation_k = float(params.get("depth_coverage_inflation_k", 2.0))
        self.depth_spread_inflation_k = float(params.get("depth_spread_inflation_k", 2.0))
        self.depth_staleness_inflation_k = float(params.get("depth_staleness_inflation_k", 3.0))
        self.use_depth_overlap_roi = bool(params.get("use_depth_overlap_roi", True))
        self.use_orientation_compensation = bool(params.get("use_orientation_compensation", True))
        self.min_orientation_ray_projection = float(params.get("min_orientation_ray_projection", 0.15))
        self.debug_render_scale = float(params.get("debug_render_scale", 2.0))

        self.feature_params = dict(
            maxCorners=int(params.get("max_corners", 45)),
            qualityLevel=float(params.get("quality_level", 0.01)),
            minDistance=float(params.get("min_distance", 5.0)),
            blockSize=int(params.get("block_size", 7)),
        )
        win_size = int(params.get("win_size", 19))
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=int(params.get("max_level", 2)),
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(params.get("criteria_count", 7)),
                float(params.get("criteria_epsilon", 0.03)),
            ),
        )
        self.vel_tr = np.array(
            params.get("velocity_transform_matrix", [1.0, 0.0, 0.0, 1.0]),
            dtype=np.float32,
        ).reshape(2, 2)

        timeout_sec = float(params.get("depth_cloud_timeout_sec", 0.15))
        self.depth_grid = BottomDepthGrid(DepthGridConfig(timeout_sec=timeout_sec))
        self.height_grid = BottomDepthGrid(DepthGridConfig(timeout_sec=timeout_sec))

        self._gray_bufs = [None, None]
        self._buf_idx = 0
        self.prev_gray = None
        self.curr_pts = None
        self.frame_counter = 0
        self.initialised = False
        self._last_img_stamp_s: float | None = None
        self._local_depth_requested: bool | None = None

        self.altitude = 0.0
        self.last_lidar_alt: float | None = None
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

        self._base_roi_mask = None
        self._depth_roi_mask = None
        self._cached_mask_shape = None
        self._image_shape = None
        self._cxcy = None
        self._flow_dist_thresh_sq = self.flow_distance_threshold ** 2
        self._cos_dir_thresh = math.cos(math.radians(self.direction_threshold_deg))

        self.orig_K = None
        self.orig_D = None
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

        calibration_file = params.get("calibration_file")
        if calibration_file and (not self.use_camera_info):
            self._load_calibration_data(Path(str(calibration_file)))

    def ingest(self, topic: str, msg) -> OfflineDebugFrame | None:
        """Ingest one message and optionally return a rendered debug frame."""
        if isinstance(msg, CameraInfo):
            self.handle_camera_info(msg)
            return None
        if isinstance(msg, Imu):
            self.handle_imu(msg)
            return None
        if isinstance(msg, LaserScan):
            self.handle_lidar(msg)
            return None
        if isinstance(msg, PointCloud2):
            self.handle_depth_points(msg)
            return None
        if isinstance(msg, CompressedImage):
            return self.handle_image(msg)
        return None

    def handle_camera_info(self, msg: CameraInfo) -> None:
        """Update camera intrinsics from `CameraInfo`."""
        k = np.array(msg.k, dtype=np.float32).reshape(3, 3)
        if k[0, 0] <= 0.0 or k[1, 1] <= 0.0:
            return
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

    def handle_imu(self, msg: Imu) -> None:
        """Update the latest IMU orientation."""
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
        self.latest_roll, self.latest_pitch, self.latest_yaw = quaternion_to_euler_xyz(qx, qy, qz, qw)
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

    def handle_lidar(self, msg: LaserScan) -> None:
        """Update the filtered scalar height."""
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

    def handle_depth_points(self, msg: PointCloud2) -> None:
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

    def handle_image(self, msg: CompressedImage) -> OfflineDebugFrame | None:
        """Process one image and return a rendered debug frame if possible."""
        if self.orig_K is None:
            return None
        frame_small = self._decode_and_downsample_image(msg)
        if frame_small is None:
            return None

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
            self._last_img_stamp_s = stamp_to_seconds(msg.header.stamp)
            self.initialised = True
            return None

        dt, image_stamp_s = self._compute_dt(msg)
        if dt <= 0.0 or self.altitude < self.min_altitude:
            return None

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
                return None

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
            return None

        status_mask = status.ravel() == 1
        if not np.any(status_mask):
            self.curr_pts = None
            return None

        good_prev = self.curr_pts[status_mask].reshape(-1, 2).astype(np.float32)
        good_next = next_pts[status_mask].reshape(-1, 2).astype(np.float32)
        lk_error = err.ravel()[status_mask].astype(np.float32) if err is not None else None
        if good_prev.shape[0] < 2:
            self.curr_pts = None
            return None

        flow = (good_next - good_prev).astype(np.float32)
        inlier_mask, _ = compute_flow_inlier_mask(
            flow,
            self._flow_dist_thresh_sq,
            self._cos_dir_thresh,
        )
        if not np.any(inlier_mask):
            self.curr_pts = None
            return None

        flow = flow[inlier_mask]
        good_prev = good_prev[inlier_mask]
        good_next = good_next[inlier_mask]
        if lk_error is not None:
            lk_error = lk_error[inlier_mask]
        self.curr_pts = good_next.reshape(-1, 1, 2)
        if flow.shape[0] < 2:
            self.curr_pts = None
            return None

        flow_result = compensate_planar_motion(good_prev, flow, self._cxcy)
        corrected_flow = flow_result.corrected_flow
        if corrected_flow.shape[0] < 2:
            self.curr_pts = None
            return None

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
            return None

        self.last_lidar_alt = max(self.altitude, self.min_altitude)
        return self._render_frame(
            undistorted,
            image_stamp_s,
            corrected_flow,
            lk_error,
            dt,
            local_estimate,
            legacy_estimate,
            estimate.mode,
        )

    def _render_frame(
        self,
        gray_image: np.ndarray,
        image_stamp_s: float | None,
        corrected_flow: np.ndarray,
        lk_error: np.ndarray | None,
        dt: float,
        local_estimate: OfflineVelocityEstimate | None,
        legacy_estimate: OfflineVelocityEstimate | None,
        mode: str,
    ) -> OfflineDebugFrame | None:
        local_feature_debug = local_estimate.feature_debug if local_estimate is not None else None
        center_height_m = self._sample_center_height(image_stamp_s)
        single_velocity = None
        local_velocity = None
        if local_feature_debug is not None:
            single_velocity = self._compute_single_height_comparison(
                corrected_flow,
                local_feature_debug,
                lk_error,
                dt,
                center_height_m,
            )
            local_velocity = local_estimate.body_velocity.astype(np.float32)
        elif legacy_estimate is not None:
            single_velocity = legacy_estimate.body_velocity.astype(np.float32)

        if single_velocity is not None:
            single_velocity = (self.vel_tr @ single_velocity).astype(np.float32)
        if local_velocity is not None:
            local_velocity = (self.vel_tr @ local_velocity).astype(np.float32)

        if local_feature_debug is None:
            overlap_bounds = (0, gray_image.shape[1], 0, gray_image.shape[0])
        else:
            overlap_bounds = self.depth_grid.overlap_bounds(
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                gray_image.shape[1],
                gray_image.shape[0],
            )
        grid_centers = self.depth_grid.grid_cell_centers_image(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
        )
        comparison_debug = VelocityComparisonDebug(
            local_depth_velocity_xy=local_velocity,
            single_height_velocity_xy=single_velocity,
            scalar_height_m=float(max(self.altitude, 0.0)),
            single_height_m=center_height_m,
        )
        canvas = render_depth_debug_canvas(
            gray_image,
            overlap_bounds,
            self._scalar_grid_for_metric_support().grid,
            grid_centers,
            local_feature_debug,
            comparison_debug,
            render_scale=self.debug_render_scale,
        )
        chosen_velocity = local_velocity if (mode == "local_depth" and local_velocity is not None) else single_velocity
        speed = float(np.linalg.norm(chosen_velocity)) if chosen_velocity is not None else 0.0
        stamp_s = image_stamp_s if image_stamp_s is not None else 0.0
        return OfflineDebugFrame(
            image_bgr=canvas,
            stamp_s=stamp_s,
            speed_mps=speed,
            mode=mode,
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
    ) -> OfflineVelocityEstimate | None:
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None

        reference_stamp_s = image_stamp_s if image_stamp_s is not None else 0.0
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
        matched_rows, matched_cols, _ = scalar_grid.nearest_cell_indices(
            next_points_xy,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
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
        selected_mask = np.zeros(total_count, dtype=bool)
        selected_indices = np.flatnonzero(valid_mask)[robust_mask]
        selected_mask[selected_indices] = True

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
        return OfflineVelocityEstimate(
            body_velocity=body_velocity.astype(np.float32),
            covariance=vel_cov,
            annotated_prev=kept_prev,
            annotated_next=kept_next,
            mode="local_depth",
            feature_debug=FeatureDepthDebug(
                prev_points_xy=prev_points_xy.astype(np.float32),
                next_points_xy=next_points_xy.astype(np.float32),
                valid_mask=valid_mask,
                selected_mask=selected_mask,
                matched_depths_m=height_samples.depths_m.astype(np.float32),
                matched_rows=matched_rows,
                matched_cols=matched_cols,
                pointcloud_age_sec=float(height_samples.age_sec),
                valid_ratio=float(valid_ratio),
            ),
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
    ) -> OfflineVelocityEstimate | None:
        if self.fx is None or self.fy is None or corrected_flow.shape[0] < 2:
            return None
        alt = max(self.altitude, self.min_altitude)
        heights = np.full(corrected_flow.shape[0], alt, dtype=np.float32)
        axial_depths, valid_mask = self._axial_depths_for_points(heights, next_points_xy)
        if int(valid_mask.sum()) < 2:
            return None

        velocity_samples = self._velocity_samples_from_depths(
            corrected_flow[valid_mask],
            axial_depths[valid_mask],
            dt,
        )
        if velocity_samples.shape[0] < 2:
            return None
        robust_mask = robust_velocity_mask(velocity_samples)
        if int(robust_mask.sum()) < 2:
            return None

        weights = np.ones(velocity_samples.shape[0], dtype=np.float32)
        if lk_error is not None:
            weights = lk_error_weights(lk_error[valid_mask])
            if weights.size == 0:
                weights = np.ones(velocity_samples.shape[0], dtype=np.float32)

        kept_velocities = velocity_samples[robust_mask]
        kept_weights = weights[robust_mask]
        body_velocity, vel_cov = covariance2x2(kept_velocities, kept_weights)
        kept_flow = corrected_flow[valid_mask][robust_mask]
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
        return OfflineVelocityEstimate(
            body_velocity=body_velocity.astype(np.float32),
            covariance=vel_cov,
            annotated_prev=prev_points_xy[valid_mask][robust_mask],
            annotated_next=next_points_xy[valid_mask][robust_mask],
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

    def _sample_center_height(self, reference_stamp_s: float | None) -> float | None:
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None
        center_point = np.asarray([[self.cx, self.cy]], dtype=np.float32)
        center_height = self._scalar_grid_for_metric_support().sample_values(
            center_point,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            reference_stamp_s,
        )
        if center_height.valid_mask[0]:
            return float(center_height.depths_m[0])
        if self.altitude >= self.min_altitude:
            return float(self.altitude)
        return None

    def _compute_single_height_comparison(
        self,
        corrected_flow: np.ndarray,
        feature_debug: FeatureDepthDebug,
        lk_error: np.ndarray | None,
        dt: float,
        center_height_m: float | None,
    ) -> np.ndarray | None:
        if center_height_m is None:
            return None
        valid_mask = feature_debug.valid_mask
        selected_mask = feature_debug.selected_mask
        if not np.any(valid_mask) or not np.any(selected_mask):
            return None

        heights = np.full(int(valid_mask.sum()), float(center_height_m), dtype=np.float32)
        depths, axial_valid_mask = self._axial_depths_for_points(
            heights,
            feature_debug.next_points_xy[valid_mask],
        )
        if int(axial_valid_mask.sum()) < 2:
            return None
        velocity_samples = self._velocity_samples_from_depths(
            corrected_flow[valid_mask][axial_valid_mask],
            depths[axial_valid_mask],
            dt,
        )
        selected_valid_mask = selected_mask[valid_mask][axial_valid_mask]
        if int(selected_valid_mask.sum()) < 2:
            return None

        weights = np.ones(velocity_samples.shape[0], dtype=np.float32)
        if lk_error is not None:
            weights = lk_error_weights(lk_error[valid_mask][axial_valid_mask])
            if weights.size == 0:
                weights = np.ones(velocity_samples.shape[0], dtype=np.float32)
        body_velocity, _ = covariance2x2(
            velocity_samples[selected_valid_mask],
            weights[selected_valid_mask],
        )
        return body_velocity.astype(np.float32)

    def _orientation_compensation_ready(self) -> bool:
        return bool(self.use_orientation_compensation and self._have_imu_orientation)

    def _scalar_grid_for_metric_support(self) -> BottomDepthGrid:
        return self.height_grid if self._orientation_compensation_ready() else self.depth_grid

    def _axial_depths_for_points(
        self,
        scalar_heights_m: np.ndarray,
        image_points_xy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        if not self._orientation_compensation_ready():
            return float(raw_range_m)
        center_projection = float(self._gravity_dir_sensor[0])
        if center_projection <= self.min_orientation_ray_projection:
            return None
        return float(raw_range_m) * center_projection

    def _compute_dt(self, msg: CompressedImage) -> tuple[float, float | None]:
        stamp_s = stamp_to_seconds(msg.header.stamp)
        if stamp_s is None:
            return 0.0, None
        if self._last_img_stamp_s is None:
            self._last_img_stamp_s = stamp_s
            return 0.0, stamp_s
        dt = stamp_s - self._last_img_stamp_s
        self._last_img_stamp_s = stamp_s
        return float(max(dt, 0.0)), stamp_s

    def _store_tracking_frame(self, undistorted: np.ndarray) -> None:
        self._buf_idx = 1 - self._buf_idx
        np.copyto(self._gray_bufs[self._buf_idx], undistorted)
        self.prev_gray = self._gray_bufs[self._buf_idx]

    def _refresh_image_geometry_cache(self, image_shape: tuple[int, int]) -> None:
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
        return self._clahe.apply(image)

    def _init_scaled_camera_matrix(self) -> None:
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
        return self.distortion_model in {"equidistant", "fisheye"}

    def _init_undistort_maps(self, width: int, height: int) -> None:
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
        if self.map1 is None:
            return frame
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def _load_calibration_data(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        self.orig_K = storage.getNode("K").mat().astype(np.float32)
        self.orig_D = storage.getNode("D").mat().astype(np.float32).reshape(-1, 1)
        distortion_model_node = storage.getNode("distortion_model")
        if not distortion_model_node.empty():
            model = distortion_model_node.string().strip().lower()
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
        storage.release()

    def _clamp_covariance(self, covariance: np.ndarray) -> np.ndarray:
        cov = np.asarray(covariance, dtype=np.float32).copy()
        diag_max = float(np.max(np.diag(cov)))
        if diag_max > self.max_covariance:
            cov *= self.max_covariance / diag_max
        diag = np.diag(cov).copy()
        diag = np.maximum(diag, self.min_covariance)
        np.fill_diagonal(cov, diag)
        return cov.astype(np.float32)
