#!/usr/bin/env python3

import math
from typing import Any, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped, TransformStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Bool, Float32
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

try:
    from .quaternion_utils import (
        quat_to_euler_xyz,
        euler_xyz_to_quat,
        _quat_inverse,
        _quat_multiply,
        _quat_normalize,
        _rotation_matrix_to_quat,
    )
    from .pose_correction import innovation_mahalanobis_squared
    from .odometry_modes import (
        OpticalFlowHealthParams,
        OpticalFlowHealthTracker,
        apply_command_velocity_prior,
        apply_zero_velocity_constraint,
        zupt_candidate,
    )
except ImportError:  # pragma: no cover - direct script execution
    from quaternion_utils import (
        quat_to_euler_xyz,
        euler_xyz_to_quat,
        _quat_inverse,
        _quat_multiply,
        _quat_normalize,
        _rotation_matrix_to_quat,
    )
    from pose_correction import innovation_mahalanobis_squared
    from odometry_modes import (
        OpticalFlowHealthParams,
        OpticalFlowHealthTracker,
        apply_command_velocity_prior,
        apply_zero_velocity_constraint,
        zupt_candidate,
    )


MAX_PATH_LEN = 200

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "verbose": True,
    "publish_track": True,
    "route_threshold": 0.5,
    "max_path_points": 100,
    "tf_broadcast_rate": 15.0,
    "publish_position_uncertainty_marker": True,
    "kf_process_noise_diagonal": [0.002, 0.002, 0.002, 0.05, 0.05, 0.05],
    "zero_vel_threshold": 0.05,
    "zero_vel_time": 0.25,
    "of_zero_vel_threshold": 0.05,
    "of_zero_vel_time": 0.25,
    "zupt_of_degraded_time_s": 1.0,
    "publish_zupt_markers": True,
    "imu_stationary_lpf_alpha": 0.2,
    "imu_stationary_acc_threshold": 0.35,
    "imu_stationary_tilt_threshold_rad": 0.05,
    "imu_stationary_timeout_s": 0.2,
    "zupt_exit_imu_motion_time": 0.05,
    "zupt_process_noise_scale_position": 0.01,
    "zupt_process_noise_scale_velocity": 0.01,
    "of_zero_innovation_chi2_gate": 9.0,
    "of_zero_imu_motion_R_scale": 100.0,
    "of_R_moving_scale": 5.0,
    "of_R_speed_inflate_gain": 1.0,
    "of_compensate_orientation_in_kf": False,
    "command_velocity_topic": "/crazyflie/cmd_vel",
    "command_velocity_timeout_s": 0.5,
    "of_usable_max_variance": 0.49,
    "of_degraded_timeout_s": 1.0,
    "of_degraded_recovery_frames": 3,
    "of_command_mismatch_threshold": 0.25,
    "of_command_mismatch_slow_speed_max": 0.3,
    "of_command_mismatch_dwell_s": 0.75,
    "command_fallback_enabled": True,
    "command_fallback_sigma": 0.8,
    "command_fallback_gain": 0.1,
    "command_fallback_max_speed": 0.8,
    "command_fallback_max_gyro_norm": 0.3,
    "command_fallback_update_rate_hz": 10.0,
    "of_command_mismatch_topic": "/drone/of_command_mismatch",
    "command_fallback_active_topic": "/drone/command_fallback_active",
    "of_degraded_topic": "/drone/odometry/of_degraded",
    "pose_correction_applied_topic": "/localization/pose_correction_applied",
    "rpy_of_inflate_gain": 2.0,
    "max_imu_update_rate_hz": 0.0,
    "imu_transformation": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    "lidar_min_cos": 0.2,
    "lidar_mount_height": 0.0,
    "alt_lpf_alpha": 0.2,
    "R_lidar_alt_init": 0.8,
    "R_lidar_vz": 0.0005,
    "lidar_alt_gate_sigma": 1.5,
    "lidar_update_rate_hz": 0.0,
    "lidar_intensity_max": 250.0,
    "lidar_R_scale_min": 0.005,
    "lidar_R_scale_max": 2.0,
    "lidar_vz_gate": 1.2,
    "lidar_vz_cov_multiplier": 2500.0,
    "lidar_vz_residual_gate": 1.2,
    "imu_topic": "/crazyflie/imu/acc_derived",
    "lidar_topic": "/crazyflie/lidar/height",
    "of_odometry_topic": "/drone/of_odometry",
    "pose_correction_innovation_chi2_gate": 9.21,
    # Receiver-side cap only; the matcher already applies a route-aware limit.
    "pose_correction_max_translation_m": 10.0,
    "debug_print_imu_accels": False,
    "debug_print_imu_accels_period_s": 0.0,
}


def fix_imu(msg_in: Imu, t3: np.ndarray, q_map_inv: Optional[np.ndarray] = None) -> Imu:
    """Apply a 3x3 remap to IMU accel/gyro, optionally remapping orientation."""
    msg_out = Imu()
    msg_out.header = msg_in.header

    acc_in = np.array(
        [
            msg_in.linear_acceleration.x,
            msg_in.linear_acceleration.y,
            msg_in.linear_acceleration.z,
        ],
        dtype=np.float64,
    )
    acc_out = t3 @ acc_in
    msg_out.linear_acceleration.x = float(acc_out[0])
    msg_out.linear_acceleration.y = float(acc_out[1])
    msg_out.linear_acceleration.z = float(acc_out[2])
    msg_out.linear_acceleration_covariance = msg_in.linear_acceleration_covariance

    ang_in = np.array(
        [
            msg_in.angular_velocity.x,
            msg_in.angular_velocity.y,
            msg_in.angular_velocity.z,
        ],
        dtype=np.float64,
    )
    ang_out = t3 @ ang_in
    msg_out.angular_velocity.x = float(ang_out[0])
    msg_out.angular_velocity.y = float(ang_out[1])
    msg_out.angular_velocity.z = float(ang_out[2])
    msg_out.angular_velocity_covariance = msg_in.angular_velocity_covariance

    if q_map_inv is not None:
        q_in = np.array(
            [
                msg_in.orientation.x,
                msg_in.orientation.y,
                msg_in.orientation.z,
                msg_in.orientation.w,
            ],
            dtype=np.float64,
        )
        q_out = _quat_multiply(q_in, q_map_inv)
        q_out = _quat_normalize(q_out)
        msg_out.orientation.x = float(q_out[0])
        msg_out.orientation.y = float(q_out[1])
        msg_out.orientation.z = float(q_out[2])
        msg_out.orientation.w = float(q_out[3])
    else:
        msg_out.orientation = msg_in.orientation
    msg_out.orientation_covariance = msg_in.orientation_covariance

    return msg_out


class SimpleKFOdometryNode(Node):
    """6D linear KF [px, py, pz, vx, vy, vz] in odom from IMU attitude, optical flow, lidar, and ZUPT."""

    def __init__(self) -> None:
        super().__init__("simple_kf_odometry_node")

        self.init_params()

        self.state_dim = 6
        self._state_idx = np.arange(self.state_dim)
        self.state = np.zeros(self.state_dim, dtype=float)
        self.covariance = np.eye(self.state_dim, dtype=float) * 1e-3

        # Preallocated to avoid per-callback allocations.
        self._F = np.eye(self.state_dim, dtype=float)
        self._H_flow = np.zeros((2, self.state_dim), dtype=float)
        self._H_flow[0, 3] = 1.0
        self._H_flow[1, 4] = 1.0
        self._H_vz = np.zeros((1, self.state_dim), dtype=float)
        self._H_vz[0, 5] = 1.0
        self._H_zupt = np.zeros((3, self.state_dim), dtype=float)
        self._H_zupt[0, 3] = 1.0
        self._H_zupt[1, 4] = 1.0
        self._H_zupt[2, 5] = 1.0
        self._H_alt = np.zeros((1, self.state_dim), dtype=float)
        self._H_alt[0, 2] = 1.0
        # Event-matcher pose corrections fuse XY only.
        self._H_pos_xy = np.zeros((2, self.state_dim), dtype=float)
        self._H_pos_xy[0, 0] = 1.0
        self._H_pos_xy[1, 1] = 1.0
        self._R_zupt = 1e-4 * np.eye(3, dtype=float)

        diag_array = np.array(self.kf_process_noise_diagonal, dtype=float)
        if diag_array.shape[0] < self.state_dim:
            needed = self.state_dim - diag_array.shape[0]
            diag_array = np.concatenate([diag_array, [diag_array[-1]] * needed])
        self.Q_moving = np.diag(diag_array)

        self.zupt_process_noise_scale_position = float(
            getattr(self, "zupt_process_noise_scale_position", 0.01)
        )
        self.zupt_process_noise_scale_velocity = float(
            getattr(self, "zupt_process_noise_scale_velocity", 0.01)
        )
        self.Q_stationary = self._build_stationary_process_noise(diag_array)

        # Bases before IMU adaptive scaling.
        self._q_base_moving = self.Q_moving.copy()
        self._q_base_stationary = self.Q_stationary.copy()
        self._q_base_active = self._q_base_moving.copy()

        # Degraded-OF fallback uses a noisy command observation, not accel-integrated velocity.
        self._imu_q_scale_vel = 1.0

        # IMU motion evidence is gravity-removed accel + tilt; gyro only gates command fallback.
        self.imu_stationary_lpf_alpha = float(
            getattr(self, "imu_stationary_lpf_alpha", 0.2)
        )
        self.imu_stationary_acc_threshold = float(
            getattr(self, "imu_stationary_acc_threshold", 0.35)
        )
        self.imu_stationary_tilt_threshold_rad = float(
            getattr(self, "imu_stationary_tilt_threshold_rad", 0.05)
        )
        self.zupt_exit_imu_motion_time = float(
            getattr(self, "zupt_exit_imu_motion_time", 0.05)
        )
        self._imu_lin_acc_norm_lpf = 0.0
        self._imu_gyro_norm_lpf = 0.0
        self._last_valid_imu_time_s: Optional[float] = None
        self._zupt_imu_motion_time = 0.0
        self._zupt_last_check_time_s: Optional[float] = None

        # Down-weight OF near-zero when KF/IMU still show motion.
        self.of_zero_innovation_chi2_gate = float(
            getattr(self, "of_zero_innovation_chi2_gate", 9.0)
        )
        self.of_zero_imu_motion_R_scale = float(
            getattr(self, "of_zero_imu_motion_R_scale", 100.0)
        )

        self._update_q_active_from_imu_scale()

        # Orientation only; accel is not a KF measurement.
        self.latest_roll_imu = 0.0
        self.latest_pitch_imu = 0.0
        self.latest_yaw_imu = 0.0

        self.imu_transformation = np.array(
            self.imu_transformation, dtype=float
        ).reshape(3, 3)

        det_t = float(np.linalg.det(self.imu_transformation))
        if det_t > 0.0:
            self._imu_q_map = _rotation_matrix_to_quat(self.imu_transformation)
            self._imu_q_map_inv = _quat_inverse(self._imu_q_map)
            if bool(getattr(self, "verbose", False)):
                self.get_logger().info(
                    f"IMU axis remap enabled (proper rotation): det={det_t:.3f} "
                    f"imu_transformation={self.imu_transformation.tolist()}"
                )
        else:
            self._imu_q_map = None
            self._imu_q_map_inv = None
            self.get_logger().warning(
                "imu_transformation is NOT a proper rotation (det<=0). "
                "This means we remap linear_acceleration/angular_velocity but cannot "
                "consistently remap orientation via quaternion mapping. "
                "Gravity removal / world-frame accelerations will likely be wrong. "
                f"det={det_t:.3f} imu_transformation={self.imu_transformation.tolist()}"
            )

        self.max_imu_update_rate_hz = float(
            getattr(self, "max_imu_update_rate_hz", 0.0)
        )
        self._imu_model_accum_dt = 0.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"
        self.last_path_pos = None

        self.prev_lidar_alt = 0.0
        self.prev_lidar_time = None
        self.alt_lpf: Optional[float] = None
        self._last_alt_f: Optional[float] = None
        self._last_lidar_stamp: Optional[float] = None
        self._last_lidar_update_time: Optional[float] = None

        self._last_of_stamp: Optional[float] = None
        self._last_imu_stamp: Optional[float] = None
        self._last_imu_accel_debug_ns: int = 0

        self.stationary_time = 0.0
        self.in_zupt_mode = False
        self.speed_of: Optional[float] = None
        self._of_measurement_usable = False
        self._latest_cmd_body = np.zeros(3, dtype=float)
        self._latest_cmd_time_s: Optional[float] = None
        self._last_command_fallback_update_s: Optional[float] = None
        self._of_health = OpticalFlowHealthTracker(OpticalFlowHealthParams(
            stale_timeout_s=float(self.of_degraded_timeout_s),
            recovery_frames=int(self.of_degraded_recovery_frames),
            mismatch_threshold_mps=float(self.of_command_mismatch_threshold),
            slow_speed_max_mps=float(self.of_command_mismatch_slow_speed_max),
            mismatch_dwell_s=float(self.of_command_mismatch_dwell_s),
        ))
        self._of_degraded = True

        self.last_time = self.get_clock().now()

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.odom_pub = self.create_publisher(
            Odometry, "/drone/state_estimate", qos_profile
        )

        self.path_pub = None
        if getattr(self, "publish_track", True):
            self.path_pub = self.create_publisher(Path, "/drone/route", qos_profile)

        self.marker_pub = None
        if getattr(self, "publish_zupt_markers", False):
            self.marker_pub = self.create_publisher(
                Marker, "/drone/zupt_markers", qos_profile
            )

        self.pos_uncertainty_marker_pub = None
        if getattr(self, "publish_position_uncertainty_marker", False):
            self.pos_uncertainty_marker_pub = self.create_publisher(
                Marker, "/drone/position_uncertainty", qos_profile
            )

        # OF vs command mismatch explains why degraded/fallback became active.
        mismatch_topic = str(getattr(self, "of_command_mismatch_topic", "/drone/of_command_mismatch") or "").strip()
        self.of_command_mismatch_pub = (
            self.create_publisher(Float32, mismatch_topic, qos_profile)
            if mismatch_topic
            else None
        )

        active_topic = str(
            getattr(self, "command_fallback_active_topic", "/drone/command_fallback_active") or ""
        ).strip()
        self.command_fallback_active_pub = (
            self.create_publisher(Bool, active_topic, qos_profile)
            if active_topic
            else None
        )

        # Downstream uses this for conservative progress/map-retention while OF is down.
        of_degraded_topic = str(
            getattr(self, "of_degraded_topic", "/drone/odometry/of_degraded") or ""
        ).strip()
        self.of_degraded_pub = (
            self.create_publisher(Bool, of_degraded_topic, qos_profile)
            if of_degraded_topic
            else None
        )

        correction_applied_topic = str(
            getattr(
                self,
                "pose_correction_applied_topic",
                "/localization/pose_correction_applied",
            )
            or ""
        ).strip()
        self.pose_correction_applied_pub = (
            self.create_publisher(
                PoseWithCovarianceStamped,
                correction_applied_topic,
                qos_reliable,
            )
            if correction_applied_topic
            else None
        )

        self.imu_stationary_pub = self.create_publisher(
            Bool, "/drone/imu_stationary", qos_profile
        )
        self.command_speed_est_pub = self.create_publisher(
            Float32, "/drone/command_speed_est", qos_profile
        )
        self.of_speed_est_pub = self.create_publisher(
            Float32, "/drone/of_speed_est", qos_profile
        )
        self.zupt_active_pub = self.create_publisher(
            Bool, "/drone/zupt_active", qos_profile
        )

        self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_profile)
        self.create_subscription(
            LaserScan, self.lidar_topic, self.lidar_callback, qos_profile
        )
        self.create_subscription(
            Odometry, self.of_odometry_topic, self.optflow_callback, qos_profile
        )
        self.create_subscription(
            Twist,
            str(self.command_velocity_topic),
            self.command_velocity_callback,
            qos_profile,
        )

        self.timer_tf = self.create_timer(
            1.0 / float(self.tf_broadcast_rate), self.timer_tf_callback
        )

        # Event-matcher pose as an absolute XY measurement.
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/localization/pose_correction",
            self.pose_correction_callback,
            qos_profile,
        )

        # Cache flags to avoid getattr in the hot path.
        self._verbose = bool(getattr(self, "verbose", False))
        self._publish_zupt_markers = bool(getattr(self, "publish_zupt_markers", False))
        self._publish_pos_uncertainty = bool(
            getattr(self, "publish_position_uncertainty_marker", False)
        )

        self._of_R_moving_scale = float(getattr(self, "of_R_moving_scale", 1.0))
        self._of_R_speed_inflate_gain = float(getattr(self, "of_R_speed_inflate_gain", 0.0))
        self._of_compensate_orientation_in_kf = bool(
            getattr(self, "of_compensate_orientation_in_kf", False)
        )
        if self._of_R_moving_scale < 0.0:
            self._of_R_moving_scale = 0.0
        if self._of_R_speed_inflate_gain < 0.0:
            self._of_R_speed_inflate_gain = 0.0

        self._command_velocity_timeout_s = max(
            0.0, float(getattr(self, "command_velocity_timeout_s", 0.5))
        )

        if self._verbose:
            self.get_logger().info(
                "SimpleKFOdometryNode started (6D linear KF: [x,y,z,vx,vy,vz])."
            )

    def init_params(self) -> None:
        """Declare ROS parameters and expose them as attributes."""
        for key, value in _DEFAULT_PARAMETERS.items():
            self.declare_parameter(key, value)
            setattr(self, key, self.get_parameter(key).value)

    def _build_stationary_process_noise(self, diag_array: np.ndarray) -> np.ndarray:
        diag_stationary = np.array(diag_array, dtype=float).copy()
        if diag_stationary.shape[0] >= 3:
            diag_stationary[0:3] *= self.zupt_process_noise_scale_position
        if diag_stationary.shape[0] >= 6:
            diag_stationary[3:6] *= self.zupt_process_noise_scale_velocity
        return np.diag(diag_stationary)

    def _set_process_noise_mode(self, stationary: bool) -> None:
        """Switch moving/stationary process-noise base, then re-apply IMU scaling."""
        self._q_base_active = (
            self._q_base_stationary if stationary else self._q_base_moving
        )
        self._update_q_active_from_imu_scale()

    def _update_q_active_from_imu_scale(self) -> None:
        """Rebuild Q_active from the current base and IMU vx/vy scale."""
        q = self._q_base_active.copy()
        scale = float(self._imu_q_scale_vel)
        if scale != 1.0:
            for idx in (3, 4):  # vx, vy
                if idx < q.shape[0]:
                    q[idx, idx] *= scale
        self.Q_active = q

    def _build_F(self, dt: float) -> np.ndarray:
        """Update and return the pre-allocated state transition matrix."""
        self._F[0, 3] = dt
        self._F[1, 4] = dt
        self._F[2, 5] = dt
        return self._F

    def kf_predict(self, dt: float) -> None:
        if dt <= 0.0:
            return

        f = self._build_F(dt)
        p = self.covariance

        self.state = f @ self.state
        self.covariance = f @ p @ f.T + (self.Q_active * dt)
        self._ensure_covariance_valid()

    def _kf_update(
        self,
        z: np.ndarray,
        h: np.ndarray,
        r: np.ndarray,
        *,
        preserve_position_mean: bool = False,
    ) -> None:
        """Linear KF update. Masked position gain rows avoid a stop-time jump from delayed velocity corrections."""
        if z.size == 0:
            return

        p = self.covariance
        hp = h @ p
        s = hp @ h.T + r

        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            return

        k = hp.T @ s_inv
        if preserve_position_mean:
            k[0:3, :] = 0.0
        innovation = z - h @ self.state
        self.state = self.state + k @ innovation

        i_kh = np.eye(self.state_dim, dtype=float) - k @ h
        # Joseph form stays symmetric/PSD when selected gain rows are masked.
        self.covariance = i_kh @ p @ i_kh.T + k @ r @ k.T
        self._ensure_covariance_valid()

    def _ensure_covariance_valid(self) -> None:
        """Symmetrize covariance and enforce a small positive variance floor."""
        p = self.covariance
        np.add(p, p.T, out=p)
        p *= 0.5
        diag_view = p.ravel()[:: self.state_dim + 1]
        np.maximum(diag_view, 1e-12, out=diag_view)

    def imu_callback(self, msg: Imu) -> None:
        msg = fix_imu(msg, self.imu_transformation, self._imu_q_map_inv)

        qx = msg.orientation.x
        qy = msg.orientation.y
        qz = msg.orientation.z
        qw = msg.orientation.w
        roll, pitch, yaw = quat_to_euler_xyz(qx, qy, qz, qw)
        self.latest_roll_imu = roll
        self.latest_pitch_imu = pitch
        self.latest_yaw_imu = yaw

        # Prefer header-based dt
        if hasattr(msg, "header") and msg.header.stamp is not None:
            stamp = msg.header.stamp
            stamp_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
            if self._last_imu_stamp is None:
                dt = 1e-3
            else:
                dt = stamp_s - self._last_imu_stamp
            if dt <= 0.0:
                dt = 1e-6
            if dt > 0.1:
                dt = 0.1
            self._last_imu_stamp = stamp_s
        else:
            dt = self.get_delta_t()

        # Gravity-removed accel and tilt for ZUPT; gyro only gates command fallback during spin.
        ax = float(msg.linear_acceleration.x)
        ay = float(msg.linear_acceleration.y)
        az = float(msg.linear_acceleration.z)
        acc_body = np.array([ax, ay, az], dtype=float)
        r_wb = self._euler_to_rotmat_world_from_body(roll, pitch, yaw)
        acc_world_with_g = r_wb @ acc_body
        acc_world_no_g = acc_world_with_g.copy()
        acc_world_no_g[2] -= 9.81  # gravity along +Z world
        self._maybe_log_imu_accels(acc_body, acc_world_with_g, acc_world_no_g)

        wx = float(msg.angular_velocity.x)
        wy = float(msg.angular_velocity.y)
        wz = float(msg.angular_velocity.z)
        gyro_body = np.array([wx, wy, wz], dtype=float)
        imu_values = np.concatenate(
            (acc_world_no_g, gyro_body, np.array([roll, pitch, yaw], dtype=float))
        )
        imu_values_valid = bool(np.all(np.isfinite(imu_values)))
        if imu_values_valid:
            self._update_imu_stationary_metrics(acc_world_no_g, gyro_body)

        perform_model_update = True
        dt_model = dt
        if self.max_imu_update_rate_hz > 0.0:
            min_period = 1.0 / self.max_imu_update_rate_hz
            self._imu_model_accum_dt += dt
            if self._imu_model_accum_dt < min_period:
                perform_model_update = False
            else:
                dt_model = self._imu_model_accum_dt
                self._imu_model_accum_dt = 0.0

        if perform_model_update:
            self.kf_predict(dt_model)

        now_s = self.get_clock().now().nanoseconds * 1e-9
        if imu_values_valid:
            self._last_valid_imu_time_s = now_s
        self._update_of_degraded_status(now_s)
        self._apply_command_velocity_fallback(now_s)

        self.finish_callback_loop(dt)

    def _maybe_log_imu_accels(
        self,
        acc_body: np.ndarray,
        acc_world_with_g: np.ndarray,
        acc_world_no_g: np.ndarray,
    ) -> None:
        """Optionally print remapped body, world, and gravity-removed IMU accelerations."""
        if not bool(getattr(self, "debug_print_imu_accels", False)):
            return

        period_s = float(getattr(self, "debug_print_imu_accels_period_s", 0.5))
        if period_s < 0.0:
            period_s = 0.0

        now_ns = int(self.get_clock().now().nanoseconds)
        if period_s > 0.0:
            min_dt_ns = int(period_s * 1e9)
            if (now_ns - int(self._last_imu_accel_debug_ns)) < min_dt_ns:
                return
        self._last_imu_accel_debug_ns = now_ns

        abx, aby, abz = (float(acc_body[0]), float(acc_body[1]), float(acc_body[2]))
        awx, awy, awz = (
            float(acc_world_with_g[0]),
            float(acc_world_with_g[1]),
            float(acc_world_with_g[2]),
        )
        anx, any_, anz = (
            float(acc_world_no_g[0]),
            float(acc_world_no_g[1]),
            float(acc_world_no_g[2]),
        )
        self.get_logger().info(
            "IMU acc [m/s^2] "
            f"body(remap)=({abx:+.3f}, {aby:+.3f}, {abz:+.3f}) "
            f"world(rot)=({awx:+.3f}, {awy:+.3f}, {awz:+.3f}) "
            f"world(no_g)=({anx:+.3f}, {any_:+.3f}, {anz:+.3f})"
        )

    def _euler_to_rotmat_world_from_body(
        self, roll: float, pitch: float, yaw: float
    ) -> np.ndarray:
        """Body-to-world rotation using XYZ (roll, pitch, yaw)."""
        sr = math.sin(roll)
        cr = math.cos(roll)
        sp = math.sin(pitch)
        cp = math.cos(pitch)
        sy = math.sin(yaw)
        cy = math.cos(yaw)

        r00 = cp * cy
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr
        r10 = cp * sy
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr
        r20 = -sp
        r21 = cp * sr
        r22 = cp * cr

        return np.array(
            [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
            dtype=float,
        )

    def _update_imu_stationary_metrics(
        self, lin_acc_world_no_g: np.ndarray, gyro_body: np.ndarray
    ) -> None:
        """LPF IMU cues for ZUPT veto; quiet values are necessary but not sufficient for zero velocity."""
        acc_norm = float(np.linalg.norm(lin_acc_world_no_g))
        gyro_norm = float(np.linalg.norm(gyro_body))

        alpha = float(self.imu_stationary_lpf_alpha)
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        self._imu_lin_acc_norm_lpf = (1.0 - alpha) * self._imu_lin_acc_norm_lpf + alpha * acc_norm
        self._imu_gyro_norm_lpf = (1.0 - alpha) * self._imu_gyro_norm_lpf + alpha * gyro_norm

    def _imu_is_fresh(self, now_s: Optional[float] = None) -> bool:
        if self._last_valid_imu_time_s is None:
            return False
        now = (
            self.get_clock().now().nanoseconds * 1e-9
            if now_s is None
            else float(now_s)
        )
        timeout = max(0.0, float(self.imu_stationary_timeout_s))
        return timeout <= 0.0 or now - self._last_valid_imu_time_s <= timeout

    def _imu_is_stationary(self, now_s: Optional[float] = None) -> bool:
        if not self._imu_is_fresh(now_s):
            return False
        acc_th = float(self.imu_stationary_acc_threshold)
        if acc_th < 0.0:
            acc_th = 0.0
        tilt_th = max(0.0, float(self.imu_stationary_tilt_threshold_rad))
        tilt = math.hypot(float(self.latest_roll_imu), float(self.latest_pitch_imu))
        return self._imu_lin_acc_norm_lpf <= acc_th and tilt <= tilt_th

    def command_velocity_callback(self, msg: Twist) -> None:
        """Cache the most recent body-frame velocity command for degraded OF periods."""
        vx = float(msg.linear.x)
        vy = float(msg.linear.y)
        vz = float(msg.linear.z)
        if not all(math.isfinite(value) for value in (vx, vy, vz)):
            return
        self._latest_cmd_body[:] = (vx, vy, vz)
        self._latest_cmd_time_s = self.get_clock().now().nanoseconds * 1e-9

    def _command_state(self, now_s: float) -> tuple[bool, np.ndarray, float]:
        fresh = (
            self._latest_cmd_time_s is not None
            and (
                self._command_velocity_timeout_s <= 0.0
                or now_s - self._latest_cmd_time_s <= self._command_velocity_timeout_s
            )
        )
        body = self._latest_cmd_body.copy() if fresh else np.zeros(3, dtype=float)
        return bool(fresh), body, float(np.linalg.norm(body))

    def _command_world_xy(self, body_command: np.ndarray) -> np.ndarray:
        vx_body = float(body_command[0])
        vy_body = float(body_command[1])
        yaw = float(self.latest_yaw_imu)
        return np.array(
            [
                vx_body * math.cos(yaw) - vy_body * math.sin(yaw),
                vx_body * math.sin(yaw) + vy_body * math.cos(yaw),
            ],
            dtype=float,
        )

    def _publish_bool(self, publisher, value: bool) -> None:
        if publisher is None:
            return
        msg = Bool()
        msg.data = bool(value)
        publisher.publish(msg)

    def _update_of_degraded_status(self, now_s: float) -> None:
        self._of_degraded = bool(self._of_health.tick(now_s))
        self._publish_bool(self.of_degraded_pub, self._of_degraded)
        self._publish_bool(self.imu_stationary_pub, self._imu_is_stationary(now_s))
        self._publish_bool(self.zupt_active_pub, self.in_zupt_mode)

    def _apply_command_velocity_fallback(self, now_s: float) -> None:
        command_fresh, command_body, _command_speed = self._command_state(now_s)
        horizontal_command_speed = float(np.linalg.norm(command_body[:2]))
        gyro_limit = max(0.0, float(self.command_fallback_max_gyro_norm))
        rotation_ok = (
            gyro_limit <= 0.0 or self._imu_gyro_norm_lpf <= gyro_limit
        )
        eligible = (
            bool(self.command_fallback_enabled)
            and self._of_degraded
            and not self.in_zupt_mode
            and command_fresh
            and horizontal_command_speed > float(self.zero_vel_threshold)
            and self._imu_is_fresh(now_s)
            and rotation_ok
        )

        self._publish_bool(self.command_fallback_active_pub, eligible)
        if not eligible:
            return

        rate_hz = max(0.0, float(self.command_fallback_update_rate_hz))
        if (
            rate_hz > 0.0
            and self._last_command_fallback_update_s is not None
            and now_s - self._last_command_fallback_update_s < 1.0 / rate_hz
        ):
            return

        command_world = self._command_world_xy(command_body)
        max_speed = max(0.0, float(self.command_fallback_max_speed))
        norm = float(np.linalg.norm(command_world))
        if max_speed > 0.0 and norm > max_speed and norm > 1e-9:
            command_world *= max_speed / norm
        self.state, self.covariance = apply_command_velocity_prior(
            state=self.state,
            covariance=self.covariance,
            command_world_xy=command_world,
            gain=float(self.command_fallback_gain),
            sigma_mps=float(self.command_fallback_sigma),
        )
        self._last_command_fallback_update_s = now_s

    def _lpf_alt(self, alt_m: float) -> float:
        if self.alt_lpf is None:
            self.alt_lpf = float(alt_m)
            return self.alt_lpf
        alpha = float(self.alt_lpf_alpha)
        self.alt_lpf = (1.0 - alpha) * self.alt_lpf + alpha * float(alt_m)
        return self.alt_lpf

    def lidar_callback(self, msg: LaserScan) -> None:
        dt = 0.0

        if not msg.ranges:
            self.finish_callback_loop(dt)
            return

        raw_range_m = float(msg.ranges[0])
        if raw_range_m < 0.0:
            raw_range_m = 0.0
        if raw_range_m < 0.0 or raw_range_m > msg.range_max:
            self.finish_callback_loop(dt)
            return

        roll = self.latest_roll_imu
        pitch = self.latest_pitch_imu
        vertical_frac = math.cos(roll) * math.cos(pitch)
        if vertical_frac < self.lidar_min_cos:
            self.finish_callback_loop(dt)
            return
        alt_measured_m = raw_range_m * vertical_frac + self.lidar_mount_height

        r_intensity_scale = 1.0
        if msg.intensities:
            lidar_intensity = float(msg.intensities[0])
            imax = float(getattr(self, "lidar_intensity_max", 250.0))
            lidar_intensity = min(imax, lidar_intensity)
            frac = lidar_intensity / imax
            scale_min = float(getattr(self, "lidar_R_scale_min", 0.5))
            scale_max = float(getattr(self, "lidar_R_scale_max", 2.0))
            r_intensity_scale = scale_max - frac * (scale_max - scale_min)

        stamp = msg.header.stamp
        stamp_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
        allow_rate_update = True
        if (
            hasattr(self, "lidar_update_rate_hz")
            and self.lidar_update_rate_hz
            and self.lidar_update_rate_hz > 0.0
        ):
            min_period = 1.0 / float(self.lidar_update_rate_hz)
            if (
                self._last_lidar_update_time is not None
                and (stamp_s - self._last_lidar_update_time) < min_period
            ):
                allow_rate_update = False

        # First lidar frame: gated absolute altitude.
        if self.prev_lidar_time is None:
            vertical_frac_eff = max(
                1e-3, math.cos(self.latest_roll_imu) * math.cos(self.latest_pitch_imu)
            )
            r_alt = (float(self.R_lidar_alt_init) * r_intensity_scale) / (
                vertical_frac_eff * vertical_frac_eff
            )

            alt_resid = alt_measured_m - float(self.state[2])
            sigma_alt = math.sqrt(
                max(float(self.covariance[2, 2]), 1e-12) + float(r_alt)
            )

            if allow_rate_update and (
                abs(alt_resid) <= self.lidar_alt_gate_sigma * sigma_alt
            ):
                z_alt = np.array([alt_measured_m], dtype=float)
                r_mat = np.array([[r_alt]], dtype=float)
                self._kf_update(z_alt, self._H_alt, r_mat)

            self.prev_lidar_alt = alt_measured_m
            self.prev_lidar_time = self.get_clock().now()
            self._last_lidar_stamp = stamp_s
            if allow_rate_update:
                self._last_lidar_update_time = stamp_s
            self.finish_callback_loop(dt)
            return

        # Later frames: vz from filtered altitude derivative.
        vz_for_cov = None
        if self._last_lidar_stamp is not None:
            dt_lidar = stamp_s - self._last_lidar_stamp
        else:
            dt_lidar = None

        if dt_lidar is not None and dt_lidar >= 1e-4:
            alt_f = self._lpf_alt(alt_measured_m)
            if self._last_alt_f is None:
                self._last_alt_f = alt_f
            else:
                vz_for_cov = (alt_f - self._last_alt_f) / dt_lidar
            self._last_alt_f = alt_f

        if (vz_for_cov is not None) and allow_rate_update:
            r_vz = float(self.R_lidar_vz) * r_intensity_scale
            angle_mag = abs(self.latest_roll_imu) + abs(self.latest_pitch_imu)
            r_vz *= 1.0 + angle_mag * self.rpy_of_inflate_gain

            vz_gate_mag = float(getattr(self, "lidar_vz_gate", 3.0))
            vz_resid = vz_for_cov - float(self.state[5])
            vz_resid_gate = float(getattr(self, "lidar_vz_residual_gate", 0.6))
            vz_cov_mult = float(getattr(self, "lidar_vz_cov_multiplier", 25.0))
            if (abs(vz_for_cov) > vz_gate_mag) or (abs(vz_resid) > vz_resid_gate):
                r_vz *= vz_cov_mult

            z_vz = np.array([vz_for_cov], dtype=float)
            r_mat_vz = np.array([[r_vz]], dtype=float)
            self._kf_update(
                z_vz,
                self._H_vz,
                r_mat_vz,
                preserve_position_mean=True,
            )

        self.prev_lidar_alt = alt_measured_m
        self.prev_lidar_time = self.get_clock().now()
        self._last_lidar_stamp = stamp_s
        if allow_rate_update:
            self._last_lidar_update_time = stamp_s

        self.finish_callback_loop(dt)

    def optflow_callback(self, odom_msg: Odometry) -> None:
        dt = 0.0

        if hasattr(odom_msg, "header") and odom_msg.header.stamp is not None:
            stamp = odom_msg.header.stamp
            stamp_s = float(stamp.sec) + 1e-9 * float(stamp.nanosec)
            if self._last_of_stamp is not None:
                dt = stamp_s - self._last_of_stamp
                if dt <= 0.0:
                    dt = 0.0
                elif dt > 0.1:
                    dt = 0.1
            self._last_of_stamp = stamp_s

        now_s = self.get_clock().now().nanoseconds * 1e-9
        vx_body = float(odom_msg.twist.twist.linear.x)
        # of_localization already applies camera-to-body; no extra Y sign flip here.
        vy_body = float(odom_msg.twist.twist.linear.y)
        self.speed_of = math.hypot(vx_body, vy_body)

        imu_stationary = self._imu_is_stationary(now_s)

        cov_vx_x = float(odom_msg.twist.covariance[0])
        cov_vx_y = float(odom_msg.twist.covariance[1])
        cov_vy_x = float(odom_msg.twist.covariance[6])
        cov_vy_y = float(odom_msg.twist.covariance[7])
        r_of_mat_body = np.array(
            [[cov_vx_x, cov_vx_y], [cov_vy_x, cov_vy_y]],
            dtype=float,
        )

        # Large/invalid covariance means no usable flow; a numerical-zero velocity is not health proof.
        raw_values = np.array(
            [vx_body, vy_body, cov_vx_x, cov_vx_y, cov_vy_x, cov_vy_y],
            dtype=float,
        )
        max_variance = max(0.0, float(self.of_usable_max_variance))
        measurement_usable = bool(
            np.all(np.isfinite(raw_values))
            and np.allclose(r_of_mat_body, r_of_mat_body.T, rtol=1e-3, atol=1e-9)
            and cov_vx_x > 0.0
            and cov_vy_y > 0.0
            and (max_variance <= 0.0 or max(cov_vx_x, cov_vy_y) <= max_variance)
            and float(np.linalg.det(r_of_mat_body)) >= -1e-9
        )
        self._of_measurement_usable = measurement_usable

        # Inflate OF R when moving or at high speed so the filter trusts OF less.
        r_scale = 1.0
        if (not imu_stationary) and (self._of_R_moving_scale != 1.0):
            r_scale *= float(self._of_R_moving_scale)
        if self._of_R_speed_inflate_gain > 0.0:
            r_scale *= 1.0 + float(self._of_R_speed_inflate_gain) * float(self.speed_of) * float(self.speed_of)
        if r_scale != 1.0:
            r_of_mat_body *= r_scale

        angle_mag = math.sqrt(
            self.latest_roll_imu * self.latest_roll_imu
            + self.latest_pitch_imu * self.latest_pitch_imu
        )
        factor = 1.0 + angle_mag * self.rpy_of_inflate_gain
        r_of_mat_body *= factor

        # Default: OF already compensated roll/pitch; optional flag applies full RPY here.
        if self._of_compensate_orientation_in_kf:
            r_wb = self._euler_to_rotmat_world_from_body(
                self.latest_roll_imu,
                self.latest_pitch_imu,
                self.latest_yaw_imu,
            )
            vel_body = np.array([vx_body, vy_body, 0.0], dtype=float)
            vel_world = r_wb @ vel_body
            vx_world = float(vel_world[0])
            vy_world = float(vel_world[1])
            r_xy = r_wb[0:2, 0:2]
            r_of_mat_world = r_xy @ r_of_mat_body @ r_xy.T
        else:
            yaw = self.latest_yaw_imu
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            vx_world = vx_body * cos_y - vy_body * sin_y
            vy_world = vx_body * sin_y + vy_body * cos_y

            r_yaw = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=float)
            r_of_mat_world = r_yaw @ r_of_mat_body @ r_yaw.T

        # Floor variance so tiny upstream R cannot make the KF overconfident.
        min_var = 1e-8
        diag = np.diag(r_of_mat_world)
        diag = np.where(diag < min_var, min_var, diag)
        np.fill_diagonal(r_of_mat_world, diag)

        z_flow = np.array([vx_world, vy_world], dtype=float)

        command_fresh, command_body, _command_speed = self._command_state(now_s)
        command_world = self._command_world_xy(command_body)
        horizontal_command_speed = float(np.linalg.norm(command_world))
        speed_meas = float(np.linalg.norm(z_flow))
        mismatch = (
            float(np.linalg.norm(command_world - z_flow)) if command_fresh else 0.0
        )

        self._of_degraded = bool(self._of_health.observe(
            now_s=now_s,
            measurement_usable=measurement_usable,
            of_speed_mps=speed_meas,
            command_fresh=command_fresh,
            command_speed_mps=horizontal_command_speed,
            command_mismatch_mps=mismatch,
        ))

        self._publish_bool(self.imu_stationary_pub, imu_stationary)
        self._publish_bool(self.zupt_active_pub, self.in_zupt_mode)
        self._publish_bool(self.of_degraded_pub, self._of_degraded)
        if self.command_speed_est_pub is not None:
            m = Float32()
            m.data = float(horizontal_command_speed)
            self.command_speed_est_pub.publish(m)
        if self.of_speed_est_pub is not None:
            m = Float32()
            m.data = float(speed_meas)
            self.of_speed_est_pub.publish(m)
        if self.of_command_mismatch_pub is not None:
            m = Float32()
            m.data = mismatch
            self.of_command_mismatch_pub.publish(m)

        # Skip fusion when OF is unusable/degraded; command fallback may run on IMU ticks.
        if not measurement_usable or self._of_degraded:
            self.finish_callback_loop(dt)
            return

        # Near-zero OF vs IMU/KF motion: inflate R and chi2-gate the update.
        if speed_meas <= float(self.of_zero_vel_threshold):
            if (not imu_stationary) and (self.of_zero_imu_motion_R_scale > 1.0):
                r_of_mat_world = r_of_mat_world * float(self.of_zero_imu_motion_R_scale)

            gate = float(self.of_zero_innovation_chi2_gate)
            if gate > 0.0:
                v_pred = np.array([float(self.state[3]), float(self.state[4])], dtype=float)
                delta = z_flow - v_pred
                s = self.covariance[3:5, 3:5] + r_of_mat_world
                try:
                    s_inv = np.linalg.inv(s)
                except np.linalg.LinAlgError:
                    s_inv = None
                if s_inv is not None:
                    d2 = float(delta.T @ s_inv @ delta)
                    if d2 > gate:
                        self.finish_callback_loop(dt)
                        return

        self._kf_update(
            z_flow,
            self._H_flow,
            r_of_mat_world,
            preserve_position_mean=True,
        )
        self.finish_callback_loop(dt)

    def check_and_apply_zupt(self, _callback_dt: float) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self._zupt_last_check_time_s is None:
            elapsed_s = 0.0
        else:
            elapsed_s = min(max(now_s - self._zupt_last_check_time_s, 0.0), 0.1)
        self._zupt_last_check_time_s = now_s

        self._of_degraded = bool(self._of_health.tick(now_s))
        command_fresh, _command_body, command_speed = self._command_state(now_s)
        imu_fresh = self._imu_is_fresh(now_s)
        imu_stationary = self._imu_is_stationary(now_s)
        tilt = math.hypot(float(self.latest_roll_imu), float(self.latest_pitch_imu))
        speed_threshold = min(
            max(0.0, float(self.zero_vel_threshold)),
            max(0.0, float(self.of_zero_vel_threshold)),
        )
        candidate = zupt_candidate(
            command_fresh=command_fresh,
            command_speed_mps=command_speed,
            imu_fresh=imu_fresh,
            imu_acceleration_mps2=float(self._imu_lin_acc_norm_lpf),
            tilt_rad=tilt,
            imu_acceleration_threshold_mps2=float(self.imu_stationary_acc_threshold),
            tilt_threshold_rad=float(self.imu_stationary_tilt_threshold_rad),
            of_degraded=self._of_degraded,
            of_measurement_usable=self._of_measurement_usable,
            of_speed_mps=self.speed_of,
            speed_threshold_mps=speed_threshold,
        )

        if candidate:
            self.stationary_time += elapsed_s
        else:
            self.stationary_time = 0.0

        dwell_s = (
            float(self.zupt_of_degraded_time_s)
            if self._of_degraded
            else max(float(self.zero_vel_time), float(self.of_zero_vel_time))
        )
        if self.stationary_time >= max(0.0, dwell_s):
            if not self.in_zupt_mode:
                self.in_zupt_mode = True
                self.stationary_time = 0.0
                self._zupt_imu_motion_time = 0.0
                self._set_process_noise_mode(stationary=True)
                if self._verbose:
                    mode = "OF_DEGRADED" if self._of_degraded else "OF_NOMINAL"
                    self.get_logger().info(f"ZUPT triggered ({mode}).")
                self.kf_update_zero_velocity()
                if self._publish_zupt_markers:
                    px, py, pz = self.state[0:3]
                    self.publish_zupt_marker(px, py, pz)

        if self.in_zupt_mode:
            immediate_exit = (
                not command_fresh
                or command_speed > speed_threshold
                or not imu_fresh
                or (
                    not self._of_degraded
                    and self._of_measurement_usable
                    and self.speed_of is not None
                    and self.speed_of > float(self.of_zero_vel_threshold)
                )
            )
            if immediate_exit:
                self.in_zupt_mode = False
                self.stationary_time = 0.0
                self._zupt_imu_motion_time = 0.0
                self._set_process_noise_mode(stationary=False)
                if self._verbose:
                    self.get_logger().info(
                        "Exit ZUPT due to command/OF motion evidence."
                    )
            elif imu_stationary:
                self._zupt_imu_motion_time = 0.0
            else:
                self._zupt_imu_motion_time += elapsed_s

            if self._zupt_imu_motion_time >= float(self.zupt_exit_imu_motion_time):
                self.in_zupt_mode = False
                self._set_process_noise_mode(stationary=False)
                self._zupt_imu_motion_time = 0.0
                if self._verbose:
                    self.get_logger().info("Exit ZUPT due to IMU motion.")

        self._publish_bool(self.of_degraded_pub, self._of_degraded)
        self._publish_bool(self.imu_stationary_pub, imu_stationary)
        self._publish_bool(self.zupt_active_pub, self.in_zupt_mode)

    def kf_update_zero_velocity(self) -> None:
        """Constrain velocity without moving position; a ZUPT is not a new position observation."""
        self.state, self.covariance = apply_zero_velocity_constraint(
            state=self.state,
            covariance=self.covariance,
            measurement_variance=float(self._R_zupt[0, 0]),
        )
        self._ensure_covariance_valid()

    def publish_odom(self) -> None:
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"

        px, py, pz = (float(self.state[0]), float(self.state[1]), float(self.state[2]))
        vx, vy, vz = (float(self.state[3]), float(self.state[4]), float(self.state[5]))

        msg.pose.pose.position.x = px
        msg.pose.pose.position.y = py
        msg.pose.pose.position.z = pz

        roll = self.latest_roll_imu
        pitch = self.latest_pitch_imu
        yaw = self.latest_yaw_imu
        qx, qy, qz, qw = euler_xyz_to_quat(roll, pitch, yaw)
        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)

        msg.twist.twist.linear.x = vx
        msg.twist.twist.linear.y = vy
        msg.twist.twist.linear.z = vz

        pose_cov = np.zeros((6, 6), dtype=float)
        pose_cov[0:3, 0:3] = self.covariance[0:3, 0:3]
        msg.pose.covariance = pose_cov.flatten().tolist()

        twist_cov = np.zeros((6, 6), dtype=float)
        twist_cov[0:3, 0:3] = self.covariance[3:6, 3:6]
        msg.twist.covariance = twist_cov.flatten().tolist()

        self.odom_pub.publish(msg)
        self.publish_position_uncertainty_marker_msg()

    def publish_path(self) -> None:
        if self.path_pub is None:
            return

        px = float(self.state[0])
        py = float(self.state[1])
        pz = float(self.state[2])
        yaw = self.latest_yaw_imu

        roll = self.latest_roll_imu
        pitch = self.latest_pitch_imu

        if self.last_path_pos is None:
            self.last_path_pos = (px, py, pz)

        dx = px - self.last_path_pos[0]
        dy = py - self.last_path_pos[1]
        dz = pz - self.last_path_pos[2]
        dist_moved = math.sqrt(dx * dx + dy * dy + dz * dz)

        if dist_moved < float(getattr(self, "route_threshold", 0.5)):
            return

        self.last_path_pos = (px, py, pz)

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "odom"
        ps.pose.position.x = px
        ps.pose.position.y = py
        ps.pose.position.z = pz

        qx, qy, qz, qw = euler_xyz_to_quat(roll, pitch, yaw)
        ps.pose.orientation.x = float(qx)
        ps.pose.orientation.y = float(qy)
        ps.pose.orientation.z = float(qz)
        ps.pose.orientation.w = float(qw)

        self.path_msg.poses.append(ps)
        max_len = int(getattr(self, "max_path_points", MAX_PATH_LEN))
        if len(self.path_msg.poses) > max_len:
            self.path_msg.poses.pop(0)

        self.path_msg.header.stamp = ps.header.stamp
        self.path_pub.publish(self.path_msg)

    def timer_tf_callback(self) -> None:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "crazyflie/base_link"

        px = float(self.state[0])
        py = float(self.state[1])
        pz = float(self.state[2])

        roll = self.latest_roll_imu
        pitch = self.latest_pitch_imu
        yaw = self.latest_yaw_imu

        t.transform.translation.x = px
        t.transform.translation.y = py
        t.transform.translation.z = pz

        qx, qy, qz, qw = euler_xyz_to_quat(roll, pitch, yaw)
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)

        self.tf_broadcaster.sendTransform(t)

    def publish_zupt_marker(self, x: float, y: float, z: float) -> None:
        if self.marker_pub is None:
            return
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = int(self.get_clock().now().nanoseconds % 1000000)
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.lifetime.sec = 0
        self.marker_pub.publish(marker)

    def publish_position_uncertainty_marker_msg(self) -> None:
        if (not self._publish_pos_uncertainty) or (self.pos_uncertainty_marker_pub is None):
            return

        px = float(self.state[0])
        py = float(self.state[1])
        pz = float(self.state[2])

        cov = self.covariance
        sigma_x = math.sqrt(max(float(cov[0, 0]), 0.0))
        sigma_y = math.sqrt(max(float(cov[1, 1]), 0.0))
        sigma_z = math.sqrt(max(float(cov[2, 2]), 0.0))

        min_scale = 0.02
        scale_x = max(2.0 * sigma_x, min_scale)
        scale_y = max(2.0 * sigma_y, min_scale)
        scale_z = max(2.0 * sigma_z, min_scale)

        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "position_uncertainty"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = px
        marker.pose.position.y = py
        marker.pose.position.z = pz
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale_x
        marker.scale.y = scale_y
        marker.scale.z = scale_z
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.4
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 0

        self.pos_uncertainty_marker_pub.publish(marker)

    def finish_callback_loop(self, dt: float) -> None:
        self.check_and_apply_zupt(dt)
        self.publish_odom()
        self.publish_path()

    def get_delta_t(self) -> float:
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0:
            dt = 1e-6
        if dt > 0.1:
            dt = 0.1
        self.last_time = now
        return dt

    def pose_correction_callback(self, msg: PoseWithCovarianceStamped) -> None:
        """Fuse an event-matcher pose as an absolute XY (not Z) measurement in odom."""
        frame_dbg = getattr(msg.header, "frame_id", "") or ""
        x_dbg = float(msg.pose.pose.position.x)
        y_dbg = float(msg.pose.pose.position.y)
        if len(msg.pose.covariance) == 36:
            rxx_dbg = float(msg.pose.covariance[0])
            ryy_dbg = float(msg.pose.covariance[7])
            rxy_dbg = float(msg.pose.covariance[1])
        else:
            rxx_dbg = float("nan")
            ryy_dbg = float("nan")
            rxy_dbg = float("nan")
        self.get_logger().info(
            f"pose_correction rx: frame_id='{frame_dbg}' pos=({x_dbg:.3f}, {y_dbg:.3f}) "
            f"Rxx={rxx_dbg:.3g} Ryy={ryy_dbg:.3g} Rxy={rxy_dbg:.3g}"
        )

        frame_id = getattr(msg.header, "frame_id", "") or ""
        if frame_id not in ("odom", "", None):
            self.get_logger().warning(
                f"pose_correction ignored: unexpected frame_id='{frame_id}'"
            )
            return

        px = float(msg.pose.pose.position.x)
        py = float(msg.pose.pose.position.y)
        z = np.array([px, py], dtype=float)
        if not np.all(np.isfinite(z)):
            self.get_logger().warning("pose_correction ignored: non-finite XY")
            return

        cov_list = list(msg.pose.covariance)
        if len(cov_list) != 36:
            self.get_logger().warning(
                f"pose_correction ignored: covariance len={len(cov_list)} (expected 36)"
            )
            return
        r_full = np.array(cov_list, dtype=float).reshape(6, 6)
        r = r_full[:2, :2]
        if not np.all(np.isfinite(r)):
            self.get_logger().warning("pose_correction ignored: non-finite R")
            return

        # Symmetrize and floor R so upstream numerical issues cannot break the KF.
        r = 0.5 * (r + r.T)
        diag = np.diag(r)
        min_var = 1e-8
        diag = np.where(diag < min_var, min_var, diag)
        idx = np.arange(2)
        r[idx, idx] = diag

        try:
            np.linalg.cholesky(r + np.eye(2, dtype=float) * 1e-9)
        except np.linalg.LinAlgError:
            self.get_logger().warning("pose_correction ignored: R not PD")
            return

        pre_xy = (float(self.state[0]), float(self.state[1]))
        innovation_xy = np.array(
            [float(px - pre_xy[0]), float(py - pre_xy[1])],
            dtype=float,
        )
        innovation_norm = float(np.linalg.norm(innovation_xy))
        max_translation = max(
            0.0,
            float(getattr(self, "pose_correction_max_translation_m", 0.0)),
        )
        if max_translation > 0.0 and innovation_norm > max_translation:
            self.get_logger().warning(
                "pose_correction ignored: "
                f"innovation_norm={innovation_norm:.3f}m exceeds "
                f"{max_translation:.3f}m"
            )
            return

        innovation_m2 = innovation_mahalanobis_squared(
            innovation_xy,
            self.covariance[:2, :2],
            r,
        )
        innovation_gate = max(
            0.0,
            float(getattr(self, "pose_correction_innovation_chi2_gate", 0.0)),
        )
        if innovation_gate > 0.0 and innovation_m2 > innovation_gate:
            self.get_logger().warning(
                "pose_correction ignored: "
                f"innovation_chi2={innovation_m2:.3f} exceeds "
                f"{innovation_gate:.3f}"
            )
            return

        self._kf_update(z, self._H_pos_xy, r)
        post_xy = (float(self.state[0]), float(self.state[1]))
        if self.pose_correction_applied_pub is not None:
            applied = PoseWithCovarianceStamped()
            applied.header.stamp = self.get_clock().now().to_msg()
            applied.header.frame_id = "odom"
            applied.pose.pose.position.x = post_xy[0]
            applied.pose.pose.position.y = post_xy[1]
            applied.pose.pose.position.z = float(self.state[2])
            applied.pose.pose.orientation.w = 1.0
            applied_covariance = np.zeros((6, 6), dtype=float)
            applied_covariance[:2, :2] = self.covariance[:2, :2]
            applied.pose.covariance = applied_covariance.reshape(-1).tolist()
            self.pose_correction_applied_pub.publish(applied)
        dx = float(post_xy[0] - pre_xy[0])
        dy = float(post_xy[1] - pre_xy[1])
        self.get_logger().info(
            f"pose_correction applied: innovation=({innovation_xy[0]:.3f}, {innovation_xy[1]:.3f}) "
            f"chi2={innovation_m2:.3f} "
            f"dxy=({dx:.3f}, {dy:.3f}) new=({post_xy[0]:.3f}, {post_xy[1]:.3f})"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimpleKFOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
