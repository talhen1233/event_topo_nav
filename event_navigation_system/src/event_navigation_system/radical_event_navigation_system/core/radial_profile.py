#!/usr/bin/env python3
"""1D periodic radial distance profile with frustum-gated confidence and filtering."""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.ndimage import median_filter, gaussian_filter1d


@dataclass
class SensorConfig:
    """TOF sensor frustum and range configuration."""
    directions: List[float]
    fov_deg: float
    max_range: float
    # Per-sensor ranges limit each ray by the longest covering frustum; else max_range is uniform.
    max_ranges: Optional[List[float]] = None


class RadialProfile:
    """360° radial distances with frustum-gated confidence and periodic filtering."""

    def __init__(
        self,
        angular_resolution_deg: float = 5.0,
        sensor_config: Optional[SensorConfig] = None,
        confidence_decay_rate: float = 0.35,
        voxel_proximity_stop: float = 0.2,
        median_window: int = 7,
        gaussian_sigma_samples: float = 1.5,
        vertical_pitch_half_angle_deg: float = 0.0,
        confidence_rise_rate: float = 1.0,
    ):
        self.angular_resolution = float(angular_resolution_deg)
        self.num_rays = int(360.0 / self.angular_resolution)

        if sensor_config is None:
            self.sensor_config = SensorConfig(
                directions=[0.0, 90.0, 180.0, 270.0],
                fov_deg=45.0,
                max_range=4.0,
            )
        else:
            self.sensor_config = sensor_config
        
        self.fov_half = self.sensor_config.fov_deg / 2.0
        self.confidence_decay_rate = float(confidence_decay_rate)
        # 1.0 jumps observed rays to full confidence immediately.
        self.confidence_rise_rate = float(confidence_rise_rate)
        self.voxel_proximity_stop = float(voxel_proximity_stop)
        self.median_window = int(median_window) if int(median_window) % 2 == 1 else int(median_window) + 1
        self.gaussian_sigma = float(gaussian_sigma_samples)
        self.vertical_pitch_half_angle_deg = float(max(0.0, vertical_pitch_half_angle_deg))
        self._vertical_pitch_num_samples = 3
        self.angles = np.linspace(0.0, 360.0, self.num_rays, endpoint=False)
        self.angles_rad = np.deg2rad(self.angles).astype(np.float32)
        self.cos_angles = np.cos(self.angles_rad).astype(np.float32)
        self.sin_angles = np.sin(self.angles_rad).astype(np.float32)
        self.raw_distances = np.full(self.num_rays, self.sensor_config.max_range, dtype=np.float32)
        self.confidence = np.zeros(self.num_rays, dtype=np.float32)
        self.filtered_distances = np.full(self.num_rays, self.sensor_config.max_range, dtype=np.float32)
        self.last_measurement_time = np.full(self.num_rays, -np.inf, dtype=np.float64)
        self._effective_max_ranges = np.full(
            self.num_rays, float(self.sensor_config.max_range), dtype=np.float32
        )
    
    def update_from_voxels(
        self,
        voxel_map: Dict[Tuple[int, int, int], any],
        drone_position: np.ndarray,
        drone_yaw: float,
        voxel_size: float,
        current_time: float,
    ) -> None:
        """Cast rays through the voxel map and update per-ray distances and confidence."""
        max_range_default = float(self.sensor_config.max_range)
        if self.vertical_pitch_half_angle_deg <= 0.0:
            pitch_samples_rad = np.array([0.0], dtype=np.float32)
        else:
            half = float(self.vertical_pitch_half_angle_deg)
            pitch_samples_deg = np.linspace(-half, half, self._vertical_pitch_num_samples, dtype=np.float32)
            pitch_samples_rad = np.deg2rad(pitch_samples_deg).astype(np.float32)
        
        for i in range(self.num_rays):
            angle_deg = float(self.angles[i])
            cos_yaw = self.cos_angles[i]
            sin_yaw = self.sin_angles[i]
            max_range = self._get_effective_max_range(angle_deg, drone_yaw, max_range_default)
            self._effective_max_ranges[i] = float(max_range)

            # Pitch sweep keeps the longest free distance, not the first hit.
            best_distance = 0.0
            for pitch_rad in pitch_samples_rad:
                cos_pitch = float(np.cos(pitch_rad))
                sin_pitch = float(np.sin(pitch_rad))
                ray_dir = np.array(
                    [cos_yaw * cos_pitch, sin_yaw * cos_pitch, sin_pitch],
                    dtype=np.float32,
                )
                hit_distance = self._cast_ray_through_voxels(
                    drone_position,
                    ray_dir,
                    voxel_map,
                    voxel_size,
                    max_range,
                )
                if hit_distance > best_distance:
                    best_distance = hit_distance
            hit_distance = best_distance
            in_frustum = self._is_in_frustum(angle_deg, drone_yaw)
            self.raw_distances[i] = hit_distance

            if in_frustum:
                self.confidence[i] = min(
                    1.0,
                    float(self.confidence[i]) + self.confidence_rise_rate,
                )
                self.last_measurement_time[i] = current_time
            else:
                # Decay from the current value, not from 1.0.
                dt = current_time - self.last_measurement_time[i]
                if dt >= 0:
                    self.confidence[i] = float(
                        self.confidence[i] * np.exp(-self.confidence_decay_rate * dt)
                    )
                else:
                    self.confidence[i] = 0.0
                # Stamp every update so decay dt is relative to the previous cycle, not last frustum hit.
                self.last_measurement_time[i] = current_time

    def reset(self) -> None:
        """Reset all per-ray state as if the profile was freshly constructed."""
        max_r = float(self.sensor_config.max_range)
        self.raw_distances.fill(max_r)
        self.filtered_distances.fill(max_r)
        self.confidence.fill(0.0)
        self.last_measurement_time.fill(-np.inf)
        self._effective_max_ranges.fill(max_r)

    def _get_effective_max_range(
        self,
        angle_deg: float,
        drone_yaw_rad: float,
        default_max_range: float,
    ) -> float:
        """Longest covering per-sensor range, or default_max_range if none apply."""
        per_sensor = getattr(self.sensor_config, "max_ranges", None)
        if not per_sensor:
            return float(default_max_range)

        drone_yaw_deg = float(np.rad2deg(drone_yaw_rad) % 360.0)
        best_range = 0.0
        for sensor_dir, sensor_range in zip(self.sensor_config.directions, per_sensor):
            global_dir = (float(sensor_dir) + drone_yaw_deg) % 360.0
            diff = abs(self._angle_diff(angle_deg, global_dir))
            if diff <= self.fov_half:
                best_range = max(best_range, float(sensor_range))

        if best_range <= 0.0:
            return float(default_max_range)
        return best_range

    def _is_in_frustum(self, angle_deg: float, drone_yaw_rad: float) -> bool:
        """Check if angle is within any sensor's field of view."""
        drone_yaw_deg = np.rad2deg(drone_yaw_rad) % 360.0
        
        for sensor_dir in self.sensor_config.directions:
            global_dir = (sensor_dir + drone_yaw_deg) % 360.0
            diff = abs(self._angle_diff(angle_deg, global_dir))
            if diff <= self.fov_half:
                return True
        
        return False
    
    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Compute smallest angle difference (handling wrap-around)."""
        diff = (a1 - a2) % 360.0
        if diff > 180.0:
            diff -= 360.0
        return diff
    
    def _cast_ray_through_voxels(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        voxel_map: Dict[Tuple[int, int, int], any],
        voxel_size: float,
        max_range: float,
    ) -> float:
        """First voxel intersection along the ray, or max_range if none."""
        if not voxel_map:
            return max_range

        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-9:
            return max_range
        direction = direction / dir_norm
        step_size = voxel_size * 1.0
        num_steps = int(max_range / step_size)
        
        for step in range(num_steps):
            t = step * step_size
            point = origin + t * direction
            if self._is_point_near_occupied_voxel(point, voxel_map, voxel_size, self.voxel_proximity_stop):
                return t
        
        return max_range

    def _is_point_near_occupied_voxel(
        self,
        point: np.ndarray,
        voxel_map: Dict[Tuple[int, int, int], any],
        voxel_size: float,
        threshold: float,
    ) -> bool:
        """True if point is within threshold of an occupied voxel (squared-distance check)."""
        if not voxel_map:
            return False
        ix = int(np.floor(point[0] / voxel_size))
        iy = int(np.floor(point[1] / voxel_size))
        iz = int(np.floor(point[2] / voxel_size))
        r = 1 if threshold > 0.5 * voxel_size else 0
        thr_sq = float(threshold * threshold)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    v_idx = (ix + dx, iy + dy, iz + dz)
                    if v_idx in voxel_map:
                        voxel = voxel_map[v_idx]
                        if hasattr(voxel, 'occupancy') and not voxel.occupancy:
                            continue
                        vc = np.array([
                            (v_idx[0] + 0.5) * voxel_size,
                            (v_idx[1] + 0.5) * voxel_size,
                            (v_idx[2] + 0.5) * voxel_size,
                        ], dtype=np.float32)
                        diff = point - vc
                        if (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]) <= thr_sq:
                            return True
        return False

    def apply_filtering(self, confidence_threshold: float = 0.3) -> None:
        """Fill low-confidence gaps, then median+Gaussian-smooth the periodic profile."""
        high_conf_mask = self.confidence >= confidence_threshold
        gaps = self._find_contiguous_gaps(high_conf_mask)
        interpolated = self._interpolate_low_confidence_regions(confidence_threshold)
        raw_periodic = np.concatenate([
            interpolated[-self.median_window//2:],
            interpolated,
            interpolated[:self.median_window//2],
        ])
        
        median_filtered = median_filter(raw_periodic, size=self.median_window, mode='nearest')
        median_result = median_filtered[self.median_window//2 : self.median_window//2 + self.num_rays]
        # Wrap Gaussian so the 0°/360° seam does not create an edge artifact.
        smooth = gaussian_filter1d(median_result, sigma=self.gaussian_sigma, mode='wrap')
        self.filtered_distances = smooth.astype(np.float32)

        for gap_start, gap_end in gaps:
            left_idx = self._find_nearest_high_conf_backward(gap_start, high_conf_mask)
            right_idx = self._find_nearest_high_conf_forward(gap_end, high_conf_mask)
            if left_idx is None or right_idx is None:
                continue
            gap_indices = self._get_gap_indices(gap_start, gap_end)
            L = len(gap_indices)
            if L == 0:
                continue
            y0 = float(self.filtered_distances[left_idx])
            y1 = float(self.filtered_distances[right_idx])
            span = float(L + 1)
            deriv = self._central_derivative(self.filtered_distances)
            m0 = float(deriv[left_idx])
            m1 = float(deriv[right_idx])
            t = (np.arange(L, dtype=np.float32) + 1.0) / span
            h00 = (2.0 * t - 3.0) * t * t + 1.0
            h10 = (t * t * (t - 2.0) + t)
            h01 = (-2.0 * t + 3.0) * t * t
            h11 = (t * t * (t - 1.0))
            y_hermite = h00 * y0 + h10 * (span * m0) + h01 * y1 + h11 * (span * m1)
            self.filtered_distances[np.array(gap_indices, dtype=np.int64)] = y_hermite.astype(np.float32)
    
    def _interpolate_low_confidence_regions(self, threshold: float = 0.3) -> np.ndarray:
        """Linearly fill low-confidence gaps from neighboring high-confidence rays."""
        result = self.raw_distances.copy()
        high_conf_mask = self.confidence >= threshold
        if np.sum(high_conf_mask) < 3:
            return result
        gaps = self._find_contiguous_gaps(high_conf_mask)
        for gap_start, gap_end in gaps:
            left_idx = self._find_nearest_high_conf_backward(gap_start, high_conf_mask)
            right_idx = self._find_nearest_high_conf_forward(gap_end, high_conf_mask)
            if left_idx is not None and right_idx is not None:
                left_val = self.raw_distances[left_idx]
                right_val = self.raw_distances[right_idx]
                gap_indices = self._get_gap_indices(gap_start, gap_end)
                n_points = len(gap_indices)
                if n_points > 0:
                    for idx, gap_i in enumerate(gap_indices):
                        t = (idx + 1) / (n_points + 1)
                        result[gap_i] = left_val + (right_val - left_val) * t
        
        return result
    
    def _find_contiguous_gaps(self, high_conf_mask: np.ndarray) -> List[Tuple[int, int]]:
        """Return (start, end) index pairs for low-confidence runs, including wraparound."""
        gaps = []
        in_gap = False
        gap_start = 0
        
        for i in range(self.num_rays):
            if not high_conf_mask[i]:
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    gaps.append((gap_start, i - 1))
                    in_gap = False

        if in_gap:
            wrap_end = 0
            while wrap_end < self.num_rays and not high_conf_mask[wrap_end]:
                wrap_end += 1
            
            if wrap_end > 0:
                gaps.append((gap_start, wrap_end - 1))
            else:
                gaps.append((gap_start, self.num_rays - 1))
        
        return gaps
    
    def _find_nearest_high_conf_backward(self, idx: int, high_conf_mask: np.ndarray) -> Optional[int]:
        """Find nearest high-confidence ray searching backward (with wraparound)."""
        for offset in range(1, self.num_rays):
            check_idx = (idx - offset) % self.num_rays
            if high_conf_mask[check_idx]:
                return check_idx
        return None
    
    def _find_nearest_high_conf_forward(self, idx: int, high_conf_mask: np.ndarray) -> Optional[int]:
        """Find nearest high-confidence ray searching forward (with wraparound)."""
        for offset in range(1, self.num_rays):
            check_idx = (idx + offset) % self.num_rays
            if high_conf_mask[check_idx]:
                return check_idx
        return None

    def _central_derivative(self, data: np.ndarray) -> np.ndarray:
        """Circular central differences with unit sample spacing."""
        prev = np.roll(data, 1)
        nxt = np.roll(data, -1)
        return 0.5 * (nxt - prev)

    def _get_gap_indices(self, start: int, end: int) -> List[int]:
        """Indices in [start, end], wrapping through 0 when end < start."""
        if end >= start:
            return list(range(start, end + 1))
        return list(range(start, self.num_rays)) + list(range(0, end + 1))

    def get_statistics(self) -> Dict[str, float]:
        """Rotation-invariant radius statistics from high-confidence rays when possible."""
        high_conf_mask = self.confidence > 0.5
        if np.sum(high_conf_mask) < 10:
            data = self.filtered_distances
        else:
            data = self.filtered_distances[high_conf_mask]
        return {
            'mean_radius': float(np.mean(data)),
            'std_radius': float(np.std(data)),
            'median_radius': float(np.median(data)),
            'range': float(np.ptp(data)),
        }

    def compute_fourier_features(self, num_harmonics: int = 8) -> np.ndarray:
        """Rotation-invariant FFT magnitudes, excluding DC."""
        fft = np.fft.fft(self.filtered_distances)
        magnitudes = np.abs(fft)
        return magnitudes[1:num_harmonics+1].astype(np.float32)


