#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

import numpy as np
import math
from typing import Dict, Tuple
from numba import njit

from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
from sensor_msgs_py import point_cloud2
import tf2_ros
import tf2_py

from .voxel_decay import (
    acquisition_age_s,
    active_voxel_ttl_s,
    decay_voxel_state,
    fresh_cloud_input_count,
    tf_fallback_is_fresh,
    voxel_is_expired,
)


def _pc2_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract XYZ as (N, 3) float32 without building an intermediate list."""
    gen = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)

    rec = np.fromiter(
        gen,
        dtype=[('x', np.float32), ('y', np.float32), ('z', np.float32)],
        count=-1
    )

    if rec.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    return rec.view(np.float32).reshape(-1, 3)


def transform_to_matrix(t: TransformStamped) -> np.ndarray:
    """4x4 homogeneous transform (float32)."""
    rx, ry, rz, rw = t.transform.rotation.x, t.transform.rotation.y, \
                     t.transform.rotation.z, t.transform.rotation.w
    tx, ty, tz = t.transform.translation.x, t.transform.translation.y, \
                 t.transform.translation.z
    # faster than tf_transformations
    n = rx*rx + ry*ry + rz*rz + rw*rw
    s = 2.0 / n if n else 0.0
    xs, ys, zs = rx*s, ry*s, rz*s
    wx, wy, wz = rw*xs, rw*ys, rw*zs
    xx, xy, xz = rx*xs, rx*ys, rx*zs
    yy, yz, zz = ry*ys, ry*zs, rz*zs
    rot = np.array([
        [1.0-(yy+zz), xy-wz,       xz+wy],
        [xy+wz,       1.0-(xx+zz), yz-wx],
        [xz-wy,       yz+wx,       1.0-(xx+yy)]
    ], dtype=np.float32)
    tfm = np.eye(4, dtype=np.float32)
    tfm[:3, :3], tfm[:3, 3] = rot, [tx, ty, tz]
    return tfm


@njit(cache=True)
def _compute_voxel_indices(points: np.ndarray, voxel_size_inv: float) -> np.ndarray:
    """Fast voxel index computation."""
    return np.floor(points * voxel_size_inv).astype(np.int32)


@njit(cache=True)
def _filter_by_distance(points: np.ndarray, robot_pos: np.ndarray, min_dist: float) -> np.ndarray:
    """Filter points by minimum distance from robot."""
    n = points.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    min_dist_sq = min_dist * min_dist
    for i in range(n):
        dx = points[i, 0] - robot_pos[0]
        dy = points[i, 1] - robot_pos[1]
        dz = points[i, 2] - robot_pos[2]
        dist_sq = dx*dx + dy*dy + dz*dz
        keep[i] = dist_sq >= min_dist_sq
    return keep


@njit(cache=True)
def _filter_by_region(points: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Filter points within axis-aligned bounding box."""
    n = points.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        keep[i] = (points[i, 0] >= lower[0] and points[i, 0] <= upper[0] and
                   points[i, 1] >= lower[1] and points[i, 1] <= upper[1] and
                   points[i, 2] >= lower[2] and points[i, 2] <= upper[2])
    return keep


@njit(cache=True)
def _check_voxel_in_region(voxel_centers: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Check which voxel centers are within region."""
    n = voxel_centers.shape[0]
    keep = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        keep[i] = (voxel_centers[i, 0] >= lower[0] and voxel_centers[i, 0] <= upper[0] and
                   voxel_centers[i, 1] >= lower[1] and voxel_centers[i, 1] <= upper[1] and
                   voxel_centers[i, 2] >= lower[2] and voxel_centers[i, 2] <= upper[2])
    return keep


@njit(cache=True)
def _hash_voxel_indices(vox_idx: np.ndarray) -> np.ndarray:
    """Hash 3D indices; assumes [-50000, 50000] per axis (~±7.5 km at 0.15 m)."""
    n = vox_idx.shape[0]
    hashes = np.empty(n, dtype=np.int64)
    offset = 100000
    for i in range(n):
        ix = vox_idx[i, 0] + offset
        iy = vox_idx[i, 1] + offset
        iz = vox_idx[i, 2] + offset
        hashes[i] = ix + iy * 200000 + iz * 40000000000
    return hashes


@njit(cache=True)
def _unique_with_counts(hashes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unique hashes and counts; faster than np.unique here."""
    if hashes.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    sorted_idx = np.argsort(hashes)
    sorted_hashes = hashes[sorted_idx]

    n = len(sorted_hashes)
    n_unique = 1
    for i in range(1, n):
        if sorted_hashes[i] != sorted_hashes[i - 1]:
            n_unique += 1

    unique_vals = np.empty(n_unique, dtype=np.int64)
    counts = np.empty(n_unique, dtype=np.int64)

    unique_vals[0] = sorted_hashes[0]
    count = 1
    idx = 0
    for i in range(1, n):
        if sorted_hashes[i] != sorted_hashes[i - 1]:
            counts[idx] = count
            idx += 1
            unique_vals[idx] = sorted_hashes[i]
            count = 1
        else:
            count += 1
    counts[idx] = count

    return unique_vals, counts


@njit(cache=True)
def _unhash_voxel_indices(hashes: np.ndarray) -> np.ndarray:
    """Convert hashes back to 3D voxel indices."""
    n = hashes.shape[0]
    indices = np.empty((n, 3), dtype=np.int32)
    offset = 100000
    for i in range(n):
        h = hashes[i]
        iz = h // 40000000000
        remainder = h - iz * 40000000000
        iy = remainder // 200000
        ix = remainder - iy * 200000
        indices[i, 0] = ix - offset
        indices[i, 1] = iy - offset
        indices[i, 2] = iz - offset
    return indices


class MinimalVoxelMapping(Node):
    DISCOVERY_PERIOD = 3.0
    DECAY_PERIOD     = 2.0

    # exclude from sensor-cloud discovery
    MAP_TOPIC = '/map_pointcloud'

    def __init__(self):
        super().__init__('minimal_voxel_mapping')

        # sensors are fire-and-forget
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.declare_parameter('map_frame',               'odom')
        self.declare_parameter('voxel_size',              0.15)   # m
        self.declare_parameter('shift_threshold',         3.0)    # m
        self.declare_parameter('region_size',            10.0)    # m (square)
        self.declare_parameter('map_publish_rate',        5.0)    # Hz
        self.declare_parameter('sensor_cloud_timeout_s',  1.25)   # s
        self.declare_parameter('min_fresh_sensor_clouds', 1)
        self.declare_parameter('max_tf_fallback_lag_s',   0.08)   # s
        self.declare_parameter('min_votes',               4)
        self.declare_parameter('consistency_time_thresh', 0.5)    # s
        self.declare_parameter('min_consistency',         3)
        self.declare_parameter('max_consistency',         20.0)
        self.declare_parameter('consistency_decay_rate',  0.5)    # 1/s
        self.declare_parameter('vote_decay_rate',         0.5)    # 1/s
        self.declare_parameter('decay_delay',             0.0)    # s
        self.declare_parameter('confidence_decay_bonus',  1.0)    # extra delay (s) per consistency unit
        self.declare_parameter('confidence_decay_scale',  0.15)    # decay rate reduction factor per consistency unit
        # Hard TTL so high-confidence voxels cannot retain ghost geometry indefinitely.
        self.declare_parameter('voxel_ttl_s',             5.0)
        self.declare_parameter('of_degraded_voxel_ttl_s', 4.0)
        self.declare_parameter('of_degraded_topic',       '/drone/odometry/of_degraded')
        self.declare_parameter('of_degraded_timeout_s',   1.0)
        self.declare_parameter('keep_ratio',              1.0)    # 0-1
        self.declare_parameter('crop_enabled',            True)
        self.declare_parameter('z_crop_max',              3.0)    # m
        self.declare_parameter('min_distance',            0.1)    # m
        # unused here; declared so launch remappings can set them
        self.declare_parameter('odom_topic',              '/drone/state_estimate')
        self.declare_parameter('map_topic',               '/map_pointcloud')

        p = self.get_parameter
        self.map_frame            = p('map_frame').value
        self.voxel_size           = p('voxel_size').value
        self.voxel_size_inv       = 1.0 / self.voxel_size
        self.shift_threshold      = p('shift_threshold').value
        self.region_size          = p('region_size').value
        self.map_pub_rate         = p('map_publish_rate').value
        self.sensor_cloud_timeout_s = max(
            0.0, float(p('sensor_cloud_timeout_s').value)
        )
        self.min_fresh_sensor_clouds = max(
            1, int(p('min_fresh_sensor_clouds').value)
        )
        self.max_tf_fallback_lag_s = max(
            0.0, float(p('max_tf_fallback_lag_s').value)
        )
        self.min_votes            = p('min_votes').value
        self.ct_thresh            = p('consistency_time_thresh').value
        self.min_consistency      = p('min_consistency').value
        self.max_consistency      = float(p('max_consistency').value)
        self.consistency_decay    = p('consistency_decay_rate').value
        self.vote_decay           = p('vote_decay_rate').value
        self.decay_delay          = p('decay_delay').value
        self.conf_decay_bonus     = p('confidence_decay_bonus').value
        self.conf_decay_scale     = p('confidence_decay_scale').value
        self.voxel_ttl_s          = max(0.0, float(p('voxel_ttl_s').value))
        self.of_degraded_voxel_ttl_s = max(
            0.0, float(p('of_degraded_voxel_ttl_s').value)
        )
        self.of_degraded_timeout_s = max(
            0.0, float(p('of_degraded_timeout_s').value)
        )
        self.keep_ratio           = max(0.01, min(1.0, p('keep_ratio').value))
        self.crop_enabled         = p('crop_enabled').value
        self.z_crop_max           = p('z_crop_max').value
        self.min_distance         = p('min_distance').value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.map_pub = self.create_publisher(PointCloud2, self.MAP_TOPIC, qos)
        self.create_timer(1.0 / max(0.1, self.map_pub_rate), self._pub_map)
        self.create_timer(self.DISCOVERY_PERIOD, self._refresh_subscribers)
        self.create_timer(self.DECAY_PERIOD,     self._apply_decay)

        self.create_subscription(Odometry, '/drone/state_estimate',
                                 self._odom_cb, qos)
        of_degraded_topic = str(p('of_degraded_topic').value or '').strip()
        self._of_degraded_sub = (
            self.create_subscription(
                Bool, of_degraded_topic, self._of_degraded_cb, qos
            )
            if of_degraded_topic
            else None
        )

        self._subs: Dict[str, rclpy.subscription.Subscription] = {}
        # key -> [vote, last_hit_s, last_decay_s, bounded_consistency]
        self._voxel_votes: Dict[Tuple[int, int, int], list] = {}
        self._robot_pos = np.zeros(3, dtype=np.float32)
        self._have_pose = False
        self._sensor_cloud_times: Dict[str, float] = {}
        self._of_degraded = False
        self._of_degraded_last_update_s = -1.0
        # reuse one PointField schema instead of reallocating on each publish
        self._pc_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        self.get_logger().info("MinimalVoxelMapping started")

    def _of_degraded_cb(self, msg: Bool):
        self._of_degraded = bool(msg.data)
        self._of_degraded_last_update_s = (
            self.get_clock().now().nanoseconds * 1e-9
        )

    def _is_of_degraded(self, now_s: float) -> bool:
        if not self._of_degraded or self._of_degraded_last_update_s < 0.0:
            return False
        if self.of_degraded_timeout_s <= 0.0:
            return True
        age_s = float(now_s) - self._of_degraded_last_update_s
        return 0.0 <= age_s <= self.of_degraded_timeout_s

    def _active_voxel_ttl_s(self, now_s: float) -> float:
        return active_voxel_ttl_s(
            nominal_ttl_s=self.voxel_ttl_s,
            degraded_ttl_s=self.of_degraded_voxel_ttl_s,
            of_degraded=self._is_of_degraded(now_s),
        )

    def _odom_cb(self, msg: Odometry):
        pos = np.array([msg.pose.pose.position.x,
                        msg.pose.pose.position.y,
                        msg.pose.pose.position.z], dtype=np.float32)
        if not self._have_pose:
            self._robot_pos   = pos
            self._have_pose   = True
            self._last_shift  = pos.copy()
            return

        self._robot_pos = pos
        # drop voxels that left the rolling window
        if np.linalg.norm(pos - self._last_shift) > self.shift_threshold:
            self._trim_to_region(pos)
            self._last_shift = pos.copy()

    def _refresh_subscribers(self):
        """Scan ROS graph and (un)subscribe to PointCloud2 topics."""
        graph = self.get_topic_names_and_types()
        current_topics = {g[0] for g in graph}

        for topic, topic_type in graph:
            is_sensor_cloud = (
                '/points' in topic
                or topic.endswith('/pointcloud')
            )
            is_valid = (
                topic != self.MAP_TOPIC
                and is_sensor_cloud
                and topic_type[0] == 'sensor_msgs/msg/PointCloud2'
                and topic not in self._subs
            )
            if is_valid:
                callback = (
                    lambda msg, source_topic=topic:
                    self._cloud_cb(msg, source_topic=source_topic)
                )
                self._subs[topic] = self.create_subscription(
                    PointCloud2, topic, callback,
                    QoSProfile(
                        reliability=QoSReliabilityPolicy.BEST_EFFORT,
                        history=QoSHistoryPolicy.KEEP_LAST,
                        depth=1
                    )
                )
                self.get_logger().info(f"Subscribed {topic}")

        for t in list(self._subs):
            if t not in current_topics:
                self.destroy_subscription(self._subs.pop(t))
                self._sensor_cloud_times.pop(t, None)
                self.get_logger().info(f"Unsubscribed {t}")

    def _cloud_cb(self, msg: PointCloud2, *, source_topic: str = ""):
        source_key = source_topic or str(msg.header.frame_id) or "unknown"
        received_ros_s = self.get_clock().now().nanoseconds * 1e-9
        self._sensor_cloud_times.pop(source_key, None)
        stamp_ros_s = (
            float(msg.header.stamp.sec)
            + 1e-9 * float(msg.header.stamp.nanosec)
        )
        now_ros_s = self.get_clock().now().nanoseconds * 1e-9
        initial_age_s = acquisition_age_s(now_ros_s, stamp_ros_s)
        if (
            not math.isfinite(initial_age_s)
            or (
                self.sensor_cloud_timeout_s > 0.0
                and initial_age_s > self.sensor_cloud_timeout_s
            )
        ):
            return
        # Fresh empty cloud still proves source liveness; stale stamps do not refresh the publish quorum.
        self._sensor_cloud_times[source_key] = received_ros_s
        if not self._have_pose:
            return

        points = _pc2_xyz(msg)
        if points.size > 0:
            points = points[np.all(np.isfinite(points), axis=1)]
        if points.size == 0:
            return

        # strided downsample is cheaper than random
        if self.keep_ratio < 0.999:
            step = max(1, int(1.0 / self.keep_ratio))
            points = points[::step]
            if points.size == 0:
                return

        try:
            stamp = msg.header.stamp
            lookup_time = (
                rclpy.time.Time()
                if int(stamp.sec) == 0 and int(stamp.nanosec) == 0
                else rclpy.time.Time.from_msg(stamp)
            )
            tf = self.tf_buffer.lookup_transform(
                self.map_frame,
                msg.header.frame_id,
                lookup_time,
            )
        except tf2_py.ExtrapolationException:
            # Estimator TF trails the stamp; waiting here would stall the TF listener, so use latest TF only within max_tf_fallback_lag_s.
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    msg.header.frame_id,
                    rclpy.time.Time(),
                )
            except (tf2_py.LookupException, tf2_py.ExtrapolationException):
                return
            transform_stamp_s = (
                float(tf.header.stamp.sec)
                + 1e-9 * float(tf.header.stamp.nanosec)
            )
            if not tf_fallback_is_fresh(
                sample_stamp_s=stamp_ros_s,
                transform_stamp_s=transform_stamp_s,
                max_lag_s=self.max_tf_fallback_lag_s,
            ):
                return
        except tf2_py.LookupException:
            return
        tfm = transform_to_matrix(tf)
        points_map = points @ tfm[:3, :3].T + tfm[:3, 3]
        # crop Z in the map frame, not the sensor frame
        points_map = points_map[points_map[:, 2] < self.z_crop_max]
        if points_map.size == 0:
            return

        if self.min_distance > 0.0:
            keep_mask = _filter_by_distance(points_map, self._robot_pos, self.min_distance)
            points_map = points_map[keep_mask]
            if points_map.size == 0:
                return

        if self.crop_enabled:
            half = self.region_size * 0.5
            lower = self._robot_pos - half
            upper = self._robot_pos + half
            keep_mask = _filter_by_region(points_map, lower, upper)
            points_map = points_map[keep_mask]
            if points_map.size == 0:
                return

        vox_idx = _compute_voxel_indices(points_map, self.voxel_size_inv)
        hashes = _hash_voxel_indices(vox_idx)
        uniq_hashes, counts = _unique_with_counts(hashes)
        uniq_indices = _unhash_voxel_indices(uniq_hashes)

        now = self.get_clock().now().nanoseconds * 1e-9
        vote_decay = self.vote_decay
        ct_thresh = self.ct_thresh
        voxel_votes = self._voxel_votes
        active_ttl_s = self._active_voxel_ttl_s(now)

        for i in range(len(uniq_indices)):
            key = (int(uniq_indices[i, 0]), int(uniq_indices[i, 1]), int(uniq_indices[i, 2]))
            c = counts[i]
            if (
                key in voxel_votes
                and not voxel_is_expired(
                    last_hit_s=voxel_votes[key][1],
                    now_s=now,
                    ttl_s=active_ttl_s,
                )
            ):
                vote, last_hit, last_decay, cons = voxel_votes[key]
                vote, cons, _ = decay_voxel_state(
                    vote=vote,
                    consistency=cons,
                    last_hit_s=last_hit,
                    last_decay_s=last_decay,
                    now_s=now,
                    vote_decay_rate=vote_decay,
                    consistency_decay_rate=self.consistency_decay,
                    decay_delay_s=self.decay_delay,
                    confidence_delay_per_unit_s=self.conf_decay_bonus,
                    confidence_decay_scale=self.conf_decay_scale,
                )
                dt_hit = now - last_hit
                vote += float(c)
                cons = (cons + 1.0) if dt_hit < ct_thresh else max(cons - 1.0, 0.0)
                cons = min(cons, self.max_consistency)
            else:
                vote, cons = float(c), 1.0
            voxel_votes[key] = [vote, now, now, cons]

    def _apply_decay(self):
        """Decay untouched voxels; high confidence delays and slows decay."""
        if not self._voxel_votes:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        keys_to_delete = []
        keys_to_update = []
        vote_decay = self.vote_decay
        consistency_decay = self.consistency_decay
        decay_delay = self.decay_delay
        conf_decay_bonus = self.conf_decay_bonus
        conf_decay_scale = self.conf_decay_scale
        active_ttl_s = self._active_voxel_ttl_s(now)

        for key, (vote, last_hit, last_decay, cons) in self._voxel_votes.items():
            if voxel_is_expired(
                last_hit_s=last_hit,
                now_s=now,
                ttl_s=active_ttl_s,
            ):
                keys_to_delete.append(key)
                continue
            vote_new, cons_new, decay_stamp = decay_voxel_state(
                vote=vote,
                consistency=cons,
                last_hit_s=last_hit,
                last_decay_s=last_decay,
                now_s=now,
                vote_decay_rate=vote_decay,
                consistency_decay_rate=consistency_decay,
                decay_delay_s=decay_delay,
                confidence_delay_per_unit_s=conf_decay_bonus,
                confidence_decay_scale=conf_decay_scale,
            )

            if vote_new < 0.1 and cons_new < 0.1:
                keys_to_delete.append(key)
            else:
                keys_to_update.append((key, vote_new, cons_new, last_hit, decay_stamp))

        for k in keys_to_delete:
            del self._voxel_votes[k]

        for key, vote_new, cons_new, last_hit, decay_stamp in keys_to_update:
            self._voxel_votes[key] = [vote_new, last_hit, decay_stamp, cons_new]

    def _trim_to_region(self, centre: np.ndarray):
        """Throw away voxels outside region_size centred on centre."""
        if not self._voxel_votes:
            return

        half = self.region_size * 0.5
        lower, upper = centre - half, centre + half

        keys = np.array(list(self._voxel_votes.keys()), dtype=np.int32)
        voxel_centers = (keys.astype(np.float32) + 0.5) * self.voxel_size
        keep_mask = _check_voxel_in_region(voxel_centers, lower, upper)

        keys_to_remove = keys[~keep_mask]
        for k in keys_to_remove:
            del self._voxel_votes[tuple(k)]

        self.get_logger().info(f"Map shifted: {len(self._voxel_votes)} voxels kept")

    def _pub_map(self):
        """Publish voxels whose (vote, consistency) exceed thresholds."""
        now_ros_s = self.get_clock().now().nanoseconds * 1e-9
        fresh_sources = fresh_cloud_input_count(
            last_inputs_s=self._sensor_cloud_times.values(),
            now_s=now_ros_s,
            timeout_s=self.sensor_cloud_timeout_s,
        )
        if fresh_sources < self.min_fresh_sensor_clouds:
            return

        min_votes = self.min_votes
        min_consistency = self.min_consistency
        active_ttl_s = self._active_voxel_ttl_s(now_ros_s)

        valid_keys = [
            key for key, (vote, last_hit, _, cons) in self._voxel_votes.items()
            if vote >= min_votes and cons >= min_consistency
            and not voxel_is_expired(
                last_hit_s=last_hit,
                now_s=now_ros_s,
                ttl_s=active_ttl_s,
            )
        ]

        # Publish an empty cloud when sources are alive so downstream can tell that from a silent pipeline.
        if valid_keys:
            keys_array = np.array(valid_keys, dtype=np.int32)
            voxel_centers = (
                keys_array.astype(np.float32) + 0.5
            ) * self.voxel_size
        else:
            voxel_centers = np.empty((0, 3), dtype=np.float32)

        # bytes path avoids tuple conversion
        pc2_msg = PointCloud2()
        pc2_msg.header.frame_id = self.map_frame
        pc2_msg.header.stamp = self.get_clock().now().to_msg()
        pc2_msg.height = 1
        pc2_msg.width = len(voxel_centers)
        pc2_msg.fields = self._pc_fields
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 12
        pc2_msg.row_step = 12 * len(voxel_centers)
        pc2_msg.data = voxel_centers.astype(np.float32).tobytes()
        pc2_msg.is_dense = True

        self.map_pub.publish(pc2_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MinimalVoxelMapping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
