"""Tests for external pose-correction innovation gating."""

import math

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseWithCovarianceStamped

from drone_odometry.kf_odometry_node import SimpleKFOdometryNode
from drone_odometry.pose_correction import innovation_mahalanobis_squared


class _Logger:
    def info(self, _message: str) -> None:
        pass

    def warning(self, _message: str) -> None:
        pass


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message: PoseWithCovarianceStamped) -> None:
        self.messages.append(message)


class _Clock:
    def now(self):
        class _Now:
            @staticmethod
            def to_msg() -> Time:
                return Time(sec=12, nanosec=345)

        return _Now()


class _CorrectionHarness:
    def __init__(self) -> None:
        self.state = np.zeros(9, dtype=float)
        self.covariance = np.eye(9, dtype=float) * 0.1
        self._H_pos_xy = np.zeros((2, 9), dtype=float)
        self._H_pos_xy[0, 0] = 1.0
        self._H_pos_xy[1, 1] = 1.0
        self.pose_correction_max_translation_m = 2.5
        self.pose_correction_innovation_chi2_gate = 9.21
        self.pose_correction_applied_pub = _Publisher()
        self.update_count = 0

    def get_logger(self) -> _Logger:
        return _Logger()

    def get_clock(self) -> _Clock:
        return _Clock()

    def _kf_update(self, measurement, _matrix, _covariance) -> None:
        self.update_count += 1
        self.state[:2] = measurement


def _correction(x: float, y: float, variance: float = 0.4):
    message = PoseWithCovarianceStamped()
    message.header.frame_id = "odom"
    message.pose.pose.position.x = x
    message.pose.pose.position.y = y
    covariance = np.zeros((6, 6), dtype=float)
    covariance[0, 0] = variance
    covariance[1, 1] = variance
    message.pose.covariance = covariance.reshape(-1).tolist()
    return message


def test_pose_innovation_uses_state_and_measurement_covariance() -> None:
    result = innovation_mahalanobis_squared(
        np.array([0.5, 0.0]),
        np.eye(2) * 0.1,
        np.eye(2) * 0.4,
    )
    assert math.isclose(result, 0.5, rel_tol=1e-8)


def test_pose_innovation_rejects_singular_covariance() -> None:
    result = innovation_mahalanobis_squared(
        np.array([1.0, 0.0]),
        np.zeros((2, 2)),
        np.zeros((2, 2)),
    )
    assert result > 1e8


def test_rejected_pose_correction_does_not_publish_applied_ack() -> None:
    harness = _CorrectionHarness()

    SimpleKFOdometryNode.pose_correction_callback(
        harness,
        _correction(3.0, 0.0),
    )

    assert harness.update_count == 0
    assert harness.pose_correction_applied_pub.messages == []


def test_accepted_pose_correction_publishes_posterior_ack() -> None:
    harness = _CorrectionHarness()

    SimpleKFOdometryNode.pose_correction_callback(
        harness,
        _correction(0.5, -0.25),
    )

    assert harness.update_count == 1
    assert len(harness.pose_correction_applied_pub.messages) == 1
    applied = harness.pose_correction_applied_pub.messages[0]
    assert applied.header.frame_id == "odom"
    assert applied.header.stamp.sec == 12
    assert applied.pose.pose.position.x == 0.5
    assert applied.pose.pose.position.y == -0.25
