import numpy as np

from drone_ukf_odometry.odometry_modes import (
    OpticalFlowHealthParams,
    OpticalFlowHealthTracker,
    apply_command_velocity_prior,
    apply_zero_velocity_constraint,
    zupt_candidate,
)


def test_of_degraded_on_invalid_or_stale_and_recovers_with_hysteresis() -> None:
    tracker = OpticalFlowHealthTracker(
        OpticalFlowHealthParams(stale_timeout_s=0.5, recovery_frames=3)
    )
    assert tracker.degraded
    for index in range(2):
        assert tracker.observe(
            now_s=0.1 * index,
            measurement_usable=True,
            of_speed_mps=0.2,
            command_fresh=True,
            command_speed_mps=0.2,
            command_mismatch_mps=0.0,
        )
    assert not tracker.observe(
        now_s=0.2,
        measurement_usable=True,
        of_speed_mps=0.2,
        command_fresh=True,
        command_speed_mps=0.2,
        command_mismatch_mps=0.0,
    )
    assert tracker.tick(0.8)


def test_persistent_zero_of_during_nonzero_command_is_degraded() -> None:
    tracker = OpticalFlowHealthTracker(
        OpticalFlowHealthParams(recovery_frames=1, mismatch_dwell_s=0.1)
    )
    tracker.observe(
        now_s=0.0,
        measurement_usable=True,
        of_speed_mps=0.0,
        command_fresh=True,
        command_speed_mps=0.5,
        command_mismatch_mps=0.5,
    )
    assert tracker.observe(
        now_s=0.11,
        measurement_usable=True,
        of_speed_mps=0.0,
        command_fresh=True,
        command_speed_mps=0.5,
        command_mismatch_mps=0.5,
    )


def test_default_mismatch_gate_accepts_quarter_mps_motion_disagreement() -> None:
    tracker = OpticalFlowHealthTracker(
        OpticalFlowHealthParams(recovery_frames=1, mismatch_dwell_s=0.1)
    )
    assert not tracker.observe(
        now_s=-0.1,
        measurement_usable=True,
        of_speed_mps=0.25,
        command_fresh=True,
        command_speed_mps=0.25,
        command_mismatch_mps=0.0,
    )
    assert not tracker.observe(
        now_s=0.0,
        measurement_usable=True,
        of_speed_mps=0.0,
        command_fresh=True,
        command_speed_mps=0.25,
        command_mismatch_mps=0.25,
    )
    assert tracker.observe(
        now_s=0.11,
        measurement_usable=True,
        of_speed_mps=0.0,
        command_fresh=True,
        command_speed_mps=0.25,
        command_mismatch_mps=0.25,
    )


def test_stale_of_clears_partial_mismatch_dwell() -> None:
    tracker = OpticalFlowHealthTracker(
        OpticalFlowHealthParams(stale_timeout_s=0.5, recovery_frames=1)
    )
    tracker.observe(
        now_s=0.0,
        measurement_usable=True,
        of_speed_mps=0.0,
        command_fresh=True,
        command_speed_mps=0.5,
        command_mismatch_mps=0.5,
    )
    assert tracker._mismatch_start_s == 0.0

    assert tracker.tick(0.6)
    assert tracker._mismatch_start_s is None


def test_single_unusable_of_sample_does_not_immediately_degrade() -> None:
    tracker = OpticalFlowHealthTracker(
        OpticalFlowHealthParams(stale_timeout_s=1.0, recovery_frames=1)
    )
    assert not tracker.observe(
        now_s=0.0,
        measurement_usable=True,
        of_speed_mps=0.2,
        command_fresh=True,
        command_speed_mps=0.2,
        command_mismatch_mps=0.0,
    )
    assert not tracker.observe(
        now_s=0.2,
        measurement_usable=False,
        of_speed_mps=0.0,
        command_fresh=True,
        command_speed_mps=0.2,
        command_mismatch_mps=0.2,
    )
    assert not tracker.tick(0.9)
    assert tracker.tick(1.01)


def test_repeated_command_prior_never_creates_false_confidence() -> None:
    state = np.zeros(6, dtype=float)
    covariance = np.eye(6, dtype=float) * 0.01
    initial_position_variance = np.diag(covariance)[:3].copy()

    for _ in range(50):
        previous = covariance.copy()
        state, covariance = apply_command_velocity_prior(
            state=state,
            covariance=covariance,
            command_world_xy=np.array([0.8, 0.0]),
            gain=0.2,
            sigma_mps=0.8,
        )
        assert np.all(np.diag(covariance) >= np.diag(previous))

    assert state[3] > 0.79
    assert abs(state[4]) < 1e-12
    assert np.allclose(np.diag(covariance)[:3], initial_position_variance)
    assert covariance[3, 3] >= 0.8 ** 2
    assert covariance[4, 4] >= 0.8 ** 2


def test_zero_velocity_constraint_does_not_move_position_mean() -> None:
    state = np.array([4.0, -2.0, 1.5, 0.6, -0.4, 0.2], dtype=float)
    covariance = np.eye(6, dtype=float)
    covariance[0, 3] = covariance[3, 0] = 0.8
    covariance[1, 4] = covariance[4, 1] = -0.7
    position_before = state[:3].copy()

    state, covariance = apply_zero_velocity_constraint(
        state=state,
        covariance=covariance,
        measurement_variance=1e-4,
    )

    np.testing.assert_allclose(state[:3], position_before)
    assert np.linalg.norm(state[3:6]) < 1e-3
    np.testing.assert_allclose(covariance, covariance.T)


def test_uncertain_zero_of_cannot_arm_nominal_zupt() -> None:
    common = dict(
        command_fresh=True,
        command_speed_mps=0.0,
        imu_fresh=True,
        imu_acceleration_mps2=0.01,
        tilt_rad=0.01,
        imu_acceleration_threshold_mps2=0.35,
        tilt_threshold_rad=0.05,
        of_speed_mps=0.0,
        speed_threshold_mps=0.05,
    )
    assert not zupt_candidate(
        **common, of_degraded=False, of_measurement_usable=False
    )
    assert zupt_candidate(
        **common, of_degraded=False, of_measurement_usable=True
    )


def test_command_tilt_and_acceleration_each_veto_zupt() -> None:
    common = dict(
        command_fresh=True,
        command_speed_mps=0.0,
        imu_fresh=True,
        imu_acceleration_mps2=0.01,
        tilt_rad=0.01,
        imu_acceleration_threshold_mps2=0.35,
        tilt_threshold_rad=0.05,
        of_degraded=True,
        of_measurement_usable=False,
        of_speed_mps=None,
        speed_threshold_mps=0.05,
    )
    assert zupt_candidate(**common)
    assert not zupt_candidate(**{**common, "command_speed_mps": 0.2})
    assert not zupt_candidate(**{**common, "tilt_rad": 0.1})
    assert not zupt_candidate(**{**common, "imu_acceleration_mps2": 0.5})
    assert not zupt_candidate(**{**common, "imu_fresh": False})
