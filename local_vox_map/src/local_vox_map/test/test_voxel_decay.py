import math
from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from local_vox_map.voxel_decay import (  # noqa: E402
    acquisition_age_s,
    active_voxel_ttl_s,
    cloud_input_is_fresh,
    decay_voxel_state,
    fresh_cloud_input_count,
    tf_fallback_is_fresh,
    voxel_is_expired,
)


def test_acquisition_age_uses_stamp_and_rejects_future_time():
    assert acquisition_age_s(10.0, 9.7) == pytest.approx(0.3)
    assert acquisition_age_s(10.0, 0.0) == 0.0
    assert math.isinf(acquisition_age_s(10.0, 10.1))


def _decay(vote, consistency, last_decay, now):
    return decay_voxel_state(
        vote=vote,
        consistency=consistency,
        last_hit_s=0.0,
        last_decay_s=last_decay,
        now_s=now,
        vote_decay_rate=0.5,
        consistency_decay_rate=0.5,
        decay_delay_s=0.0,
        confidence_delay_per_unit_s=0.0,
        confidence_decay_scale=0.0,
    )


def test_periodic_decay_consumes_each_time_interval_once():
    vote_1, cons_1, stamp_1 = _decay(10.0, 4.0, 0.0, 1.0)
    vote_2, cons_2, stamp_2 = _decay(vote_1, cons_1, stamp_1, 2.0)
    vote_direct, cons_direct, _ = _decay(10.0, 4.0, 0.0, 2.0)

    assert vote_2 == pytest.approx(vote_direct)
    assert cons_2 == pytest.approx(cons_direct)
    assert stamp_2 == 2.0


def test_grace_period_only_delays_unconsumed_decay_interval():
    params = dict(
        vote_decay_rate=1.0,
        consistency_decay_rate=1.0,
        decay_delay_s=2.0,
        confidence_delay_per_unit_s=0.0,
        confidence_decay_scale=0.0,
    )
    vote_1, cons_1, stamp_1 = decay_voxel_state(
        vote=10.0,
        consistency=2.0,
        last_hit_s=0.0,
        last_decay_s=0.0,
        now_s=1.0,
        **params,
    )
    vote_2, _, _ = decay_voxel_state(
        vote=vote_1,
        consistency=cons_1,
        last_hit_s=0.0,
        last_decay_s=stamp_1,
        now_s=3.0,
        **params,
    )

    assert vote_1 == 10.0
    assert vote_2 == pytest.approx(10.0 * math.exp(-1.0))


def test_sensor_cloud_freshness_does_not_refresh_stale_geometry():
    assert not cloud_input_is_fresh(last_input_s=None, now_s=5.0, timeout_s=0.5)
    assert cloud_input_is_fresh(last_input_s=4.6, now_s=5.0, timeout_s=0.5)
    assert not cloud_input_is_fresh(last_input_s=4.4, now_s=5.0, timeout_s=0.5)
    assert fresh_cloud_input_count(
        last_inputs_s=[4.9, 4.8, 4.0],
        now_s=5.0,
        timeout_s=0.5,
    ) == 2


def test_voxel_ttl_is_refreshed_only_by_a_new_hit():
    assert not voxel_is_expired(last_hit_s=10.0, now_s=15.9, ttl_s=6.0)
    assert voxel_is_expired(last_hit_s=10.0, now_s=16.0, ttl_s=6.0)
    assert not voxel_is_expired(last_hit_s=15.0, now_s=16.0, ttl_s=6.0)
    assert not voxel_is_expired(last_hit_s=10.0, now_s=100.0, ttl_s=0.0)


def test_of_degraded_selects_shorter_voxel_ttl():
    assert active_voxel_ttl_s(
        nominal_ttl_s=6.0,
        degraded_ttl_s=4.0,
        of_degraded=False,
    ) == 6.0
    assert active_voxel_ttl_s(
        nominal_ttl_s=6.0,
        degraded_ttl_s=4.0,
        of_degraded=True,
    ) == 4.0
    # A misconfigured degraded value must never lengthen retention.
    assert active_voxel_ttl_s(
        nominal_ttl_s=6.0,
        degraded_ttl_s=8.0,
        of_degraded=True,
    ) == 6.0


def test_tf_fallback_accepts_only_a_recent_preceding_transform():
    assert tf_fallback_is_fresh(
        sample_stamp_s=10.0,
        transform_stamp_s=9.95,
        max_lag_s=0.08,
    )
    assert not tf_fallback_is_fresh(
        sample_stamp_s=10.0,
        transform_stamp_s=9.90,
        max_lag_s=0.08,
    )
    assert not tf_fallback_is_fresh(
        sample_stamp_s=10.0,
        transform_stamp_s=10.01,
        max_lag_s=0.08,
    )
    assert not tf_fallback_is_fresh(
        sample_stamp_s=math.nan,
        transform_stamp_s=9.95,
        max_lag_s=0.08,
    )
