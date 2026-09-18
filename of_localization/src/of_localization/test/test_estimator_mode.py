from of_localization.estimator_mode import request_local_depth


def test_first_sample_uses_center_threshold() -> None:
    assert request_local_depth(1.0, 1.0, 0.15, None) is True
    assert request_local_depth(1.01, 1.0, 0.15, None) is False


def test_hysteresis_holds_mode_inside_the_band() -> None:
    assert request_local_depth(1.10, 1.0, 0.15, True) is True
    assert request_local_depth(0.90, 1.0, 0.15, False) is False


def test_hysteresis_switches_only_after_leaving_the_band() -> None:
    assert request_local_depth(1.16, 1.0, 0.15, True) is False
    assert request_local_depth(0.85, 1.0, 0.15, False) is True


def test_zero_hysteresis_is_a_hard_threshold() -> None:
    assert request_local_depth(1.0, 1.0, 0.0, True) is True
    assert request_local_depth(1.01, 1.0, 0.0, True) is False
    assert request_local_depth(1.0, 1.0, 0.0, False) is True
