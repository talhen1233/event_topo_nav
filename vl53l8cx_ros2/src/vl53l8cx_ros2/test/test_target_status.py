"""Tests for conservative VL53L8CX range-status filtering."""

from vl53l8cx_ros2.vl53l8cx_ros2_node import is_valid_target_status


def test_only_explicitly_accepted_statuses_are_range_returns() -> None:
    accepted = (5, 9)
    assert is_valid_target_status(5, accepted)
    assert is_valid_target_status(9, accepted)
    for status in (0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 255):
        assert not is_valid_target_status(status, accepted)
