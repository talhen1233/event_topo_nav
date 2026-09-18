import math

import numpy as np
import pytest

from of_localization.orientation_utils import (
    axial_depths_from_heights,
    gravity_direction_sensor_frame,
    heights_from_sensor_points,
)


SENSOR_FROM_BODY = np.asarray(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def test_gravity_direction_sensor_frame_is_optical_axis_when_level():
    gravity_sensor = gravity_direction_sensor_frame(
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        sensor_from_body=SENSOR_FROM_BODY,
    )
    np.testing.assert_allclose(gravity_sensor, np.asarray([1.0, 0.0, 0.0], dtype=np.float32), atol=1e-6)


def test_gravity_projection_matches_center_ray_tilt_factor():
    roll = 0.18
    pitch = -0.11
    gravity_sensor = gravity_direction_sensor_frame(
        roll=roll,
        pitch=pitch,
        yaw=0.7,
        sensor_from_body=SENSOR_FROM_BODY,
    )
    expected_projection = math.cos(roll) * math.cos(pitch)
    assert gravity_sensor[0] == pytest.approx(expected_projection, abs=1e-5)


def test_axial_depths_from_heights_preserves_optical_axis_depth():
    gravity_sensor = np.asarray([0.95, 0.0, 0.0], dtype=np.float32)
    heights = np.asarray([0.5, 0.5], dtype=np.float32)
    image_points = np.asarray([[80.0, 60.0], [140.0, 60.0]], dtype=np.float32)
    depths, valid_mask, projections = axial_depths_from_heights(
        heights,
        image_points,
        gravity_sensor,
        fx=120.0,
        fy=120.0,
        cx=80.0,
        cy=60.0,
        min_projection=0.1,
    )

    assert valid_mask.tolist() == [True, True]
    assert depths[0] == pytest.approx(0.5 / 0.95, abs=1e-6)
    # Axial depth is the coefficient of the unnormalised pinhole ray
    # [1, -x/fx, -y/fy], not Euclidean range along a unit ray.  With gravity
    # parallel to the optical axis, equal heights therefore have equal axial
    # depth even away from the principal point.
    assert depths[1] == pytest.approx(depths[0], abs=1e-6)
    assert projections[1] == pytest.approx(projections[0], abs=1e-6)


def test_heights_from_sensor_points_projects_onto_gravity():
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    gravity_sensor = np.asarray([0.8, 0.6, 0.0], dtype=np.float32)
    heights = heights_from_sensor_points(points, gravity_sensor)
    np.testing.assert_allclose(heights, np.asarray([0.8, 2.2], dtype=np.float32), atol=1e-6)
