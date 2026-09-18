import math
from typing import Tuple

import numpy as np


def quat_to_euler_xyz(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
    """Convert quaternion to XYZ roll, pitch, yaw."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_xyz_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert XYZ Euler angles to quaternion (x, y, z, w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize [x, y, z, w]; identity if the norm is degenerate."""
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of [x, y, z, w] quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float64)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    """Inverse of a unit quaternion [x, y, z, w]."""
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _rotation_matrix_to_quat(Rm: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation to quaternion [x, y, z, w]."""
    m00, m01, m02 = float(Rm[0, 0]), float(Rm[0, 1]), float(Rm[0, 2])
    m10, m11, m12 = float(Rm[1, 0]), float(Rm[1, 1]), float(Rm[1, 2])
    m20, m21, m22 = float(Rm[2, 0]), float(Rm[2, 1]), float(Rm[2, 2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        S = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / S
        x = (m21 - m12) * S
        y = (m02 - m20) * S
        z = (m10 - m01) * S
    elif (m00 > m11) and (m00 > m22):
        S = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / S
        x = 0.25 * S
        y = (m01 + m10) / S
        z = (m02 + m20) / S
    elif m11 > m22:
        S = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / S
        x = (m01 + m10) / S
        y = 0.25 * S
        z = (m12 + m21) / S
    else:
        S = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / S
        x = (m02 + m20) / S
        y = (m12 + m21) / S
        z = 0.25 * S
    q = np.array([x, y, z, w], dtype=np.float64)
    return _quat_normalize(q)


