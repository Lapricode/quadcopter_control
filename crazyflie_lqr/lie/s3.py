"""Unit quaternions (S^3), Hamilton convention, q = [w, x, y, z].

The tangent vector used everywhere is the *rotation vector* tau (same as in
``so3``), so ``Exp``/``Log`` here are the full-angle versions and
``Exp(tau)`` represents the same rotation as ``so3.Exp(tau)``.
"""

import numpy as np


def identity():
    return np.array([1.0, 0.0, 0.0, 0.0])


def compose(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def inverse(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(q):
    return q / np.linalg.norm(q)


def Exp(tau):
    tau = np.asarray(tau, dtype=float)
    theta = np.linalg.norm(tau)
    if theta < 1e-8:
        return normalize(np.concatenate([[1.0], 0.5 * tau]))
    half = 0.5 * theta
    return np.concatenate([[np.cos(half)], np.sin(half) / theta * tau])


def Log(q):
    """Quaternion -> rotation vector, always the *shortest* rotation (angle <= pi).

    (The original implementation did not flip q when w < 0, which returns
    non-minimal angles in (pi, 2*pi].)
    """
    if q[0] < 0.0:  # q and -q are the same rotation: take the short way around
        q = -q
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2.0 * v / q[0]
    return 2.0 * np.arctan2(n, q[0]) / n * v


def plus_right(q, tau):
    return normalize(compose(q, Exp(tau)))


def minus_right(q1, q2):
    return Log(compose(inverse(q2), q1))


def to_matrix(q):
    """Quaternion -> rotation matrix (q is normalised internally)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def from_matrix(R):
    """Rotation matrix -> quaternion with w >= 0.

    Shepperd's method: accurate for *every* rotation, including tiny ones
    (the sqrt-of-diagonal formula used before loses ~1e-8 rad near identity).
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if tr > R[i, i]:
        w = 0.5 * np.sqrt(1.0 + tr)
        f = 0.25 / w
        q = np.array([w, (R[2, 1] - R[1, 2]) * f, (R[0, 2] - R[2, 0]) * f, (R[1, 0] - R[0, 1]) * f])
    elif i == 0:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif i == 1:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def from_rpy(roll, pitch, yaw):
    """ZYX Euler angles -> quaternion."""
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def yaw_of(q):
    R = to_matrix(q)
    return np.arctan2(R[1, 0], R[0, 0])
