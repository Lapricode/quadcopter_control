"""SO(3) Lie-group utilities (rotation matrices, right/left conventions).

Conventions
-----------
* ``tau in R^3`` is the rotation vector (angle * axis), the "Cartesian" element.
* ``hat(tau)`` is the 3x3 skew-symmetric matrix of the algebra, ``vee`` its inverse.
* ``plus_right(R, tau) = R @ Exp(tau)`` and ``minus_right(R1, R2) = Log(R2.T @ R1)``.
  Perturbations therefore live in the *body* frame.

Numerics
--------
Everything stays accurate for tiny rotations (no hard-thresholding of the
angle: Taylor series are used instead) and never produces NaN for rounding
noise (``Log`` uses ``atan2`` instead of ``arccos``).
"""

import numpy as np

_SERIES_TOL = 1e-4  # below this angle Taylor expansions replace the closed forms
_PI_TOL = 1e-3  # distance to pi below which Log() uses the symmetric-part formula


def hat(v):
    """R^3 -> so(3) (skew-symmetric matrix)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def vee(M):
    """so(3) -> R^3 (inverse of :func:`hat`). Always returns shape (3,)."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def _abcd(theta):
    """Coefficients shared by Exp and the Jacobians.

    A = sin t / t,  B = (1 - cos t) / t^2,  C = (1 - A) / t^2,
    D = 1/t^2 - (1 + cos t) / (2 t sin t)   (appears in the inverse Jacobians).
    """
    t2 = theta * theta
    if theta < _SERIES_TOL:
        A = 1.0 - t2 / 6.0 + t2 * t2 / 120.0
        B = 0.5 - t2 / 24.0 + t2 * t2 / 720.0
        C = 1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0
        D = 1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / t2
        C = (1.0 - A) / t2
        D = (1.0 - A / (2.0 * B)) / t2
    return A, B, C, D


def Exp(tau):
    """Rotation vector -> rotation matrix (Rodrigues)."""
    tau = np.asarray(tau, dtype=float)
    theta = np.linalg.norm(tau)
    A, B, _, _ = _abcd(theta)
    K = hat(tau)
    return np.eye(3) + A * K + B * (K @ K)


def Log(R):
    """Rotation matrix -> rotation vector with angle in [0, pi]."""
    w = 0.5 * vee(R - R.T)  # = sin(theta) * axis
    s = np.linalg.norm(w)
    c = 0.5 * (np.trace(R) - 1.0)
    theta = np.arctan2(s, c)
    if theta < _SERIES_TOL:
        return w * (1.0 + theta * theta / 6.0)
    if np.pi - theta > _PI_TOL:
        return w * (theta / s)
    # theta close to pi: R is almost symmetric, recover the axis from R + I
    S = 0.5 * (R + R.T)
    uuT = (S - c * np.eye(3)) / (1.0 - c)
    u = uuT[:, np.argmax(np.diag(uuT))]
    u = u / np.linalg.norm(u)
    if u @ w < 0.0:
        u = -u
    return theta * u


def jacobian_right(tau):
    theta = np.linalg.norm(tau)
    _, B, C, _ = _abcd(theta)
    K = hat(tau)
    return np.eye(3) - B * K + C * (K @ K)


def jacobian_left(tau):
    theta = np.linalg.norm(tau)
    _, B, C, _ = _abcd(theta)
    K = hat(tau)
    return np.eye(3) + B * K + C * (K @ K)


def jacobian_right_inverse(tau):
    theta = np.linalg.norm(tau)
    _, _, _, D = _abcd(theta)
    K = hat(tau)
    return np.eye(3) + 0.5 * K + D * (K @ K)


def jacobian_left_inverse(tau):
    theta = np.linalg.norm(tau)
    _, _, _, D = _abcd(theta)
    K = hat(tau)
    return np.eye(3) - 0.5 * K + D * (K @ K)


def plus_right(R, tau):
    return R @ Exp(tau)


def plus_left(R, tau):
    return Exp(tau) @ R


def minus_right(R1, R2):
    return Log(R2.T @ R1)


def minus_left(R1, R2):
    return Log(R1 @ R2.T)


def adjoint(R):
    return R


# Jacobians of the right-plus / right-minus operators ------------------------
def jacobian_plus_right_1(R, tau):
    """d(R + tau)/dR  (right perturbation)."""
    return Exp(tau).T


def jacobian_plus_right_2(R, tau):
    """d(R + tau)/dtau."""
    return jacobian_right(tau)


def jacobian_minus_right_1(R1, R2):
    return jacobian_right_inverse(minus_right(R1, R2))


def jacobian_minus_right_2(R1, R2):
    return -jacobian_left_inverse(minus_right(R1, R2))


def jacobian_rotation_action_1(R, x):
    """d(R x)/dR (right perturbation)."""
    return -R @ hat(x)


def jacobian_rotation_action_2(R, x):
    return R


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
