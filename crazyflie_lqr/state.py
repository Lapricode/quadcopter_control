"""State manifold  M = R^3 x SO(3) x R^3 x R^3.

Stored state (13,):  x = [r_w (3), q_wb (4), v_b (3), omega_b (3)]
Tangent / error  (12,): dx = [dr_w, dtheta_b, dv_b, domega_b]

``plus``/``minus`` use the *right* convention of the original project
(``plus``: q <- q (x) Exp(dtheta); ``minus``: Log(q2^-1 q1)), and are exactly what
the C++ controller computes in ``compute_state_error``.
"""

import numpy as np

from .lie import s3

R_SL = slice(0, 3)
Q_SL = slice(3, 7)
V_SL = slice(7, 10)
W_SL = slice(10, 13)


def make_state(r=(0.0, 0.0, 0.0), q=None, v=(0.0, 0.0, 0.0), omega=(0.0, 0.0, 0.0), yaw=0.0):
    if q is None:
        q = s3.from_rpy(0.0, 0.0, yaw)
    return np.concatenate([np.asarray(r, float), np.asarray(q, float), np.asarray(v, float), np.asarray(omega, float)])


def rotation(x):
    """R_wb of a state."""
    return s3.to_matrix(x[Q_SL])


def plus(x, dx):
    out = np.empty(13)
    out[R_SL] = x[R_SL] + dx[0:3]
    out[Q_SL] = s3.plus_right(x[Q_SL], dx[3:6])
    out[V_SL] = x[V_SL] + dx[6:9]
    out[W_SL] = x[W_SL] + dx[9:12]
    return out


def minus(x1, x2):
    out = np.empty(12)
    out[0:3] = x1[R_SL] - x2[R_SL]
    out[3:6] = s3.minus_right(x1[Q_SL], x2[Q_SL])
    out[6:9] = x1[V_SL] - x2[V_SL]
    out[9:12] = x1[W_SL] - x2[W_SL]
    return out
