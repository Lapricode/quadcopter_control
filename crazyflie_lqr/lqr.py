"""LQR / TVLQR synthesis for the Crazyflie error dynamics."""

import numpy as np
from scipy.linalg import solve_discrete_are

from . import dynamics as dy
from . import state as st
from .lie import so3
from .params import CrazyflieParams


# ------------------------------------------------------------------ weights helpers
def bryson_weights(pos=0.1, att=0.2, vel=0.5, rate=4.0, thrust=0.05, yaw=None):
    """Bryson's rule: weight = 1 / (max acceptable deviation)^2.

    pos [m], att [rad], vel [m/s], rate [rad/s], thrust [N, per motor].
    ``yaw`` (rad) optionally gives yaw its own tolerance.
    Returns (Q, R) with Q in R^{12x12}, R in R^{4x4}.
    """
    q = np.concatenate(
        [np.full(3, pos**-2), np.full(3, att**-2), np.full(3, vel**-2), np.full(3, rate**-2)]
    )
    if yaw is not None:
        q[5] = yaw**-2
    return np.diag(q), np.eye(4) * thrust**-2


# ---------------------------------------------------------------------- infinite horizon
def dare_gain(A, B, Q, R):
    """Infinite-horizon discrete LQR gain (replaces the 10 000-step Riccati loop)."""
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return K, P


def hover_gain(p: CrazyflieParams, Q, R, dt, yaw=0.0, discretization="zoh"):
    """Infinite-horizon hover gain for a given yaw, so that  dF = -K (x (-) x_ref).

    The hover error dynamics are invariant to yaw *once the position error is
    expressed in the yaw-aligned frame*, so a single Riccati solve at yaw = 0 is
    enough:  K(yaw) = K0 @ blockdiag(Rz(yaw)^T, I, I, I).  This replaces the
    table of 10 pre-computed gains of the old code and is exact for any yaw
    (it also requires the *same* weight for x and y, see README).
    """
    x0 = st.make_state(yaw=0.0)
    F0 = p.hover_thrust * np.ones(4)
    disc = dy.discretize_zoh if discretization == "zoh" else dy.discretize_euler
    A, B = disc(x0, F0, p, dt)
    K0, P0 = dare_gain(A, B, Q, R)
    return rotate_gain_to_yaw(K0, yaw), P0


def rotate_gain_to_yaw(K0, yaw):
    T = np.eye(12)
    T[0:3, 0:3] = so3.rot_z(yaw).T
    return K0 @ T


def thrust_gain_to_speed_gain(K_F, p: CrazyflieParams):
    """Gain in rotor-speed units (rad/s per unit error) for the *hover* operating point.

    dF = 2 Kf u_hover du   =>   K_u = K_F / (2 Kf u_hover)
    This is the matrix that goes into the firmware (``Kinf`` in the C++ file).
    """
    return K_F / (2.0 * p.Kf * p.hover_speed)


# ----------------------------------------------------------------------- finite horizon
def tvlqr(A_list, B_list, Q, R, Qf):
    """Time-varying LQR (backward Riccati, Joseph-stabilised).

    A_list[k], B_list[k] map the error at step k to step k+1.  Returns
    (K_list, P_list) with len(K_list) == len(A_list); the control at step k is
    dF_k = -K_k dx_k.
    """
    n = len(A_list)
    K_list = [None] * n
    P = Qf.copy()
    P_list = [None] * (n + 1)
    P_list[n] = P
    for k in range(n - 1, -1, -1):
        A, B = A_list[k], B_list[k]
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        Acl = A - B @ K
        P = Q + K.T @ R @ K + Acl.T @ P @ Acl  # Joseph form: symmetric by construction
        P = 0.5 * (P + P.T)
        K_list[k] = K
        P_list[k] = P
    return K_list, P_list


def tvlqr_along_reference(ref, p: CrazyflieParams, Q, R, Qf=None, discretization="zoh"):
    """Linearise the plant along (x_ref[k], F_ref[k]) and solve TVLQR.

    The last reference sample has no successor, so we get len(ref)-1 gains;
    the last one is repeated so that the controller can run for len(ref) steps.
    If ``Qf`` is None the terminal cost is the DARE solution of the last point.
    """
    disc = dy.discretize_zoh if discretization == "zoh" else dy.discretize_euler
    n = len(ref.t)
    A_list, B_list = [], []
    for k in range(n - 1):
        A, B = disc(ref.x[k], ref.F[k], p, ref.dt)
        A_list.append(A)
        B_list.append(B)
    if Qf is None:
        _, Qf = dare_gain(A_list[-1], B_list[-1], Q, R)
    K_list, _ = tvlqr(A_list, B_list, Q, R, Qf)
    K_list.append(K_list[-1])
    return K_list
