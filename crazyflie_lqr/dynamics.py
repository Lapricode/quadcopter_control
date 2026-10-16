"""Crazyflie rigid-body dynamics, Jacobians and integrators.

Model (same as the original project)
------------------------------------
    r_dot       = R v
    R_dot       = R hat(omega)
    v_dot       = (1/m) (R^T [0,0,-m g] + [0,0,T]) - omega x v
    omega_dot   = I^-1 (tau - omega x I omega)

with the *motor thrusts* F_i = Kf u_i^2 as control input.  Working with thrusts
instead of rotor speeds makes the input matrix B constant (the model is affine
in F), which is what an LQR design wants; speeds are recovered with
``u_i = sqrt(F_i / Kf)``.  The wrench is  [T, tau] = M F  (see
``CrazyflieParams.mixer``).
"""

import numpy as np
from scipy.linalg import expm

from . import state as st
from .lie import s3, so3
from .params import CrazyflieParams


# ----------------------------------------------------------------------------- model
def dynamics(x, F, p: CrazyflieParams):
    """Continuous dynamics, returns the 12-d tangent vector [r_dot, omega, v_dot, omega_dot]."""
    R = st.rotation(x)
    v, w = x[st.V_SL], x[st.W_SL]
    wrench = p.mixer @ F
    T, tau = wrench[0], wrench[1:]
    f_body = R.T @ np.array([0.0, 0.0, -p.mass * p.g]) + np.array([0.0, 0.0, T])
    v_dot = f_body / p.mass - np.cross(w, v)
    w_dot = p.inertia_inv @ (tau - np.cross(w, p.inertia @ w))
    return np.concatenate([R @ v, w, v_dot, w_dot])


def jacobians_continuous(x, F, p: CrazyflieParams):
    """Error-state Jacobians (A_c, B_c) with right perturbations  dx = x (-) x_ref.

        d(dr)/dt     = -R hat(v) dtheta + R dv
        d(dtheta)/dt = -hat(w) dtheta + dw
        d(dv)/dt     = hat(R^T g_w) dtheta - hat(w) dv + hat(v) dw
        d(dw)/dt     = -I^-1 (hat(w) I - hat(I w)) dw + I^-1 d(tau)
    """
    R = st.rotation(x)
    v, w = x[st.V_SL], x[st.W_SL]
    I, Iinv = p.inertia, p.inertia_inv
    A = np.zeros((12, 12))
    A[0:3, 3:6] = -R @ so3.hat(v)
    A[0:3, 6:9] = R
    A[3:6, 3:6] = -so3.hat(w)
    A[3:6, 9:12] = np.eye(3)
    A[6:9, 3:6] = so3.hat(R.T @ np.array([0.0, 0.0, -p.g]))
    A[6:9, 6:9] = -so3.hat(w)
    A[6:9, 9:12] = so3.hat(v)
    A[9:12, 9:12] = -Iinv @ (so3.hat(w) @ I - so3.hat(I @ w))
    M = p.mixer
    B = np.zeros((12, 4))
    B[8, :] = M[0] / p.mass
    B[9:12, :] = Iinv @ M[1:4]
    return A, B


def discretize_zoh(x, F, p, dt):
    """Exact zero-order-hold discretisation of the frozen continuous error model."""
    A, B = jacobians_continuous(x, F, p)
    n, m = B.shape
    E = np.zeros((n + m, n + m))
    E[:n, :n] = A
    E[:n, n:] = B
    Ed = expm(E * dt)
    return Ed[:n, :n], Ed[:n, n:]


def discretize_euler(x, F, p, dt):
    """Analytic Jacobians of ``step_euler`` (explicit Euler with SO(3) retraction)."""
    A, B = jacobians_continuous(x, F, p)
    Ad = np.eye(12) + A * dt
    w = x[st.W_SL]
    Ad[3:6, 3:6] = so3.Exp(w * dt).T
    Ad[3:6, 9:12] = so3.jacobian_right(w * dt) * dt
    return Ad, B * dt


def finite_difference_jacobians(step, x, F, dt, eps=1e-6):
    """Central finite differences of a one-step map ``x+ = step(x, F, dt)`` in tangent coordinates."""
    A = np.zeros((12, 12))
    B = np.zeros((12, 4))
    for i in range(12):
        e = np.zeros(12)
        e[i] = eps
        A[:, i] = st.minus(step(st.plus(x, e), F, dt), step(st.plus(x, -e), F, dt)) / (2 * eps)
    for j in range(4):
        e = np.zeros(4)
        e[j] = eps
        B[:, j] = st.minus(step(x, F + e, dt), step(x, F - e, dt)) / (2 * eps)
    return A, B


# ------------------------------------------------------------------------ integrators
def step_euler(x, F, p, dt):
    return st.plus(x, dynamics(x, F, p) * dt)


def step_rk4(x, F, p, dt):
    k1 = dynamics(x, F, p)
    k2 = dynamics(st.plus(x, 0.5 * dt * k1), F, p)
    k3 = dynamics(st.plus(x, 0.5 * dt * k2), F, p)
    k4 = dynamics(st.plus(x, dt * k3), F, p)
    return st.plus(x, dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))


def step_semi_implicit(x, F, p, dt):
    """Semi-implicit (symplectic) Euler: update the velocities first, then the poses with the *new* velocities.

        omega+ = omega + dt * I^-1 (tau - omega x I omega)
        v+     = v + dt * ( (R^T [0,0,-m g] + [0,0,T]) / m - omega+ x v )     (old R, new omega)
        q+     = q (x) Exp(omega+ dt)
        r+     = r + dt * R+ v+                                               (new R, new v)

    First-order accurate, like explicit Euler (measured order 1.0).  It is NOT more accurate than explicit
    Euler in general: its error constant is smaller in some closed-loop tests and larger in open-loop
    tumbling, and it can blow up at very coarse steps (80 ms).  Its advantages here are cost (analytic
    Jacobians, no matrix exponential) and a clean, exactly differentiable discrete model.  RK4 remains the
    default wherever accuracy matters (README, section 9).
    """
    R = st.rotation(x)
    v, w = x[st.V_SL], x[st.W_SL]
    wrench = p.mixer @ F
    T, tau = wrench[0], wrench[1:]
    w_new = w + dt * (p.inertia_inv @ (tau - np.cross(w, p.inertia @ w)))
    f_body = R.T @ np.array([0.0, 0.0, -p.mass * p.g]) + np.array([0.0, 0.0, T])
    v_new = v + dt * (f_body / p.mass - np.cross(w_new, v))
    q_new = s3.plus_right(x[st.Q_SL], w_new * dt)
    r_new = x[st.R_SL] + dt * (s3.to_matrix(q_new) @ v_new)
    return np.concatenate([r_new, q_new, v_new, w_new])


def discretize_semi_implicit(x, F, p, dt):
    """Analytic Jacobians (A, B) of ``step_semi_implicit`` in tangent coordinates (right perturbations).

    Built by chaining the four sub-updates; each row block is a (3 x 16) map from [d x (12), d F (4)].
    """
    R = st.rotation(x)
    v, w = x[st.V_SL], x[st.W_SL]
    I, Iinv, M, m = p.inertia, p.inertia_inv, p.mixer, p.mass
    I3 = np.eye(3)
    w_new = w + dt * (Iinv @ (M[1:4] @ F - np.cross(w, I @ w)))
    v_new = step_semi_implicit(x, F, p, dt)[st.V_SL]
    R_new = R @ so3.Exp(w_new * dt)

    Dw = np.zeros((3, 16))
    Dw[:, 9:12] = I3 + dt * (-Iinv @ (so3.hat(w) @ I - so3.hat(I @ w)))
    Dw[:, 12:16] = dt * Iinv @ M[1:4]

    Dv = np.zeros((3, 16))
    Dv[:, 3:6] = dt * so3.hat(R.T @ np.array([0.0, 0.0, -p.g]))
    Dv[:, 6:9] = I3 - dt * so3.hat(w_new)
    Dv[2, 12:16] = dt * M[0] / m
    Dv += dt * so3.hat(v) @ Dw

    Dth = np.zeros((3, 16))
    Dth[:, 3:6] = so3.Exp(w_new * dt).T
    Dth += dt * so3.jacobian_right(w_new * dt) @ Dw

    Dr = np.zeros((3, 16))
    Dr[:, 0:3] = I3
    Dr += dt * (-R_new @ so3.hat(v_new) @ Dth + R_new @ Dv)

    J = np.vstack([Dr, Dth, Dv, Dw])
    return J[:, :12], J[:, 12:]


# name -> integrator / matching discrete-time Jacobians (used by simulate, lqr and ilqr)
INTEGRATORS = {"euler": step_euler, "semi_implicit": step_semi_implicit, "rk4": step_rk4}
DISCRETIZERS = {"euler": discretize_euler, "semi_implicit": discretize_semi_implicit, "rk4": discretize_zoh, "zoh": discretize_zoh}


# ---------------------------------------------------------------------------- helpers
def speeds_to_thrusts(u, p):
    return p.Kf * np.square(u)


def thrusts_to_speeds(F, p):
    return np.sqrt(np.clip(F, 0.0, None) / p.Kf)
