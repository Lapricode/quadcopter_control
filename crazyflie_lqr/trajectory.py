"""Dynamically feasible reference trajectories via differential flatness.

The quadrotor is differentially flat in (r_w(t), yaw(t)).  Given a smooth
position/yaw profile we recover the *whole* nominal state and the nominal
motor thrusts:

    z_b = (a + g e_z) / |a + g e_z|            thrust direction
    T   = m |a + g e_z|                        collective thrust
    x_b, y_b from z_b and the yaw              -> R_wb
    omega_b = Log(R_k^-1 R_k+1) / dt           (manifold finite differences)
    domega_b = d(omega_b)/dt
    tau = I domega_b + omega_b x I omega_b
    F   = M^-1 [T, tau]                        nominal motor thrusts

A reference built this way satisfies the dynamics (up to O(dt^2)), which is what
makes TVLQR work: the feed-forward input F_ref keeps the vehicle on the
trajectory and the TVLQR feedback only has to fight disturbances.  (The old
commented-out circle used zero reference velocities with hover thrust, i.e. a
reference the vehicle could never follow.)
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.interpolate import CubicSpline

from . import state as st
from .lie import s3, so3
from .params import CrazyflieParams


@dataclass
class Reference:
    t: np.ndarray  # (N,)
    x: np.ndarray  # (N, 13) nominal states
    F: np.ndarray  # (N, 4) nominal motor thrusts [N]
    pos: np.ndarray  # (N, 3) world position
    vel: np.ndarray  # (N, 3) world velocity
    acc: np.ndarray  # (N, 3) world acceleration
    yaw: np.ndarray  # (N,)
    name: str = "trajectory"

    @property
    def dt(self):
        return float(self.t[1] - self.t[0])

    def __len__(self):
        return len(self.t)


# --------------------------------------------------------------- flatness -> reference
def flat_reference(t, pos, vel, acc, yaw, p: CrazyflieParams, name="trajectory"):
    t = np.asarray(t)
    dt = float(t[1] - t[0])
    n = len(t)
    a_t = acc + np.array([0.0, 0.0, p.g])
    norm = np.linalg.norm(a_t, axis=1, keepdims=True)
    zb = a_t / norm
    T = p.mass * norm[:, 0]
    xc = np.stack([np.cos(yaw), np.sin(yaw), np.zeros(n)], axis=1)
    yb = np.cross(zb, xc)
    yb /= np.linalg.norm(yb, axis=1, keepdims=True)
    xb = np.cross(yb, zb)
    Rs = np.stack([xb, yb, zb], axis=2)  # (N,3,3), columns are the body axes

    omega = np.zeros((n, 3))
    for k in range(1, n - 1):
        omega[k] = so3.Log(Rs[k - 1].T @ Rs[k + 1]) / (2 * dt)
    omega[0] = so3.Log(Rs[0].T @ Rs[1]) / dt
    omega[-1] = so3.Log(Rs[-2].T @ Rs[-1]) / dt
    omega_dot = np.gradient(omega, dt, axis=0, edge_order=2)

    I = p.inertia
    tau = omega_dot @ I.T + np.cross(omega, omega @ I.T)
    wrench = np.column_stack([T, tau])
    F = wrench @ p.mixer_inv.T

    x = np.zeros((n, 13))
    for k in range(n):
        x[k, st.R_SL] = pos[k]
        x[k, st.Q_SL] = s3.from_matrix(Rs[k])
        x[k, st.V_SL] = Rs[k].T @ vel[k]
        x[k, st.W_SL] = omega[k]
    return Reference(t=t, x=x, F=F, pos=pos, vel=vel, acc=acc, yaw=np.asarray(yaw), name=name)


# ------------------------------------------------------------ smooth progress profile
def _smoother(x):
    return 6 * x**5 - 15 * x**4 + 10 * x**3


def _dsmoother(x):
    return 30 * x**2 * (1 - x) ** 2


def _ismoother(x):  # integral of _smoother from 0 to x
    return x**6 - 3 * x**5 + 2.5 * x**4


def progress_profile(t, T, ramp, rate):
    """C^2 progress phi(t) that accelerates to ``rate`` (rad/s) in ``ramp`` s, cruises,
    then brakes to rest at ``T``.  Returns (phi, dphi, ddphi)."""
    t = np.asarray(t, float)
    phi = np.zeros_like(t)
    dphi = np.zeros_like(t)
    ddphi = np.zeros_like(t)
    up = t < ramp
    dn = t > T - ramp
    mid = ~up & ~dn
    x = t[up] / ramp
    phi[up], dphi[up], ddphi[up] = rate * ramp * _ismoother(x), rate * _smoother(x), rate / ramp * _dsmoother(x)
    phi[mid] = rate * ramp / 2 + rate * (t[mid] - ramp)
    dphi[mid] = rate
    phi_tot = rate * (T - ramp)
    x = np.clip((T - t[dn]) / ramp, 0, 1)
    phi[dn] = phi_tot - rate * ramp * _ismoother(x)
    dphi[dn] = rate * _smoother(x)
    ddphi[dn] = -rate / ramp * _dsmoother(x)
    return phi, dphi, ddphi


# ----------------------------------------------------------------------------- paths
def _circle(phi, center, radius, climb):
    c, s = np.cos(phi), np.sin(phi)
    z = np.zeros_like(phi)
    p = center + np.stack([radius * c, radius * s, climb * phi], axis=1)
    dp = np.stack([-radius * s, radius * c, climb + z], axis=1)
    ddp = np.stack([-radius * c, -radius * s, z], axis=1)
    return p - np.array([radius, 0.0, 0.0]), dp, ddp


def _figure8(phi, center, ax, ay, az):
    c, s, c2, s2 = np.cos(phi), np.sin(phi), np.cos(2 * phi), np.sin(2 * phi)
    p = center + np.stack([ax * s, 0.5 * ay * s2, az * s], axis=1)
    dp = np.stack([ax * c, ay * c2, az * c], axis=1)
    ddp = np.stack([-ax * s, -2 * ay * s2, -az * s], axis=1)
    return p, dp, ddp


def path_trajectory(
    kind,
    p: CrazyflieParams,
    T=12.0,
    dt=0.004,
    center=(0.0, 0.0, 1.0),
    size=0.8,
    rate=1.2,
    ramp=2.0,
    climb=0.0,
    yaw_mode="fixed",
    yaw0=0.0,
):
    """Build a reference that follows a geometric path at the progress rate ``rate`` rad/s.

    kind      : "circle" (radius=size), "figure8" (half-width=size, optional z-wobble=climb) or "helix"
    yaw_mode  : "fixed" (yaw0 constant) or "tangent" (nose follows the direction of travel)
    """
    t = np.arange(0.0, T + 0.5 * dt, dt)
    phi, dphi, ddphi = progress_profile(t, T, ramp, rate)
    center = np.asarray(center, float)
    if kind in ("circle", "helix"):
        pth, dp, ddp = _circle(phi, center, size, climb if kind == "helix" else 0.0)
        heading = phi + 0.5 * np.pi
    elif kind == "figure8":
        pth, dp, ddp = _figure8(phi, center, size, size, climb)
        heading = np.arctan2(dp[:, 1], dp[:, 0])
    else:
        raise ValueError(f"unknown path '{kind}'")
    vel = dp * dphi[:, None]
    acc = ddp * (dphi**2)[:, None] + dp * ddphi[:, None]
    yaw = np.full_like(t, yaw0) if yaw_mode == "fixed" else np.unwrap(heading)
    return flat_reference(t, pth, vel, acc, yaw, p, name=f"{kind}")


def velocity_profile_trajectory(
    knot_t, knot_v, p: CrazyflieParams, r0=(0.0, 0.0, 1.0), dt=0.004, yaw=0.0, name="velocity profile"
):
    """Reference from a *velocity profile*: world-frame velocity knots (K,3) at times knot_t (K,).

    A clamped cubic spline (zero acceleration at both ends) interpolates the
    knots; position is its integral (Simpson), acceleration its analytic derivative.
    The first and last knot velocities should be zero for a start/stop manoeuvre.
    """
    knot_t = np.asarray(knot_t, float)
    knot_v = np.asarray(knot_v, float)
    t = np.arange(knot_t[0], knot_t[-1] + 0.5 * dt, dt)
    cs = CubicSpline(knot_t, knot_v, bc_type="clamped", axis=0)
    vel = cs(t)
    acc = cs.derivative(1)(t)
    pos = np.asarray(r0, float) + cumulative_simpson(vel, dx=dt, axis=0, initial=0.0)
    yaw_arr = np.full_like(t, yaw) if np.isscalar(yaw) else yaw(t)
    return flat_reference(t - t[0], pos, vel, acc, yaw_arr, p, name=name)


def subsample(ref: Reference, n: int) -> Reference:
    """Keep every n-th sample (e.g. plan at 20 ms from a reference computed at 4 ms)."""
    return Reference(t=ref.t[::n], x=ref.x[::n], F=ref.F[::n], pos=ref.pos[::n], vel=ref.vel[::n],
                     acc=ref.acc[::n], yaw=ref.yaw[::n], name=ref.name)
