"""Closed-loop simulation with zero-order-hold control."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import dynamics as dy
from . import lqr as lq
from . import state as st
from .params import CrazyflieParams


# ------------------------------------------------------------------------ controllers
class HoverController:
    """u = F_hover - K (x (-) x_ref)   (regulation to a fixed configuration)."""

    def __init__(self, K, x_ref, p: CrazyflieParams):
        self.K, self.x_ref = K, x_ref
        self.F_hover = p.hover_thrust * np.ones(4)

    def __call__(self, i, t, x):
        return self.F_hover - self.K @ st.minus(x, self.x_ref)


class TVLQRTracker:
    """u_k = F_ref[k] - K_k (x (-) x_ref[k])   (feed-forward + time-varying feedback)."""

    def __init__(self, ref, K_list):
        self.ref, self.K = ref, K_list

    def __call__(self, i, t, x):
        k = min(i, len(self.ref) - 1)
        return self.ref.F[k] - self.K[k] @ st.minus(x, self.ref.x[k])


# ------------------------------------------------------------------------ results
@dataclass
class SimResult:
    t: np.ndarray  # (N,)
    x: np.ndarray  # (N,13)
    F: np.ndarray  # (N,4) applied (saturated) motor thrusts
    u: np.ndarray  # (N,4) applied motor speeds [rad/s]
    dt: float
    params: CrazyflieParams

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pr = self.params
        np.savez_compressed(
            path, t=self.t, x=self.x, F=self.F, u=self.u, dt=self.dt,
            mass=pr.mass, arm=pr.arm, g=pr.g, Kf=pr.Kf, Kt=pr.Kt,
            body_yaw0=pr.body_yaw0, speed_max=pr.speed_max, inertia=pr.inertia,
        )

    @staticmethod
    def load_npz(path):
        d = np.load(path)
        p = CrazyflieParams(
            mass=float(d["mass"]), arm=float(d["arm"]), g=float(d["g"]), Kf=float(d["Kf"]),
            Kt=float(d["Kt"]), body_yaw0=float(d["body_yaw0"]), inertia=d["inertia"],
            max_rpm=float(d["speed_max"]) * 60 / (2 * np.pi),
        )
        return SimResult(d["t"], d["x"], d["F"], d["u"], float(d["dt"]), p)

    def export_legacy_csv(self, folder):
        """Write the five files the *old* visualiser expected (states/controls/positions/orientations/params)."""
        import json

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        pr = self.params
        np.savetxt(folder / "crazyflie_states.csv", self.x, delimiter=",")
        np.savetxt(folder / "crazyflie_controls.csv", self.u, delimiter=",")
        np.savetxt(folder / "crazyflie_positions.csv", self.x[:, :3], delimiter=",")
        Rs = np.array([st.rotation(x).ravel() for x in self.x])
        np.savetxt(folder / "crazyflie_orientations.csv", Rs, delimiter=",")
        with open(folder / "crazyflie_params.json", "w") as f:
            json.dump(
                {"freq": 1 / self.dt, "dt": self.dt, "mass": pr.mass, "arm_length": pr.arm,
                 "body_yaw0": float(pr.body_yaw0), "u_max": [pr.speed_max] * 4, "g": pr.g,
                 "Kf": pr.Kf, "Kt": pr.Kt}, f, indent=4)


# ------------------------------------------------------------------------ simulator
def simulate(x0, policy, plant: CrazyflieParams, T, dt_sim=1e-3, dt_ctrl=4e-3, meas_noise=None, rng=None,
             model: CrazyflieParams = None, motor_tau=None, integrator="rk4"):
    """Integrate the plant (RK4) while ``policy`` is called every ``dt_ctrl`` seconds (ZOH).

    ``policy(i, t, x_measured) -> F`` returns the 4 *commanded* motor thrusts [N] in the units of the
    controller's ``model``.  They become rotor-speed commands ``u = sqrt(F / model.Kf)`` (saturated to
    [0, speed_max]), exactly like the firmware, and the plant produces ``plant.Kf * u**2``.  Hence
    ``plant`` may differ from ``model`` in mass, inertia, Kf, Kt, arm... (default: model = plant).
    ``motor_tau`` (s) adds a first-order rotor-speed lag.
    ``integrator`` selects the plant integrator: "rk4" (default), "semi_implicit" or "euler" (step = ``dt_sim``).
    ``meas_noise`` = (sigma_pos, sigma_att, sigma_vel, sigma_rate) adds Gaussian noise to the state
    the controller sees (not to the plant).
    """
    model = model or plant
    step = dy.INTEGRATORS[integrator]
    n_steps = int(round(T / dt_sim))
    hold = max(1, int(round(dt_ctrl / dt_sim)))
    rng = rng or np.random.default_rng(0)
    xs = np.zeros((n_steps + 1, 13))
    Fs = np.zeros((n_steps + 1, 4))
    us = np.zeros((n_steps + 1, 4))
    ts = np.arange(n_steps + 1) * dt_sim
    x = np.array(x0, float)
    u_cmd = model.hover_speed * np.ones(4)
    u = u_cmd.copy()
    sig = None
    if meas_noise is not None:
        sig = np.concatenate([np.full(3, meas_noise[0]), np.full(3, meas_noise[1]),
                              np.full(3, meas_noise[2]), np.full(3, meas_noise[3])])
    for k in range(n_steps + 1):
        if k % hold == 0:
            x_meas = x if sig is None else st.plus(x, rng.normal(size=12) * sig)
            F_cmd = np.clip(policy(k // hold, ts[k], x_meas), 0.0, None)
            u_cmd = np.clip(np.sqrt(F_cmd / model.Kf), 0.0, plant.speed_max)
        u = u_cmd if motor_tau is None else u + dt_sim / motor_tau * (u_cmd - u)
        F = plant.Kf * u**2
        xs[k], Fs[k], us[k] = x, F, u
        if k < n_steps:
            x = step(x, F, plant, dt_sim)
    return SimResult(ts, xs, Fs, us, dt_sim, plant)


# ------------------------------------------------------------------------ metrics
def tracking_errors(res: SimResult, ref):
    """Position / yaw-aware errors of a run against a Reference sampled at its own dt."""
    idx = np.clip(np.round(res.t / ref.dt).astype(int), 0, len(ref) - 1)
    e_pos = res.x[:, :3] - ref.pos[idx]
    e_att = np.array([np.linalg.norm(st.minus(x, ref.x[i])[3:6]) for x, i in zip(res.x, idx)])
    e_vel = res.x[:, 7:10] - ref.x[idx, 7:10]
    return e_pos, e_att, e_vel, idx


class SpeedGainController:
    """Firmware-style law  u = u_hover - K_u (x (-) x_ref)  in rotor-speed units, saturated to [0, speed_max].

    ``u_hover`` is the feed-forward speed the firmware uses (model hover speed + ``hover_adjust``);
    ``plant`` provides the thrust coefficient used to convert speeds to thrusts.
    """

    def __init__(self, K_u, u_hover, x_ref, plant: CrazyflieParams, yaw_rotate=True):
        from .lqr import rotate_gain_to_yaw
        from .lie import s3

        self.K = rotate_gain_to_yaw(K_u, s3.yaw_of(x_ref[3:7])) if yaw_rotate else K_u
        self.u_hover, self.x_ref, self.plant = u_hover, x_ref, plant

    def __call__(self, i, t, x):
        u = np.clip(self.u_hover - self.K @ st.minus(x, self.x_ref), 0.0, self.plant.speed_max)
        return self.plant.Kf * u**2


# ------------------------------------------------------------- integral controllers
class _IntegralMixin:
    """Integrator state with conditional integration (anti-windup) and clamping."""

    def _init_integral(self, dt, axes, xi_max, plant_thrust_max):
        self.dt, self.axes, self.xi_max, self.Fmax = dt, tuple(axes), xi_max, plant_thrust_max
        self.xi = np.zeros(len(self.axes))

    def reset(self):
        self.xi[:] = 0.0

    def _law(self, F_ff, K, e):
        z = np.concatenate([e, self.xi])
        F = F_ff - K @ z
        saturated = np.any(F < 0.0) or np.any(F > self.Fmax)
        if not saturated:  # conditional integration: freeze the integrator while motors saturate
            self.xi = np.clip(self.xi + self.dt * e[list(self.axes)], -self.xi_max, self.xi_max)
        return F


class IntegralHoverController(_IntegralMixin):
    def __init__(self, K_aug, x_ref, p: CrazyflieParams, dt, axes=(0, 1, 2), xi_max=1.0):
        self.K, self.x_ref, self.F_hover = K_aug, x_ref, p.hover_thrust * np.ones(4)
        self._init_integral(dt, axes, xi_max, p.thrust_max)

    def __call__(self, i, t, x):
        return self._law(self.F_hover, self.K, st.minus(x, self.x_ref))


class IntegralTVLQRTracker(_IntegralMixin):
    def __init__(self, ref, K_list, p: CrazyflieParams, axes=(0, 1, 2), xi_max=1.0):
        self.ref, self.K = ref, K_list
        self._init_integral(ref.dt, axes, xi_max, p.thrust_max)

    def __call__(self, i, t, x):
        k = min(i, len(self.ref) - 1)
        return self._law(self.ref.F[k], self.K[k], st.minus(x, self.ref.x[k]))
