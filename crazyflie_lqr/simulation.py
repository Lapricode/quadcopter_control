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
def simulate(x0, policy, plant: CrazyflieParams, T, dt_sim=1e-3, dt_ctrl=4e-3, meas_noise=None, rng=None):
    """Integrate the plant (RK4) while ``policy`` is called every ``dt_ctrl`` seconds (ZOH).

    ``policy(i, t, x_measured) -> F`` returns the 4 motor thrusts [N]; they are
    saturated to [0, thrust_max] (equivalently, rotor speeds in [0, speed_max]).
    ``meas_noise`` = (sigma_pos, sigma_att, sigma_vel, sigma_rate) adds Gaussian
    noise to the state the controller sees (not to the plant).
    """
    n_steps = int(round(T / dt_sim))
    hold = max(1, int(round(dt_ctrl / dt_sim)))
    rng = rng or np.random.default_rng(0)
    xs = np.zeros((n_steps + 1, 13))
    Fs = np.zeros((n_steps + 1, 4))
    ts = np.arange(n_steps + 1) * dt_sim
    x = np.array(x0, float)
    F = plant.hover_thrust * np.ones(4)
    for k in range(n_steps + 1):
        if k % hold == 0:
            x_meas = x
            if meas_noise is not None:
                sig = np.concatenate([np.full(3, meas_noise[0]), np.full(3, meas_noise[1]),
                                      np.full(3, meas_noise[2]), np.full(3, meas_noise[3])])
                x_meas = st.plus(x, rng.normal(size=12) * sig)
            F = np.clip(policy(k // hold, ts[k], x_meas), 0.0, plant.thrust_max)
        xs[k], Fs[k] = x, F
        if k < n_steps:
            x = dy.step_rk4(x, F, plant, dt_sim)
    return SimResult(ts, xs, Fs, dy.thrusts_to_speeds(Fs, plant), dt_sim, plant)


# ------------------------------------------------------------------------ metrics
def tracking_errors(res: SimResult, ref):
    """Position / yaw-aware errors of a run against a Reference sampled at its own dt."""
    idx = np.clip(np.round(res.t / ref.dt).astype(int), 0, len(ref) - 1)
    e_pos = res.x[:, :3] - ref.pos[idx]
    e_att = np.array([np.linalg.norm(st.minus(x, ref.x[i])[3:6]) for x, i in zip(res.x, idx)])
    e_vel = res.x[:, 7:10] - ref.x[idx, 7:10]
    return e_pos, e_att, e_vel, idx
