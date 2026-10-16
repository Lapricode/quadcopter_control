"""Constrained planning/control: iLQR with input bounds + soft state constraints, and receding-horizon MPC.

    python -m examples.constrained --scenario step       # big 3-D step: LQR flips, iLQR respects limits (~30 s)
    python -m examples.constrained --scenario obstacle   # figure-8 with a sphere on the path (~2 min)
"""

import argparse
import time
from pathlib import Path

import numpy as np

from crazyflie_lqr import ilqr
from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr import trajectory as tj
from crazyflie_lqr.params import CrazyflieParams

OUT = Path(__file__).resolve().parents[1] / "results"


def _tilt(res):
    return np.degrees(np.arccos(np.clip([st.rotation(x)[2, 2] for x in res.x], -1, 1)))


def scenario_step(p):
    import matplotlib.pyplot as plt

    x0 = st.make_state(r=[0, 0, 0.5])
    tgt = np.array([1.5, 1.0, 1.5])
    x_goal = st.make_state(r=tgt)
    T, dt = 6.0, 0.02
    runs = {}
    with np.errstate(all="ignore"):
        for tol in (0.1, 0.3):  # LQR baselines at 100 Hz
            Q, R = lq.bryson_weights(tol, 0.1, 0.5, 4.0, 0.05)
            K, _ = lq.hover_gain(p, Q, R, 0.01)
            runs[f"LQR (pos tol {tol} m)"] = sim.simulate(x0, sim.HoverController(K, x_goal, p), p, T, 1e-3, 0.01)
        N = int(T / dt)
        Q, R = lq.bryson_weights(0.2, 0.2, 0.5, 4.0, 0.05)
        F_h = p.hover_thrust * np.ones(4)
        for name, pens in [("iLQR (thrust bounds)", []), ("iLQR (+ tilt <= 30 deg)", [ilqr.TiltLimit(30.0, 1e5)])]:
            prob = ilqr.Problem(np.tile(x_goal, (N + 1, 1)), np.tile(F_h, (N, 1)), Q, R, 100 * Q, dt, pens, 0.0, p.thrust_max, 2)
            t0 = time.time()
            sol = ilqr.solve(prob, x0, np.tile(F_h, (N, 1)), p, iters=60)
            print(f"  {name}: {sol.iters} iterations, {time.time() - t0:.0f} s")
            runs[name] = sim.simulate(x0, ilqr.ILQRPolicy(sol, p), p, T, 1e-3, dt)
    print(f"{'controller':<26}{'final err':>10}{'settle 2cm':>12}{'max tilt':>10}{'saturated':>11}")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, r in runs.items():
        e = np.linalg.norm(r.x[:, :3] - tgt, axis=1)
        idx = np.where(e > 0.02)[0]
        ok = np.isfinite(r.x).all()
        tl = _tilt(r)
        sat = ((r.F >= p.thrust_max - 1e-9) | (r.F <= 1e-12)).mean()
        print(f"{name:<26}{e[-1] if ok else np.nan:10.3f}{(r.t[idx[-1]] if len(idx) else 0):11.2f}s{np.nanmax(tl):9.0f}°{100 * sat:10.1f}%")
        ax[0].plot(r.t, np.clip(e, 0, 3), label=name)
        ax[1].plot(r.t, np.clip(tl, 0, 180))
        ax[2].plot(r.t, r.F.max(axis=1) / p.thrust_max)
    ax[0].set_title("distance to target [m] (clipped at 3)"); ax[1].set_title("tilt [deg]")
    ax[2].set_title("largest motor thrust / F_max"); ax[0].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("t [s]")
    fig.tight_layout(); OUT.mkdir(exist_ok=True); fig.savefig(OUT / "constrained_step.png", dpi=110)


def scenario_obstacle(p, mass_error=0.05):
    import matplotlib.pyplot as plt

    fine = tj.path_trajectory("figure8", p, T=14.0, dt=0.004, size=0.9, rate=1.1, ramp=2.5, climb=0.15, yaw_mode="tangent")
    ref = tj.subsample(fine, 5)
    dt = ref.dt
    c = ref.pos[int(6.0 / dt)] + np.array([0.0, 0.1, 0.0])
    obs = ilqr.SphereObstacle(center=tuple(c), radius=0.25, margin=0.10, weight=1e5)
    Q, R = lq.bryson_weights(0.1, 0.2, 0.5, 4.0, 0.05)
    x0 = ref.x[0].copy()
    x0[:3] += [0.1, -0.05, 0.05]
    plant = p.with_(mass=p.mass * (1 + mass_error))
    runs = {}
    runs["TVLQR (ignores obstacle)"] = sim.simulate(x0, sim.TVLQRTracker(ref, lq.tvlqr_along_reference(ref, p, Q, R)), plant, 14.0, 1e-3, dt)
    prob = ilqr.Problem(ref.x, ref.F[:-1], Q, R, 10 * Q, dt, [obs], 0.0, p.thrust_max, 2)
    t0 = time.time()
    sol = ilqr.solve(prob, ref.x[0], ref.F[:-1], p, iters=40)
    print(f"  offline iLQR plan: {sol.iters} iterations, {time.time() - t0:.0f} s")
    runs["iLQR plan + its feedback"] = sim.simulate(x0, ilqr.ILQRPolicy(sol, p), plant, 14.0, 1e-3, dt)
    mpc = ilqr.ILQRMPC(ref, p, Q, R, 10 * Q, horizon=25, penalties=[obs], replan_every=5, iters=4)
    runs["iLQR MPC (0.5 s horizon)"] = sim.simulate(x0, mpc, plant, 14.0, 1e-3, dt)
    print(f"  MPC solve time: mean {np.mean(mpc.solve_times):.2f} s, max {np.max(mpc.solve_times):.2f} s per re-plan (every {5 * dt:.2f} s)")
    print(f"(plant mass +{mass_error:.0%}; obstacle radius 0.25 m + 0.10 m margin)")
    print(f"{'controller':<28}{'min dist to centre':>20}{'rms dev. from ref':>20}{'thrust bounds ok':>18}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, r in runs.items():
        d = np.linalg.norm(r.x[:, :3] - c, axis=1)
        dev = np.linalg.norm(r.x[:, :3] - ref.pos[np.clip(np.round(r.t / dt).astype(int), 0, len(ref) - 1)], axis=1)
        print(f"{name:<28}{d.min():17.3f} m{1000 * np.sqrt((dev ** 2).mean()):17.0f} mm{str(bool(r.F.min() >= 0 and r.F.max() <= p.thrust_max + 1e-12)):>18}")
        ax[0].plot(r.x[:, 0], r.x[:, 1], label=name)
        ax[1].plot(r.t, d, label=name)
    ax[0].plot(ref.pos[:, 0], ref.pos[:, 1], "k--", lw=0.8, label="reference")
    ax[0].add_patch(plt.Circle(c[:2], 0.25, color="r", alpha=0.35)); ax[0].add_patch(plt.Circle(c[:2], 0.35, fill=False, color="r", ls=":"))
    ax[0].set_aspect("equal"); ax[0].set_title("top view (red = obstacle, dotted = margin)"); ax[0].legend(fontsize=7)
    ax[1].axhline(0.25, color="r"); ax[1].axhline(0.35, color="r", ls=":"); ax[1].set_title("distance to obstacle centre [m]"); ax[1].set_xlabel("t [s]")
    fig.tight_layout(); OUT.mkdir(exist_ok=True); fig.savefig(OUT / "constrained_obstacle.png", dpi=110)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="step", choices=["step", "obstacle", "all"])
    a = ap.parse_args(argv)
    p = CrazyflieParams()
    if a.scenario in ("step", "all"):
        print("== large 3-D step (0,0,0.5) -> (1.5,1,1.5), controller at 100/50 Hz")
        scenario_step(p)
    if a.scenario in ("obstacle", "all"):
        print("== figure-8 with a spherical obstacle on the path")
        scenario_obstacle(p)


if __name__ == "__main__":
    main()
