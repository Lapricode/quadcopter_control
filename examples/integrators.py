"""Compare the three integrators: explicit Euler, semi-implicit Euler, RK4.

    python -m examples.integrators              # accuracy + closed-loop tests (~15 s)
    python -m examples.integrators --planning   # + iLQR planning cost per integrator (~1 min)
"""

import argparse
import time

import numpy as np

from crazyflie_lqr import dynamics as dy
from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr.lie import s3
from crazyflie_lqr.params import CrazyflieParams

NAMES = ["euler", "semi_implicit", "rk4"]


def accuracy(p):
    x0 = st.make_state(r=[0, 0, 1], q=s3.from_rpy(0.3, -0.2, 0.5), v=[0.8, -0.4, 0.3], omega=[2.0, -1.5, 1.0])
    F = p.hover_thrust * np.ones(4) + np.array([0.006, -0.004, 0.002, -0.003])

    def run(step, dt, T=1.0):
        x = x0.copy()
        for _ in range(int(round(T / dt))):
            x = step(x, F, p, dt)
        return x

    ref = run(dy.step_rk4, 1e-4)
    print("1) open-loop accuracy after 1 s of a lively manoeuvre (v = 0.9 m/s, w = 2.7 rad/s, uneven thrust)")
    print("   error vs RK4 with dt = 0.1 ms:   position [m]   attitude [rad]")
    print(f"   {'integrator':<15}" + "".join(f"{f'dt={d * 1000:.0f} ms':>22}" for d in (10e-3, 4e-3, 1e-3)))
    errs = {}
    for n in NAMES:
        cells = []
        for dt in (10e-3, 4e-3, 1e-3):
            e = st.minus(run(dy.INTEGRATORS[n], dt), ref)
            errs[(n, dt)] = np.linalg.norm(e[0:3])
            cells.append(f"{np.linalg.norm(e[0:3]):9.1e} {np.linalg.norm(e[3:6]):11.1e}")
        print(f"   {n:<15}" + "".join(f"{c:>22}" for c in cells))
    print("   observed order (position, 4 ms -> 1 ms): " +
          ", ".join(f"{n} {np.log(errs[(n, 4e-3)] / errs[(n, 1e-3)]) / np.log(4):.2f}" for n in NAMES))
    x = x0.copy()
    print("   cost per step: " + ", ".join(f"{n} {_time(dy.INTEGRATORS[n], x, F, p):.0f} us" for n in NAMES))


def _time(step, x, F, p, n=2000):
    t0 = time.time()
    for _ in range(n):
        x = step(x, F, p, 1e-3)
    return (time.time() - t0) / n * 1e6


def closed_loop(p):
    Q, R = lq.bryson_weights(0.3, 0.1, 0.5, 4.0, 0.05)
    x_ref = st.make_state(r=[0.5, 0.5, 0.5])
    print("\n2) closed loop: plant integrated with the SAME coarse step as the controller (gain from ZOH design)")
    print("   deviation of the position at t = 2 s from the RK4 / 1 ms run with the same gain [m]")
    print(f"   {'controller dt':<15}" + "".join(f"{n:>16}" for n in NAMES))
    for dt in (0.01, 0.02, 0.04, 0.08):
        K, _ = lq.hover_gain(p, Q, R, dt)
        truth = sim.simulate(st.make_state(), sim.HoverController(K, x_ref, p), p, 3.0, 1e-3, dt)
        cells = []
        for n in NAMES:
            r = sim.simulate(st.make_state(), sim.HoverController(K, x_ref, p), p, 3.0, dt, dt, integrator=n)
            d = np.linalg.norm(r.x[int(round(2 / dt)), :3] - truth.x[2000, :3])
            cells.append("diverged" if not np.isfinite(d) or d > 1 else f"{d:.1e}")
        print(f"   {dt * 1000:>5.0f} ms       " + "".join(f"{c:>16}" for c in cells))


def planning(p):
    from crazyflie_lqr import ilqr

    print("\n3) iLQR planning of the 3-D step (6 s, 20 ms), plan executed on the RK4 plant")
    x0, tgt = st.make_state(r=[0, 0, 0.5]), np.array([1.5, 1.0, 1.5])
    x_goal = st.make_state(r=tgt)
    dt, N = 0.02, 300
    Q, R = lq.bryson_weights(0.2, 0.2, 0.5, 4.0, 0.05)
    F_h = p.hover_thrust * np.ones(4)
    print(f"   {'planner':<28}{'iters':>6}{'s/iter':>9}{'final err':>11}{'settle':>8}")
    with np.errstate(all="ignore"):
        for integ, sub in [("rk4", 2), ("semi_implicit", 2), ("semi_implicit", 4), ("euler", 2)]:
            prob = ilqr.Problem(np.tile(x_goal, (N + 1, 1)), np.tile(F_h, (N, 1)), Q, R, 100 * Q, dt,
                                [ilqr.TiltLimit(30.0, 1e5)], 0.0, p.thrust_max, sub, integ)
            t0 = time.time()
            sol = ilqr.solve(prob, x0, np.tile(F_h, (N, 1)), p, iters=60)
            tt = time.time() - t0
            r = sim.simulate(x0, ilqr.ILQRPolicy(sol, p), p, 6.0, 1e-3, dt)
            e = np.linalg.norm(r.x[:, :3] - tgt, axis=1)
            idx = np.where(e > 0.02)[0]
            print(f"   {f'{integ}, {sub} substeps':<28}{sol.iters:>6}{tt / sol.iters:>8.2f}s{1000 * e[-1]:>9.1f}mm{r.t[idx[-1]]:>7.2f}s")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--planning", action="store_true")
    a = ap.parse_args(argv)
    p = CrazyflieParams()
    accuracy(p)
    closed_loop(p)
    if a.planning:
        planning(p)


if __name__ == "__main__":
    main()
