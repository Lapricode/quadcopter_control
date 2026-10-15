"""Regulate the Crazyflie to a fixed position + yaw (replaces speeds_LQR_crazyflie.py's __main__).

    python -m examples.hover [--animate] [--yaw 90]
"""

import argparse
from pathlib import Path

import numpy as np

from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr.io import export_c_gain
from crazyflie_lqr.params import CrazyflieParams

OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    ap.add_argument("--yaw", type=float, default=90.0, help="target yaw [deg]")
    ap.add_argument("--T", type=float, default=8.0)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--export", action="store_true", help="write results/Kinf_speeds.txt for the firmware")
    a = ap.parse_args()

    p = CrazyflieParams()
    dt = 4e-3
    Q, R = lq.bryson_weights(pos=0.3, att=0.1, vel=0.5, rate=4.0, thrust=0.05)  # gentle enough for a 1.7 m step
    yaw = np.radians(a.yaw)
    K, _ = lq.hover_gain(p, Q, R, dt, yaw=yaw)  # exact for any yaw, no look-up table
    x_ref = st.make_state(r=a.target, yaw=yaw)
    res = sim.simulate(st.make_state(), sim.HoverController(K, x_ref, p), p, a.T, 1e-3, dt)
    print("final error:", np.round(st.minus(res.x[-1], x_ref), 6))
    print("saturated fraction: %.3f" % (res.F >= p.thrust_max - 1e-9).mean())
    res.save_npz(OUT / "hover.npz")
    if a.export:
        export_c_gain(lq.thrust_gain_to_speed_gain(K, p), OUT / "Kinf_speeds.txt")
    if a.animate:
        from crazyflie_lqr.visualize import quadrotor_visualize

        quadrotor_visualize(res, states_step=20)


if __name__ == "__main__":
    main()
