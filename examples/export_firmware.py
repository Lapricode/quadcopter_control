"""Design a firmware gain that is consistent with the *real* loop rate and hover speed, check it, export it.

    python -m examples.export_firmware --rate 100 --hover-adjust 300 --cpp firmware/LQR_controller_cpp.orig.cpp

The firmware computes  u = (u_hover_model + hover_adjust) - K e  at ``--rate`` Hz (ZOH).  The plant
implied by that hover speed (same mass, thrust coefficient rescaled) is used for the design, so the
gain has the right effectiveness and the right discretisation.
"""

import argparse
from pathlib import Path

import numpy as np

from crazyflie_lqr import dynamics as dy
from crazyflie_lqr import firmware as fw
from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr.io import export_c_gain
from crazyflie_lqr.params import CrazyflieParams

OUT = Path(__file__).resolve().parents[1] / "firmware"


def step_metrics(K_u, u_hover, plant, rate, target=(0.5, 0.5, 0.5), T=6.0):
    x_ref = st.make_state(r=target)
    res = sim.simulate(st.make_state(), sim.SpeedGainController(K_u, u_hover, x_ref, plant), plant, T, 1e-3, 1.0 / rate)
    e = np.linalg.norm(res.x[:, :3] - np.array(target), axis=1)
    idx = np.where(e > 0.02)[0]
    ok = np.isfinite(res.x).all()
    return dict(
        final=float(e[-1]) if ok else np.nan,
        settle=float(res.t[idx[-1]]) if len(idx) else 0.0,
        tilt=float(np.degrees(np.arccos(np.clip(min(st.rotation(x)[2, 2] for x in res.x[::10]), -1, 1)))),
        rate_max=float(np.linalg.norm(res.x[:, 10:13], axis=1).max()),
        sat=float(((res.F >= plant.thrust_max - 1e-9) | (res.F <= 1e-12)).mean()),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=100.0, help="rate at which the firmware evaluates the law [Hz]")
    ap.add_argument("--hover-adjust", type=float, default=None, help="rad/s added to the model hover speed (default: parsed from --cpp)")
    ap.add_argument("--cpp", default=str(OUT / "LQR_controller_cpp.orig.cpp"))
    ap.add_argument("--pos", type=float, default=0.3)
    ap.add_argument("--att", type=float, default=0.1)
    ap.add_argument("--vel", type=float, default=0.5)
    ap.add_argument("--rate-tol", type=float, default=4.0, help="body-rate tolerance [rad/s]")
    ap.add_argument("--thrust-tol", type=float, default=0.05)
    a = ap.parse_args(argv)

    model = CrazyflieParams()
    adj = fw.parse_hover_adjust(a.cpp) if a.hover_adjust is None else a.hover_adjust
    u_hover = model.hover_speed + adj
    plant = fw.effective_plant(model, u_hover)
    print(f"hover speed: model {model.hover_speed:.0f} rad/s, firmware {u_hover:.0f} rad/s "
          f"-> implied Kf/Kf_model = {plant.Kf / model.Kf:.3f}, thrust headroom {(plant.speed_max / u_hover) ** 2:.2f}x")

    dt = 1.0 / a.rate
    Q, R = lq.bryson_weights(a.pos, a.att, a.vel, a.rate_tol, a.thrust_tol)
    K_F, _ = lq.hover_gain(plant, Q, R, dt, yaw=0.0)
    K_u = K_F / (2.0 * plant.Kf * u_hover)  # thrust gain -> speed gain at the firmware hover speed
    rho = fw.closed_loop_radius(K_u, plant, dt, 0.0, u_hover)
    print(f"generated gain: spectral radius at {a.rate:.0f} Hz = {rho:.4f}")

    print("\n0.5 m diagonal step, plant = firmware-implied plant, controller at the firmware rate")
    print(f"{'gain':<22}{'rho':>8}{'final err':>11}{'settle(2cm)':>13}{'tilt':>8}{'max rate':>10}{'sat':>7}")
    cands = {"generated": K_u}
    for n, K in fw.parse_c_gain_tables(a.cpp).items():
        cands[n] = K
    for n, K in cands.items():
        m = step_metrics(K, u_hover, plant, a.rate)
        r = fw.closed_loop_radius(K, plant, dt, 0.0, u_hover)
        print(f"{n:<22}{r:8.3f}{m['final']:11.4f}{m['settle']:12.2f}s{m['tilt']:7.1f}°{m['rate_max']:9.1f}{m['sat']:7.2f}")

    export_c_gain(K_u, OUT / "Kinf_generated.txt")
    print(f"\nwrote {OUT / 'Kinf_generated.txt'}  (yaw-0 gain; the patched firmware rotates it with the setpoint yaw)")


if __name__ == "__main__":
    main()
