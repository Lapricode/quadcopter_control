"""Trajectory / velocity-profile tracking with TVLQR.

    python -m examples.track --traj figure8            # circle | figure8 | helix | velocity
    python -m examples.track --traj velocity --vel-priority
    python -m examples.track --traj circle --mass-error 0.1 --noise --animate
"""

import argparse
from pathlib import Path

import numpy as np

from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import trajectory as tj
from crazyflie_lqr.params import CrazyflieParams

OUT = Path(__file__).resolve().parents[1] / "results"

# a "stop-and-go" square-ish tour defined only by its world-frame velocity knots (1 s apart)
VEL_KNOTS_T = np.arange(13.0)
VEL_KNOTS_V = np.array(
    [[0, 0, 0], [.5, 0, 0], [.5, 0, 0], [.25, .25, 0], [0, .5, .1], [0, .5, .1], [-.25, .25, 0],
     [-.5, 0, -.1], [-.5, 0, -.1], [-.25, -.25, 0], [0, -.5, 0], [0, -.5, 0], [0, 0, 0]]
)


def build_reference(kind, p, dt):
    if kind == "circle":
        return tj.path_trajectory("circle", p, T=14.0, dt=dt, size=0.8, rate=1.2, ramp=2.0)
    if kind == "figure8":
        return tj.path_trajectory("figure8", p, T=14.0, dt=dt, size=0.9, rate=1.1, ramp=2.5,
                                  climb=0.15, yaw_mode="tangent")
    if kind == "helix":
        return tj.path_trajectory("helix", p, T=14.0, dt=dt, size=0.7, rate=1.3, ramp=2.0, climb=0.08,
                                  center=(0, 0, 0.5))
    if kind == "velocity":
        return tj.velocity_profile_trajectory(VEL_KNOTS_T, VEL_KNOTS_V, p, r0=(0, 0, 1.0), dt=dt)
    raise ValueError(kind)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="figure8", choices=["circle", "figure8", "helix", "velocity"])
    ap.add_argument("--vel-priority", action="store_true", help="weight velocity error more than position error")
    ap.add_argument("--offset", type=float, nargs=3, default=[0.15, -0.10, 0.10], help="initial position error [m]")
    ap.add_argument("--mass-error", type=float, default=0.0, help="plant mass = (1+e) * model mass")
    ap.add_argument("--noise", action="store_true", help="noisy state measurements")
    ap.add_argument("--dt-ctrl", type=float, default=4e-3)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--gif", type=str, default=None)
    a = ap.parse_args(argv)

    model = CrazyflieParams()
    plant = model.with_(mass=model.mass * (1 + a.mass_error))
    ref = build_reference(a.traj, model, a.dt_ctrl)
    if a.vel_priority:
        Q, R = lq.bryson_weights(pos=1.0, att=0.2, vel=0.1, rate=4.0, thrust=0.05)
    else:
        Q, R = lq.bryson_weights(pos=0.1, att=0.2, vel=0.5, rate=4.0, thrust=0.05)
    K = lq.tvlqr_along_reference(ref, model, Q, R)
    x0 = ref.x[0].copy()
    x0[:3] += a.offset
    noise = (0.002, 0.005, 0.01, 0.05) if a.noise else None
    res = sim.simulate(x0, sim.TVLQRTracker(ref, K), plant, ref.t[-1], 1e-3, a.dt_ctrl, meas_noise=noise)

    e_pos, e_att, e_vel, _ = sim.tracking_errors(res, ref)
    half = len(res.t) // 4  # skip the initial transient caused by the offset
    print(f"[{a.traj}] steady-state (t>{res.t[half]:.1f}s)  pos RMS {1000*np.sqrt((e_pos[half:]**2).sum(1).mean()):.2f} mm, "
          f"max {1000*np.linalg.norm(e_pos[half:], axis=1).max():.2f} mm, "
          f"vel RMS {np.sqrt((e_vel[half:]**2).sum(1).mean()):.4f} m/s, att RMS {np.degrees(np.sqrt((e_att[half:]**2).mean())):.3f} deg")
    print("saturated fraction: %.4f" % (res.F >= model.thrust_max - 1e-9).mean())
    tag = a.traj + ("_velprio" if a.vel_priority else "")
    res.save_npz(OUT / f"{tag}.npz")
    from crazyflie_lqr.visualize import plot_summary, quadrotor_visualize

    plot_summary(res, ref, OUT / f"{tag}.png", title=f"TVLQR tracking: {tag}")
    if a.gif:
        quadrotor_visualize(res, ref, states_step=80, save_path=a.gif, fps=12)
    if a.animate:
        quadrotor_visualize(res, ref, states_step=20)
    return res, ref


if __name__ == "__main__":
    main()
