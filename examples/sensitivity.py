"""Which parameter errors hurt?  Figure-8 TVLQR tracking with the *plant* perturbed, controller on nominal model.

    python -m examples.sensitivity            # writes results/sensitivity.png + prints a table
"""

import argparse
from pathlib import Path

import numpy as np

from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import trajectory as tj
from crazyflie_lqr.params import CrazyflieParams

OUT = Path(__file__).resolve().parents[1] / "results"
DT = 4e-3


def perturbations(model, frac):
    I = model.inertia
    cases = {
        "nominal": (model, None),
        f"mass +{frac:.0%}": (model.with_(mass=model.mass * (1 + frac)), None),
        f"mass -{frac:.0%}": (model.with_(mass=model.mass * (1 - frac)), None),
        f"Kf +{frac:.0%}": (model.with_(Kf=model.Kf * (1 + frac)), None),
        f"Kf -{frac:.0%}": (model.with_(Kf=model.Kf * (1 - frac)), None),
        f"Kt +{frac:.0%}": (model.with_(Kt=model.Kt * (1 + frac)), None),
        f"Kt -{frac:.0%}": (model.with_(Kt=model.Kt * (1 - frac)), None),
        f"Ixx,Iyy +{frac:.0%}": (model.with_(inertia=_scale_xy(I, 1 + frac)), None),
        f"Ixx,Iyy -{frac:.0%}": (model.with_(inertia=_scale_xy(I, 1 - frac)), None),
        f"Izz +{frac:.0%}": (model.with_(inertia=_scale_z(I, 1 + frac)), None),
        f"Izz -{frac:.0%}": (model.with_(inertia=_scale_z(I, 1 - frac)), None),
        f"arm +{frac:.0%}": (model.with_(arm=model.arm * (1 + frac)), None),
        f"arm -{frac:.0%}": (model.with_(arm=model.arm * (1 - frac)), None),
        "motor lag 10 ms": (model, 0.010),
        "motor lag 20 ms": (model, 0.020),
        "motor lag 40 ms": (model, 0.040),
    }
    return cases


def _scale_xy(I, s):
    S = np.diag([np.sqrt(s), np.sqrt(s), 1.0])
    return S @ I @ S


def _scale_z(I, s):
    S = np.diag([1.0, 1.0, np.sqrt(s)])
    return S @ I @ S


def run_case(plant, tau, model, ref, policy):
    x0 = ref.x[0].copy()
    x0[:3] += [0.15, -0.10, 0.10]
    res = sim.simulate(x0, policy, plant, ref.t[-1], 1e-3, DT, model=model, motor_tau=tau)
    if not np.isfinite(res.x).all():
        return np.inf
    e, _, _, _ = sim.tracking_errors(res, ref)
    n = len(e)
    r = 1000 * np.sqrt((e[int(0.6 * n):] ** 2).sum(1).mean())
    return r if r < 500 else np.inf


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2)
    a = ap.parse_args(argv)
    model = CrazyflieParams()
    ref = tj.path_trajectory("figure8", model, T=14.0, dt=DT, size=0.9, rate=1.1, ramp=2.5, climb=0.15, yaw_mode="tangent")
    Q, R = lq.bryson_weights(0.1, 0.2, 0.5, 4.0, 0.05)
    K0 = lq.tvlqr_along_reference(ref, model, Q, R)
    Ki = lq.tvlqr_along_reference_integral(ref, model, Q, R, int_tol=0.05)
    rows = []
    for name, (plant, tau) in perturbations(model, a.frac).items():
        r0 = run_case(plant, tau, model, ref, sim.TVLQRTracker(ref, K0))
        r1 = run_case(plant, tau, model, ref, sim.IntegralTVLQRTracker(ref, Ki, model))
        rows.append((name, r0, r1))
    print(f"{'perturbation of the plant':<20}{'TVLQR [mm]':>12}{'+ integral [mm]':>18}")
    for n, r0, r1 in rows:
        f = lambda v: "UNSTABLE" if not np.isfinite(v) else f"{v:.1f}"
        print(f"{n:<20}{f(r0):>12}{f(r1):>18}")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(rows))
    cap = 100.0
    for off, idx, lab, c in [(-0.2, 1, "TVLQR", "tab:blue"), (0.2, 2, "TVLQR + integral", "tab:green")]:
        v = np.array([min(r[idx], cap) for r in rows])
        ax.bar(x + off, v, 0.4, label=lab, color=c)
        for xi, r in zip(x, rows):
            if not np.isfinite(r[idx]):
                ax.text(xi + off, cap, "X", ha="center", va="bottom", color="r", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=60, ha="right")
    ax.set_ylabel("steady-state position RMS [mm] (clipped at 100)"); ax.set_yscale("log"); ax.legend()
    ax.set_title("Sensitivity of figure-8 tracking to plant/model mismatch (X = unstable)")
    fig.tight_layout(); fig.savefig(OUT / "sensitivity.png", dpi=110)


if __name__ == "__main__":
    main()
