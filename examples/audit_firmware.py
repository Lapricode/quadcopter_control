"""Audit the gains compiled into the firmware C++ file against the Crazyflie model.

    python -m examples.audit_firmware [path/to/LQR_controller_cpp.cpp]
"""

import sys
from pathlib import Path

import numpy as np

from crazyflie_lqr import firmware as fw
from crazyflie_lqr.params import CrazyflieParams

FW = Path(__file__).resolve().parents[1] / "firmware"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cpp = Path(argv[0]) if argv else FW / "LQR_controller_cpp.orig.cpp"
    p = CrazyflieParams()
    tabs = fw.parse_c_gain_tables(cpp)
    for f in ("Kinf_speeds.txt", "Kinf_thrusts.txt"):
        if (FW / f).exists():
            tabs[f] = next(iter(fw.parse_c_gain_tables(FW / f).values()))
    adj = fw.parse_hover_adjust(cpp)
    u_fw = p.hover_speed + adj
    peff = fw.effective_plant(p, u_fw)
    print(f"file: {cpp.name}")
    print(f"hover speed: model {p.hover_speed:.0f} rad/s + hover_adjust {adj:.0f} = {u_fw:.0f} rad/s")
    print(f"  -> firmware commands {(u_fw / p.hover_speed) ** 2:.2f}x the model's hover thrust; implied Kf/m is {peff.Kf / p.Kf:.2f}x the model's")
    print(f"  -> thrust headroom above hover: {(p.speed_max / u_fw) ** 2:.2f}x (model: {(p.speed_max / p.hover_speed) ** 2:.2f}x)")
    rates = [1000, 500, 250, 200, 150, 100]
    print("\nspectral radius of the closed loop (<1 = stable), model plant, yaw = 0, ZOH at the given rate:")
    print(f"{'table':<18}{'K_pos_z':>9}" + "".join(f"{r:>8d}Hz" for r in rates) + f"{'100Hz yaw90*':>14}")
    for n, K in tabs.items():
        row = [fw.closed_loop_radius(K, p, 1.0 / r) for r in rates]
        y90 = fw.closed_loop_radius(K, p, 1e-2, yaw=np.pi / 2)
        print(f"{n:<18}{K[0, 2]:9.1f}" + "".join(f"{v:10.3f}" for v in row) + f"{y90:14.4f}")
    print("* yaw 90 deg with the gain NOT rotated (what the unpatched firmware does)")


if __name__ == "__main__":
    main()
