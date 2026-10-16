import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from crazyflie_lqr import firmware as fw
from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr import trajectory as tj
from crazyflie_lqr.params import CrazyflieParams

p = CrazyflieParams()
FW = Path(__file__).resolve().parents[1] / "firmware"
Q, R = lq.bryson_weights(0.3, 0.1, 0.5, 4.0, 0.05)


# ------------------------------------------------------------------ integral action
def test_integral_hover_removes_mass_offset():
    dt, yaw = 4e-3, np.radians(90)
    x_ref = st.make_state(r=[0.5, 0.5, 0.5], yaw=yaw)
    plant = p.with_(mass=p.mass * 1.15)
    K = lq.hover_gain_integral(p, Q, R, dt, yaw=yaw, int_tol=0.3)
    res = sim.simulate(st.make_state(), sim.IntegralHoverController(K, x_ref, p, dt), plant, 20.0, 1e-3, dt)
    assert np.linalg.norm(res.x[-1, :3] - [0.5, 0.5, 0.5]) < 2e-3
    K0, _ = lq.hover_gain(p, Q, R, dt, yaw=yaw)  # same plant without integrator: clearly offset
    res0 = sim.simulate(st.make_state(), sim.HoverController(K0, x_ref, p), plant, 20.0, 1e-3, dt)
    assert np.linalg.norm(res0.x[-1, :3] - [0.5, 0.5, 0.5]) > 0.05


def test_integral_hover_gain_yaw_rotation_is_exact():
    from crazyflie_lqr import dynamics as dy

    dt = 4e-3
    for yaw in [0.6, -2.0]:
        x = st.make_state(yaw=yaw)
        A, B = dy.discretize_zoh(x, p.hover_thrust * np.ones(4), p, dt)
        Aa, Ba = lq.augment_integral(A, B, dt)
        Qa = np.zeros((15, 15)); Qa[:12, :12] = Q; Qa[12:, 12:] = lq.integral_weight(0.3) * np.eye(3)
        Kd, _ = lq.dare_gain(Aa, Ba, Qa, R)
        assert np.allclose(Kd, lq.hover_gain_integral(p, Q, R, dt, yaw=yaw, int_tol=0.3), atol=1e-6)


def test_integral_tracking_reduces_error_under_mass_mismatch():
    ref = tj.path_trajectory("figure8", p, T=12.0, size=0.8, yaw_mode="tangent")
    Qt, Rt = lq.bryson_weights(0.1, 0.2, 0.5, 4.0, 0.05)
    plant = p.with_(mass=p.mass * 1.1)
    x0 = ref.x[0].copy()
    r0 = sim.simulate(x0, sim.TVLQRTracker(ref, lq.tvlqr_along_reference(ref, p, Qt, Rt)), plant, 12.0, 1e-3, 4e-3)
    K = lq.tvlqr_along_reference_integral(ref, p, Qt, Rt, int_tol=0.05)
    r1 = sim.simulate(x0, sim.IntegralTVLQRTracker(ref, K, p), plant, 12.0, 1e-3, 4e-3)
    e0 = np.linalg.norm(sim.tracking_errors(r0, ref)[0][-3000:], axis=1).mean()
    e1 = np.linalg.norm(sim.tracking_errors(r1, ref)[0][-3000:], axis=1).mean()
    assert e1 < 0.2 * e0


# ------------------------------------------------------------------ firmware audit
CPP, ORIG = FW / "LQR_controller_cpp.cpp", FW / "LQR_controller_cpp.orig.cpp"


def test_cpp_overflow_is_fixed_in_patched_file_only():
    assert "float state_error[3];" in ORIG.read_text()
    fixed = CPP.read_text()
    assert "float state_error[3];" not in fixed and "float state_error[12];" in fixed


def test_parse_firmware_tables_and_hover_adjust():
    tabs = fw.parse_c_gain_tables(ORIG)
    assert set(tabs) == {"Kinf_default", "Kinf_alt#1", "Kinf_alt#2", "Kinf_alt#3"}
    assert all(K.shape == (4, 12) for K in tabs.values())
    assert fw.parse_hover_adjust(ORIG) == 300.0


def test_compiled_default_gain_is_unstable_at_100hz_but_stable_at_1khz():
    K = fw.parse_c_gain_tables(ORIG)["Kinf_default"]
    assert fw.closed_loop_radius(K, p, 1e-3) < 1.0
    assert fw.closed_loop_radius(K, p, 1e-2) > 1.0


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_c_yaw_rotation_matches_python(tmp_path):
    exe = tmp_path / "rotk"
    subprocess.run(["gcc", "-O1", "-o", str(exe), str(FW / "test_gain_rotation.c"), "-lm"], check=True)
    K = fw.parse_c_gain_tables(ORIG)["Kinf_default"]
    for yaw in [0.0, 0.7, -2.0]:
        inp = f"{yaw}\n" + "\n".join(f"{v:.9g}" for v in K.ravel())
        out = subprocess.run([str(exe)], input=inp, capture_output=True, text=True, check=True).stdout.split()
        assert np.allclose(np.array(out, float).reshape(4, 12), lq.rotate_gain_to_yaw(K, yaw), rtol=1e-5, atol=1e-3)
