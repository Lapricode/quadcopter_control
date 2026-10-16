import numpy as np

from crazyflie_lqr import dynamics as dy
from crazyflie_lqr import sysid
from crazyflie_lqr import state as st
from crazyflie_lqr.params import CrazyflieParams, load_params, save_params


def test_thrust_fit_recovers_kf():
    rng = np.random.default_rng(0)
    u = np.linspace(800, 2400, 40)
    T = 1.9e-8 * u**2 + rng.normal(0, 2e-4, u.size)
    assert abs(sysid.fit_thrust_coefficient(u, T) - 1.9e-8) / 1.9e-8 < 0.03


def test_hover_speed_to_kf_roundtrip():
    p = CrazyflieParams()
    assert abs(sysid.kf_from_hover_speed(p.hover_speed, p.mass) - p.Kf) / p.Kf < 1e-12


def test_bifilar_formula():
    # I = 0.01 kg m^2 -> T = 2 pi sqrt(I L / (m g a^2))
    m, a, L, I = 0.5, 0.1, 1.0, 0.01
    T = 2 * np.pi * np.sqrt(I * L / (m * 9.81 * a**2))
    assert abs(sysid.inertia_bifilar(T, m, a, L) - I) < 1e-12


def test_identify_from_synthetic_log():
    """Excite a 'true' vehicle that differs from the model; recover its coefficients from gyro/accel/speeds."""
    model = CrazyflieParams()
    true = model.with_(mass=0.0345, Kf=1.7e-8, Kt=1.5e-10, inertia=model.inertia * 1.15)
    rng = np.random.default_rng(1)
    dt, n = 1e-3, 6000
    u_h = true.hover_speed
    x = st.make_state()
    t = np.arange(n) * dt
    gyro, acc, speeds = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 4))
    u = u_h * np.ones(4)
    for k in range(n):
        if k % 20 == 0:  # random binary excitation of the four rotors, 20 ms hold
            u = u_h + 60.0 * rng.choice([-1, 1], size=4)
        F = true.Kf * u**2
        f = dy.dynamics(x, F, true)
        # accelerometer = specific force in the body frame = thrust / m
        acc[k] = np.array([0, 0, (true.mixer @ F)[0] / true.mass])
        gyro[k] = x[10:13]
        speeds[k] = u
        x = dy.step_rk4(x, F, true, dt)
        x[0:3] = 0.0  # keep it in place: only attitude dynamics matter here
        x[7:10] = 0.0
    gyro_n = gyro + rng.normal(0, 0.01, gyro.shape)
    acc_n = acc + rng.normal(0, 0.05, acc.shape)
    res = sysid.identify_from_log(t, gyro_n, acc_n, speeds, model, mass=true.mass)
    assert abs(res.params.Kf - true.Kf) / true.Kf < 0.02
    assert abs(res.alpha[0] - true.arm * true.Kf / true.inertia[0, 0]) / res.alpha[0] < 0.1
    assert abs(res.alpha[2] - true.Kt / true.inertia[2, 2]) / res.alpha[2] < 0.1
    assert abs(res.params.inertia[0, 0] - true.inertia[0, 0]) / true.inertia[0, 0] < 0.1


def test_params_json_roundtrip(tmp_path):
    p = CrazyflieParams(mass=0.031)
    save_params(p, tmp_path / "p.json")
    q = load_params(tmp_path / "p.json")
    assert q.mass == 0.031 and np.allclose(q.inertia, p.inertia)


def test_fit_first_order_lag():
    rng = np.random.default_rng(0)
    t = np.arange(0, 0.5, 1e-3)
    tau, t0 = 0.03, 0.1
    y = 9.81 + 0.6 * (1 - np.exp(-np.clip(t - t0, 0, None) / tau)) * (t >= t0) + rng.normal(0, 0.02, t.size)
    assert abs(sysid.fit_first_order_lag(t, y, t0) - tau) / tau < 0.1
