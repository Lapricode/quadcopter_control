import numpy as np

from crazyflie_lqr import dynamics as dy
from crazyflie_lqr import state as st
from crazyflie_lqr.lie import s3
from crazyflie_lqr.params import CrazyflieParams

p = CrazyflieParams()
X = st.make_state(r=[0.1, 0.2, 0.3], q=s3.from_rpy(0.2, -0.3, 0.7), v=[0.3, -0.2, 0.1], omega=[1.0, -0.5, 0.8])
F = p.hover_thrust * np.ones(4) + np.array([0.003, -0.002, 0.001, 0.0005])


def test_hover_is_an_equilibrium():
    x = st.make_state(yaw=0.4)
    assert np.allclose(dy.dynamics(x, p.hover_thrust * np.ones(4), p), 0.0, atol=1e-12)


def test_mixer_roundtrip_and_hover_speed():
    assert np.allclose(p.mixer_inv @ p.mixer, np.eye(4))
    assert abs(p.hover_speed - 1896.6) < 1.0
    assert p.thrust_max * 4 > p.mass * p.g  # can lift off


def test_euler_jacobians_match_finite_differences():
    dt = 1e-3
    A, B = dy.discretize_euler(X, F, p, dt)
    Af, Bf = dy.finite_difference_jacobians(lambda x, f, d: dy.step_euler(x, f, p, d), X, F, dt)
    assert np.abs(A - Af).max() < 1e-8
    assert np.abs(B - Bf).max() < 1e-8


def test_zoh_matches_rk4_jacobians():
    dt = 1e-3
    A, B = dy.discretize_zoh(X, F, p, dt)
    Af, Bf = dy.finite_difference_jacobians(lambda x, f, d: dy.step_rk4(x, f, p, d), X, F, dt)
    assert np.abs(A - Af).max() < 1e-4
    assert np.abs(B - Bf).max() < 1e-4


def test_rk4_keeps_quaternion_normalised():
    x = X.copy()
    for _ in range(2000):
        x = dy.step_rk4(x, F, p, 1e-3)
    assert abs(np.linalg.norm(x[3:7]) - 1) < 1e-12
