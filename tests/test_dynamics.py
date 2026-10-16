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


# ------------------------------------------------------------ semi-implicit Euler
def test_semi_implicit_jacobians_match_finite_differences():
    for dt in (1e-3, 2e-2):
        A, B = dy.discretize_semi_implicit(X, F, p, dt)
        Af, Bf = dy.finite_difference_jacobians(lambda x, f, d: dy.step_semi_implicit(x, f, p, d), X, F, dt)
        assert np.abs(A - Af).max() < 1e-7 and np.abs(B - Bf).max() < 1e-7


def test_semi_implicit_matches_definition():
    """One step, written out by hand: velocities first, then poses with the *new* velocities."""
    dt = 5e-3
    R = st.rotation(X)
    v, w = X[7:10], X[10:13]
    wrench = p.mixer @ F
    w1 = w + dt * p.inertia_inv @ (wrench[1:] - np.cross(w, p.inertia @ w))
    v1 = v + dt * ((R.T @ np.array([0, 0, -p.mass * p.g]) + np.array([0, 0, wrench[0]])) / p.mass - np.cross(w1, v))
    q1 = s3.plus_right(X[3:7], w1 * dt)
    r1 = X[0:3] + dt * s3.to_matrix(q1) @ v1
    assert np.allclose(dy.step_semi_implicit(X, F, p, dt), np.concatenate([r1, q1, v1, w1]), atol=1e-14)


def test_integrator_orders_and_hover_equilibrium():
    x0 = st.make_state(r=[0, 0, 1], q=s3.from_rpy(0.3, -0.2, 0.5), v=[0.8, -0.4, 0.3], omega=[2.0, -1.5, 1.0])
    Fm = p.hover_thrust * np.ones(4) + np.array([0.006, -0.004, 0.002, -0.003])

    def run(step, dt):
        x = x0.copy()
        for _ in range(int(round(0.5 / dt))):
            x = step(x, Fm, p, dt)
        return x

    ref = run(dy.step_rk4, 1e-4)
    order = lambda step: np.log(np.linalg.norm(st.minus(run(step, 4e-3), ref)[:3]) / np.linalg.norm(st.minus(run(step, 1e-3), ref)[:3])) / np.log(4)
    assert 0.8 < order(dy.step_semi_implicit) < 1.3  # first order
    assert 0.8 < order(dy.step_euler) < 1.3
    assert order(dy.step_rk4) > 1.7
    xh = st.make_state(yaw=0.3)
    for name, step in dy.INTEGRATORS.items():
        assert np.allclose(step(xh, p.hover_thrust * np.ones(4), p, 1e-2), xh, atol=1e-12), name
