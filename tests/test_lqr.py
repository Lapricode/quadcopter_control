import numpy as np

from crazyflie_lqr import dynamics as dy
from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr.params import CrazyflieParams

p = CrazyflieParams()
Q, R = lq.bryson_weights(0.3, 0.1, 0.5, 4.0, 0.05)
DT = 4e-3


def test_yaw_rotated_gain_equals_gain_solved_at_that_yaw():
    for yaw in [0.0, 0.7, 2.0, -1.3]:
        x = st.make_state(yaw=yaw)
        A, B = dy.discretize_zoh(x, p.hover_thrust * np.ones(4), p, DT)
        K_direct, _ = lq.dare_gain(A, B, Q, R)
        K_rot, _ = lq.hover_gain(p, Q, R, DT, yaw=yaw)
        assert np.allclose(K_direct, K_rot, atol=1e-7)


def test_closed_loop_is_stable():
    A, B = dy.discretize_zoh(st.make_state(), p.hover_thrust * np.ones(4), p, DT)
    K, _ = lq.dare_gain(A, B, Q, R)
    assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1.0


def test_tvlqr_converges_to_dare_far_from_horizon_end():
    x, F = st.make_state(), p.hover_thrust * np.ones(4)
    A, B = dy.discretize_zoh(x, F, p, DT)
    K_inf, P_inf = lq.dare_gain(A, B, Q, R)
    K_list, _ = lq.tvlqr([A] * 3000, [B] * 3000, Q, R, 10 * Q)
    assert np.allclose(K_list[0], K_inf, atol=1e-6)


def test_hover_regulation_to_offset_and_yaw():
    yaw = np.radians(90)
    K, _ = lq.hover_gain(p, Q, R, DT, yaw=yaw)
    x_ref = st.make_state(r=[1, 1, 1], yaw=yaw)
    res = sim.simulate(st.make_state(), sim.HoverController(K, x_ref, p), p, 8.0, 1e-3, DT)
    assert np.linalg.norm(st.minus(res.x[-1], x_ref)) < 1e-3


def test_speed_gain_conversion_is_consistent_with_linearisation():
    K_F, _ = lq.hover_gain(p, Q, R, DT)
    K_u = lq.thrust_gain_to_speed_gain(K_F, p)
    e = np.random.default_rng(0).normal(size=12) * 1e-2
    du = -K_u @ e
    dF = -K_F @ e
    u = p.hover_speed + du
    assert np.allclose(p.Kf * u**2 - p.hover_thrust, dF, rtol=1e-1, atol=1e-6)


def test_semi_implicit_design_and_plant_are_consistent():
    """Gain designed on the semi-implicit model, plant integrated semi-implicitly: stable and converging."""
    dt = 1e-2
    K, _ = lq.hover_gain(p, Q, R, dt, discretization="semi_implicit")
    A, B = dy.discretize_semi_implicit(st.make_state(), p.hover_thrust * np.ones(4), p, dt)
    assert np.abs(np.linalg.eigvals(A - B @ K)).max() < 1.0
    x_ref = st.make_state(r=[0.5, 0.5, 0.5])
    res = sim.simulate(st.make_state(), sim.HoverController(K, x_ref, p), p, 8.0, dt, dt, integrator="semi_implicit")
    assert np.linalg.norm(st.minus(res.x[-1], x_ref)) < 1e-3
