import numpy as np

from crazyflie_lqr import dynamics as dy
from crazyflie_lqr import ilqr
from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr import trajectory as tj
from crazyflie_lqr.lie import s3
from crazyflie_lqr.params import CrazyflieParams

p = CrazyflieParams()


def test_penalty_gradients_match_finite_differences():
    x = st.make_state(r=[0.1, 0.2, 0.9], q=s3.from_rpy(0.4, -0.3, 0.5))
    for pen in [ilqr.TiltLimit(10), ilqr.SphereObstacle(center=(0.1, 0.2, 1.0)), ilqr.FloorLimit(1.0)]:
        v, g = pen.g(x)
        gn = np.zeros(12)
        for i in range(12):
            d = np.zeros(12)
            d[i] = 1e-6
            gn[i] = (pen.g(st.plus(x, d))[0] - pen.g(st.plus(x, -d))[0]) / 2e-6
        assert v > 0 and np.abs(g - gn).max() < 1e-7


def test_box_qp_respects_bounds_and_is_optimal():
    rng = np.random.default_rng(0)
    for _ in range(50):
        M = rng.normal(size=(4, 4))
        Quu = M @ M.T + 0.1 * np.eye(4)
        Qu = rng.normal(size=4) * 3
        lo, hi = -np.abs(rng.normal(size=4)), np.abs(rng.normal(size=4))
        d, _ = ilqr.box_qp(Quu, Qu, lo, hi)
        assert np.all(d >= lo - 1e-12) and np.all(d <= hi + 1e-12)
        f = lambda z: 0.5 * z @ Quu @ z + Qu @ z
        for _ in range(30):  # no random feasible point does better
            z = rng.uniform(lo, hi)
            assert f(d) <= f(z) + 1e-9


def test_ilqr_recovers_lqr_gain_near_hover():
    dt, N = 0.01, 200
    Q, R = lq.bryson_weights(0.3, 0.1, 0.5, 4.0, 0.05)
    x_h, F_h = st.make_state(r=[0, 0, 1.0]), p.hover_thrust * np.ones(4)
    A, B = dy.discretize_zoh(x_h, F_h, p, dt)
    Kinf, P = lq.dare_gain(A, B, Q, R)
    prob = ilqr.Problem(np.tile(x_h, (N + 1, 1)), np.tile(F_h, (N, 1)), Q, R, P, dt, [], 0.0, p.thrust_max, 2)
    x0 = st.plus(x_h, np.array([0.05, -0.03, 0.04, 0.02, 0.01, -0.03, 0, 0, 0, 0, 0, 0]))
    sol = ilqr.solve(prob, x0, np.tile(F_h, (N, 1)), p, iters=15)
    assert sol.converged and sol.cost[-1] <= sol.cost[0]
    assert np.abs(sol.K[0] + Kinf).max() / np.abs(Kinf).max() < 0.1  # iLQR feedback is u = U + K dx, LQR is u = -K dx
    e0 = np.linalg.norm(st.minus(x0, x_h))
    assert np.linalg.norm(st.minus(sol.X[-1], x_h)) < 0.25 * e0  # converging (gentle weights, 2 s horizon)


def _step_problem(F_max, pens, T=2.5, dt=0.025):
    N = int(T / dt)
    x_goal = st.make_state(r=[1.0, 0.0, 1.0])
    Q, R = lq.bryson_weights(0.2, 0.2, 0.5, 4.0, 0.05)
    F_h = p.hover_thrust * np.ones(4)
    prob = ilqr.Problem(np.tile(x_goal, (N + 1, 1)), np.tile(F_h, (N, 1)), Q, R, 100 * Q, dt, pens, 0.0, F_max, 2)
    return prob, st.make_state(r=[0, 0, 0.5]), np.tile(F_h, (N, 1))


def test_input_bounds_are_respected():
    prob, x0, U0 = _step_problem(F_max=0.10, pens=[])
    sol = ilqr.solve(prob, x0, U0, p, iters=30)
    assert sol.U.min() >= -1e-12 and sol.U.max() <= 0.10 + 1e-12
    assert sol.cost[-1] < sol.cost[0]
    assert np.all(np.diff(sol.cost) <= 1e-9)  # monotone decrease


def test_tilt_limit_is_enforced():
    prob, x0, U0 = _step_problem(F_max=p.thrust_max, pens=[ilqr.TiltLimit(25.0, 1e5)])
    sol = ilqr.solve(prob, x0, U0, p, iters=40)
    tilt = np.degrees(np.arccos(min(st.rotation(x)[2, 2] for x in sol.X)))
    assert tilt < 26.5


def test_obstacle_avoidance_and_mpc_smoke():
    ref = tj.subsample(tj.path_trajectory("circle", p, T=6.0, dt=0.004, size=0.6, rate=1.0, ramp=1.5), 5)
    c = ref.pos[len(ref) // 2] + np.array([0.0, 0.05, 0.0])
    obs = ilqr.SphereObstacle(center=tuple(c), radius=0.2, margin=0.05, weight=1e5)
    Q, R = lq.bryson_weights(0.1, 0.2, 0.5, 4.0, 0.05)
    mpc = ilqr.ILQRMPC(ref, p, Q, R, 10 * Q, horizon=15, penalties=[obs], replan_every=5, iters=3)
    res = sim.simulate(ref.x[0], mpc, p, 6.0, 1e-3, ref.dt)
    assert np.isfinite(res.x).all()
    assert np.linalg.norm(res.x[:, :3] - c, axis=1).min() > 0.2
    assert res.F.min() >= 0 and res.F.max() <= p.thrust_max + 1e-12


def test_ilqr_with_semi_implicit_integrator_converges_and_flies_on_rk4_plant():
    prob, x0, U0 = _step_problem(F_max=p.thrust_max, pens=[])
    prob.integrator = "semi_implicit"
    sol = ilqr.solve(prob, x0, U0, p, iters=30)
    assert sol.cost[-1] < sol.cost[0] and np.all(np.diff(sol.cost) <= 1e-9)
    res = sim.simulate(x0, ilqr.ILQRPolicy(sol, p), p, 2.5, 1e-3, prob.dt)  # executed on the RK4 plant
    assert np.linalg.norm(res.x[-1, :3] - [1.0, 0.0, 1.0]) < 0.05
