import numpy as np
import pytest

from crazyflie_lqr import dynamics as dy
from crazyflie_lqr import lqr as lq
from crazyflie_lqr import simulation as sim
from crazyflie_lqr import state as st
from crazyflie_lqr import trajectory as tj
from crazyflie_lqr.params import CrazyflieParams

p = CrazyflieParams()


@pytest.mark.parametrize("kind,kw", [("circle", {}), ("figure8", {"yaw_mode": "tangent"}), ("helix", {"climb": 0.1})])
def test_reference_is_dynamically_feasible(kind, kw):
    ref = tj.path_trajectory(kind, p, T=10.0, **kw)
    assert ref.F.min() > 0 and ref.F.max() < p.thrust_max
    for k in range(1, len(ref) - 1, 5):
        xdot = st.minus(ref.x[k + 1], ref.x[k - 1]) / (2 * ref.dt)
        assert np.abs(xdot - dy.dynamics(ref.x[k], ref.F[k], p)).max() < 1e-3


def test_reference_starts_and_ends_at_rest():
    ref = tj.path_trajectory("figure8", p, T=10.0)
    assert np.linalg.norm(ref.vel[0]) < 1e-9 and np.linalg.norm(ref.vel[-1]) < 1e-9
    assert np.allclose(ref.F[0], p.hover_thrust, atol=1e-4)  # within 0.1 % of hover


@pytest.mark.parametrize("kind", ["circle", "figure8"])
def test_tvlqr_tracks_with_initial_offset(kind):
    ref = tj.path_trajectory(kind, p, T=10.0, size=0.8)
    Q, R = lq.bryson_weights(0.1, 0.2, 0.5, 4.0, 0.05)
    K = lq.tvlqr_along_reference(ref, p, Q, R)
    x0 = ref.x[0].copy()
    x0[:3] += [0.15, -0.1, 0.1]
    res = sim.simulate(x0, sim.TVLQRTracker(ref, K), p, ref.t[-1], 1e-3, 4e-3)
    e_pos, _, _, _ = sim.tracking_errors(res, ref)
    assert np.linalg.norm(e_pos[len(e_pos) // 2:], axis=1).max() < 5e-3


def test_velocity_profile_reference_integrates_to_its_position():
    t = np.arange(5.0)
    v = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0.5, 0, 0], [0, 0, 0]])
    ref = tj.velocity_profile_trajectory(t, v, p, r0=(0, 0, 1))
    dp = np.gradient(ref.pos, ref.dt, axis=0, edge_order=2)
    assert np.abs(dp - ref.vel).max() < 1e-3
    assert ref.F.min() > 0
