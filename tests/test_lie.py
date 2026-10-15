import numpy as np
import pytest

from crazyflie_lqr.lie import s3, so3

rng = np.random.default_rng(0)


def rand_rot(scale=0.7):
    return so3.Exp(rng.normal(size=3) * scale)


def test_log_of_identity_is_finite_for_rounding_noise():
    # regression: arccos((tr-1)/2) returned NaN for ~3% of random rotations
    for _ in range(3000):
        R = rand_rot()
        assert np.all(np.isfinite(so3.minus_right(R, R)))


@pytest.mark.parametrize("theta", [1e-12, 1e-9, 1e-6, 1e-3, 0.5, 2.0, 3.1, np.pi - 1e-4])
def test_exp_log_roundtrip(theta):
    tau = theta * np.array([0.6, -0.8, 0.0])
    assert np.allclose(so3.Log(so3.Exp(tau)), tau, atol=1e-12, rtol=1e-9)


def test_tiny_rotations_are_not_dropped():
    # regression: Exp() used to return identity below |tau| = 1e-7 (omega < 1e-4 rad/s at 1 kHz)
    R = so3.Exp(np.array([0.0, 0.0, 5e-8]))
    assert abs(R[1, 0] - 5e-8) < 1e-15


def test_vee_shape():
    assert so3.vee(np.zeros((3, 3))).shape == (3,)


def test_quaternion_matches_matrix():
    for _ in range(200):
        tau = rng.normal(size=3)
        assert np.allclose(s3.to_matrix(s3.Exp(tau)), so3.Exp(tau), atol=1e-12)


@pytest.mark.parametrize("theta", [1e-9, 1e-8, 1e-6, 1e-3, 1.0, 3.0])
def test_matrix_to_quaternion_accuracy(theta):
    # regression: sqrt-of-diagonal formula had ~1e-8 absolute error near identity
    tau = theta * np.array([1.0, 0.0, 0.0])
    q = s3.from_matrix(so3.Exp(tau))
    assert np.allclose(s3.Log(q), tau, atol=1e-14 + 1e-9 * theta)


def test_s3_log_is_minimal():
    q = s3.Exp(np.array([0.0, 0.0, 0.5]))
    assert np.allclose(s3.Log(-q), s3.Log(q))
    assert np.linalg.norm(s3.Log(-q)) <= np.pi + 1e-12


def test_right_jacobians_are_inverses():
    for _ in range(50):
        tau = rng.normal(size=3)
        assert np.allclose(so3.jacobian_right(tau) @ so3.jacobian_right_inverse(tau), np.eye(3), atol=1e-10)
        assert np.allclose(so3.jacobian_left(tau) @ so3.jacobian_left_inverse(tau), np.eye(3), atol=1e-10)


def test_plus_minus_consistency():
    for _ in range(50):
        R, tau = rand_rot(), rng.normal(size=3) * 0.5
        assert np.allclose(so3.minus_right(so3.plus_right(R, tau), R), tau, atol=1e-10)
        q = s3.from_matrix(R)
        assert np.allclose(s3.minus_right(s3.plus_right(q, tau), q), tau, atol=1e-10)
