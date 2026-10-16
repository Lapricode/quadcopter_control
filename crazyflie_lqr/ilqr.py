"""Constrained iLQR / receding-horizon MPC for the Crazyflie on the manifold R^3 x SO(3) x R^3 x R^3.

* Dynamics: RK4 (default), semi-implicit Euler or explicit Euler (``Problem.integrator``), optionally with
  sub-steps; control = motor thrusts F [N].
* Derivatives: exact zero-order-hold Jacobians for RK4 (``dynamics.discretize_zoh``), analytic Jacobians of the
  discrete map for the Euler variants.
  Perturbations are right perturbations, so the backward pass lives in tangent coordinates and the
  forward pass uses  u = u_bar + alpha k + K (x (-) x_bar).
* **Input constraints** 0 <= F_i <= F_max are handled *exactly* in the backward pass with a
  projected-Newton box QP (Tassa, Mansard, Todorov 2014): clamped controls get zero feedback.
* **State constraints** (tilt, obstacles, floor) are soft quadratic hinge penalties with
  Gauss-Newton Hessians; weights set how hard they are.

Cost  J = sum_k  1/2 e_x' Q e_x + 1/2 e_u' R e_u + penalties(x_k)  +  1/2 e_N' Qf e_N
with e_x = x (-) x_ref[k], e_u = F - F_ref[k].
"""

from dataclasses import dataclass, field

import numpy as np

from . import dynamics as dy
from . import state as st
from .lie import so3
from .params import CrazyflieParams


# =============================================================== soft state constraints
class Penalty:
    """Quadratic hinge  1/2 w max(0, g(x))^2  with g > 0 meaning *violated*."""

    weight = 1.0

    def g(self, x):  # -> (value, gradient wrt the 12-d right perturbation)
        raise NotImplementedError

    def evaluate(self, x):
        val, grad = self.g(x)
        if val <= 0.0:
            return 0.0, np.zeros(12), np.zeros((12, 12))
        return 0.5 * self.weight * val**2, self.weight * val * grad, self.weight * np.outer(grad, grad)


@dataclass
class TiltLimit(Penalty):
    """Keep the thrust axis within ``max_deg`` of vertical: R_zz >= cos(max_deg)."""

    max_deg: float = 30.0
    weight: float = 1e4

    def g(self, x):
        R = st.rotation(x)
        ez = np.array([0.0, 0.0, 1.0])
        grad = np.zeros(12)
        grad[3:6] = -np.cross(ez, R.T @ ez)  # d(-R_zz)/d(theta), right perturbation
        return np.cos(np.radians(self.max_deg)) - R[2, 2], grad


@dataclass
class SphereObstacle(Penalty):
    """Stay at least ``radius + margin`` away from ``center``."""

    center: tuple = (0.0, 0.0, 1.0)
    radius: float = 0.3
    margin: float = 0.1
    weight: float = 1e4

    def g(self, x):
        d = x[0:3] - np.asarray(self.center)
        n = np.linalg.norm(d) + 1e-12
        grad = np.zeros(12)
        grad[0:3] = -d / n
        return self.radius + self.margin - n, grad


@dataclass
class FloorLimit(Penalty):
    z_min: float = 0.15
    weight: float = 1e4

    def g(self, x):
        grad = np.zeros(12)
        grad[2] = -1.0
        return self.z_min - x[2], grad


# ===================================================================== problem / result
@dataclass
class Problem:
    x_ref: np.ndarray  # (N+1, 13)
    F_ref: np.ndarray  # (N, 4)
    Q: np.ndarray
    R: np.ndarray
    Qf: np.ndarray
    dt: float
    penalties: list = field(default_factory=list)
    F_min: float = 0.0
    F_max: float = 1.0
    substeps: int = 2
    integrator: str = "rk4"  # "rk4" | "semi_implicit" | "euler" (used for rollouts and Jacobians)

    @property
    def N(self):
        return len(self.F_ref)


@dataclass
class ILQRResult:
    X: np.ndarray  # (N+1, 13)
    U: np.ndarray  # (N, 4)
    K: list  # N gains (4 x 12)
    cost: list
    iters: int
    converged: bool


# ======================================================================= pieces
def _step(x, F, p, dt, substeps, integrator="rk4"):
    h = dt / substeps
    step = dy.INTEGRATORS[integrator]
    for _ in range(substeps):
        x = step(x, F, p, h)
    return x


def _linearize(prob, x, F, p):
    """Discrete Jacobians (A, B) of one planning step, matching the chosen integrator.

    RK4: exact ZOH of the frozen error dynamics.  Euler / semi-implicit: analytic Jacobians, chained
    over the sub-steps.
    """
    if prob.integrator == "rk4":
        return dy.discretize_zoh(x, F, p, prob.dt)
    h = prob.dt / prob.substeps
    disc, step = dy.DISCRETIZERS[prob.integrator], dy.INTEGRATORS[prob.integrator]
    A, B = np.eye(12), np.zeros((12, 4))
    for _ in range(prob.substeps):
        Ak, Bk = disc(x, F, p, h)
        A, B = Ak @ A, Ak @ B + Bk
        x = step(x, F, p, h)
    return A, B


def _stage(prob, k, x, F):
    """cost, lx, lu, lxx, luu of stage k (Gauss-Newton Hessian)."""
    e = st.minus(x, prob.x_ref[k])
    J = np.eye(12)
    J[3:6, 3:6] = so3.jacobian_right_inverse(e[3:6])
    eu = F - prob.F_ref[k]
    c = 0.5 * e @ prob.Q @ e + 0.5 * eu @ prob.R @ eu
    lx = J.T @ (prob.Q @ e)
    lxx = J.T @ prob.Q @ J
    for pen in prob.penalties:
        cp, gp, Hp = pen.evaluate(x)
        c += cp
        lx = lx + gp
        lxx = lxx + Hp
    return c, lx, prob.R @ eu, lxx, prob.R


def _terminal(prob, x):
    e = st.minus(x, prob.x_ref[prob.N])
    J = np.eye(12)
    J[3:6, 3:6] = so3.jacobian_right_inverse(e[3:6])
    return 0.5 * e @ prob.Qf @ e, J.T @ (prob.Qf @ e), J.T @ prob.Qf @ J


def total_cost(prob, X, U):
    c = sum(_stage(prob, k, X[k], U[k])[0] for k in range(prob.N))
    return c + _terminal(prob, X[prob.N])[0]


def rollout(x0, U, p, dt, substeps, integrator="rk4"):
    X = np.zeros((len(U) + 1, 13))
    X[0] = x0
    for k in range(len(U)):
        X[k + 1] = _step(X[k], U[k], p, dt, substeps, integrator)
    return X


def box_qp(Quu, Qu, lo, hi, iters=6):
    """min 1/2 d'Quu d + Qu'd  s.t. lo <= d <= hi  (4-d, projected active-set Newton).

    Returns (d, free_mask)."""
    n = len(Qu)
    clamped = np.zeros(n, bool)
    d = np.zeros(n)
    for _ in range(iters):
        free = ~clamped
        d = np.where(clamped, d, 0.0)
        if free.any():
            rhs = -(Qu[free] + Quu[np.ix_(free, clamped)] @ d[clamped])
            d[free] = np.linalg.solve(Quu[np.ix_(free, free)], rhs)
        low, high = d < lo, d > hi
        viol = (low | high) & free
        d = np.clip(d, lo, hi)
        if not viol.any():
            # release clamped variables whose gradient points back into the feasible set
            g = Quu @ d + Qu
            rel = clamped & (((d <= lo + 1e-12) & (g < 0)) | ((d >= hi - 1e-12) & (g > 0)))
            if not rel.any():
                break
            clamped &= ~rel
        clamped |= viol
    return d, ~clamped


def backward_pass(prob, X, U, p, mu):
    N = prob.N
    _, Vx, Vxx = _terminal(prob, X[N])
    kff = np.zeros((N, 4))
    K = [np.zeros((4, 12)) for _ in range(N)]
    dV1 = dV2 = 0.0
    I12 = np.eye(12)
    for k in range(N - 1, -1, -1):
        A, B = _linearize(prob, X[k], U[k], p)
        _, lx, lu, lxx, luu = _stage(prob, k, X[k], U[k])
        Qx = lx + A.T @ Vx
        Qu = lu + B.T @ Vx
        Vreg = Vxx + mu * I12
        Qxx = lxx + A.T @ Vxx @ A
        Quu = luu + B.T @ Vreg @ B
        Qux = B.T @ Vreg @ A
        Quu = 0.5 * (Quu + Quu.T)
        try:
            np.linalg.cholesky(Quu)
        except np.linalg.LinAlgError:
            return None
        lo, hi = prob.F_min - U[k], prob.F_max - U[k]
        d, free = box_qp(Quu, Qu, lo, hi)
        Kk = np.zeros((4, 12))
        if free.any():
            Kk[free] = -np.linalg.solve(Quu[np.ix_(free, free)], Qux[free])
        kff[k], K[k] = d, Kk
        Vx = Qx + Kk.T @ Quu @ d + Kk.T @ Qu + Qux.T @ d
        Vxx = Qxx + Kk.T @ Quu @ Kk + Kk.T @ Qux + Qux.T @ Kk
        Vxx = 0.5 * (Vxx + Vxx.T)
        dV1 += d @ Qu
        dV2 += 0.5 * d @ Quu @ d
    return kff, K, dV1, dV2


def forward_pass(prob, x0, X, U, kff, K, alpha, p):
    Xn = np.zeros_like(X)
    Un = np.zeros_like(U)
    Xn[0] = x0
    for k in range(prob.N):
        dx = st.minus(Xn[k], X[k])
        Un[k] = np.clip(U[k] + alpha * kff[k] + K[k] @ dx, prob.F_min, prob.F_max)
        Xn[k + 1] = _step(Xn[k], Un[k], p, prob.dt, prob.substeps, prob.integrator)
        if not np.all(np.isfinite(Xn[k + 1])):
            return None, None
    return Xn, Un


def solve(prob: Problem, x0, U_init, p: CrazyflieParams, iters=40, tol=1e-4, mu0=1e-6, verbose=False):
    """Run iLQR from the control guess ``U_init`` (N,4).  (Line-search trials may overflow: that is expected.)"""
    with np.errstate(all="ignore"):
        return _solve(prob, x0, U_init, p, iters, tol, mu0, verbose)


def _solve(prob, x0, U_init, p, iters, tol, mu0, verbose):
    U = np.clip(np.array(U_init, float), prob.F_min, prob.F_max)
    X = rollout(x0, U, p, prob.dt, prob.substeps, prob.integrator)
    J = total_cost(prob, X, U)
    hist = [J]
    mu, mu_min, mu_fac = mu0, 1e-9, 4.0
    K = [np.zeros((4, 12)) for _ in range(prob.N)]
    converged = False
    it = 0
    for it in range(1, iters + 1):
        bp = backward_pass(prob, X, U, p, mu)
        while bp is None:
            mu = max(mu * mu_fac, 1e-6)
            bp = backward_pass(prob, X, U, p, mu)
            if mu > 1e8:
                break
        if bp is None:
            break
        kff, K, dV1, dV2 = bp
        accepted = False
        for alpha in 0.5 ** np.arange(0, 8):
            Xn, Un = forward_pass(prob, x0, X, U, kff, K, alpha, p)
            if Xn is None:
                continue
            Jn = total_cost(prob, Xn, Un)
            expected = -(alpha * dV1 + alpha**2 * dV2)
            if Jn < J and (J - Jn) > 1e-4 * max(expected, 1e-12):
                accepted = True
                break
        if not accepted:
            mu = max(mu * mu_fac, 1e-6)
            if mu > 1e8:
                break
            continue
        dJ = J - Jn
        X, U, J = Xn, Un, Jn
        hist.append(J)
        mu = max(mu / mu_fac, mu_min)
        if verbose:
            print(f"iter {it:3d}  J={J:.5f}  dJ={dJ:.2e}  alpha={alpha:.3f}  mu={mu:.1e}")
        if dJ < tol * max(abs(J), 1.0):
            converged = True
            break
    # make sure the returned gains match the returned trajectory
    bp = backward_pass(prob, X, U, p, 1e-9)
    if bp is not None:
        K = bp[1]
    return ILQRResult(X=X, U=U, K=K, cost=hist, iters=it, converged=converged)


# ================================================================ closed-loop policies
class ILQRPolicy:
    """Execute an iLQR solution:  F_k = U_k + K_k (x (-) X_k)  (clamped later by the simulator)."""

    def __init__(self, sol: ILQRResult, p: CrazyflieParams):
        self.sol, self.Fmax = sol, p.thrust_max

    def __call__(self, i, t, x):
        k = min(i, len(self.sol.U) - 1)
        return np.clip(self.sol.U[k] + self.sol.K[k] @ st.minus(x, self.sol.X[k]), 0.0, self.Fmax)


class ILQRMPC:
    """Receding-horizon iLQR.  Re-plans every ``replan_every`` control steps over ``horizon`` steps,
    warm-started with the shifted previous plan, then follows the plan's own feedback in between."""

    def __init__(self, ref, p: CrazyflieParams, Q, R, Qf, horizon, penalties=(), replan_every=5,
                 iters=6, substeps=2, F_max=None, integrator="rk4"):
        self.ref, self.p, self.Q, self.R, self.Qf = ref, p, Q, R, Qf
        self.H, self.pen, self.every, self.iters, self.sub = horizon, list(penalties), replan_every, iters, substeps
        self.Fmax = p.thrust_max if F_max is None else F_max
        self.integrator = integrator
        self.sol, self.k0, self.U_prev = None, 0, None
        self.solve_times = []

    def _problem(self, i):
        n = len(self.ref)
        idx = np.clip(np.arange(i, i + self.H + 1), 0, n - 1)
        return Problem(x_ref=self.ref.x[idx], F_ref=self.ref.F[idx[:-1]], Q=self.Q, R=self.R, Qf=self.Qf,
                       dt=self.ref.dt, penalties=self.pen, F_max=self.Fmax, substeps=self.sub, integrator=self.integrator)

    def __call__(self, i, t, x):
        import time

        if self.sol is None or (i - self.k0) % self.every == 0 or i - self.k0 >= len(self.sol.U):
            prob = self._problem(i)
            if self.U_prev is None:
                guess = prob.F_ref.copy()
            else:  # shift the previous plan
                shift = i - self.k0
                guess = np.vstack([self.sol.U[shift:], prob.F_ref[len(self.sol.U) - shift:]])[: self.H]
            t0 = time.time()
            self.sol = solve(prob, x, guess, self.p, iters=self.iters)
            self.solve_times.append(time.time() - t0)
            self.k0, self.U_prev = i, self.sol.U
        j = min(i - self.k0, len(self.sol.U) - 1)
        return np.clip(self.sol.U[j] + self.sol.K[j] @ st.minus(x, self.sol.X[j]), 0.0, self.p.thrust_max)
