"""Parameter identification for the Crazyflie model.

None of this needs a motion-capture system: it works from quantities the Crazyflie
logs itself (gyro, accelerometer, commanded rotor speeds) plus a few bench measurements.

Bench measurements
------------------
* ``fit_thrust_coefficient``  thrust-stand data (rotor speed -> thrust)  -> Kf
* ``inertia_bifilar``         bifilar-pendulum oscillation period        -> one moment of inertia
* mass: kitchen scale (with battery and decks!)

Flight-log regressions (``identify_from_log``)
----------------------------------------------
With the specific force measured by the accelerometer a_b and the gyro w_b:

    a_b,z        = (Kf/m) * sum(u_i^2)                              -> k = Kf/m
    d w_x/dt     = (l Kf/Ixx) * [c (u1^2-u3^2) - s (u4^2-u2^2)]     -> alpha_x
    d w_y/dt     = (l Kf/Iyy) * [s (u1^2-u3^2) + c (u4^2-u2^2)]     -> alpha_y
    d w_z/dt     = (Kt/Izz)  * (-u1^2 + u2^2 - u3^2 + u4^2)         -> alpha_z

(c, s = cos/sin of ``body_yaw0``).  Angular accelerations are regressed in *integrated* form
(w[k+n] - w[k] against the sum of the regressors) so gyro noise is not amplified by differentiation.
The gyroscopic term w x Iw and rotor drag are neglected (they are small near hover); excite the vehicle
with a small random-binary (PRBS) pattern on the four rotor speeds around hover, tethered or in a net.
"""

from dataclasses import dataclass

import numpy as np

from .params import CrazyflieParams


# ------------------------------------------------------------------ bench measurements
def fit_thrust_coefficient(speeds, thrust, affine=False):
    """Least-squares Kf in  thrust = Kf * speed^2  (optionally with an offset)."""
    s2 = np.asarray(speeds, float) ** 2
    T = np.asarray(thrust, float)
    if not affine:
        return float(s2 @ T / (s2 @ s2))
    A = np.column_stack([s2, np.ones_like(s2)])
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    return float(coef[0]), float(coef[1])


def kf_from_hover_speed(hover_speed, mass, g=9.81):
    """Kf implied by the rotor speed that holds hover (all four equal)."""
    return mass * g / (4.0 * hover_speed**2)


def inertia_bifilar(period, mass, half_separation, wire_length, g=9.81):
    """Moment of inertia about the vertical axis of a bifilar pendulum.

    T = 2 pi sqrt(I L / (m g a^2))  ->  I = m g a^2 T^2 / (4 pi^2 L), a = half the wire separation.
    Hang the vehicle with the axis you want to measure vertical.
    """
    return mass * g * half_separation**2 * period**2 / (4.0 * np.pi**2 * wire_length)


# ------------------------------------------------------------------ flight-log regression
@dataclass
class IdResult:
    k_thrust: float  # Kf / m                [N / (rad/s)^2 / kg]
    alpha: np.ndarray  # (3,) = [l Kf/Ixx, l Kf/Iyy, Kt/Izz]
    r2: dict  # coefficients of determination of each regression
    params: CrazyflieParams  # updated parameters


def _windowed_diff(x, n):
    return x[n:] - x[:-n]


def _windowed_sum(r, n, dt):
    c = np.cumsum(r, axis=0)
    c = np.vstack([np.zeros((1, r.shape[1])), c])
    return (c[n:] - c[:-n])[: len(r) - n + 1][: len(r) - n] * dt


def _ls(y, a):
    alpha = float(a @ y / (a @ a))
    ss_res = float(np.sum((y - alpha * a) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1.0
    return alpha, 1.0 - ss_res / ss_tot


def identify_from_log(t, gyro, acc_body, speeds, p: CrazyflieParams, mass=None, window=20):
    """Estimate (Kf, Kt, diagonal inertia) from a hover-excitation log.

    t [s] (N,), gyro [rad/s] (N,3), acc_body specific force [m/s^2] (N,3), speeds [rad/s] (N,4).
    ``mass`` is the *measured* mass (defaults to ``p.mass``); Kf = k * mass.
    The nominal off-diagonal inertia terms are kept, scaled with the identified diagonal.
    """
    t = np.asarray(t)
    dt = float(np.mean(np.diff(t)))
    u2 = np.asarray(speeds, float) ** 2
    mass = p.mass if mass is None else mass

    # collective thrust:  a_bz = k * sum(u^2)
    s = u2.sum(axis=1)
    k, r2_k = _ls(np.asarray(acc_body)[:, 2], s)

    c, sn = np.cos(p.body_yaw0), np.sin(p.body_yaw0)
    reg = np.column_stack(
        [
            c * (u2[:, 0] - u2[:, 2]) - sn * (u2[:, 3] - u2[:, 1]),
            sn * (u2[:, 0] - u2[:, 2]) + c * (u2[:, 3] - u2[:, 1]),
            -u2[:, 0] + u2[:, 1] - u2[:, 2] + u2[:, 3],
        ]
    )
    reg = reg - reg.mean(axis=0)  # remove the (rotor-offset) trim so only the excitation is regressed
    gy = np.asarray(gyro, float)
    dw = _windowed_diff(gy, window)
    rs = _windowed_sum(reg, window, dt)
    alpha, r2 = np.zeros(3), {"thrust": r2_k}
    for i, name in enumerate(["roll", "pitch", "yaw"]):
        alpha[i], r2[name] = _ls(dw[:, i] - dw[:, i].mean(), rs[:, i] - rs[:, i].mean())

    Kf = k * mass
    I = p.inertia.copy()
    Ixx, Iyy = p.arm * Kf / alpha[0], p.arm * Kf / alpha[1]
    Izz_new = None
    scale = np.array([Ixx / I[0, 0], Iyy / I[1, 1], 1.0])
    I_new = I * np.outer(np.sqrt(scale), np.sqrt(scale))  # keep the nominal off-diagonal structure
    # yaw: alpha_z = Kt / Izz.  Izz is not separable from Kt with gyro data alone -> keep nominal Izz.
    Izz_new = I_new[2, 2]
    Kt = alpha[2] * Izz_new
    new = p.with_(mass=mass, Kf=Kf, Kt=Kt, inertia=I_new)
    return IdResult(k_thrust=k, alpha=alpha, r2=r2, params=new)


# ------------------------------------------------------------------ motor lag
def fit_first_order_lag(t, y, t_step):
    """Time constant tau of a first-order response to a step applied at ``t_step``.

    y(t) = y0 + (y1 - y0) (1 - exp(-(t - t_step)/tau)).  Use any signal that is proportional to rotor
    speed *change* (e.g. the accelerometer z reading after a step in the commanded collective speed
    in a tethered hover; thrust ~ speed^2, so keep the step small, +-3 % of hover speed).
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = t >= t_step
    tt, yy = t[m] - t_step, y[m]
    y0 = float(np.mean(y[~m])) if np.any(~m) else float(yy[0])
    y1_guess = float(np.mean(yy[-max(3, len(yy) // 10):]))

    def f(tt, y1, tau):
        return y0 + (y1 - y0) * (1.0 - np.exp(-tt / tau))

    (y1, tau), _ = curve_fit(f, tt, yy, p0=[y1_guess, 0.02], bounds=([-np.inf, 1e-4], [np.inf, 1.0]))
    return float(tau)
