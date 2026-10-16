"""Physical parameters of the Crazyflie (single source of truth).

All the numbers that used to be copy-pasted in ``speeds_LQR_crazyflie.py`` and
``thrusts_LQR_crazyflie.py`` live here.
"""

from dataclasses import dataclass, field, replace

import numpy as np

N_STATE = 12  # tangent-space dimension of the state  [r, theta, v_b, omega_b]
N_STATE_Q = 13  # stored state: [r(3), q(4), v_b(3), omega_b(3)]
N_INPUT = 4


def _default_inertia():
    return np.array(
        [
            [16.6e-6, 0.83e-6, 0.72e-6],
            [0.83e-6, 16.6e-6, 1.8e-6],
            [0.72e-6, 1.8e-6, 29.3e-6],
        ]
    )


@dataclass(eq=False)
class CrazyflieParams:
    mass: float = 0.033  # kg
    arm: float = 0.046  # m (centre -> motor)
    g: float = 9.81  # m/s^2
    Kf: float = 2.25e-8  # N / (rad/s)^2
    Kt: float = 1.34e-10  # N m / (rad/s)^2
    body_yaw0: float = -0.75 * np.pi  # rotation of the motor cross w.r.t. the body frame
    max_rpm: float = 25000.0
    inertia: np.ndarray = field(default_factory=_default_inertia)

    # ------------------------------------------------------------------ derived
    @property
    def inertia_inv(self):
        return np.linalg.inv(self.inertia)

    @property
    def speed_max(self):
        """Maximum motor speed in rad/s."""
        return self.max_rpm * 2.0 * np.pi / 60.0

    @property
    def thrust_max(self):
        """Maximum thrust of a single motor in N."""
        return self.Kf * self.speed_max**2

    @property
    def hover_thrust(self):
        """Thrust per motor needed to hover (N)."""
        return self.mass * self.g / 4.0

    @property
    def hover_speed(self):
        """Motor speed needed to hover (rad/s)."""
        return float(np.sqrt(self.hover_thrust / self.Kf))

    @property
    def mixer(self):
        """4x4 matrix M with  [T, tau_x, tau_y, tau_z] = M @ [F1..F4]  (thrust -> wrench)."""
        c, s, l = np.cos(self.body_yaw0), np.sin(self.body_yaw0), self.arm
        k = self.Kt / self.Kf
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [l * c, l * s, -l * c, -l * s],
                [l * s, -l * c, -l * s, l * c],
                [-k, k, -k, k],
            ]
        )

    @property
    def mixer_inv(self):
        return np.linalg.inv(self.mixer)

    def with_(self, **kw):
        """Copy with some fields replaced (handy for plant/model mismatch studies)."""
        return replace(self, **kw)


# ---------------------------------------------------------------- (de)serialisation
def params_to_dict(p: CrazyflieParams):
    d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.__dict__.items()}
    return d


def save_params(p: CrazyflieParams, path):
    import json

    with open(path, "w") as f:
        json.dump(params_to_dict(p), f, indent=2)


def load_params(path) -> CrazyflieParams:
    """Load parameters saved by :func:`save_params` (missing keys keep their default)."""
    import json

    with open(path) as f:
        d = json.load(f)
    if "inertia" in d:
        d["inertia"] = np.array(d["inertia"], dtype=float)
    return CrazyflieParams(**d)
