"""Tools to audit the gains that are compiled into the Crazyflie firmware (``LQR_controller_cpp.cpp``)."""

import re
from pathlib import Path

import numpy as np

from . import dynamics as dy
from . import lqr as lq
from . import state as st
from .params import CrazyflieParams

_NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def parse_c_gain_tables(path):
    """Return {label: 4x12 array} for every ``Kinf*`` 4x12 float table found in a C/C++/txt file.

    Labels are the C identifier, with ``Kinf_alt`` occurrences numbered by order
    ("Kinf_alt#1"...), matching the ``choice == n`` branches of ``Kinf_LQR_init``.
    """
    text = Path(path).read_text().replace("\r", "")
    out, counts = {}, {}
    for m in re.finditer(r"float\s+(Kinf\w*)\s*\[4\]\s*\[12\]\s*=\s*\{(.*?)\};", text, re.S):
        name, body = m.group(1), m.group(2)
        vals = [float(v) for v in re.findall(_NUM, body.replace("f", " "))]
        if len(vals) != 48:
            continue
        counts[name] = counts.get(name, 0) + 1
        label = name if name != "Kinf_alt" else f"Kinf_alt#{counts[name]}"
        out[label] = np.array(vals).reshape(4, 12)
    return out


def parse_hover_adjust(path):
    text = Path(path).read_text().replace("\r", "")
    m = re.search(r"hover_adjust\s*=\s*(" + _NUM + r")f", text)
    return float(m.group(1)) if m else 0.0


def effective_plant(p: CrazyflieParams, hover_speed_fw):
    """Plant whose hover speed is ``hover_speed_fw`` (same mass/inertia, Kf rescaled)."""
    Kf = p.mass * p.g / (4.0 * hover_speed_fw**2)
    return p.with_(Kf=Kf, Kt=p.Kt * Kf / p.Kf)


def speed_gain_to_thrust_gain(K_u, p, hover_speed):
    return K_u * (2.0 * p.Kf * hover_speed)


def closed_loop_radius(K_u, p, dt, yaw=0.0, hover_speed=None):
    """Spectral radius of the ZOH closed loop  x+ = (A - B K_F) x  at 'p' (speed-gain K_u)."""
    u_h = p.hover_speed if hover_speed is None else hover_speed
    K_F = speed_gain_to_thrust_gain(K_u, p, u_h)
    A, B = dy.discretize_zoh(st.make_state(yaw=yaw), p.hover_thrust * np.ones(4), p, dt)
    return float(np.abs(np.linalg.eigvals(A - B @ K_F)).max())
