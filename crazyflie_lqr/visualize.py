"""Crazyflie animation + static summary plots.

    quadrotor_visualize(result, ref=None, states_step=20)   # interactive 3-D animation
    quadrotor_visualize(result, ref, save_path="run.gif")   # headless: write a gif
    plot_summary(result, ref, "summary.png")                # static figure

Keyboard (interactive mode)
---------------------------
Space  pause / resume          R  restart
Left / Right  step one frame back / forward (auto-pauses)          Q  quit

Differences w.r.t. the old ``visualize_crazyflie.py``: reads a ``SimResult``
(no CSV round-trip), the reference trajectory can be overlaid, the animation
interval is real-time (``dt * states_step``) and the documented arrow keys are
the ones that are actually bound.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from . import state as st


def _yaw(res):
    R = np.array([st.rotation(x) for x in res.x])
    return np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0])), R


def _ylim(arr, margin=0.1):
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = (hi - lo) or 1.0
    return lo - margin * span, hi + margin * span


def quadrotor_visualize(res, ref=None, states_step=20, save_path=None, fps=25, show=True, dpi=60):
    p = res.params
    l, u_max = p.arm, p.speed_max
    pos, states, controls, dt = res.x[:, :3], res.x, res.u, res.dt
    yaw, Rs = _yaw(res)
    n = len(res.t)
    total_time = res.t[-1]
    display_frames = (n - 1) // states_step + 1

    cy, sy = np.cos(p.body_yaw0), np.sin(p.body_yaw0)
    body_orient0 = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    pts = np.vstack(
        [
            [0, 0, 0],
            0.75 * l * np.array([-1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]),  # antenna
            [0, l, 0], [l, 0, 0], [0, -l, 0], [-l, 0, 0],  # motors 1..4
        ]
    )
    body_points = body_orient0 @ pts.T  # (3, 6)

    allp = pos if ref is None else np.vstack([pos, ref.pos])
    lo, hi = allp.min(axis=0) - l, allp.max(axis=0) + l
    rng_xy = max(hi[0] - lo[0], hi[1] - lo[1])
    m = 0.1

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Crazyflie LQR simulation", fontsize=13)
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_x = fig.add_subplot(2, 4, 3)
    ax_v = fig.add_subplot(2, 4, 4)
    ax_om = fig.add_subplot(2, 4, 7)
    ax_u = fig.add_subplot(2, 4, 8)

    ax3d.set_title("3-D trajectory")
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.view_init(elev=35, azim=40)
    ax3d.set_xlim(lo[0] - m, lo[0] + rng_xy + m)
    ax3d.set_ylim(lo[1] - m, lo[1] + rng_xy + m)
    ax3d.set_zlim(max(lo[2], 0.0) - 0.0, hi[2] + m)
    ax3d.set_box_aspect((1, 1, (hi[2] - max(lo[2], 0.0) + m) / (rng_xy + 2 * m)))
    if ref is not None:
        ax3d.plot(*ref.pos.T, "--", color="tab:orange", lw=1.2, label="reference")
        ax3d.legend(loc="upper right", fontsize=8)

    cxyz, cu = ("r", "g", "b"), ("r", "g", "b", "m")
    ax_x.set_title("Position (m) / yaw (rad)"); ax_x_r = ax_x.twinx()
    ax_v.set_title("Body velocity $v_b$ (m/s)")
    ax_om.set_title("Angular velocity $\\omega_b$ (rad/s)")
    ax_u.set_title("Motor speeds (rad/s)")
    for ax, data in [(ax_x, pos), (ax_v, states[:, 7:10]), (ax_om, states[:, 10:13]), (ax_u, controls)]:
        ax.set_xlabel("t (s)"); ax.set_xlim(0.0, total_time); ax.set_ylim(*_ylim(data))
    ax_x_r.set_ylim(*_ylim(yaw))
    ax_u.axhline(u_max, color="k", ls=":", lw=0.8)

    trace = ax3d.scatter([], [], [], c="k", s=6)
    body_sc = ax3d.scatter([], [], [], c="k", s=60)
    ant_sc = ax3d.scatter([], [], [], c="k", s=20)
    mot_sc = ax3d.scatter([], [], [], c="g", s=60)
    arms = [ax3d.plot([], [], [], "r-", lw=2)[0] for _ in range(4)]
    bars = [ax3d.plot([], [], [], "b-", lw=2)[0] for _ in range(4)]
    ant = ax3d.plot([], [], [], "k-", lw=1)[0]
    labels = [ax3d.text(0, 0, 0, str(i + 1), fontsize=9, color="blue", ha="center") for i in range(4)]
    ttxt = ax3d.text2D(0.04, 0.93, "", transform=ax3d.transAxes, fontsize=9)
    plines = [ax_x.plot([], [], color=c, lw=1, label=s)[0] for c, s in zip(cxyz, "xyz")]
    yline = ax_x_r.plot([], [], color="k", lw=1.5, label="yaw")[0]
    vlines = [ax_v.plot([], [], color=c, lw=1, label=s)[0] for c, s in zip(cxyz, ("vx", "vy", "vz"))]
    wlines = [ax_om.plot([], [], color=c, lw=1, label=s)[0] for c, s in zip(cxyz, ("wx", "wy", "wz"))]
    ulines = [ax_u.plot([], [], color=c, lw=1, label=f"u{i + 1}")[0] for i, c in enumerate(cu)]
    if ref is not None:
        for i, c in enumerate(cxyz):
            ax_x.plot(ref.t, ref.pos[:, i], "--", color=c, lw=0.7, alpha=0.6)
    for ax in (ax_x, ax_v, ax_om, ax_u):
        ax.legend(fontsize=7, loc="upper right")
    cursors = [a.axvline(0, ls="--", lw=0.8, color="k") for a in (ax_x, ax_v, ax_om, ax_u)]

    trace_max = max(1, n // (4 * states_step))
    state = {"paused": False, "frame": 0}

    def draw(frame):
        k = frame * states_step
        r, R, u = pos[k], Rs[k], controls[k]
        pw = (R @ body_points).T + r
        pB, pA, pMs = pw[0], pw[1], pw[2:]
        cp = np.copy(body_points[:, 2:])
        cp[2, :] += (l * u / u_max).clip(0.0, l)
        pCs = (R @ cp).T + r
        tr = pos[max(0, k - trace_max * states_step): k + 1: states_step]
        trace._offsets3d = (tr[:, 0], tr[:, 1], tr[:, 2])
        body_sc._offsets3d = ([pB[0]], [pB[1]], [pB[2]])
        ant_sc._offsets3d = ([pA[0]], [pA[1]], [pA[2]])
        mot_sc._offsets3d = (pMs[:, 0], pMs[:, 1], pMs[:, 2])
        ant.set_data([pB[0], pA[0]], [pB[1], pA[1]]); ant.set_3d_properties([pB[2], pA[2]])
        for i in range(4):
            arms[i].set_data([pB[0], pMs[i, 0]], [pB[1], pMs[i, 1]]); arms[i].set_3d_properties([pB[2], pMs[i, 2]])
            bars[i].set_data([pMs[i, 0], pCs[i, 0]], [pMs[i, 1], pCs[i, 1]]); bars[i].set_3d_properties([pMs[i, 2], pCs[i, 2]])
            labels[i].set_position((pCs[i, 0], pCs[i, 1])); labels[i].set_3d_properties(pCs[i, 2], zdir=None)
        ttxt.set_text(f"t = {res.t[k]:.2f} s")
        idx = np.arange(0, k + 1, states_step)
        th = res.t[idx]
        for i, ln in enumerate(plines): ln.set_data(th, pos[idx, i])
        yline.set_data(th, yaw[idx])
        for i, ln in enumerate(vlines): ln.set_data(th, states[idx, 7 + i])
        for i, ln in enumerate(wlines): ln.set_data(th, states[idx, 10 + i])
        for i, ln in enumerate(ulines): ln.set_data(th, controls[idx, i])
        for c in cursors: c.set_xdata([res.t[k], res.t[k]])

    interval = dt * states_step * 1000.0  # real-time playback

    if save_path is not None:  # headless / file output
        ani = FuncAnimation(fig, draw, frames=display_frames, interval=interval, blit=False)
        ani.save(save_path, writer="pillow", fps=fps, dpi=dpi)
        plt.close(fig)
        return

    def tick(_):
        if state["frame"] >= display_frames:
            state["frame"] = 0
        draw(state["frame"])
        if not state["paused"]:
            state["frame"] += 1

    def on_key(ev):
        key = (ev.key or "").lower()
        if key == " ":
            state["paused"] = not state["paused"]
        elif key == "r":
            state.update(paused=False, frame=0)
        elif key in ("right", "left"):
            step = 1 if key == "right" else -1
            state["paused"] = True
            state["frame"] = int(np.clip(state["frame"] + step, 0, display_frames - 1))
            draw(state["frame"]); fig.canvas.draw_idle()
        elif key == "q":
            plt.close("all")

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout()
    ani = FuncAnimation(fig, tick, interval=interval, blit=False, cache_frame_data=False)  # noqa: F841
    if show:
        plt.show()


def plot_summary(res, ref=None, path=None, title=None):
    """Static figure: xy/xz path, position error, tilt/thrust, motor speeds."""
    yaw, Rs = _yaw(res)
    tilt = np.degrees(np.arccos(np.clip(Rs[:, 2, 2], -1, 1)))
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5))
    ax = ax.ravel()
    ax[0].plot(res.x[:, 0], res.x[:, 1], label="flown")
    if ref is not None:
        ax[0].plot(ref.pos[:, 0], ref.pos[:, 1], "--", label="reference")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].set_title("Top view"); ax[0].legend()
    for i, s in enumerate("xyz"):
        ax[1].plot(res.t, res.x[:, i], label=s)
        if ref is not None:
            ax[1].plot(ref.t, ref.pos[:, i], "k--", lw=0.6)
    ax[1].set_title("Position (dashed = reference)"); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("m"); ax[1].legend()
    if ref is not None:
        idx = np.clip(np.round(res.t / ref.dt).astype(int), 0, len(ref) - 1)
        e = res.x[:, :3] - ref.pos[idx]
        for i, s in enumerate("xyz"):
            ax[2].plot(res.t, 1000 * e[:, i], label=f"e_{s}")
        ax[2].plot(res.t, 1000 * np.linalg.norm(e, axis=1), "k", label="|e|")
        ax[2].set_title("Position error"); ax[2].set_ylabel("mm"); ax[2].legend()
    ax[2].set_xlabel("t (s)")
    ax[3].plot(res.t, tilt); ax[3].set_title("Tilt angle"); ax[3].set_ylabel("deg"); ax[3].set_xlabel("t (s)")
    ax[4].plot(res.t, np.linalg.norm(res.x[:, 7:10], axis=1), label="|v|")
    ax[4].plot(res.t, np.linalg.norm(res.x[:, 10:13], axis=1), label="|omega|")
    ax[4].set_title("Speed (m/s) and body rate (rad/s)"); ax[4].set_xlabel("t (s)"); ax[4].legend()
    for i in range(4):
        ax[5].plot(res.t, res.u[:, i], lw=0.8, label=f"u{i + 1}")
    ax[5].axhline(res.params.speed_max, color="k", ls=":"); ax[5].axhline(0, color="k", ls=":")
    ax[5].set_title("Motor speeds (rad/s), dotted = limits"); ax[5].set_xlabel("t (s)"); ax[5].legend(ncol=4, fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
