"""Known-map, four-drone space-time planning and Crazyflow execution.

A deterministic prioritized A* planner reserves each previously planned path in space and time.
Execution uses Crazyflow first-principles dynamics and state feedback; visualization uses actual
simulator positions. The scenario is not an unknown-map or physical-flight experiment.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

N = 9
CELL = 0.6
DURATION_PER_CELL = 2.0
HORIZON = 30
MIN_PLAN_DISTANCE = 0.8  # grid units (0.48 m)
BLOCKED = {(0, 4), (4, 1), (4, 4), (4, 7), (8, 4)}
STARTS = [(0, 1), (0, 7), (8, 1), (8, 7)]
GOALS = [(8, 7), (8, 1), (0, 7), (0, 1)]
COLORS = ["#e64b35", "#4dbbd5", "#00a087", "#b09cdb"]


Cell = tuple[int, int]
PathCells = list[Cell]


def point(path: PathCells, t: int) -> Cell:
    """Return the held goal position when the path ends."""
    return path[min(t, len(path) - 1)]


def safe_step(a: Cell, b: Cell, c: Cell, d: Cell) -> bool:
    """Exact minimum separation for two synchronous line segments, in cell units."""
    x, y = a[0] - c[0], a[1] - c[1]
    vx, vy = b[0] - a[0] - d[0] + c[0], b[1] - a[1] - d[1] + c[1]
    den = vx * vx + vy * vy
    u = min(1.0, max(0.0, -(x * vx + y * vy) / den)) if den else 0.0
    return math.hypot(x + u * vx, y + u * vy) >= MIN_PLAN_DISTANCE


def plan(start: Cell, goal: Cell, prior: list[PathCells]) -> PathCells:
    """Plan on a known static grid, respecting other paths and their held goals."""

    def estimate(p: Cell) -> int:
        return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

    queue = [(estimate(start), 0, start)]
    parent = {(start, 0): None}
    while queue:
        _, t, pos = heapq.heappop(queue)
        if pos == goal and all(
            all(safe_step(pos, pos, point(p, k), point(p, k + 1)) for k in range(t, HORIZON))
            for p in prior
        ):
            path = []
            state = (pos, t)
            while state is not None:
                path.append(state[0])
                state = parent[state]
            return path[::-1]
        if t >= HORIZON:
            continue
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)):
            nxt = (pos[0] + dx, pos[1] + dy)
            if not (0 <= nxt[0] < N and 0 <= nxt[1] < N) or nxt in BLOCKED:
                continue
            if not all(safe_step(pos, nxt, point(p, t), point(p, t + 1)) for p in prior):
                continue
            state = (nxt, t + 1)
            if state not in parent:
                parent[state] = (pos, t)
                heapq.heappush(queue, (t + 1 + estimate(nxt), t + 1, nxt))
    raise RuntimeError(f"No space-time path for {start} -> {goal} within {HORIZON} time steps")


def plan_all() -> list[PathCells]:
    """Plan all drones and check every pair of reserved segments."""
    paths = []
    for start, goal in zip(STARTS, GOALS, strict=True):
        paths.append(plan(start, goal, paths))
    assert all(
        path[0] == s and path[-1] == g for path, s, g in zip(paths, STARTS, GOALS, strict=True)
    )
    for t in range(HORIZON):
        for i in range(len(paths)):
            for j in range(i):
                assert safe_step(
                    point(paths[i], t),
                    point(paths[i], t + 1),
                    point(paths[j], t),
                    point(paths[j], t + 1),
                )
    return paths


def xy(cell: Cell) -> tuple[float, float]:
    """Convert a grid cell to a world-frame position."""
    return ((cell[0] - 4) * CELL, (cell[1] - 4) * CELL)


def main():
    """Execute one deterministic scene and save measurements and visual evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import imageio.v2 as imageio
    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import mujoco
    import numpy as np
    from matplotlib.patches import Rectangle

    from crazyflow.control import Control
    from crazyflow.control.transform import motor_force2rotor_vel
    from crazyflow.sim import Sim
    from crazyflow.sim.visualize import draw_line

    out = Path(__file__).resolve().parent
    paths = plan_all()
    duration = max(len(path) - 1 for path in paths) * DURATION_PER_CELL + 3.0
    sim = Sim(n_drones=4, control=Control.state, dynamics="first_principles", device=args.device)
    obstacle_xml = """
    <mujoco><worldbody><body name="obstacle">
      <geom type="box" size="0.16 0.16 0.6" rgba="0.75 0.25 0.2 1"/>
    </body></worldbody></mujoco>
    """
    spec = mujoco.MjSpec.from_string(obstacle_xml)
    frame = sim.spec.worldbody.add_frame()
    for i, cell in enumerate(sorted(BLOCKED)):
        x, y = xy(cell)
        frame.attach_body(spec.body("obstacle"), "", f":{i}").pos = [x, y, 0.6]
    sim.build_mjx()
    sim.reset()
    start_pos = np.array([[*xy(s), 1.0] for s in STARTS], dtype=np.float32)[None]
    thrust = sim.data.params.mass * -sim.data.params.gravity_vec[-1]
    rpm = motor_force2rotor_vel(thrust / 4, sim.data.params.rpm2thrust)
    states = sim.data.states.replace(
        pos=jnp.asarray(start_pos),
        vel=jnp.zeros_like(sim.data.states.vel),
        rotor_vel=jnp.broadcast_to(rpm, sim.data.states.rotor_vel.shape),
    )
    sim.data = sim.data.replace(states=states, core=sim.data.core.replace(mjx_synced=False))
    actual, desired, frames, times, contacts = [], [], [], [], []
    step_count = round(duration * sim.control_freq)
    for k in range(step_count + 1):
        t = k / sim.control_freq
        slot = int(t / DURATION_PER_CELL)
        u = min(1.0, max(0.0, (t / DURATION_PER_CELL) - slot))
        blend = 10 * u**3 - 15 * u**4 + 6 * u**5
        blend_v = (30 * u**2 - 60 * u**3 + 30 * u**4) / DURATION_PER_CELL
        blend_a = (60 * u - 180 * u**2 + 120 * u**3) / DURATION_PER_CELL**2
        cmd = np.zeros((1, 4, 16), dtype=np.float32)
        cmd[..., 12] = 1.0
        for i, path in enumerate(paths):
            p0, p1 = np.array(xy(point(path, slot))), np.array(xy(point(path, slot + 1)))
            cmd[0, i, :2] = p0 + blend * (p1 - p0)
            cmd[0, i, 2] = 1.0
            cmd[0, i, 3:5] = blend_v * (p1 - p0)
            cmd[0, i, 6:8] = blend_a * (p1 - p0)
        pos = np.asarray(sim.data.states.pos[0])
        actual.append(pos.copy())
        desired.append(cmd[0, :, :3].copy())
        times.append(t)
        if k % max(1, sim.control_freq // 10) == 0:
            contacts.append(np.asarray(sim.contacts()[0]).tolist())
            if not args.no_video:
                history = np.asarray(actual)[:: max(1, sim.control_freq // 2)]
                rgba = (
                    [0.90, 0.20, 0.15, 1.0],
                    [0.10, 0.55, 0.95, 1.0],
                    [0.10, 0.75, 0.35, 1.0],
                    [0.75, 0.30, 0.85, 1.0],
                )
                if len(history) >= 2:
                    for i in range(4):
                        draw_line(sim, history[:, i, :], np.asarray(rgba[i]), 2.0, 2.0)
                frame = sim.render(
                    mode="rgb_array",
                    width=960,
                    height=720,
                    cam_config={
                        "distance": 8.0,
                        "azimuth": 90,
                        "elevation": -65,
                        "lookat": np.array([0.0, 0.0, 0.5]),
                    },
                )
                frames.append(frame.copy())
        if k < step_count:
            sim.state_control(cmd)
            sim.step(sim.freq // sim.control_freq)
    sim.close()
    actual, desired, times = np.asarray(actual), np.asarray(desired), np.asarray(times)
    errors = np.linalg.norm(actual - desired, axis=2)
    dists = np.stack(
        [np.linalg.norm(actual[:, i] - actual[:, j], axis=1) for i in range(4) for j in range(i)],
        axis=1,
    )
    # Horizontal distance to the surface of each obstacle's axis-aligned box. This is a centre
    # clearance, not a collision-volume or continuous-time safety guarantee.
    centers = np.array([xy(cell) for cell in sorted(BLOCKED)])
    offsets = np.maximum(np.abs(actual[:, :, None, :2] - centers[None, None, :, :]) - 0.16, 0)
    obstacle_dist = np.linalg.norm(offsets, axis=-1)
    final_err = np.linalg.norm(actual[-1] - desired[-1], axis=1)
    metrics = {
        "scope": (
            "single known static grid, deterministic prioritized space-time A*, "
            "Crazyflow closed-loop"
        ),
        "device": args.device,
        "jax": jax.__version__,
        "mujoco": mujoco.__version__,
        "drones": 4,
        "worlds": 1,
        "cell_m": CELL,
        "cell_duration_s": DURATION_PER_CELL,
        "starts": STARTS,
        "goals": GOALS,
        "obstacles": sorted(BLOCKED),
        "paths": paths,
        "duration_s": float(times[-1]),
        "control_hz": sim.control_freq,
        "simulation_hz": sim.freq,
        "tracking_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "max_tracking_error_m": float(errors.max()),
        "min_inter_drone_center_distance_m": float(dists.min()),
        "min_horizontal_obstacle_surface_distance_m": float(obstacle_dist.min()),
        "final_goal_errors_m": final_err.tolist(),
        "goal_within_0_2m": (final_err < 0.2).tolist(),
        "samples": len(times),
        "contact_sampling_hz": 10,
        "sampled_contact_frames": len(contacts),
        "sampled_contact_positive_frames": sum(any(row) for row in contacts),
    }
    (out / "swarm_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(
        out / "swarm_data.npz",
        time=times,
        actual=actual,
        desired=desired,
        pair_distances=dists,
        contacts=contacts,
    )
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for cell in sorted(BLOCKED):
        x, y = xy(cell)
        ax[0].add_patch(Rectangle((x - 0.16, y - 0.16), 0.32, 0.32, color="#a94442", alpha=0.8))
    for i, (path, color) in enumerate(zip(paths, COLORS, strict=True)):
        route = np.array([xy(p) for p in path])
        ax[0].plot(route[:, 0], route[:, 1], "--", color=color, alpha=0.5, lw=1.5)
        ax[0].plot(actual[:, i, 0], actual[:, i, 1], color=color, lw=2, label=f"Drone {i} measured")
        ax[0].scatter(*xy(STARTS[i]), marker="o", color=color, s=55)
        ax[0].scatter(*xy(GOALS[i]), marker="*", color=color, s=100)
    ax[0].set(
        title="Known-map space-time A* + Crazyflow execution",
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
    )
    ax[0].grid(alpha=0.2)
    ax[0].legend(fontsize=8)
    ax[1].plot(times, dists.min(axis=1), label="Minimum drone-centre separation", color="#3b6fb6")
    ax[1].plot(
        times,
        obstacle_dist.min(axis=(1, 2)),
        label="Minimum horizontal obstacle surface distance",
        color="#ce6845",
    )
    ax[1].set(
        title="Measured distances (sampled at control rate)",
        xlabel="Time (s)",
        ylabel="Distance (m)",
    )
    ax[1].grid(alpha=0.2)
    ax[1].legend(fontsize=8)
    fig.savefig(out / "swarm_trajectory.png", dpi=160)
    plt.close(fig)
    if frames:
        imageio.mimsave(out / "swarm_simulation.mp4", frames, fps=10, codec="libx264", quality=8)
        imageio.imwrite(out / "swarm_simulation.png", frames[len(frames) // 2])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
