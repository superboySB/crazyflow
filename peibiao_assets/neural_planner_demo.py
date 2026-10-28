"""Train and evaluate a limited single-obstacle neural trajectory planner.

The planner maps obstacle position/radius to all path ordinates in one forward pass. A separate
Crazyflow state controller tracks one held-out predicted path; it is not part of the network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import imageio.v2 as imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import optax
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline

from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Sim
from crazyflow.sim.visualize import draw_line

HERE = Path(__file__).resolve().parent
N_WAYPOINTS = 31
X = np.linspace(-1.0, 1.0, N_WAYPOINTS, dtype=np.float32)


def dataset(seed: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate analytic supervision for left-to-right paths over one obstacle."""
    rng = np.random.default_rng(seed)
    obs = np.column_stack(
        (
            rng.uniform(-0.25, 0.25, count),
            rng.uniform(-0.10, 0.10, count),
            rng.uniform(0.15, 0.24, count),
        )
    ).astype(np.float32)
    dx = X[None, :] - obs[:, 0, None]
    hump = np.exp(-0.5 * (dx / 0.48) ** 2)
    # Endpoints are prescribed. Interior label deliberately passes above every obstacle.
    envelope = np.sin(np.pi * (X + 1) / 2)[None, :]
    y = (obs[:, 1, None] + obs[:, 2, None] + 0.28) * hump * envelope
    return obs, y.astype(np.float32)


def init_planner(seed: int) -> list[tuple[jax.Array, jax.Array]]:
    """Initialize a 3 -> 64 -> 64 -> 31 multilayer trajectory network."""
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    dimensions = (3, 64, 64, N_WAYPOINTS)
    return [
        (jax.random.normal(key, (din, dout)) * np.sqrt(2 / (din + dout)), jnp.zeros(dout))
        for key, din, dout in zip(keys, dimensions[:-1], dimensions[1:], strict=True)
    ]


def predict(parameters: list[tuple[jax.Array, jax.Array]], obstacles: jax.Array) -> jax.Array:
    """Predict all waypoint ordinates in one forward pass."""
    hidden = obstacles
    for weight, bias in parameters[:-1]:
        hidden = jnp.tanh(hidden @ weight + bias)
    weight, bias = parameters[-1]
    return (hidden @ weight + bias) * jnp.sin(jnp.linspace(0, jnp.pi, N_WAYPOINTS))


def clearances(obs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Minimum center-to-cylinder surface distance along each polyline, sampled densely."""
    query_x = np.linspace(-1.0, 1.0, 601)
    result = []
    for o, y in zip(obs, ys, strict=True):
        query_y = np.interp(query_x, X, y)
        result.append(np.min(np.hypot(query_x - o[0], query_y - o[1]) - o[2]))
    return np.asarray(result)


def main() -> None:
    """Train on fixed synthetic seed and run held-out planning and simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    train_x, train_y = dataset(100, 2048)
    test_x, test_y = dataset(200, 256)
    parameters = init_planner(1701)
    optimizer = optax.adam(0.003)
    opt_state = optimizer.init(parameters)

    @jax.jit
    def train_step(params: list, state: object) -> tuple:
        def loss_fn(candidate: list) -> jax.Array:
            return jnp.mean((predict(candidate, jnp.asarray(train_x)) - jnp.asarray(train_y)) ** 2)

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state, loss

    losses = []
    for _ in range(args.epochs):
        parameters, opt_state, loss = train_step(parameters, opt_state)
        losses.append(float(loss))
    test_prediction = np.asarray(predict(parameters, jnp.asarray(test_x)))
    clearance = clearances(test_x, test_prediction)
    # Select the first held-out scene before looking at the outcome.
    obstacle = test_x[0]
    planned_y = test_prediction[0]
    path = np.stack((X, planned_y, np.ones_like(X)), axis=1)
    weights = HERE / "neural_planner_weights.npz"
    np.savez_compressed(
        weights,
        **{
            f"layer{i}_{name}": np.asarray(value)
            for i, (weight, bias) in enumerate(parameters)
            for name, value in (("weight", weight), ("bias", bias))
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(np.arange(1, len(losses) + 1), losses)
    axes[0].set_yscale("log")
    axes[0].set(
        xlabel="Epoch", ylabel="Mean squared error (m²)", title="Supervised neural planner training"
    )
    axes[0].grid(alpha=0.3)
    axes[1].hist(clearance, bins=20)
    axes[1].axvline(0, color="r", label="Cylinder surface")
    axes[1].set(
        xlabel="Planned center clearance (m)",
        ylabel="Held-out scenarios",
        title="Independent synthetic test: 256 scenes",
    )
    axes[1].legend()
    fig.savefig(HERE / "neural_planner_training.png", dpi=160)
    plt.close(fig)

    sim = Sim(control="state", dynamics="first_principles", device="cpu")
    spec = mujoco.MjSpec.from_string(
        '<mujoco><worldbody><body name="obstacle"><geom name="obstacle" type="cylinder" '
        f'size="{float(obstacle[2])} 0.65" rgba="0.8 0.2 0.15 1"/>'
        "</body></worldbody></mujoco>"
    )
    attached = sim.spec.worldbody.add_frame().attach_body(spec.body("obstacle"), "", ":0")
    attached.pos = [float(obstacle[0]), float(obstacle[1]), 1.0]
    sim.build_mjx()
    sim.reset()
    weight = sim.data.params.mass * -sim.data.params.gravity_vec[-1]
    rotor = motor_force2rotor_vel(weight / 4, sim.data.params.rpm2thrust)
    states = sim.data.states.replace(
        pos=sim.data.states.pos.at[0, 0].set(jnp.asarray(path[0])),
        rotor_vel=jnp.broadcast_to(rotor, sim.data.states.rotor_vel.shape),
    )
    sim.data = sim.data.replace(states=states, core=sim.data.core.replace(mjx_synced=False))
    flight_seconds = 8.0
    path_spline = CubicSpline(np.linspace(0, flight_seconds, N_WAYPOINTS), path, axis=0)
    camera = {
        "distance": 3.4,
        "elevation": -35.0,
        "azimuth": 105.0,
        "lookat": np.array([0.0, 0.1, 0.9]),
    }
    ref, actual, contacts, frames = [], [], [], []
    frequency = int(sim.control_freq)
    writer = None
    if not args.no_render:
        writer = imageio.get_writer(
            HERE / "neural_planner_flight.mp4", fps=20, codec="libx264", quality=8
        )
    try:
        for step in range(int((flight_seconds + 2) * frequency)):
            t = step / frequency
            sample_t = min(t, flight_seconds)
            target = path_spline(sample_t)
            velocity = path_spline(sample_t, 1) if t < flight_seconds else np.zeros(3)
            acceleration = path_spline(sample_t, 2) if t < flight_seconds else np.zeros(3)
            cmd = np.zeros((1, 1, 16), np.float32)
            cmd[0, 0, :3] = target
            cmd[0, 0, 3:6] = velocity
            cmd[0, 0, 6:9] = acceleration
            cmd[0, 0, 12] = 1.0
            sim.state_control(cmd)
            sim.step(sim.freq // frequency)
            pos = np.asarray(sim.data.states.pos)[0, 0].copy()
            # Compare the post-step state with the reference at the same physical time.
            ref.append(path_spline(min((step + 1) / frequency, flight_seconds)))
            actual.append(pos)
            contacts.append(bool(np.asarray(sim.contacts())[0, 0]))
            if writer and step % max(1, frequency // 20) == 0:
                if sim.viewer is not None:
                    sim.viewer.viewer._markers.clear()
                draw_line(
                    sim, path, rgba=np.array([1.0, 0.6, 0.1, 0.9]), start_size=2.0, end_size=2.0
                )
                if len(actual) > 1:
                    draw_line(
                        sim,
                        np.asarray(actual)[::3],
                        rgba=np.array([0.1, 0.8, 1.0, 0.9]),
                        start_size=2.0,
                        end_size=2.0,
                    )
                frame = Image.fromarray(
                    sim.render(mode="rgb_array", width=800, height=528, cam_config=camera)
                )
                canvas = ImageDraw.Draw(frame)
                canvas.rectangle((0, 0, 800, 70), fill=(18, 25, 39))
                font = ImageFont.truetype("DejaVuSans.ttf", 18)
                canvas.text(
                    (12, 8),
                    "Neural trajectory planner | held-out obstacle | Crazyflow",
                    font=font,
                    fill="white",
                )
                canvas.text(
                    (12, 36),
                    f"t = {t:.2f} s | orange: plan | blue: flown trajectory",
                    font=font,
                    fill="white",
                )
                pixels = np.asarray(frame)
                writer.append_data(pixels)
                if not frames or abs(t - 7.0) < 0.03:
                    frames = [pixels.copy()]
    finally:
        if writer:
            writer.close()
        sim.close()
    actual, ref = np.asarray(actual), np.asarray(ref)
    if frames:
        Image.fromarray(frames[0]).save(HERE / "neural_planner_flight.png")
    surface = np.hypot(actual[:, 0] - obstacle[0], actual[:, 1] - obstacle[1]) - obstacle[2]
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(path[:, 0], path[:, 1], "--", color="darkorange", label="Neural planned trajectory")
    ax.plot(actual[:, 0], actual[:, 1], color="royalblue", label="Simulated trajectory")
    ax.add_patch(
        plt.Circle(obstacle[:2], obstacle[2], color="firebrick", alpha=0.7, label="Obstacle")
    )
    ax.scatter(
        [path[0, 0], path[-1, 0]],
        [path[0, 1], path[-1, 1]],
        c=["green", "black"],
        label="Start / goal",
        zorder=3,
    )
    ax.axis("equal")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Held-out neural plan and Crazyflow execution")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(HERE / "neural_planner_trajectory.png", dpi=160)
    plt.close(fig)
    result = {
        "description": "A network predicts the path; a separate state controller tracks it",
        "jax_version": jax.__version__,
        "mujoco_version": mujoco.__version__,
        "backend": jax.default_backend(),
        "train_seed": 100,
        "train_scenes": len(train_x),
        "test_seed": 200,
        "test_scenes": len(test_x),
        "jax_seed": 1701,
        "epochs": args.epochs,
        "final_training_mse_m2": losses[-1],
        "test_waypoint_rmse_m": float(np.sqrt(np.mean((test_prediction - test_y) ** 2))),
        "test_center_clearance_min_m": float(clearance.min()),
        "test_plans_intersecting_cylinder": int((clearance < 0).sum()),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "heldout_index": 0,
        "obstacle_xyr_m": obstacle.tolist(),
        "physics": "Crazyflow cf21B_500 first_principles; 500 Hz, state control 100 Hz",
        "flight_seconds": flight_seconds,
        "settle_seconds": 2,
        "trajectory_tracking_rmse_m": float(np.sqrt(np.mean(np.sum((actual - ref) ** 2, axis=1)))),
        "actual_min_center_clearance_m": float(surface.min()),
        "actual_contact_steps": int(sum(contacts)),
        "final_goal_error_m": float(np.linalg.norm(actual[-1] - path[-1])),
        "render_enabled": bool(writer),
        "limitations": (
            "Synthetic single static cylinder, fixed endpoints, 2D path and known obstacle; "
            "separate controller; no real flight, sensing, or global safety guarantee."
        ),
    }
    (HERE / "neural_planner_results.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        HERE / "neural_planner_trajectory.npz",
        planned=path,
        reference=ref,
        actual=actual,
        contacts=contacts,
        obstacle=obstacle,
        test_obstacles=test_x,
        test_predictions=test_prediction,
        test_labels=test_y,
        losses=losses,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
