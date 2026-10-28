"""Reproduce first-principles rotor and disturbance response in two worlds.

A deterministic open-loop simulation, not a real-flight validation: world 1 gets
an external force pulse while both worlds receive the same rotor thrust pulse.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import crazyflow  # noqa: I001  # Configure SciPy array API before other imports.
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.dynamics import Dynamics, load_params
from crazyflow.sim import Sim

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL = ROOT / "crazyflow/dynamics/first_principles/params.toml"
FREQUENCY = 500
SAMPLE_STEPS = 5
DURATION = 5.0


def main() -> None:
    """Record the batched simulator's actual state and a rendered frame."""
    params = load_params(Dynamics.first_principles, "cf21B_500")
    mass = float(params["mass"])
    g = -float(params["gravity_vec"][2])
    nominal = float(motor_force2rotor_vel(np.asarray(mass * g / 4), params["rpm2thrust"]))
    pulse = float(motor_force2rotor_vel(np.asarray(1.1 * mass * g / 4), params["rpm2thrust"]))
    sim = Sim(
        n_worlds=2,
        n_drones=1,
        drone="cf21B_500",
        dynamics=Dynamics.first_principles,
        control=Control.rotor_vel,
        device="cpu",
        freq=FREQUENCY,
    )
    sim.reset()
    states = sim.data.states.replace(
        pos=sim.data.states.pos.at[..., 2].set(1.0),
        rotor_vel=jnp.full_like(sim.data.states.rotor_vel, nominal),
    )
    sim.data = sim.data.replace(states=states)
    records = []
    image0 = image1 = None
    steps = int(DURATION * FREQUENCY / SAMPLE_STEPS)
    try:
        for i in range(steps + 1):
            t = i * SAMPLE_STEPS / FREQUENCY
            command = pulse if 2.0 <= t < 3.0 else nominal
            force_x = 0.02 if 1.0 <= t < 2.0 else 0.0
            pos = np.asarray(sim.data.states.pos)[:, 0]
            vel = np.asarray(sim.data.states.vel)[:, 0]
            rotors = np.asarray(sim.data.states.rotor_vel)[:, 0]
            records.append(
                [
                    t,
                    command,
                    force_x,
                    rotors[0, 0],
                    pos[0, 0],
                    pos[1, 0],
                    vel[0, 0],
                    vel[1, 0],
                    pos[0, 2],
                    pos[1, 2],
                    vel[0, 2],
                    vel[1, 2],
                ]
            )
            if i == 190:
                camera = {
                    "distance": 1.5,
                    "elevation": -25,
                    "azimuth": -135,
                    "lookat": [0.2, 0.0, 1.0],
                }
                image0 = sim.render(
                    mode="rgb_array", world=0, width=640, height=480, cam_config=camera
                )
                image1 = sim.render(
                    mode="rgb_array", world=1, width=640, height=480, cam_config=camera
                )
            if i == steps:
                break
            forces = jnp.zeros_like(sim.data.states.force).at[1, 0, 0].set(force_x)
            sim.data = sim.data.replace(states=sim.data.states.replace(force=forces))
            sim.rotor_vel_control(jnp.full((2, 1, 4), command))
            sim.step(SAMPLE_STEPS)
    finally:
        sim.close()
    data = np.asarray(records)
    assert np.isfinite(data).all()
    assert data[:, 5].max() > 0.1 and np.max(np.abs(data[:, 4])) < 1e-6
    assert data[:, 3].max() > nominal and data[-1, 8] > 1.0
    header = (
        "time_s,command_rpm,world1_force_x_N,actual_rotor_rpm,world0_x_m,world1_x_m,"
        "world0_vx_mps,world1_vx_mps,world0_z_m,world1_z_m,world0_vz_mps,world1_vz_mps"
    )
    np.savetxt(HERE / "dynamics_trace.csv", data, delimiter=",", comments="", header=header)
    summary = {
        "drone": "cf21B_500",
        "dynamics": "first_principles",
        "environment": {
            "crazyflow": crazyflow.__version__,
            "jax": jax.__version__,
            "backend": jax.default_backend(),
        },
        "frequency_hz": FREQUENCY,
        "output_sample_hz": FREQUENCY / SAMPLE_STEPS,
        "duration_s": DURATION,
        "worlds": 2,
        "initial_height_m": 1.0,
        "mass_kg": mass,
        "gravity_mps2": g,
        "parameter_file": "crazyflow/dynamics/first_principles/params.toml",
        "parameter_file_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "rotor_model": "quadratic rpm-to-thrust plus first-order nonlinear rotor dynamics",
        "nominal_rotor_rpm": nominal,
        "pulse_rotor_rpm": pulse,
        "pulse_time_s": [2, 3],
        "thrust_pulse": "nominal total thrust command multiplied by 1.1",
        "world1_force_x_N": 0.02,
        "world1_force_interval_s": [1, 2],
        "world0_force_x_N": 0,
        "max_world1_x_m": float(np.max(data[:, 5])),
        "max_world0_x_m": float(np.max(np.abs(data[:, 4]))),
        "max_rotor_rpm": float(np.max(data[:, 3])),
        "final_altitude_m": float(data[-1, 8]),
        "final_world1_x_m": float(data[-1, 5]),
        "samples": len(data),
        "scope": (
            "open-loop ideal symmetric rotors; no real-flight comparison or tracking controller"
        ),
    }
    (HERE / "dynamics_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    left = fig.add_subplot(grid[0, 0])
    left.imshow(image0)
    left.axis("off")
    left.set_title("World 0: undisturbed | t = 1.9 s")
    right = fig.add_subplot(grid[0, 1])
    right.imshow(image1)
    right.axis("off")
    right.set_title("World 1: 0.02 N x-force | t = 1.9 s")
    motor = fig.add_subplot(grid[1, 0])
    motor.plot(data[:, 0], data[:, 1], "k--", label="command")
    motor.plot(data[:, 0], data[:, 3], label="actual rotor")
    motor.set(xlabel="Time (s)", ylabel="Speed (RPM)", title="Actuator response")
    motor.legend()
    wind = fig.add_subplot(grid[1, 1])
    wind.plot(data[:, 0], data[:, 2], color="tab:orange")
    wind.set(xlabel="Time (s)", ylabel="Force (N)", title="World 1 applied force")
    xaxis = fig.add_subplot(grid[2, 0])
    xaxis.plot(data[:, 0], data[:, 4], label="world 0")
    xaxis.plot(data[:, 0], data[:, 5], label="world 1")
    xaxis.set(xlabel="Time (s)", ylabel="x (m)", title="Horizontal displacement")
    xaxis.legend()
    zaxis = fig.add_subplot(grid[2, 1])
    zaxis.plot(data[:, 0], data[:, 8], label="world 0")
    zaxis.plot(data[:, 0], data[:, 9], "--", label="world 1")
    zaxis.set(xlabel="Time (s)", ylabel="z (m)", title="Vertical displacement")
    zaxis.legend()
    for ax in (motor, wind, xaxis, zaxis):
        ax.grid(alpha=0.25)
    fig.savefig(HERE / "dynamics_evidence.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
