"""Reproduce the CPU contact API and constant-thrust integrator checks.

Run from the repository root with the project installed and JAX_PLATFORM_NAME=cpu.
Writes a JSON record next to this script; no training or historical checkpoint evaluation.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from crazyflow.control import Control
from crazyflow.control.transform import motor_force2rotor_vel
from crazyflow.sim import Sim
from crazyflow.sim.integration import Integrator


def contacts_experiment() -> dict:
    """Measure and assert contacts and body pairs across three worlds."""
    sim = Sim(n_worlds=3, n_drones=3, device="cpu")
    try:
        sim.reset()
        # World 0: 0/1 overlap; world 1: drone 2 penetrates floor;
        # world 2: all drones are separated and above the floor.
        positions = jnp.array(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, -0.05]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
            ]
        )
        sim.data = sim.data.replace(states=sim.data.states.replace(pos=positions))
        sim.data = sim.data.replace(core=sim.data.core.replace(mjx_synced=False))
        body1, body2, touching = sim.contact_bodies()
        pairs = []
        for world in range(sim.n_worlds):
            active = np.flatnonzero(np.asarray(touching[world]))
            pairs.append(
                [
                    [
                        mujoco.mj_id2name(
                            sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, int(body1[world, i])
                        ),
                        mujoco.mj_id2name(
                            sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, int(body2[world, i])
                        ),
                    ]
                    for i in active
                ]
            )
        scenarios = [
            ("default", (), [[True, True, False], [False, False, True], [False, False, False]]),
            (
                "ignore_drone_0_1_pair",
                (("drone:0", "drone:1"),),
                [[False, False, False], [False, False, True], [False, False, False]],
            ),
            (
                "ignore_world",
                ("world",),
                [[True, True, False], [False, False, False], [False, False, False]],
            ),
            (
                "ignore_drone_0",
                ("drone:0",),
                [[False, False, False], [False, False, True], [False, False, False]],
            ),
        ]
        values = {}
        for name, ignored, expected in scenarios:
            sim.build_contact_fn(ignore=ignored)
            actual = np.asarray(sim.contacts()).tolist()
            assert actual == expected, (name, actual, expected)
            values[name] = actual
        pair_sets = [{frozenset(pair) for pair in world} for world in pairs]
        assert pair_sets == [
            {frozenset(("drone:0", "drone:1"))},
            {frozenset(("world", "drone:2"))},
            set(),
        ], pairs
        return {
            "positions_m": np.asarray(positions).tolist(),
            "contact_flags": values,
            "active_body_pairs": pairs,
            "contact_slots_per_world": int(touching.shape[-1]),
        }
    finally:
        sim.close()


def integrator_experiment() -> dict:
    """Record final heights for each integrator under constant thrust."""
    result = {}
    for integrator in Integrator:
        sim = Sim(control=Control.rotor_vel, integrator=integrator, device="cpu")
        try:
            sim.reset()
            thrust = 1.2 * sim.data.params.mass * -sim.data.params.gravity_vec[-1]
            rpm = motor_force2rotor_vel(thrust / 4, sim.data.params.rpm2thrust)
            sim.rotor_vel_control(np.broadcast_to(rpm, (sim.n_worlds, sim.n_drones, 4)))
            start = float(sim.data.states.pos[0, 0, 2])
            steps = int(2.0 * sim.freq)
            sim.step(steps)
            result[integrator.value] = {
                "start_z_m": start,
                "final_z_m": float(sim.data.states.pos[0, 0, 2]),
                "steps": steps,
                "frequency_hz": sim.freq,
            }
            assert np.isfinite(result[integrator.value]["final_z_m"])
        finally:
            sim.close()
    return result


def main() -> None:
    """Run both experiments and save their measured results."""
    output = {
        "method": (
            "3 worlds, 3 drones; 2 s at 1.2 times hover thrust following "
            "examples/control/integrators.py"
        ),
        "platform": platform.platform(),
        "jax": jax.__version__,
        "mujoco": mujoco.__version__,
        "backend": jax.default_backend(),
        "devices": [str(x) for x in jax.devices()],
        "contacts": contacts_experiment(),
        "integrators": integrator_experiment(),
    }
    path = Path(__file__).with_name("upstream_contact_integrator_20261028_results.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
