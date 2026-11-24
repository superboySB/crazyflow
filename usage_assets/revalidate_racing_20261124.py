"""Check the pinned racing integration on CPU after MuJoCo 4 naming changes.

Run inside Docker:
    JAX_PLATFORMS=cpu python usage_assets/revalidate_racing_20261124.py

Uses the original level0 course, two seeded worlds, its 13D StateController interface,
and the historical frozen PPO for both an adapter check and its complete 15 s course.
No training or acados build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from lsy_drone_racing.control.state_controller import StateController
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.utils import load_config

from crazyflow.sim.sim import sync_sim2mjx

ROOT = Path(__file__).resolve().parents[1]
RACING = Path("/opt/lsy_drone_racing")
REF = "065a6ecb7a7621f156bdd0bedc70020e3331a3e9"
OUT = ROOT / "usage_assets" / "upstream_20261124"


def sha256(path: Path) -> str:
    """Identify source patches, configuration, weights and raw trajectories."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_contract() -> dict:
    """Compare floor/gate/free contacts using legacy slot masks and native per-drone filters."""
    config = load_config(RACING / "config" / "level0.toml")
    config.sim.render = False
    env = VecDroneRaceEnv(
        num_envs=3,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        control_mode="attitude",
        seed=123,
        device="cpu",
    )
    try:
        env.reset(seed=123)
        env.sim.contacts()  # The original helper first populated the contact geometry arrays.
        template = env.sim.mjx_data
        original = template._impl.contact
        g1, g2 = np.asarray(original.geom1[0]), np.asarray(original.geom2[0])
        body = env.sim.mj_model.body("0:drone").id
        start, count = env.sim.mj_model.body_geomadr[body], env.sim.mj_model.body_geomnum[body]
        drone = ((g1 >= start) & (g1 < start + count)) | ((g2 >= start) & (g2 < start + count))
        world = env.sim.mj_model.body("world").id
        start, count = env.sim.mj_model.body_geomadr[world], env.sim.mj_model.body_geomnum[world]
        floor = ((g1 >= start) & (g1 < start + count)) | ((g2 >= start) & (g2 < start + count))
        legacy_mask = drone & ~floor
        positions = jnp.array([[[-1.5, 0.75, -0.01]], [[0.5, 0.25, 0.98]], [[-2.0, -1.0, 1.0]]])
        sim_data = env.data.sim_data.replace(
            states=env.data.sim_data.states.replace(pos=positions),
            core=env.data.sim_data.core.replace(mjx_synced=False),
        )
        data = env.data.replace(sim_data=sim_data)
        course_contacts = np.asarray(env.build_contact_check_fn()(data))
        _, synced = sync_sim2mjx(sim_data, template, env.sim.mjx_model)
        unfiltered = np.asarray(env.sim.build_contact_fn()(synced))
        native = np.asarray(env.sim.build_contact_fn(ignore=["world"])(synced))
        touching = np.asarray(synced._impl.contact.dist < 0)
        legacy = np.any(touching & legacy_mask[None], axis=-1)[:, None]
        expected = np.array([[False], [True], [False]])
        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(course_contacts, expected)
        np.testing.assert_array_equal(legacy, expected)
        np.testing.assert_array_equal(unfiltered, [[True], [True], [False]])
        return {
            "scenarios": ["floor penetration", "gate top bar overlap", "free flight"],
            "positions_m": np.asarray(positions).tolist(),
            "native_ignore_world": native.tolist(),
            "native_unfiltered": unfiltered.tolist(),
            "legacy_static_slot_mask": legacy.tolist(),
            "course_contact_check": course_contacts.tolist(),
            "native_matches_legacy_on_these_scenarios": bool(np.array_equal(native, legacy)),
            "slot_geom_ids_constant_across_worlds": bool(
                np.all(
                    np.asarray(synced._impl.contact.geom1)
                    == np.asarray(synced._impl.contact.geom1)[0]
                )
                and np.all(
                    np.asarray(synced._impl.contact.geom2)
                    == np.asarray(synced._impl.contact.geom2)[0]
                )
            ),
            "slot_geom_ids_match_initial_template": bool(
                np.all(np.asarray(synced._impl.contact.geom1) == g1)
                and np.all(np.asarray(synced._impl.contact.geom2) == g2)
            ),
            "scope": (
                "Controlled contract checks; no policy failure is attributed to slot caching."
            ),
        }
    finally:
        env.close()


def state_course(seed: int, output: Path) -> dict:
    """Execute both complete first episodes without resetting finished or failed worlds."""
    config_path = RACING / "config" / "level0.toml"
    config = load_config(config_path)
    config.sim.render = False
    env = VecDroneRaceEnv(
        num_envs=2,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode="state",
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=seed,
        device="cpu",
    )
    try:
        env.settings = env.settings.replace(autoreset=False)
        env._step = env.build_step_fn()
        obs, info = env.reset(seed=seed)

        def host(values: dict, world: int) -> dict:
            return {key: np.asarray(value[world]) for key, value in values.items()}

        controllers = [
            StateController(host(obs, world), host(info, world), config) for world in range(2)
        ]
        active = np.ones(2, dtype=bool)
        gates = np.zeros(2, dtype=int)
        finish_steps = np.zeros(2, dtype=int)
        positions, masks, progress = [], [], []
        for tick in range(int(20 * config.env.freq)):
            action = np.stack(
                [
                    controller.compute_control(host(obs, world), host(info, world))
                    for world, controller in enumerate(controllers)
                ]
            )
            assert action.shape == (2, 13)
            obs, reward, terminated, truncated, info = env.step(jnp.asarray(action))
            position = np.asarray(obs["pos"])
            if not np.isfinite(position).all():
                raise FloatingPointError("Nonfinite racing position")
            positions.append(position.copy())
            masks.append(active.copy())
            progress.append(np.asarray(obs["n_gates_passed"]).copy())
            gates[active] = np.asarray(obs["n_gates_passed"])[active]
            finished = np.asarray(terminated | truncated) | (gates >= 4)
            for world, controller in enumerate(controllers):
                controller_finished = controller.step_callback(
                    action[world],
                    host(obs, world),
                    float(reward[world]),
                    bool(terminated[world]),
                    bool(truncated[world]),
                    host(info, world),
                )
                finished[world] |= controller_finished
            finish_steps[active & finished] = tick + 1
            active &= ~finished
            if not active.any():
                break
        raw = output / "racing_state_cpu.npz"
        np.savez_compressed(
            raw, pos=np.asarray(positions), active=np.asarray(masks), gates=np.asarray(progress)
        )
        return {
            "seed": seed,
            "worlds": 2,
            "device": "cpu",
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "drone": config.sim.drone,
            "dynamics": config.sim.dynamics,
            "physics_frequency_hz": config.sim.freq,
            "policy_frequency_hz": config.env.freq,
            "controller": "unchanged course StateController; 13D legacy to 16D Crazyflow adapter",
            "gates_passed": gates.tolist(),
            "finish_seconds": (finish_steps / config.env.freq).tolist(),
            "successes": int((gates >= 4).sum()),
            "raw": str(raw.relative_to(ROOT)),
            "raw_sha256": sha256(raw),
        }
    finally:
        env.close()


def main() -> None:
    """Verify the fixed source/patches, seeded reset, frozen adapter and full state course."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--require-state-success", action="store_true")
    parser.add_argument("--ppo-worlds", type=int, default=2)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RACING, text=True).strip()
    if head != REF:
        raise ValueError(f"Unexpected racing source: {head}")
    patches = {}
    for name in ("compat", "inertia", "mujoco4"):
        patch = ROOT / ".devcontainer" / f"lsy_drone_racing-{name}.patch"
        subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=RACING, check=True)
        patches[str(patch.relative_to(ROOT))] = sha256(patch)
    import torch
    from racing_fulltrain_20261015 import (
        course_evaluate,
        smoke_current_image,
        verify_inertia_distribution,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    smoke = smoke_current_image()
    checkpoint = ROOT / smoke["checkpoint"]
    smoke["checkpoint_sha256"] = sha256(checkpoint)
    manifest = json.loads((ROOT / "trained_models" / "manifest_20261015.json").read_text())
    assert smoke["checkpoint_sha256"] == manifest["validation_selected_racing"]["sha256"]
    ppo_raw = args.output / "racing_ppo_cpu.npz"
    ppo = course_evaluate(
        checkpoint,
        827451,
        args.ppo_worlds,
        torch.device("cpu"),
        jax_device="cpu",
        save_trajectory=ppo_raw,
    )
    ppo["checkpoint"] = str(checkpoint.relative_to(ROOT))
    ppo["checkpoint_sha256"] = sha256(checkpoint)
    ppo["raw"] = str(ppo_raw.relative_to(ROOT))
    ppo["raw_sha256"] = sha256(ppo_raw)
    with np.load(ppo_raw) as raw:
        for name, values in raw.items():
            if not np.isfinite(values).all():
                raise FloatingPointError(f"Nonfinite racing PPO array: {name}")
    result = {
        "source_commit": head,
        "patch_sha256": patches,
        "versions": {name: version(name) for name in ("jax", "mujoco", "mujoco-mjx", "torch")},
        "smoke": smoke,
        "inertia_distribution": verify_inertia_distribution(),
        "contact_contract": contact_contract(),
        "state_course": state_course(args.seed, args.output),
        "frozen_ppo_course": ppo,
        "scope": (
            "CPU integration regression; complete first episodes of state and frozen PPO; "
            "no retraining or acados/MPC test."
        ),
    }
    target = args.output / "racing_cpu.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    if args.require_state_success:
        assert result["state_course"]["successes"] == 2, (
            "StateController did not finish both four-gate courses"
        )


if __name__ == "__main__":
    main()
