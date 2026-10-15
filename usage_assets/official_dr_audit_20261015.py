"""Audit physical constraints of the exact official DR draws used in held-out evaluation."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import official_policy_20261015 as experiment


def main() -> None:
    """Check mass positivity, inertia SPD/triangle inequalities and sampled thrust curves."""
    experiment.bootstrap(Path("/tmp/crazyflow_experiments"))
    module = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
    drone_env = importlib.import_module("crazyflow_experiments.core.envs.drone_env")
    args = module.Args.create("5_5s", jax_device="cpu")
    output = {}
    for scale in (0.0, 0.2, 1.0):
        env = module.make_jitted_envs(
            64,
            module.reward_coefs(args),
            "cpu",
            reset_rotor=True,
            reset_random=True,
            dr_scale=scale,
        )
        env, _ = env.reset(env, seed=10001)
        params = env.unwrapped.data.params
        mass = np.broadcast_to(np.asarray(params.mass), (64, 1, 1))
        inertia = np.broadcast_to(np.asarray(params.J), (64, 1, 3, 3))
        eigenvalues = np.linalg.eigvalsh(inertia)
        triangle_margin = eigenvalues[..., :2].sum(axis=-1) - eigenvalues[..., 2]
        curve = np.broadcast_to(np.asarray(params.rpm2thrust), (64, 1, 4, 3))
        bounds = drone_env.create_action_space("rotor_vel", "cf21B_500")
        rpm = np.linspace(float(bounds.low[0]), float(bounds.high[0]), 256)
        thrust = curve[..., 0, None] + curve[..., 1, None] * rpm + curve[..., 2, None] * rpm**2
        derivative = curve[..., 1, None] + 2 * curve[..., 2, None] * rpm
        result = {
            "seed": 10001,
            "worlds": 64,
            "dr_scale": scale,
            "mass_min_kg": float(mass.min()),
            "mass_max_kg": float(mass.max()),
            "inertia_min_eigenvalue": float(eigenvalues.min()),
            "inertia_triangle_violation_worlds": int((triangle_margin < 0).any(axis=-1).sum()),
            "inertia_triangle_min_margin_kg_m2": float(triangle_margin.min()),
            "inertia_triangle_valid_world_indices": np.flatnonzero(
                (triangle_margin >= 0).all(axis=-1)
            ).tolist(),
            "sampled_rpm_range": [float(rpm[0]), float(rpm[-1])],
            "rpm_grid_samples": len(rpm),
            "negative_thrust_worlds_in_sampled_rpm_range": int(
                (thrust < 0).any(axis=(-1, -2, -3)).sum()
            ),
            "negative_thrust_derivative_worlds_in_sampled_rpm_range": int(
                (derivative < 0).any(axis=(-1, -2, -3)).sum()
            ),
            "minimum_thrust_in_sampled_rpm_range_N": float(thrust.min()),
            "official_commit": experiment.OFFICIAL_COMMIT,
        }
        np.savez_compressed(
            experiment.ASSETS / f"official_dr_physical_audit_20261015_scale{scale:g}.npz",
            mass=mass,
            inertia=inertia,
            rpm2thrust=curve,
            valid_inertia=(triangle_margin >= 0).all(axis=-1),
            inertia_triangle_margin=triangle_margin,
        )
        output[str(scale)] = result
        print(result)
        env.close()
    experiment.write_json(experiment.ASSETS / "official_dr_physical_audit_20261015.json", output)


if __name__ == "__main__":
    main()
