"""Check that official TD3 reset goals, position and velocity use independent RNG draws.

The actual task has a fixed goal. A separate diagnostic only expands goal bounds to detect
key reuse; no policy is evaluated on that diagnostic and no 3D-goal performance is claimed.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import official_policy_20261015 as experiment


def main() -> None:
    """Check 1024 worlds at an independent audit seed and save exact reset populations."""
    experiment.bootstrap(Path("/tmp/crazyflow_experiments"))
    module = importlib.import_module("crazyflow_experiments.learn2fly.train")
    output = {}
    for random_goals in (False, True):
        kwargs = (
            {"goal_pmin": (-0.5, -0.5, 1.2), "goal_pmax": (0.5, 0.5, 1.8)} if random_goals else {}
        )
        args = module.Args(num_envs=1024, **kwargs)
        env = module.make_reach_pos_envs(1024, args)
        env, _ = env.reset(env, seed=13001)
        pos = np.asarray(env.unwrapped.data.states.pos)[:, 0]
        vel = np.asarray(env.unwrapped.data.states.vel)[:, 0]
        goal = np.asarray(env.unwrapped.goal_pos)
        p = (pos - np.array(args.pos_min)) / (np.array(args.pos_max) - np.array(args.pos_min))
        v = (vel - args.vel_min) / (args.vel_max - args.vel_min)
        result = {
            "seed": 13001,
            "worlds": 1024,
            "diagnostic_random_goals": random_goals,
            "official_commit": experiment.OFFICIAL_COMMIT,
            "pos_std_xyz": pos.std(axis=0).tolist(),
            "vel_std_xyz": vel.std(axis=0).tolist(),
            "goal_std_xyz": goal.std(axis=0).tolist(),
            "goal_minus_pos_std_xyz": (goal - pos).std(axis=0).tolist(),
            "normalized_pos_velocity_correlation": float(np.corrcoef(p.ravel(), v.ravel())[0, 1]),
        }
        if random_goals:
            g = (goal - np.array(args.goal_pmin)) / (
                np.array(args.goal_pmax) - np.array(args.goal_pmin)
            )
            result["normalized_pos_goal_correlation"] = float(
                np.corrcoef(p.ravel(), g.ravel())[0, 1]
            )
            if abs(result["normalized_pos_goal_correlation"]) >= 0.1:
                raise ValueError("Goal and initial position draws appear correlated")
        if abs(result["normalized_pos_velocity_correlation"]) >= 0.1:
            raise ValueError("Position and velocity draws appear correlated")
        label = "diagnostic_random_goals" if random_goals else "actual_fixed_goal_task"
        output[label] = result
        np.savez_compressed(
            experiment.ASSETS / f"official_rng_audit_20261015_{label}.npz",
            pos=pos,
            vel=vel,
            goal=goal,
            normalized_pos=p,
            normalized_vel=v,
        )
        print(result)
        env.close()
    experiment.write_json(experiment.ASSETS / "official_rng_audit_20261015.json", output)


if __name__ == "__main__":
    main()
