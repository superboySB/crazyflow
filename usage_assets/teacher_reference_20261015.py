"""Evaluate the calibrated analytic PD teacher as a control reference, without training.

This reference is not a neural policy or checkpoint candidate. It uses the same official
ReachPos/Landing tasks, independent goal stream, first-episode metrics, and 3 x 1024 held-out
cases as the frozen neural-policy study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from fast_gymnasium_20261015 import make_fused_env
from train_gymnasium import (
    RESET_PROTOCOL,
    Config,
    evaluate,
    observation_keys,
    position_teacher_from_flat_obs,
    reset_env,
    task_error,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "usage_assets/fulltrain_20261015"
TASKS = {"reach_pos": "DroneReachPos-v0", "landing": "DroneLanding-v0"}
SEEDS = (31001, 31002, 31003)


class AnalyticTeacher:
    """Expose an analytic controller through the evaluator's deterministic policy protocol."""

    def get_action_and_value(
        self, obs: torch.Tensor, deterministic: bool = True
    ) -> tuple[torch.Tensor, None, None, None]:
        """Compute calibrated PD commands directly; no weights or optimization are involved."""
        if not deterministic:
            raise ValueError("The analytic reference is deterministic")
        return position_teacher_from_flat_obs(obs, "calibrated"), None, None, None


def digest(path: Path) -> str:
    """Return an artifact's SHA256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomically save strict JSON."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def evaluate_task(task: str, output: Path) -> dict:
    """Keep every first episode, save raw per-world measurements, and recompute aggregate scores."""
    config = Config(env_id=TASKS[task], eval_envs=1024, teacher_model="calibrated")
    env = make_fused_env(config, config.eval_envs)
    rows = []
    squares = last_squares = 0.0
    observed = last_observed = successes = survivors = 0
    try:
        for seed in SEEDS:
            initial, _ = reset_env(env, seed)
            initial_goals = (
                (initial["pos"] + task_error(initial, config.env_id)).cpu().numpy().copy()
            )
            metrics, raw = evaluate(
                AnalyticTeacher(), config, observation_keys(config.env_id), eval_seed=seed, env=env
            )
            # This field is deliberately NaN for non-velocity tasks in the general evaluator.
            # It is inapplicable here; all retained raw float measurements must be finite.
            raw.pop("initial_target_vel")
            raw["initial_target_pos"] = initial_goals
            np.testing.assert_array_equal(initial["pos"].cpu().numpy(), raw["initial_pos"])
            assert all(np.isfinite(value).all() for value in raw.values())
            lengths = raw["per_env_episode_steps"].astype(int)
            rebuilt = (
                raw["per_env_survival"]
                & (lengths == round(config.episode_time * config.freq))
                & (raw["per_env_last_rmse"] < 0.1)
            )
            if task == "landing":
                rebuilt &= raw["per_env_last_speed_rmse"] < 0.1
            np.testing.assert_array_equal(rebuilt, raw["per_env_success"])
            assert lengths.shape == (1024,)
            squared = float(np.sum(raw["per_env_rmse"].astype(float) ** 2 * lengths))
            np.testing.assert_allclose(
                np.sqrt(squared / lengths.sum()), metrics["rmse"], rtol=3e-6, atol=3e-7
            )
            np.testing.assert_allclose(rebuilt.mean(), metrics["success_rate"], atol=1e-8)
            path = output / f"teacher_reference_{task}_test{seed}.npz"
            np.savez_compressed(path, **raw)
            tail_lengths = np.minimum(lengths, config.freq)
            squares += squared
            observed += int(lengths.sum())
            last_squares += float(
                np.sum(raw["per_env_last_rmse"].astype(float) ** 2 * tail_lengths)
            )
            last_observed += int(tail_lengths.sum())
            won, alive = int(rebuilt.sum()), int(raw["per_env_survival"].sum())
            successes += won
            survivors += alive
            rows.append(
                {
                    "seed": seed,
                    "episodes": 1024,
                    "successes": won,
                    "survivors": alive,
                    "metrics": metrics,
                    "raw_per_world": str(path.relative_to(ROOT)),
                    "raw_sha256": digest(path),
                }
            )
            print(task, seed, f"success={won}/1024", f"RMSE={metrics['rmse']:.6f}", flush=True)
    finally:
        env.close()
    report = {
        "schema": 1,
        "kind": "analytic_controller_reference",
        "controller": {
            "name": "calibrated PD teacher",
            "teacher_model": "calibrated",
            "desired_acceleration": (
                "4 * position_error - 4 * velocity, component-clipped to +/-4 m/s^2"
            ),
            "action_mapping": (
                "Exact steady-state inverse of fitted L250 so_rpy thrust/attitude gains."
            ),
            "training_samples": 0,
            "neural_policy": False,
        },
        "eligible_for_neural_checkpoint_selection": False,
        "protocol": "Analytic control reference only. No training and no checkpoint selection.",
        "goal_rng_protocol": RESET_PROTOCOL,
        "env_backend": "fused",
        "torch_matmul_allow_tf32": True,
        "config": {
            key: value
            for key, value in asdict(config).items()
            if key in ("env_id", "freq", "episode_time", "eval_envs", "teacher_model")
        },
        "seed_list": list(SEEDS),
        "populations": rows,
        "aggregate": {
            "episodes": 3072,
            "successes": successes,
            "survivors": survivors,
            "success_rate": successes / 3072,
            "survival_rate": survivors / 3072,
            "first_episode_sample_weighted_rmse": np.sqrt(squares / observed),
            "last_observed_second_sample_weighted_rmse": np.sqrt(last_squares / last_observed),
            "first_episode_observed_steps": observed,
        },
        "scope": (
            "Official fitted-attitude model, near-ground hover for Landing. "
            "This analytic controller reference is not evidence of neural-policy performance, "
            "direct rotor control, fixed-wing flight, or real-world transfer."
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)): digest(path)
            for path in (
                Path(__file__),
                ROOT / "usage_assets/train_gymnasium.py",
                ROOT / "usage_assets/fast_gymnasium_20261015.py",
                ROOT / "crazyflow/dynamics/so_rpy/dynamics.py",
            )
        },
    }
    write_json(output / f"teacher_reference_{task}.json", report)
    return report


def main() -> None:
    """Run the two analytic reference tasks without touching trained policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", choices=TASKS, nargs="+", default=list(TASKS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        evaluate_task(task, OUT)


if __name__ == "__main__":
    main()
