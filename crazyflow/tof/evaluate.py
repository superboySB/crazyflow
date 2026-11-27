"""Frozen first-episode evaluation of a Crazyflow ToF policy.

All worlds remain in the denominator. The environment may automatically reset a
finished world, but this evaluator records only its first episode and uses the
terminal pose supplied before that reset. MuJoCo is used separately for rendering;
this evaluator always advances the actual Crazyflow environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from crazyflow.tof.env import ToFEnv


def summarize_episodes(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool episode outcomes, including failures, with a Wilson binomial interval."""
    if not records:
        raise ValueError("at least one episode is required")
    count = len(records)
    successes = sum(record["outcome"] == "success" for record in records)
    rate = successes / count
    z = 1.959963984540054
    center = (rate + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * np.sqrt(rate * (1 - rate) / count + z * z / (4 * count**2))
    radius /= 1 + z * z / count
    finite_distances = [
        record["final_distance_m"] for record in records if record["final_distance_m"] is not None
    ]
    return {
        "episodes": count,
        "successes": successes,
        "success_rate": rate,
        "success_rate_wilson_95": [max(0.0, center - radius), min(1.0, center + radius)],
        "collisions": sum(record["outcome"] == "collision" for record in records),
        "timeouts": sum(record["outcome"] == "timeout" for record in records),
        "other_failures": sum(
            record["outcome"] not in ("success", "collision", "timeout") for record in records
        ),
        "mean_return": float(np.mean([record["return"] for record in records])),
        "mean_duration_s": float(np.mean([record["duration_s"] for record in records])),
        "mean_final_distance_m": float(np.mean(finite_distances)) if finite_distances else None,
        "per_episode": records,
    }


def rollout(
    actor: Callable[[Any], Any],
    env: ToFEnv,
    *,
    seed: int = 42,
    max_steps: int | None = None,
    record_trajectories: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one independent initial episode per batched world without replacement."""
    import jax
    import jax.numpy as jnp

    if max_steps is None:
        max_steps = env.config.max_episode_steps
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    state = env.reset(jax.random.key(seed))
    initial = jax.device_get(state)
    positions = np.asarray(initial.sim.states.pos[:, 0], dtype=np.float32)
    quaternions = np.asarray(initial.sim.states.quat[:, 0], dtype=np.float32)
    goals = np.asarray(initial.goal, dtype=np.float32)
    count = len(positions)
    active = np.ones(count, dtype=bool)
    returns = np.zeros(count, dtype=np.float64)
    lengths = np.zeros(count, dtype=np.int64)
    path_length = np.zeros(count, dtype=np.float64)
    final_positions = positions.copy()
    outcomes = np.full(count, "timeout", dtype="U16")
    position_history = [positions.copy()]
    quaternion_history = [quaternions.copy()]
    action_history = []
    depths = [np.asarray(initial.depth, dtype=np.float32)]
    actor_jit = jax.jit(actor)
    step_jit = jax.jit(env.step)
    for _ in range(max_steps):
        actions = actor_jit(state.obs)
        if np.shape(actions) != (count, 4):
            raise ValueError(f"actor must return {(count, 4)} actions, got {np.shape(actions)}")
        finite_actions = np.all(np.isfinite(np.asarray(jax.device_get(actions))), axis=1)
        # An invalid actor is an explicit failure. Keep other worlds evaluable.
        invalid_action = active & ~finite_actions
        outcomes[invalid_action] = "nonfinite"
        active[invalid_action] = False
        actions = jnp.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
        next_state, result = step_jit(state, actions)
        # Transfer only scoring/rendering fields. The full batched simulator,
        # observation history and actor/critic inputs stay on their JAX device.
        (
            next_positions,
            next_quaternions,
            rewards,
            done,
            terminated,
            success,
            collision,
            terminal_pos,
            terminal_quat,
            terminal_depth,
        ) = jax.device_get(
            (
                next_state.sim.states.pos[:, 0],
                next_state.sim.states.quat[:, 0],
                result.reward,
                result.done,
                result.terminated,
                result.success,
                result.collision,
                result.terminal_pos,
                result.terminal_quat,
                result.terminal_critic_obs[:, -64:] / 30,
            )
        )
        done = np.asarray(done, dtype=bool)
        next_positions = np.where(done[:, None], terminal_pos, next_positions)
        next_quaternions = np.where(done[:, None], terminal_quat, next_quaternions)
        finite = np.isfinite(next_positions).all(1) & np.isfinite(next_quaternions).all(1)
        invalid_state = active & ~finite
        rewards = np.asarray(rewards, dtype=float)
        invalid_reward = active & ~np.isfinite(rewards)
        invalid = invalid_state | invalid_reward
        outcomes[invalid] = "nonfinite"
        returns += np.where(active & np.isfinite(rewards), rewards, 0.0)
        lengths += active
        displacement = np.linalg.norm(next_positions - positions, axis=1)
        path_length += np.where(active & np.isfinite(displacement), displacement, 0.0)
        positions = np.where(active[:, None], next_positions, positions)
        quaternions = np.where(active[:, None], next_quaternions, quaternions)
        final_positions = np.where(active[:, None], next_positions, final_positions)
        ended = active & done & ~invalid
        # Physical failure takes priority if success and collision coincide.
        collision = np.asarray(collision, dtype=bool)
        success = np.asarray(success, dtype=bool)
        outcomes[ended & collision] = "collision"
        outcomes[ended & ~collision & success] = "success"
        outcomes[ended & ~collision & ~success & np.asarray(terminated)] = "unstable"
        active &= ~(done | invalid)
        if record_trajectories:
            position_history.append(positions.copy())
            quaternion_history.append(quaternions.copy())
            action_history.append(np.asarray(jax.device_get(actions), dtype=np.float32))
            # The final critic depth is noise-free metres times 30 and belongs
            # to the terminal pose before autoreset. Keep one packet per pose.
            depths.append(np.asarray(terminal_depth, dtype=np.float32))
        state = next_state
        if not active.any():
            break
    dt = env.config.dt
    records = []
    for index in range(count):
        distance = float(np.linalg.norm(final_positions[index] - goals[index]))
        records.append(
            {
                "episode": index,
                "seed": seed,
                "map_id": int(np.asarray(initial.map_id)[index]),
                "outcome": str(outcomes[index]),
                "return": float(returns[index]),
                "steps": int(lengths[index]),
                "duration_s": float(lengths[index] * dt),
                "path_length_m": float(path_length[index]),
                "initial_distance_m": float(
                    np.linalg.norm(position_history[0][index] - goals[index])
                ),
                "final_distance_m": distance if np.isfinite(distance) else None,
            }
        )
    report = summarize_episodes(records)
    report.update(
        {
            "backend": "crazyflow",
            "seed": seed,
            "max_steps": max_steps,
            "policy_mode": "deterministic_raw_gaussian_mean",
            "evaluation_protocol": "first_episode_per_world_no_replacement",
            "env_config": asdict(env.config),
        }
    )
    trajectory = {
        "positions": np.stack(position_history),
        "quaternions_xyzw": np.stack(quaternion_history),
        "goals": goals,
        "map_ids": np.asarray(initial.map_id),
        "lengths": lengths,
        "dt": np.asarray(dt),
        "scene_centers": np.asarray(jax.device_get(env.scene.centers)),
        "scene_half_sizes": np.asarray(jax.device_get(env.scene.half_sizes)),
        "scene_kinds": np.asarray(jax.device_get(env.scene.kinds)),
        "scene_rotations": np.asarray(jax.device_get(env.scene.rotations)),
        "front_ray_directions": np.asarray(jax.device_get(env.ray_directions)),
        "room_half_size": np.asarray(env.scene.room_half_size),
        "floor_height": np.asarray(getattr(env.scene, "floor", 0.01)),
        "ceiling_height": np.asarray(getattr(env.scene, "ceiling", 2.0)),
        "scene_counts": np.asarray(
            getattr(
                env.scene, "counts", [[env.scene.centers.shape[1], 0]] * env.scene.centers.shape[0]
            )
        ),
    }
    if record_trajectories and action_history:
        trajectory["actions"] = np.stack(action_history)
        trajectory["front_depth_m"] = np.stack(depths)
    return report, trajectory


def save_evaluation(
    report: dict[str, Any], trajectories: dict[str, np.ndarray], output: Path
) -> None:
    """Save inspectable metrics, row-oriented episode records and lossless trajectories."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["per_episode"][0]))
        writer.writeheader()
        writer.writerows(report["per_episode"])
    np.savez_compressed(output / "trajectories.npz", **trajectories)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exported policy.npz")
    parser.add_argument("--config", type=Path, help="Optional task configuration override")
    parser.add_argument("--episodes", type=int, default=256, help="One initial episode per world")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--map-seed",
        type=int,
        help="Generate new maps with this seed; preserve other checkpoint task/physics settings",
    )
    parser.add_argument(
        "--max-steps", type=int, help="Debug horizon; shorter runs count as timeouts"
    )
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--output", type=Path, default=Path("outputs/tof/evaluation"))
    return parser


def load_actor_environment(args: argparse.Namespace) -> tuple[Callable, ToFEnv, dict[str, Any]]:
    """Construct an evaluator from the exported task and actor metadata."""
    from crazyflow.tof.config import EnvConfig, env_config, load_config
    from crazyflow.tof.env import ToFEnv
    from crazyflow.tof.policy import act, load_policy

    metadata, params = load_policy(args.checkpoint)
    if args.config:
        loaded = load_config(args.config)
        cfg = env_config(loaded.get("environment", {}))
    else:
        cfg = EnvConfig(**metadata["env_config"])
    if args.map_seed is not None:
        cfg = replace(cfg, map_seed=args.map_seed)
    env = ToFEnv(cfg, num_envs=args.episodes, device=args.device)
    return lambda obs: act(params, obs), env, metadata


def main() -> None:
    """Run a fixed evaluation and report the all-episode success rate."""
    args = build_parser().parse_args()
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    actor, env, metadata = load_actor_environment(args)
    report, trajectories = rollout(actor, env, seed=args.seed, max_steps=args.max_steps)
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    report["checkpoint_metadata"] = metadata
    save_evaluation(report, trajectories, args.output)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("per_episode", "env_config", "checkpoint_metadata")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
