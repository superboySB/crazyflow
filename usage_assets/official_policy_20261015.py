"""Reproduce official Crazyflow policies at full paper budgets, with held-out evaluation.

Run inside the research Docker:
    python usage_assets/official_policy_20261015.py --algorithm bptt
    python usage_assets/official_policy_20261015.py --algorithm ppo
    python usage_assets/official_policy_20261015.py --algorithm td3
External source is checked against an immutable commit; no hardware deployment is imported.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import importlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import jax.numpy as jp
import numpy as np
import optax

OFFICIAL_COMMIT = "bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3"
OFFICIAL_URL = "https://github.com/learnsyslab/crazyflow_experiments.git"
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "usage_assets"
MODELS = ROOT / "trained_models"
EVAL_SEED = 10001
VALIDATION_SEED = 9001


def write_json(path: Path, data: Any) -> None:
    """Save strict JSON, avoiding non-standard NaN values."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2) + "\n")


def bootstrap(path: Path) -> None:
    """Fetch an immutable official source tree, then verify it before importing."""
    if not path.exists():
        subprocess.run(["git", "clone", OFFICIAL_URL, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", OFFICIAL_COMMIT], check=True)
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if head != OFFICIAL_COMMIT:
        raise ValueError(f"Expected official commit {OFFICIAL_COMMIT}, found {head}")
    if subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Official checkout must have no local changes")
    sys.path.insert(0, str(path))


def reduce_training(module: Any, args: Any, algorithm: str) -> Any:
    """Keep official updates unchanged; reduce scan outputs before materializing them."""
    original = module.make_train_fn(args)
    if algorithm == "td3":

        @jax.jit
        def reduced(env: Any, agent: Any, obs: Any, reward: Any, key: Any, buffer: Any) -> Any:
            agent, (_, data, critic_loss, actor_loss) = original(
                env, agent, obs, reward, key, buffer
            )
            done = data.terminated | data.truncated
            count = jp.maximum(done.sum(axis=(1, 2)), 1)
            curve = {
                "episode_reward": jp.where(done, data.sum_rewards, 0).sum(axis=(1, 2)) / count,
                "episode_count": done.sum(axis=(1, 2)),
                "success_rate": jp.where(done, data.success, 0).sum(axis=(1, 2)) / count,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
            }
            return agent, curve

        return reduced

    @jax.jit
    def reduced(env: Any, agent: Any, key: Any, obs: Any, done: Any, reward: Any) -> Any:
        final, outputs = original(env, agent, key, obs, done, reward)
        data = outputs[0] if algorithm == "ppo" else outputs[1]
        curve = {
            "mean_step_reward": data.rewards.mean(axis=(1, 2)),
            "done_fraction": data.dones.astype(jp.float32).mean(axis=(1, 2)),
        }
        if algorithm == "ppo":
            curve |= {
                "policy_loss": outputs[1].mean(axis=tuple(range(1, outputs[1].ndim))),
                "value_loss": outputs[2].mean(axis=tuple(range(1, outputs[2].ndim))),
            }
        else:
            curve["policy_loss"] = outputs[0].mean(axis=tuple(range(1, outputs[0].ndim)))
        return final[1], curve

    return reduced


def setup(module: Any, args: Any, algorithm: str) -> tuple[Any, Any, Any]:
    """Build the exact upstream optimizer/environment, using reduced output buffers."""
    if algorithm == "td3":
        env = module.make_reach_pos_envs(args.num_envs, args)
        obs_dim = env.single_observation_space.shape[0]
        agent = module.TD3Agent.create(
            key=jax.random.PRNGKey(args.seed + 10000),
            obs_dim=obs_dim,
            act_dim=env.single_action_space.shape[0],
            actor_obs_dim=obs_dim - 4,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            hidden_size_critic=args.hidden_size_critic,
            num_layers_critic=args.num_layers_critic,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
        )
    else:
        extra = {"dr_scale": args.dr_scale} if algorithm == "ppo" else {"reset_random": False}
        env = module.make_jitted_envs(
            num_envs=args.num_envs,
            jax_device=args.jax_device,
            coefs=module.reward_coefs(args),
            reset_rotor=True,
            **extra,
        )
        kwargs = {}
        if algorithm == "ppo":
            n = args.num_iterations * args.update_epochs * args.num_minibatches
            kwargs["critic_lr"] = (
                optax.linear_schedule(args.critic_lr, 0.0, n) if args.anneal_lr else args.critic_lr
            )
            lr = optax.linear_schedule(args.actor_lr, 0.0, n) if args.anneal_lr else args.actor_lr
        else:
            lr = (
                optax.linear_schedule(args.actor_lr, 0.0, args.num_iterations)
                if args.anneal_actor_lr
                else args.actor_lr
            )
        agent = module.Agent.create(
            key=jax.random.PRNGKey(args.seed + 10000),
            obs_dim=env.single_observation_space.shape[0],
            act_dim=env.single_action_space.shape[0],
            hidden_size=args.hidden_size,
            actor_lr=lr,
            init_logstd=args.init_logstd,
            **kwargs,
        )
    return reduce_training(module, args, algorithm), env, agent


def train_one(
    module: Any, args: Any, algorithm: str, fn: Any, env: Any, template: Any
) -> tuple[Any, dict, float]:
    """Use upstream reset, RNG splitting and full budget, without W&B network writes."""
    key = jax.random.PRNGKey(args.seed)
    if algorithm == "td3":
        key, init_key = jax.random.split(key)
    else:
        init_key, key = jax.random.split(key)
    agent = template.reinit(init_key, env.single_observation_space.shape[0])
    fresh, (obs, _) = env.reset(env, seed=args.seed)
    start = time.perf_counter()
    if algorithm == "td3":
        buffer = module.ReplayBuffer.create(
            capacity=args.buffer_size,
            obs_dim=env.single_observation_space.shape[0],
            act_dim=env.single_action_space.shape[0],
        )
        result = fn(fresh, agent, obs, jp.zeros(args.num_envs), key, jax.device_put(buffer))
    else:
        result = fn(
            fresh, agent, key, obs, jp.zeros(args.num_envs, dtype=bool), jp.zeros(args.num_envs)
        )
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    return result[0], jax.tree.map(np.asarray, result[1]), elapsed


def rollout(env: Any, agent: Any, seed: int, seconds: float) -> dict[str, np.ndarray]:
    """Record only each world's first episode; autoreset states do not count as flight."""
    n = env.num_envs
    length = int(seconds * env.unwrapped.freq)

    @jax.jit
    def execute(params: Any) -> Any:
        fresh, (obs, _) = env.reset(env, seed=seed)

        def step(carry: Any, _: Any) -> Any:
            state, observation, active = carry
            action = agent.mean_action(params, observation)
            state, (observation, reward, terminated, truncated, info) = state.step(state, action)
            raw = state.unwrapped
            success = info.get("success", jp.zeros(n, dtype=bool))
            finite = jp.isfinite(raw.data.states.pos[:, 0]).all(axis=-1)
            valid = active & ~(terminated | truncated) & finite
            target = raw.target_pos if hasattr(raw, "target_pos") else raw.goal_pos
            data = (
                raw.data.states.pos[:, 0],
                target,
                action,
                valid,
                active & ((terminated & ~success) | ~finite),
                active & truncated,
                reward,
                active & success,
                raw.data.states.vel[:, 0],
                raw.data.states.quat[:, 0],
            )
            return (state, observation, valid), data

        _, data = jax.lax.scan(step, (fresh, obs, jp.ones(n, dtype=bool)), xs=None, length=length)
        return data

    data = jax.tree.map(np.asarray, execute(agent.actor_states.params))
    env.close()
    return dict(
        zip(
            (
                "pos",
                "target",
                "action",
                "active",
                "terminated",
                "truncated",
                "reward",
                "success",
                "vel",
                "quat",
            ),
            data,
        )
    )


def metrics(raw: dict[str, np.ndarray], frequency: int) -> dict:
    """Report errors, survival and tolerance separately, without hiding failed flights."""
    active = raw["active"]
    error_sq = np.sum((raw["pos"] - raw["target"]) ** 2, axis=-1)
    crash = raw["terminated"].any(axis=0)
    count = active.sum(axis=0)
    per_world = np.sqrt((error_sq * active).sum(axis=0) / np.maximum(count, 1))
    tail = active[-frequency:]
    tail_rmse = np.sqrt(
        (error_sq[-frequency:] * tail).sum(axis=0) / np.maximum(tail.sum(axis=0), 1)
    )
    survived = (~crash) & (count >= active.shape[0] - 1)
    all_samples = active.sum()
    survivor_count = (active * survived[None]).sum()
    return {
        "worlds": int(active.shape[1]),
        "frequency_hz": frequency,
        "mean_tracked_steps": float(count.mean()),
        "crash_rate": float(crash.mean()),
        "full_horizon_completion_rate": float(survived.mean()),
        "alive_without_crash_rate": float((~crash).mean()),
        "goal_success_rate": float(raw["success"].any(axis=0).mean()),
        "mean_first_episode_seconds": float(count.mean() / frequency),
        "tracking_rmse_m": float(np.sqrt((error_sq * active).sum() / max(all_samples, 1))),
        "survivor_tracking_rmse_m": (
            float(np.sqrt((error_sq * active * survived[None]).sum() / survivor_count))
            if survivor_count
            else None
        ),
        "tail_rmse_m": float(tail_rmse[survived].mean()) if survived.any() else None,
        "success_rate_rmse_5cm": float((survived & (per_world < 0.05)).mean()),
        "success_rate_tail_10cm": float((survived & (tail_rmse < 0.10)).mean()),
        "per_world_rmse_m": per_world.tolist(),
    }


def evaluate(module: Any, args: Any, algorithm: str, agent: Any, stem: str, eval_envs: int) -> dict:
    """Evaluate nominal validation and independent perturbation seeds; save raw trajectories."""
    if algorithm == "td3":
        scenarios = [
            ("validation_hover", 0.0),
            ("holdout_hover", 0.0),
            ("holdout_figure8_15s", 15.0),
            ("holdout_figure8_5.5s", 5.5),
            ("holdout_figure8_3.5s", 3.5),
        ]
    else:
        scenarios = [
            ("validation_nominal", 0.0),
            ("holdout_initial_state", 0.0),
            ("holdout_dynamics_0.2", 0.2),
            ("boundary_dynamics_1.0", 1.0),
        ]
    results = {}
    for label, setting in scenarios:
        print(f"Evaluate {stem} {label}", flush=True)
        if algorithm == "td3":
            env = (
                module.make_reach_pos_envs(eval_envs, args)
                if not setting
                else module.make_figure8_env(eval_envs, args, setting)
            )
            seconds, freq = (7.0 if not setting else setting), 100
        else:
            ppo = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
            env = ppo.make_jitted_envs(
                eval_envs,
                module.reward_coefs(args),
                args.jax_device,
                reset_rotor=True,
                reset_random=label != "validation_nominal",
                dr_scale=setting,
            )
            seconds, freq = 20.0, 50
        eval_seed = VALIDATION_SEED if label.startswith("validation_") else EVAL_SEED
        raw = rollout(env, agent, eval_seed, seconds)
        np.savez_compressed(ASSETS / f"{stem}_{label}.npz", **raw)
        results[label] = metrics(raw, freq) | {"seed": eval_seed}
    return results


def main() -> None:
    """Train multiple independent seeds, retaining full budgets and inspectable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=Path("/tmp/crazyflow_experiments"))
    parser.add_argument("--algorithm", choices=("bptt", "ppo", "td3"), required=True)
    parser.add_argument("--presets", nargs="+", default=["10s", "5_5s"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--budget-multiplier", type=int, default=1)
    parser.add_argument("--eval-envs", type=int, default=64)
    parser.add_argument("--eval-only", action="store_true")
    cli = parser.parse_args()
    if cli.budget_multiplier < 1:
        raise ValueError("Only complete official budgets or larger are accepted")
    bootstrap(cli.experiments_dir)
    MODELS.mkdir(exist_ok=True)
    module_name = f"figure8.train_{cli.algorithm}" if cli.algorithm != "td3" else "learn2fly.train"
    module = importlib.import_module(f"crazyflow_experiments.{module_name}")
    presets = cli.presets if cli.algorithm != "td3" else ["rotor_reach"]
    for preset in presets:
        args = module.Args.create(preset) if cli.algorithm != "td3" else module.Args()
        original_budget = args.total_timesteps
        total = original_budget * cli.budget_multiplier
        args = (
            module.Args.create(preset, total_timesteps=total)
            if cli.algorithm != "td3"
            else dataclasses.replace(args, total_timesteps=total)
        )
        title = f"official_{cli.algorithm}_{preset}_budget{cli.budget_multiplier}_20261015"
        records = []
        setup_start = time.perf_counter()
        fn, env, template = setup(module, args, cli.algorithm)
        setup_seconds = time.perf_counter() - setup_start
        for seed in cli.seeds:
            stem = f"{title}_seed{seed}"
            model_path = MODELS / f"{stem}.ckpt"
            run_args = dataclasses.replace(args, seed=seed)
            print(f"Start {stem}, requested transitions={total}", flush=True)
            if cli.eval_only:
                payload = pickle.loads(model_path.read_bytes())
                agent = template.replace(
                    actor_states=template.actor_states.replace(params=payload["actor"])
                )
                previous = json.loads((ASSETS / f"{stem}.json").read_text())
                curve, duration = {}, previous["train_seconds_including_first_compile"]
                setup_seconds = previous["setup_seconds"]
            else:
                agent, curve, duration = train_one(
                    module, run_args, cli.algorithm, fn, env, template
                )
                payload = {"actor": jax.tree.map(np.asarray, agent.actor_states.params)}
                if cli.algorithm == "ppo":
                    payload["critic"] = jax.tree.map(np.asarray, agent.critic_states.params)
                payload["samples_dt"] = getattr(args, "samples_dt", None)
                model_path.write_bytes(pickle.dumps(payload))
                np.savez_compressed(ASSETS / f"{stem}_training.npz", **curve)
            actual = (
                args.num_iterations * args.batch_size
                if cli.algorithm != "td3"
                else args.learning_starts + args.num_iterations * args.rollout_batchsize
            )
            record = {
                "algorithm": cli.algorithm,
                "preset": preset,
                "seed": seed,
                "budget_multiplier": cli.budget_multiplier,
                "requested_transitions": total,
                "actual_transitions": actual,
                "warmup_training_discarded": False,
                "setup_seconds": setup_seconds,
                "train_seconds_including_first_compile": duration,
                "official_commit": OFFICIAL_COMMIT,
                "official_source": OFFICIAL_URL,
                "crazyflow_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "config": dataclasses.asdict(run_args),
                "control": "rotor_vel" if cli.algorithm == "td3" else "attitude",
                "dynamics": "first_principles",
                "drone": "cf2x_L250" if cli.algorithm == "td3" else "cf21B_500",
                "policy_frequency_hz": 100 if cli.algorithm == "td3" else 50,
                "physics_frequency_hz": 500,
                "checkpoint": str(model_path.relative_to(ROOT)),
                "checkpoint_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "evaluation": evaluate(module, run_args, cli.algorithm, agent, stem, cli.eval_envs),
            }
            write_json(ASSETS / f"{stem}.json", record)
            records.append(record)
            write_json(ASSETS / f"{title}_results.json", records)
            print(
                json.dumps(
                    {
                        k: {m: v for m, v in x.items() if m != "per_world_rmse_m"}
                        for k, x in record["evaluation"].items()
                    }
                ),
                flush=True,
            )
            del agent, curve
            gc.collect()
        env.close()
        del fn, env, template
        jax.clear_caches()
        gc.collect()


if __name__ == "__main__":
    main()
