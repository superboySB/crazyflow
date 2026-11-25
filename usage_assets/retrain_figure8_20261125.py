"""Retrain the existing 3.5-second figure-eight study on the current simulator.

Inside Docker:
    python usage_assets/retrain_figure8_20261125.py --algorithm bptt
    python usage_assets/retrain_figure8_20261125.py --algorithm ppo
    python usage_assets/retrain_figure8_20261125.py --evaluate
Keep historical policies and measurements unchanged. Select using validation only.
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
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import matplotlib.pyplot as plt
import numpy as np
import official_policy_20261015 as official
from official_report_20261015 import verify
from official_stress_20261015 import inject_reset
from train_gymnasium import simulator_source_sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "usage_assets" / "figure8_retrain_20261125"
SOURCE = Path("/tmp/crazyflow_experiments_20261125")
EXPERIMENT_COMMIT = "203ad4e31b9380fec985f17008bb7a68ad5cf768"
BUDGETS = {"bptt": 1_600_000, "ppo": 40_000_000}
SEEDS = (42, 43, 44)
SCENARIOS = (
    ("validation", 9001),
    ("position_holdout", 14001),
    ("velocity_holdout", 14002),
    ("position_unseen", 25001),
    ("velocity_unseen", 25002),
)
OLD_MODELS = {
    "bptt": "official_frontier_bptt_T3.5s_budget1_seed43",
    "ppo": "official_frontier_ppo_T3.5s_budget2_seed43",
}


def digest(path: Path) -> str:
    """Identify input and output artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap() -> dict:
    """Check the immutable current experiment checkout before importing its modules."""
    if not SOURCE.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/learnsyslab/crazyflow_experiments.git",
                str(SOURCE),
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(SOURCE), "fetch", "origin", EXPERIMENT_COMMIT], check=True)
        subprocess.run(
            ["git", "-C", str(SOURCE), "checkout", "--detach", EXPERIMENT_COMMIT], check=True
        )
    head = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != EXPERIMENT_COMMIT:
        raise ValueError(f"Unexpected experiment source: {head}")
    if subprocess.check_output(
        ["git", "-C", str(SOURCE), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Experiment checkout has local changes")
    sys.path.insert(0, str(SOURCE))
    return {
        "date": "2026-11-25",
        "python": platform.python_version(),
        "versions": {
            name: version(name) for name in ("jax", "flax", "optax", "mujoco", "mujoco-mjx")
        },
        "crazyflow_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "upstream_main": subprocess.check_output(
            ["git", "rev-parse", "upstream/main"], text=True
        ).strip(),
        "simulator_source_sha256": simulator_source_sha256(),
        "pixi_lock_sha256": digest(ROOT / "pixi.lock"),
        "experiment_source_sha256": {
            name: digest(SOURCE / "crazyflow_experiments" / name)
            for name in (
                "figure8/train_bptt.py",
                "figure8/train_ppo.py",
                "core/agents/bptt_agent.py",
                "core/agents/ppo_agent.py",
                "core/envs/drone_env.py",
                "core/envs/figure8_env.py",
                "core/envs/wrappers.py",
            )
        },
        "official_experiment_commit": EXPERIMENT_COMMIT,
        "source": "https://github.com/learnsyslab/crazyflow_experiments",
        "devices": [str(device) for device in jax.devices()],
        "protocol": {
            "period_s": 3.5,
            "seeds": list(SEEDS),
            "requested_transitions": BUDGETS,
            "training_devices": {"bptt": "cpu (official preset)", "ppo": "gpu"},
            "evaluation_device": "cpu for every policy",
            "eval_worlds": 64,
            "eval_seconds": 20,
            "policy_hz": 50,
            "physics_hz": 500,
            "selection": "validation seed9001: highest completion, then lowest overall RMSE",
            "test_seeds": {
                "position_historical": 14001,
                "velocity_historical": 14002,
                "position_unseen": 25001,
                "velocity_unseen": 25002,
            },
            "noise": "none; unchanged state observations",
            "budgets_note": (
                "Algorithm-specific historical budgets; not equal-sample or equal-time comparison."
            ),
        },
    }


def assert_runtime(saved: dict, current: dict) -> None:
    """Reject reuse across different simulator, dependency lock or experiment protocols."""
    for key in (
        "simulator_source_sha256",
        "pixi_lock_sha256",
        "official_experiment_commit",
        "experiment_source_sha256",
        "protocol",
    ):
        if saved[key] != current[key]:
            raise ValueError(f"Saved experiment fingerprint differs: {key}")


def module_for(algorithm: str) -> Any:
    """Import the unchanged official algorithm."""
    return importlib.import_module(f"crazyflow_experiments.figure8.train_{algorithm}")


def load_agent(module: Any, args: Any, env: Any, checkpoint: Path, algorithm: str) -> Any:
    """Load architecture-compatible weights without optimizer continuation."""
    if algorithm == "ppo":
        agent = module.load_agent(args, env, checkpoint)
    else:
        agent = module.Agent.create(
            key=jax.random.PRNGKey(0),
            obs_dim=env.single_observation_space.shape[0],
            act_dim=env.single_action_space.shape[0],
            hidden_size=args.hidden_size,
            init_logstd=args.init_logstd,
        )
        payload = pickle.loads(checkpoint.read_bytes())
        agent = agent.replace(actor_states=agent.actor_states.replace(params=payload["actor"]))
    return jax.device_put(agent, jax.devices("cpu")[0])


def evaluate(record: dict, label: str, seed: int) -> dict:
    """Evaluate every world's first episode using the historical masks and tolerances."""
    algorithm = record["algorithm"]
    module = module_for(algorithm)
    args = module.Args(**{**record["config"], "jax_device": "cpu"})
    checkpoint = ROOT / record["checkpoint"]
    if digest(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint changed: {checkpoint.name}")
    path = OUT / f"{checkpoint.stem}_{label}.npz"
    result_path = path.with_suffix(".json")
    if path.exists() and result_path.exists():
        previous = json.loads(result_path.read_text())
        if (
            previous["checkpoint_sha256"] != digest(checkpoint)
            or previous["seed"] != seed
            or previous["scenario"] != label
            or previous["evaluation_device"] != "cpu"
            or previous["raw_sha256"] != digest(path)
        ):
            raise ValueError("Existing evaluation fingerprint differs")
        verify(path, previous["metrics"])
        return previous
    with jax.default_device(jax.devices("cpu")[0]):
        ppo = module_for("ppo")
        env = ppo.make_jitted_envs(
            64,
            module.reward_coefs(args),
            "cpu",
            reset_rotor=True,
            reset_random=label != "validation",
            dr_scale=0.0,
        )
        if label.startswith("velocity_"):
            env = inject_reset(env, "velocity", seed)
        initial, _ = env.reset(env, seed=seed)
        state = initial.unwrapped.data.states
        initial_arrays = {
            "initial_pos": np.asarray(state.pos),
            "initial_vel": np.asarray(state.vel),
            "initial_quat": np.asarray(state.quat),
            "initial_reference": np.asarray(initial.unwrapped.target_pos),
            "initial_reference_vel": np.asarray(initial.unwrapped.target_vel),
        }
        agent = load_agent(module, args, env, checkpoint, algorithm)
        raw = official.rollout(env, agent, seed, 20.0)
    for name, value in {**raw, **initial_arrays}.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite evaluation array: {name}")
    if raw["active"].shape != (1000, 64):
        raise ValueError("Incomplete evaluation population")
    delta = initial_arrays["initial_vel"][:, 0] - initial_arrays["initial_reference_vel"]
    max_delta = float(np.abs(delta).max())
    if label.startswith("velocity_") and not 0.45 < max_delta <= 0.500001:
        raise ValueError("True initial velocity perturbation was overwritten")
    metric = official.metrics(raw, 50)
    np.savez_compressed(path, **raw, **initial_arrays)
    verify(path, metric)
    result = {
        "algorithm": algorithm,
        "checkpoint": record["checkpoint"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "scenario": label,
        "seed": seed,
        "evaluation_device": "cpu",
        "initial_velocity_delta_abs_max_mps": max_delta,
        "metrics": metric,
        "raw": str(path.relative_to(ROOT)),
        "raw_sha256": digest(path),
    }
    official.write_json(path.with_suffix(".json"), result)
    print(
        json.dumps(
            {
                "evaluated": checkpoint.stem,
                "scenario": label,
                "rmse_mm": metric["tracking_rmse_m"] * 1000,
                "complete": round(metric["full_horizon_completion_rate"] * 64),
                "success_5cm": round(metric["success_rate_rmse_5cm"] * 64),
            }
        ),
        flush=True,
    )
    return result


def train(algorithm: str, runtime: dict) -> None:
    """Run three fresh seeds using the full unchanged historical algorithm configuration."""
    module = module_for(algorithm)
    args = module.Args.create("5_5s", trajectory_time=3.5, total_timesteps=BUDGETS[algorithm])
    expected = "cpu" if algorithm == "bptt" else "gpu"
    if args.jax_device != expected:
        raise ValueError("Official training device default changed")
    with jax.default_device(jax.devices(args.jax_device)[0]):
        started = time.perf_counter()
        fn, env, template = official.setup(module, args, algorithm)
        setup_seconds = time.perf_counter() - started
        probe, (probe_obs, _) = env.reset(env, seed=12345)
        probe, (probe_obs, *_) = probe.step(
            probe, jax.numpy.zeros((args.num_envs, env.single_action_space.shape[0]))
        )
        actual_device = {
            "state_shape_after_step": list(probe.unwrapped.data.states.pos.shape),
            "state_device_after_step": str(probe.unwrapped.data.states.pos.device),
            "state_platform": probe.unwrapped.data.states.pos.device.platform,
            "observation_device_after_step": str(probe_obs.device),
            "actor_devices": sorted(
                {str(value.device) for value in jax.tree.leaves(template.actor_states.params)}
            ),
            "one_step_device_probe_discarded": True,
        }
        if actual_device["state_platform"] != expected:
            raise ValueError(f"Simulator is on the wrong device: {actual_device}")
        if actual_device["state_shape_after_step"] != [args.num_envs, 1, 3]:
            raise ValueError("Wrong parallel simulator world count")
        print(json.dumps({"actual_training_devices": actual_device}), flush=True)
        for seed in SEEDS:
            stem = f"figure8_retrain_20261125_{algorithm}_T3.5s_seed{seed}"
            path = OUT / f"{stem}.json"
            checkpoint = ROOT / "trained_models" / f"{stem}.ckpt"
            if path.exists():
                saved = json.loads(path.read_text())
                assert_runtime(saved["runtime"], runtime)
                if digest(checkpoint) != saved["checkpoint_sha256"]:
                    raise ValueError("A saved checkpoint changed")
                print(f"Reuse verified current run: {stem}", flush=True)
                continue
            if checkpoint.exists():
                raise FileExistsError(f"Refuse to overwrite unrecorded weights: {checkpoint}")
            run_args = dataclasses.replace(args, seed=seed)
            print(
                f"Train {stem}: {args.num_iterations * args.batch_size:,} transitions "
                f"on {args.jax_device}, {args.num_envs} parallel worlds",
                flush=True,
            )
            agent, curve, elapsed = official.train_one(
                module, run_args, algorithm, fn, env, template
            )
            for name, value in curve.items():
                if not np.isfinite(value).all():
                    raise FloatingPointError(f"Nonfinite training curve: {name}")
            for value in jax.tree.leaves(agent):
                if not np.isfinite(np.asarray(value)).all():
                    raise FloatingPointError("Nonfinite agent weights or optimizer state")
            payload = {
                "actor": jax.tree.map(np.asarray, agent.actor_states.params),
                "samples_dt": run_args.samples_dt,
            }
            if algorithm == "ppo":
                payload["critic"] = jax.tree.map(np.asarray, agent.critic_states.params)
            checkpoint.write_bytes(pickle.dumps(payload))
            curve_path = OUT / f"{stem}_training.npz"
            np.savez_compressed(curve_path, **curve)
            rewards = curve["mean_step_reward"]
            window = max(1, len(rewards) // 10)
            record = {
                "algorithm": algorithm,
                "seed": seed,
                "from_scratch": True,
                "runtime": runtime,
                "actual_training_devices": actual_device,
                "config": dataclasses.asdict(run_args),
                "requested_transitions": BUDGETS[algorithm],
                "actual_transitions": args.num_iterations * args.batch_size,
                "setup_seconds": setup_seconds,
                "train_seconds_including_compile": elapsed,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": digest(checkpoint),
                "training_curve": str(curve_path.relative_to(ROOT)),
                "training_curve_sha256": digest(curve_path),
                "late_learning": {
                    "last_10pct_reward": float(rewards[-window:].mean()),
                    "prior_10pct_reward": float(rewards[-2 * window : -window].mean()),
                    "last_10pct_minus_prior": float(
                        rewards[-window:].mean() - rewards[-2 * window : -window].mean()
                    ),
                },
            }
            # Persist the full completed budget before any evaluation.
            official.write_json(path, record)
            record["validation"] = evaluate(record, "validation", 9001)
            official.write_json(path, record)
            print(
                json.dumps(
                    {"trained": stem, "seconds": elapsed, "late_learning": record["late_learning"]}
                ),
                flush=True,
            )
            del agent, curve
    env.close()
    del fn, env, template
    jax.clear_caches()
    gc.collect()


def evaluate_selected(runtime: dict) -> None:
    """Freeze validation-selected models before opening either held-out test."""
    selected = {}
    for algorithm in ("bptt", "ppo"):
        records = []
        for seed in SEEDS:
            path = OUT / f"figure8_retrain_20261125_{algorithm}_T3.5s_seed{seed}.json"
            record = json.loads(path.read_text())
            assert_runtime(record["runtime"], runtime)
            if "validation" not in record:
                record["validation"] = evaluate(record, "validation", 9001)
                official.write_json(path, record)
            records.append(record)
        selected[algorithm] = min(
            records,
            key=lambda record: (
                -record["validation"]["metrics"]["full_horizon_completion_rate"],
                record["validation"]["metrics"]["tracking_rmse_m"],
            ),
        )
    official.write_json(
        OUT / "selected_before_test.json",
        {
            "rule": runtime["protocol"]["selection"],
            "checkpoints": {key: record["checkpoint"] for key, record in selected.items()},
            "validation": {key: record["validation"] for key, record in selected.items()},
        },
    )
    comparison = {"runtime": runtime, "policies": {}}
    for family, group in (("retrained", selected), ("frozen", OLD_MODELS)):
        for algorithm, item in group.items():
            record = (
                item
                if family == "retrained"
                else json.loads(
                    (
                        ROOT / "usage_assets" / "official_frontier_20261015" / f"{item}.json"
                    ).read_text()
                )
            )
            results = {}
            for label, seed in SCENARIOS:
                if label == "validation" and family == "retrained":
                    results[label] = record["validation"]
                else:
                    results[label] = evaluate(record, label, seed)
            comparison["policies"][f"{family}_{algorithm}"] = {
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "seed": record["seed"],
                "actual_training_transitions": record["actual_transitions"],
                "scenarios": results,
            }
    official.write_json(OUT / "comparison.json", comparison)
    plot()


def plot() -> None:
    """Plot all independent training curves and the fixed world-zero held-out trajectories."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for column, algorithm in enumerate(("bptt", "ppo")):
        ax = axes[0, column]
        for seed in SEEDS:
            path = OUT / f"figure8_retrain_20261125_{algorithm}_T3.5s_seed{seed}.json"
            record = json.loads(path.read_text())
            with np.load(ROOT / record["training_curve"]) as data:
                reward = data["mean_step_reward"]
            steps = np.arange(1, len(reward) + 1) * record["config"]["batch_size"]
            ax.plot(steps / 1e6, reward, label=f"seed {seed}", linewidth=0.8, alpha=0.8)
        ax.set(
            title=f"{algorithm.upper()} | all fresh seeds",
            xlabel="Transitions (millions)",
            ylabel="Mean step reward",
        )
        ax.legend()
        ax.grid(alpha=0.2)
    comparison = json.loads((OUT / "comparison.json").read_text())
    for column, scenario in enumerate(("position_holdout", "velocity_holdout")):
        ax = axes[1, column]
        for name in ("retrained_bptt", "retrained_ppo", "frozen_ppo"):
            result = comparison["policies"][name]["scenarios"][scenario]
            with np.load(ROOT / result["raw"]) as raw:
                active = raw["active"][:, 0]
                error = np.linalg.norm(raw["pos"][:, 0] - raw["target"][:, 0], axis=-1)
            ax.plot(
                np.arange(len(error))[active] / 50, error[active] * 100, label=name, linewidth=1
            )
        ax.set(title=f"{scenario} | fixed world 0", xlabel="Time (s)", ylabel="Position error (cm)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    figure.savefig(OUT / "learning_and_holdout.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    scenarios = (
        "validation",
        "position_holdout",
        "position_unseen",
        "velocity_holdout",
        "velocity_unseen",
    )
    labels = ("Nominal", "Pos 14001", "Pos 25001", "Pos+vel 14002", "Pos+vel 25002")
    x = np.arange(len(scenarios))
    for offset, name, label in (
        (-0.19, "retrained_bptt", "Retrained BPTT"),
        (0.19, "retrained_ppo", "Retrained PPO"),
    ):
        values = comparison["policies"][name]["scenarios"]
        rmse = [values[key]["metrics"]["tracking_rmse_m"] * 1000 for key in scenarios]
        success = [round(values[key]["metrics"]["success_rate_rmse_5cm"] * 64) for key in scenarios]
        bars = axes[0].bar(x + offset, rmse, width=0.38, label=label)
        axes[0].bar_label(bars, fmt="%.1f", fontsize=8)
        bars = axes[1].bar(x + offset, success, width=0.38, label=label)
        axes[1].bar_label(bars, fontsize=8)
    axes[0].set(title="All 64 worlds | includes failed flights", ylabel="Overall RMSE (mm)")
    axes[1].set(
        title="Complete 20 s and RMSE < 5 cm", ylabel="Successful worlds / 64", ylim=(0, 72)
    )
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    figure.savefig(OUT / "all_world_comparison.png", dpi=150)
    plt.close(figure)


def main() -> None:
    """Write the protocol before training and keep new artifacts isolated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("bptt", "ppo"))
    parser.add_argument("--evaluate", action="store_true")
    cli = parser.parse_args()
    if not cli.algorithm and not cli.evaluate:
        parser.error("Choose --algorithm or --evaluate")
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = bootstrap()
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        assert_runtime(json.loads(protocol_path.read_text()), runtime)
    else:
        official.write_json(protocol_path, runtime)
    if cli.algorithm:
        train(cli.algorithm, runtime)
    if cli.evaluate:
        evaluate_selected(runtime)


if __name__ == "__main__":
    main()
