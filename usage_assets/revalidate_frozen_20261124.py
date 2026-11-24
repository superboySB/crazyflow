"""Re-evaluate preselected frozen research policies after the 2026-11-24 upstream merge.

Inside Docker, with the research CUDA environment active:
    python usage_assets/revalidate_frozen_20261124.py --suite gym
    python usage_assets/revalidate_frozen_20261124.py --suite official

No training or model selection occurs. Original configurations and historical artifacts stay
unchanged; complete first-episode populations and current measurements are saved separately.
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
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")

import jax
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "usage_assets" / "frozen_revalidation_20261124"
GYM_MODELS = {
    "reach_pos": "gym_reach_pos_s19_g0.99_200000000_calibrated.pt",
    "reach_vel": "gym_reach_vel_s7_g0.99_200000000.pt",
    "landing": "gym_landing_s43_g0.99_200000000_calibrated.pt",
    "figure8": "gym_figure8_s43_g0.99_200000000.pt",
}
OFFICIAL_MODELS = {
    "ppo": "official_frontier_ppo_T3.5s_budget2_seed43.ckpt",
    "bptt": "official_frontier_bptt_T3.5s_budget1_seed43.ckpt",
    "td3": "official_td3_rotor_reach_budget2_20261015_seed42.ckpt",
}


def sha256(path: Path) -> str:
    """Identify immutable input and newly produced output artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, record: dict) -> None:
    """Write strict JSON atomically, including Python scalars rather than array objects."""

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite JSON measurement: {value}")
        return value

    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime() -> dict:
    """Record source identity even while a merge is being validated before its commit."""
    from train_gymnasium import simulator_source_sha256

    def ref(name: str) -> str:
        return subprocess.check_output(["git", "rev-parse", name], cwd=ROOT, text=True).strip()

    return {
        "date": "2026-11-24",
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("jax", "mujoco", "mujoco-mjx", "torch")},
        "jax_devices": [str(device) for device in jax.devices()],
        "torch_device": torch.cuda.get_device_name(0),
        "checkout_head_at_execution": ref("HEAD"),
        "upstream_main": ref("upstream/main"),
        "crazyflow_source_sha256": simulator_source_sha256(),
        "torch_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "purpose": "Frozen-policy compatibility measurement; no training or new model selection.",
    }


def assert_finite_metrics(metrics: dict, nullable: tuple[str, ...] = ()) -> None:
    """Reject nonfinite metrics and unexplained empty populations."""
    for name, value in metrics.items():
        if value is None:
            if name not in nullable:
                raise FloatingPointError(f"Unexpected undefined metric: {name}")
        elif not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError(f"Nonfinite metric: {name}")


def verify_checkpoint(path: Path, expected: str) -> None:
    """Reject altered weights instead of evaluating a checkpoint under a historical name."""
    if sha256(path) != expected:
        raise ValueError(f"Checkpoint SHA256 differs from its historical record: {path}")


def evaluate_gym(task: str, output: Path, environment: dict) -> dict:
    """Reuse the study's fused backend and exact first-episode evaluation definition."""
    import fast_gymnasium_20261015 as fast
    import train_gymnasium as trainer

    checkpoint = ROOT / "trained_models" / GYM_MODELS[task]
    tag = checkpoint.stem.removeprefix("gym_")
    historical_path = ROOT / "usage_assets" / "fulltrain_20261015" / f"{tag}.json"
    historical = json.loads(historical_path.read_text())
    verify_checkpoint(checkpoint, historical["checkpoint_sha256"])
    agent, config, keys, _ = trainer.load_checkpoint(checkpoint)
    original_config = dataclasses.asdict(config)
    config = dataclasses.replace(config, eval_envs=1024)
    trainer.make_env = fast.make_fused_env
    metrics, behavior = trainer.evaluate(agent, config, keys, eval_seed=31001)
    assert_finite_metrics(metrics)
    for name, values in behavior.items():
        # This established field is a NaN placeholder for tasks without a velocity target.
        if name == "initial_target_vel" and task != "reach_vel":
            continue
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite Gym evaluation array: {name}")
    raw_path = output / f"gym_{task}.npz"
    np.savez_compressed(raw_path, **behavior)
    reference = next(test for test in historical["tests"] if test["seed"] == 31001)
    result = {
        "family": "gym",
        "task": task,
        "runtime": environment,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "historical_record": str(historical_path.relative_to(ROOT)),
        "historical_record_sha256": sha256(historical_path),
        "training_config": original_config,
        "evaluation_config": dataclasses.asdict(config),
        "observation_keys": keys,
        "seed": 31001,
        "worlds": 1024,
        "seconds": config.episode_time,
        "execution_backend": "fused official NEXT_STEP environment",
        "evaluation_definition": trainer.evaluate.__doc__,
        "metrics": metrics,
        "historical_same_seed_metrics": reference["metrics"],
        "raw": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256(raw_path),
        "successes": int(behavior["per_env_success"].sum()),
        "survivors": int(behavior["per_env_survival"].sum()),
    }
    save_json(output / f"gym_{task}.json", result)
    return result


def load_official_agent(module: Any, args: Any, env: Any, checkpoint: Path, algorithm: str) -> Any:
    """Use each historical policy architecture without loading optimizer/simulator state."""
    if algorithm == "td3":
        agent = module.load_actor(checkpoint)
    elif algorithm == "ppo":
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
    return jax.device_put(agent, jax.devices(args.jax_device)[0])


def evaluate_official(algorithm: str, output: Path, environment: dict, source: Path) -> dict:
    """Reuse immutable experiment source, original configuration and predetermined held-out seed."""
    import official_policy_20261015 as official

    official.bootstrap(source)
    checkpoint = ROOT / "trained_models" / OFFICIAL_MODELS[algorithm]
    if algorithm == "td3":
        historical_path = ROOT / "usage_assets" / f"{checkpoint.stem}.json"
        raw_historical = historical_path.with_name(f"{checkpoint.stem}_holdout_hover.npz")
        seed, seconds, frequency = 10001, 7.0, 100
        label = "holdout_hover"
    else:
        directory = ROOT / "usage_assets" / "official_frontier_20261015"
        historical_path = directory / f"{checkpoint.stem}.json"
        raw_historical = directory / f"{checkpoint.stem}_test_position_holdout.npz"
        seed, seconds, frequency = 14001, 20.0, 50
        label = "test_position_holdout"
    historical = json.loads(historical_path.read_text())
    verify_checkpoint(checkpoint, historical["checkpoint_sha256"])
    name = "learn2fly.train" if algorithm == "td3" else f"figure8.train_{algorithm}"
    module = importlib.import_module(f"crazyflow_experiments.{name}")
    args = module.Args(**historical["config"])
    with jax.default_device(jax.devices(args.jax_device)[0]):
        if algorithm == "td3":
            env = module.make_reach_pos_envs(64, args)
            previous_metrics = historical["evaluation"][label]
        else:
            ppo = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
            env = ppo.make_jitted_envs(
                64,
                module.reward_coefs(args),
                args.jax_device,
                reset_rotor=True,
                reset_random=True,
                dr_scale=0.0,
            )
            previous_metrics = json.loads(raw_historical.with_suffix(".json").read_text())
        agent = load_official_agent(module, args, env, checkpoint, algorithm)
        initial, _ = env.reset(env, seed=seed)
        state = initial.unwrapped.data.states
        initial_arrays = {
            "initial_pos": np.asarray(state.pos),
            "initial_vel": np.asarray(state.vel),
            "initial_quat": np.asarray(state.quat),
        }
        raw = official.rollout(env, agent, seed, seconds)
    metric = official.metrics(raw, frequency)
    assert raw["active"].shape == (int(seconds * frequency), 64)
    for name, values in {**raw, **initial_arrays}.items():
        if not np.isfinite(values).all():
            raise FloatingPointError(f"Nonfinite official evaluation array: {name}")
    counts = raw["active"].sum(axis=0)
    complete = (~raw["terminated"].any(axis=0)) & (counts >= len(raw["active"]) - 1)
    nullable = ("survivor_tracking_rmse_m", "tail_rmse_m") if not complete.any() else ()
    assert_finite_metrics(metric, nullable)
    undefined = {
        name: "No world completed the requested first-episode horizon." for name in nullable
    }
    with np.load(raw_historical) as previous:
        recorded_pos = previous["pos"]
        if recorded_pos.ndim == 3:
            recorded_pos = recorded_pos[:, 0]
        common = min(len(recorded_pos), len(raw["pos"]))
        old_active = previous["active"][:common]
        if old_active.ndim == 2:
            old_active = old_active[:, 0]
        shared = old_active & raw["active"][:common, 0]
        difference = raw["pos"][:common, 0][shared] - recorded_pos[:common][shared]
        world0_change = float(np.max(np.abs(difference))) if difference.size else None
    raw_path = output / f"official_{algorithm}.npz"
    np.savez_compressed(raw_path, **raw, **initial_arrays)
    result = {
        "family": "official",
        "algorithm": algorithm,
        "runtime": environment,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "historical_record": str(historical_path.relative_to(ROOT)),
        "historical_record_sha256": sha256(historical_path),
        "immutable_experiment_source_commit": official.OFFICIAL_COMMIT,
        "config": dataclasses.asdict(args),
        "seed": seed,
        "scenario": label,
        "worlds": 64,
        "seconds": seconds,
        "frequency_hz": frequency,
        "metrics": metric,
        "complete_horizon_worlds": int(complete.sum()),
        "undefined_metric_reasons": undefined,
        "historical_same_seed_metrics": previous_metrics,
        "historical_world0_position_max_difference_m": world0_change,
        "raw": str(raw_path.relative_to(ROOT)),
        "raw_sha256": sha256(raw_path),
    }
    save_json(output / f"official_{algorithm}.json", result)
    return result


def main() -> None:
    """Run seven predeclared measurements without replacing any historical experiment output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "gym", "official"), default="all")
    parser.add_argument("--tasks", nargs="+", choices=tuple(GYM_MODELS), default=list(GYM_MODELS))
    parser.add_argument(
        "--algorithms", nargs="+", choices=tuple(OFFICIAL_MODELS), default=list(OFFICIAL_MODELS)
    )
    parser.add_argument("--experiments-dir", type=Path, default=Path("/tmp/crazyflow_experiments"))
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen-policy revalidation requires the research CUDA environment")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    environment = runtime()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.suite in ("all", "gym"):
        jobs += [("gym", name) for name in args.tasks]
    if args.suite in ("all", "official"):
        jobs += [("official", name) for name in args.algorithms]
    for family, name in jobs:
        print(f"Revalidate frozen {family}/{name}", flush=True)
        result = (
            evaluate_gym(name, args.output_dir, environment)
            if family == "gym"
            else evaluate_official(name, args.output_dir, environment, args.experiments_dir)
        )
        print(
            json.dumps({"family": family, "name": name, "metrics": result["metrics"]}), flush=True
        )
        gc.collect()
        jax.clear_caches()


if __name__ == "__main__":
    main()
