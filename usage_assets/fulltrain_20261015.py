"""Reproduce the multi-seed Gymnasium study inside the research Docker image.

Run ``python usage_assets/fulltrain_20261015.py suite``. Each independent worker
writes its complete configuration, training/validation history and frozen test
results. Weights are retained in trained_models, including the final checkpoint
so an earlier validation maximum cannot conceal later regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "usage_assets" / "fulltrain_20261015"
WEIGHTS = ROOT / "trained_models"
SEEDS = (7, 19, 43)
TEST_SEEDS = (31001, 31002, 31003)
TASKS = ("reach_pos", "reach_vel", "landing", "figure8")
IDS = dict(
    zip(
        TASKS,
        (
            "DroneReachPos-v0",
            "DroneReachVel-v0",
            "DroneLanding-v0",
            "DroneFigureEightTrajectory-v0",
        ),
        strict=True,
    )
)


def clean(value: object) -> object:
    """Encode numeric artifacts as standards-compliant JSON."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: object) -> None:
    """Atomically persist an experiment result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    """Identify the exact checkpoint that produced reported scores."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, count: int) -> list[float]:
    """Return the 95 percent Wilson interval for held-out episodes."""
    z = 1.959963984540054
    p = successes / count
    denom = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / denom
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denom
    return [centre - half, centre + half]


def worker(args: argparse.Namespace) -> None:
    """Train one seed without sharing mutable simulator state with other runs."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    import torch
    import train_gymnasium as trainer

    import crazyflow

    if args.env_backend == "fused":
        import fast_gymnasium_20261015 as fast
        from gymnasium.wrappers.vector import JaxToTorch

        def fused_env(config: trainer.Config, count: int) -> object:
            return JaxToTorch(
                fast.make_raw(config, count, "gpu", True), device=torch.device("cuda")
            )

        trainer.make_env = fused_env

    OUT.mkdir(parents=True, exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    jax.config.update("jax_compilation_cache_dir", "/tmp/crazyflow-fulltrain-jax-cache")
    tag = f"{args.task}_s{args.seed}_g{args.gamma:g}_{args.steps}"
    if args.teacher_model != "legacy":
        tag += f"_{args.teacher_model}"
    checkpoint = WEIGHTS / f"gym_{tag}.pt"
    result_path = OUT / f"{tag}.json"
    config = trainer.Config(
        env_id=IDS[args.task],
        seed=args.seed,
        total_timesteps=args.steps,
        num_envs=4096,
        num_steps=64,
        gamma=args.gamma,
        teacher_model=args.teacher_model,
        eval_envs=256,
        eval_interval=100,
        validation_seed=20007,
    )
    started = time.perf_counter()
    if args.evaluate_only:
        agent, config, keys, history = trainer.load_checkpoint(checkpoint)
        history = torch.load(
            checkpoint.with_name(checkpoint.stem + "_last.pt"),
            map_location="cpu",
            weights_only=False,
        )["history"]
    else:
        agent, keys, history = trainer.train(config, checkpoint, resume=args.resume)
    training_seconds = time.perf_counter() - started
    if args.evaluate_only and result_path.exists():
        previous = json.loads(result_path.read_text())
        training_seconds = previous.get("training_seconds_including_validation", training_seconds)
    if not all(torch.isfinite(p).all().item() for p in agent.parameters()):
        raise FloatingPointError("Nonfinite trained policy weights")
    if args.validation_only:
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = data["trainer_state"]
        write_json(
            result_path,
            {
                "config": asdict(config),
                "history": history,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "selected_update": state["iteration"],
                "selected_validation_score": state["best_score"],
                "protocol": "Hyperparameter screening uses validation only; no held-out tests run.",
            },
        )
        return
    config.eval_envs = 1024
    tests = []
    horizontal = []
    first_behavior = None
    for seed in TEST_SEEDS:
        metrics, behavior = trainer.evaluate(agent, config, keys, eval_seed=seed)
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite frozen-policy evaluation")
        np.savez_compressed(OUT / f"{tag}_test{seed}.npz", **behavior)
        successes = int(np.asarray(behavior["per_env_success"]).sum())
        row = {
            "seed": seed,
            "metrics": metrics,
            "successes": successes,
            "episodes": config.eval_envs,
            "success_95pct_wilson": wilson(successes, config.eval_envs),
        }
        if args.task == "reach_vel":
            vz = np.asarray(behavior["initial_target_vel"])[:, 2]
            for name, mask in (("vz_nonnegative", vz >= 0), ("vz_negative", vz < 0)):
                success = np.asarray(behavior["per_env_success"])[mask]
                row[name] = {"count": int(mask.sum()), "success_rate": float(success.mean())}
            hmetrics, hbehavior = trainer.evaluate(
                agent,
                config,
                keys,
                eval_seed=seed,
                reset_options={
                    "vel_min": jnp.array([-1.0, -1.0, 0.0]),
                    "vel_max": jnp.array([1.0, 1.0, 0.0]),
                },
            )
            np.savez_compressed(OUT / f"{tag}_horizontal_test{seed}.npz", **hbehavior)
            horizontal.append(
                {
                    "seed": seed,
                    "metrics": hmetrics,
                    "successes": int(np.asarray(hbehavior["per_env_success"]).sum()),
                    "episodes": config.eval_envs,
                }
            )
        tests.append(row)
        if first_behavior is None:
            first_behavior = behavior
    successes = sum(x["successes"] for x in tests)
    n = sum(x["episodes"] for x in tests)
    result = {
        "schema": 2,
        "goal_rng_protocol": "independent_goal_rng_v1",
        "env_backend": args.env_backend,
        "torch_matmul_allow_tf32": True,
        "env_backend_adapter_sha256": (
            sha256(Path(fast.__file__)) if args.env_backend == "fused" else None
        ),
        "env_backend_note": "Same NEXT_STEP physics/reward; fusion verified within 1e-6",
        "crazyflow_version": crazyflow.__version__,
        "official_main": "c95f69bc54f830e859fee4ac5734bf571acd81f3",
        "config": asdict(config),
        "validation_seed": 20007,
        "test_seeds": TEST_SEEDS,
        "training_seconds_including_validation": training_seconds,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "last_checkpoint_sha256": sha256(checkpoint.with_name(checkpoint.stem + "_last.pt")),
        "history": history,
        "tests": tests,
        "horizontal_tests": horizontal,
        "aggregate_successes": successes,
        "aggregate_episodes": n,
        "aggregate_success_rate": successes / n,
        "aggregate_success_95pct_wilson": wilson(successes, n),
        "protocol": (
            "validation alone selects checkpoint; first-episode test outcomes; "
            "no test-driven checkpoint selection"
        ),
    }
    # Inspect checkpoint metadata without assuming the trainer's schema details.
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("iteration", "global_step", "best_validation", "best_score", "validation_history"):
        if key in data:
            result[key] = data[key]
    state = data.get("trainer_state", {})
    result["selected_update"] = state.get("iteration")
    result["selected_validation_score"] = state.get("best_score")
    final_data = torch.load(
        checkpoint.with_name(checkpoint.stem + "_last.pt"), map_location="cpu", weights_only=False
    )
    final_state = final_data.get("trainer_state", {})
    result["completed_updates"] = final_state.get("iteration")
    result["completed_vector_step_slots"] = final_state.get("global_step")
    result["completed_valid_transitions"] = final_state.get("valid_step")
    write_json(result_path, result)
    trainer.plot_result(OUT / f"{tag}.png", config, history, tests[0]["metrics"], first_behavior)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "success_rate": successes / n,
                "rmse": [x["metrics"]["rmse"] for x in tests],
            }
        ),
        flush=True,
    )


def suite(args: argparse.Namespace) -> None:
    """Execute the preregistered task/seed matrix in isolated Python workers."""
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(task, seed, args.gamma) for task in (args.tasks or TASKS) for seed in SEEDS]
    manifest = {
        "goal_rng_protocol": "independent_goal_rng_v1",
        "tasks": {task: IDS[task] for task in (args.tasks or TASKS)},
        "teacher_model": args.teacher_model,
        "gamma": args.gamma,
        "training_seeds": SEEDS,
        "validation_seed": 20007,
        "test_seeds": TEST_SEEDS,
        "requested_vector_step_slots_per_run": args.steps,
        "complete_updates_per_run": args.steps // (4096 * 64),
        "actual_vector_step_slots_per_run": (args.steps // (4096 * 64)) * (4096 * 64),
        "transition_count_note": (
            "Ignored NEXT_STEP reset slots are excluded by each run valid_transition counter; "
            "BC samples are additional."
        ),
        "jobs": jobs,
        "frozen_test_protocol": "No model or hyperparameter choice based on test seeds.",
        "budget_rationale": (
            f"Requested {args.steps:,} slots, {args.steps // (4096 * 64)} complete PPO updates. "
            "Main 200M budget is 10-20 times historical budgets; compare complete "
            "validation histories and separately specified recipes, retaining regressions. "
            "Finite-budget evidence, without a global upper-bound claim."
        ),
    }
    protocol_name = "protocol"
    if args.teacher_model != "legacy":
        protocol_name += f"_{args.teacher_model}"
    if args.gamma != 0.99:
        protocol_name += f"_g{args.gamma:g}"
    protocol_name += ".json"
    write_json(OUT / protocol_name, manifest)

    def execute(job: tuple[str, int, float]) -> None:
        task, seed, gamma = job
        tag = f"{task}_s{seed}_g{gamma:g}_{args.steps}"
        if args.teacher_model != "legacy":
            tag += f"_{args.teacher_model}"
        if (OUT / f"{tag}.json").exists() or f"{task}:{seed}" in args.exclude_job:
            return
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--task",
            task,
            "--seed",
            str(seed),
            "--gamma",
            str(gamma),
            "--steps",
            str(args.steps),
            "--env-backend",
            args.env_backend,
            "--teacher-model",
            args.teacher_model,
        ]
        print("Starting", tag, flush=True)
        with (OUT / f"{tag}.log").open("w") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                env=os.environ | {"PYTHONUNBUFFERED": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print("Completed", tag, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(execute, jobs))


def baseline(args: argparse.Namespace) -> None:
    """Reevaluate historical policies with the same first-episode test definition."""
    import numpy as np
    import torch
    import train_gymnasium as trainer

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    for task in TASKS:
        path = WEIGHTS / f"baseline_20261009_{task}.pt"
        if not path.exists():
            continue
        agent, config, keys, _ = trainer.load_checkpoint(path)
        config.eval_envs = 256
        validation, _ = trainer.evaluate(agent, config, keys, eval_seed=20007)
        config.eval_envs = 1024
        tests = []
        for seed in TEST_SEEDS:
            metrics, behavior = trainer.evaluate(agent, config, keys, eval_seed=seed)
            if not all(np.isfinite(value) for value in metrics.values()):
                raise FloatingPointError("Nonfinite frozen-policy evaluation")
            np.savez_compressed(OUT / f"baseline_{task}_test{seed}.npz", **behavior)
            tests.append({"seed": seed, "metrics": metrics})
        write_json(
            OUT / f"baseline_{task}.json",
            {
                "checkpoint_sha256": sha256(path),
                "config": asdict(config),
                "training_goal_rng_protocol": "legacy_correlated_goal_rng",
                "evaluation_goal_rng_protocol": trainer.RESET_PROTOCOL,
                "evaluation_env_backend": "python",
                "evaluation_torch_matmul_allow_tf32": False,
                "validation": validation,
                "checkpoint": str(path.relative_to(ROOT)),
                "tests": tests,
                "protocol": "Historical weights reevaluated; first episodes only, same test seeds.",
            },
        )
        print("Reevaluated historical", task, flush=True)


def main() -> None:
    """Parse worker or complete-study arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("suite", "worker", "baseline"))
    parser.add_argument("--task", choices=TASKS, default="figure8")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--steps", type=int, default=200_000_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--env-backend", choices=("python", "fused"), default="fused")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--tasks", choices=TASKS, nargs="+")
    parser.add_argument("--teacher-model", choices=("legacy", "calibrated"), default="legacy")
    parser.add_argument("--exclude-job", action="append", default=[])
    args = parser.parse_args()
    {"worker": worker, "suite": suite, "baseline": baseline}[args.command](args)


if __name__ == "__main__":
    main()
