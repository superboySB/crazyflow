"""Local speed-specific training extension; not a reproduction of published faster-flight results.

One command inside Docker:
    python usage_assets/official_frontier_20261015/run.py
The only training-task change from the official 5_5s setup is reference trajectory_time.
PPO uses 20M transitions; BPTT uses 1.6M; both use seeds42/43/44 from scratch.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import matplotlib.pyplot as plt
import numpy as np
import official_policy_20261015 as official
from official_report_20261015 import compact, verify
from official_speed_sweep_20261015 import reference_peaks
from official_stress_20261015 import inject_reset

DIRECTORY = Path(__file__).resolve().parent
PERIODS = (3.5, 3.0, 2.5)
SEEDS = (42, 43, 44)
BUDGETS = {"ppo": 20_000_000, "bptt": 1_600_000}


def name_for(algorithm: str, period: float, seed: int, multiplier: int = 1) -> str:
    """Return a unique name isolated from the earlier official-reproduction artifacts."""
    return f"official_frontier_{algorithm}_T{period:g}s_budget{multiplier}_seed{seed}"


def load_agent(module: Any, args: Any, env: Any, checkpoint: Path, algorithm: str) -> Any:
    """Load a frozen actor while preserving its training observation architecture."""
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
    return jax.device_put(agent, jax.devices(args.jax_device)[0])


def evaluate(
    module: Any, args: Any, checkpoint: Path, algorithm: str, seed: int, label: str
) -> dict:
    """Evaluate unchanged weights on nominal validation or genuine held-out initial states."""
    ppo = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
    env = ppo.make_jitted_envs(
        64,
        module.reward_coefs(args),
        args.jax_device,
        reset_rotor=True,
        reset_random=label != "validation",
        dr_scale=0.0,
    )
    agent = load_agent(module, args, env, checkpoint, algorithm)
    if label == "velocity_holdout":
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
    velocity_delta = initial_arrays["initial_vel"][:, 0] - initial_arrays["initial_reference_vel"]
    if label == "velocity_holdout" and not (0.45 < np.max(np.abs(velocity_delta)) <= 0.500001):
        raise ValueError("Requested real velocity perturbation was overwritten")
    raw = official.rollout(env, agent, seed, 20.0)
    metric = official.metrics(raw, 50)
    active = raw["active"]
    saturation = np.abs(raw["action"][..., [0, 1, 3]]) >= 0.99
    metric |= reference_peaks(args.trajectory_time) | {
        "seed": seed,
        "checkpoint": str(checkpoint.relative_to(official.ROOT)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "initial_velocity_delta_abs_max_mps": float(np.max(np.abs(velocity_delta))),
        "normalized_effective_action_saturation_rate": float(
            (saturation & active[..., None]).sum() / max(active.sum() * 3, 1)
        ),
        "initial_position_randomization": "none" if label == "validation" else "uniform +/-0.1 m",
        "velocity_perturbation": "reference + uniform +/-0.5 m/s"
        if label == "velocity_holdout"
        else "exact reference velocity",
    }
    path = DIRECTORY / f"{checkpoint.stem}_{label}.npz"
    np.savez_compressed(path, **raw)
    compact(path, 50)
    verify(path, metric)
    with np.load(path) as content:
        reduced = {key: content[key] for key in content.files}
    np.savez_compressed(
        path, **reduced, **initial_arrays, full_effective_action_saturated=saturation
    )
    official.write_json(path.with_suffix(".json"), metric)
    return metric


def train(algorithm: str, periods: list[float], multiplier: int, seeds: list[int]) -> None:
    """Train independent seeds at the entire requested budget without changing official updates."""
    module = importlib.import_module(f"crazyflow_experiments.figure8.train_{algorithm}")
    budget = BUDGETS[algorithm] * multiplier
    for period in periods:
        args = module.Args.create("5_5s", trajectory_time=period, total_timesteps=budget)
        with jax.default_device(jax.devices(args.jax_device)[0]):
            started = time.perf_counter()
            fn, env, template = official.setup(module, args, algorithm)
            setup_seconds = time.perf_counter() - started
            records = []
            for seed in seeds:
                stem = name_for(algorithm, period, seed, multiplier)
                record_path = DIRECTORY / f"{stem}.json"
                checkpoint = official.MODELS / f"{stem}.ckpt"
                if record_path.exists():
                    record = json.loads(record_path.read_text())
                    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    if digest != record["checkpoint_sha256"]:
                        raise ValueError("Saved result and checkpoint do not match")
                    records.append(record)
                    continue
                run_args = dataclasses.replace(args, seed=seed)
                print(f"Train {stem}: requested {budget:,} transitions", flush=True)
                agent, curve, elapsed = official.train_one(
                    module, run_args, algorithm, fn, env, template
                )
                payload = {
                    "actor": jax.tree.map(np.asarray, agent.actor_states.params),
                    "samples_dt": run_args.samples_dt,
                }
                if algorithm == "ppo":
                    payload["critic"] = jax.tree.map(np.asarray, agent.critic_states.params)
                checkpoint.write_bytes(pickle.dumps(payload))
                np.savez_compressed(DIRECTORY / f"{stem}_training.npz", **curve)
                reward = curve["mean_step_reward"]
                quarter = max(1, len(reward) // 4)
                prior = float(np.mean(reward[-2 * quarter : -quarter]))
                last = float(np.mean(reward[-quarter:]))
                record = {
                    "study": "local speed-specific extension; not published-paper reproduction",
                    "algorithm": algorithm,
                    "reference_period_s": period,
                    "seed": seed,
                    "official_preset": "5_5s",
                    "budget_multiplier": multiplier,
                    "requested_transitions": budget,
                    "actual_transitions": run_args.num_iterations * run_args.batch_size,
                    "config": dataclasses.asdict(run_args),
                    "only_training_task_change": "trajectory_time",
                    "drone": "cf21B_500",
                    "dynamics": "nominal first_principles",
                    "control": "attitude",
                    "policy_frequency_hz": 50,
                    "physics_frequency_hz": 500,
                    "official_commit": official.OFFICIAL_COMMIT,
                    "source": official.OFFICIAL_URL,
                    "checkpoint": str(checkpoint.relative_to(official.ROOT)),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "setup_seconds": setup_seconds,
                    "train_seconds_including_first_compile": elapsed,
                    "late_learning": {
                        "prior_quarter_mean_step_reward": prior,
                        "last_quarter_mean_step_reward": last,
                        "last_minus_prior": last - prior,
                        "last_quarter_reward_std": float(np.std(reward[-quarter:])),
                    },
                    "validation": evaluate(
                        module, run_args, checkpoint, algorithm, 9001, "validation"
                    ),
                }
                official.write_json(record_path, record)
                records.append(record)
                print(
                    json.dumps(
                        {
                            "name": stem,
                            "actual_transitions": record["actual_transitions"],
                            "late_learning": record["late_learning"],
                            "validation_rmse_mm": record["validation"]["tracking_rmse_m"] * 1000,
                            "validation_completion": record["validation"][
                                "full_horizon_completion_rate"
                            ],
                            "validation_5cm_success": record["validation"]["success_rate_rmse_5cm"],
                        }
                    ),
                    flush=True,
                )
                del agent, curve
            selected = min(
                records,
                key=lambda r: (
                    -r["validation"]["full_horizon_completion_rate"],
                    r["validation"]["tracking_rmse_m"],
                ),
            )
            selected_path = DIRECTORY / f"selected_{algorithm}_T{period:g}s_budget{multiplier}.json"
            selected["selection_rule"] = (
                "validation seed9001: completion first, minimum RMSE second"
            )
            checkpoint = official.ROOT / selected["checkpoint"]
            chosen_args = module.Args(**selected["config"])
            selected["test_position_holdout"] = evaluate(
                module, chosen_args, checkpoint, algorithm, 14001, "test_position_holdout"
            )
            selected["test_velocity_holdout"] = evaluate(
                module, chosen_args, checkpoint, algorithm, 14002, "velocity_holdout"
            )
            official.write_json(selected_path, selected)
            print(f"Selected {selected['checkpoint']}", flush=True)
            env.close()
            del fn, env, template
            jax.clear_caches()


def report() -> None:
    """Describe all training runs, their convergence evidence and selected held-out performance."""
    records = [
        json.loads(path.read_text())
        for path in sorted(DIRECTORY.glob("official_frontier_*_seed*.json"))
        if "_validation" not in path.stem and "_holdout" not in path.stem
    ]
    records = [record for record in records if "config" in record]
    for record in records:
        stem = Path(record["checkpoint"]).stem
        with np.load(DIRECTORY / f"{stem}_training.npz") as data:
            reward = data["mean_step_reward"]
        window = max(1, len(reward) // 10)
        record["late_learning"].update(
            {
                "last_10pct_mean_step_reward": float(reward[-window:].mean()),
                "prior_10pct_mean_step_reward": float(reward[-2 * window : -window].mean()),
                "last_10pct_minus_prior": float(
                    reward[-window:].mean() - reward[-2 * window : -window].mean()
                ),
                "last_10pct_reward_std": float(reward[-window:].std()),
            }
        )
        official.write_json(DIRECTORY / f"{stem}.json", record)
    candidates = [
        json.loads(path.read_text()) for path in sorted(DIRECTORY.glob("selected_*_budget*.json"))
    ]
    by_checkpoint = {record["checkpoint"]: record for record in records}
    for candidate in candidates:
        candidate["late_learning"] = by_checkpoint[candidate["checkpoint"]]["late_learning"]
    selected = []
    for algorithm in ("ppo", "bptt"):
        for period in PERIODS:
            choices = [
                r
                for r in candidates
                if r["algorithm"] == algorithm and r["reference_period_s"] == period
            ]
            if choices:
                selected.append(
                    min(
                        choices,
                        key=lambda r: (
                            -r["validation"]["full_horizon_completion_rate"],
                            r["validation"]["tracking_rmse_m"],
                        ),
                    )
                )
    official.write_json(DIRECTORY / "selected_budget_candidates.json", candidates)
    eligible = [
        r
        for r in selected
        if r["validation"]["full_horizon_completion_rate"] == 1.0
        and r["validation"]["success_rate_rmse_5cm"] == 1.0
    ]
    if eligible:
        best = min(
            eligible, key=lambda r: (r["reference_period_s"], r["validation"]["tracking_rmse_m"])
        )
        official.write_json(DIRECTORY / "fastest_reliable_demo.json", best)
    official.write_json(DIRECTORY / "training_records.json", records)
    official.write_json(DIRECTORY / "selected_checkpoints.json", selected)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    colors = {42: "#3168b3", 43: "#e58e26", 44: "#269d79"}
    for row, algorithm in enumerate(("ppo", "bptt")):
        for column, period in enumerate(PERIODS):
            ax = axes[row, column]
            for record in records:
                if record["algorithm"] != algorithm or record["reference_period_s"] != period:
                    continue
                with np.load(DIRECTORY / f"{Path(record['checkpoint']).stem}_training.npz") as raw:
                    rewards = raw["mean_step_reward"]
                steps = np.linspace(0, record["actual_transitions"], len(rewards))
                ax.plot(
                    steps / 1e6,
                    rewards,
                    linewidth=1,
                    color=colors[record["seed"]],
                    linestyle="-" if record["budget_multiplier"] == 1 else "--",
                    label=f"seed{record['seed']} / {record['budget_multiplier']}x",
                )
            ax.set(
                title=f"{algorithm.upper()} | trained T={period:g}s",
                xlabel="Environment transitions (millions)",
                ylabel="Mean step reward",
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
    figure.savefig(DIRECTORY / "learning_curves.png", dpi=160)
    plt.close(figure)
    text = """# 本地专门高速训练扩展（2026-10-15）

这项实验针对更快参考周期从头训练，不是把5.5s策略迁移失败当项目上限，也不是声称复现了
论文中未发表的高速结果。官方源码固定bfc6414；沿用5_5s的全部网络/奖励/优化参数与preview，
唯一训练任务改动是trajectory_time，分别3.5/3.0/2.5s。PPO完整20M、BPTT完整1.6M，
各seeds42/43/44。机型cf21B_500，名义first_principles、attitude控制、物理500Hz/控制50Hz。

每种算法/周期的权重仅由validation seed9001选择：完整20s比例优先，再选最低RMSE。
独立test seed14001添加位置±0.1m（初速精确参考）；压力seed14002同时添加位置±0.1m和
真正速度扰动±0.5m/s，速度在所有reset结束后显式注入并刷新策略观测。
每项64world、20s，保留全部失败种子；到结束/终止后的autoreset状态不计入。
全程精度成功=完整20s且RMSE<5cm，不用幸存者误差替代所有world的主指标。

## 全部独立训练

| 算法 | 周期s | seed | 预算倍数 | 实际transitions | 验证RMSE mm | 完整20s | <5cm | 末10%奖励改善 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for r in records:
        v = r["validation"]
        text += (
            f"| {r['algorithm'].upper()} | {r['reference_period_s']:g} | {r['seed']} | "
            f"{r['budget_multiplier']} | {r['actual_transitions']:,} | "
            f"{v['tracking_rmse_m'] * 1000:.2f} | {v['full_horizon_completion_rate']:.2%} | "
            f"{v['success_rate_rmse_5cm']:.2%} | "
            f"{r['late_learning']['last_10pct_minus_prior']:.4f} |\n"
        )
    text += """
## 固定验证规则选出的权重与独立holdout

| 算法 | 周期s | seed/预算 | Val mm / <5cm | Pos mm / <5cm | Vel mm / <5cm |
|---|---:|---:|---|---|---|
"""
    for r in selected:
        metrics = [r[k] for k in ("validation", "test_position_holdout", "test_velocity_holdout")]
        cells = [
            f"{m['tracking_rmse_m'] * 1000:.2f} / {m['success_rate_rmse_5cm']:.2%}" for m in metrics
        ]
        text += (
            f"| {r['algorithm'].upper()} | {r['reference_period_s']:g} | "
            f"{r['seed']}/{r['budget_multiplier']}x | " + " | ".join(cells) + " |\n"
        )
    text += """
![完整预算和全部种子曲线](learning_curves.png)

“末10%奖励改善”是最后10%迭代均值减前一10%均值，JSON还保留季度指标和尾段标准差；
曲线和真实指标一起阅读，
不把正斜率自动当作充分收敛，也不把失败种子隐藏。
结果限定本地有限预算、固定官方超参数族和这些任务分布，不能声称全局最优或机型绝对物理极限。

## 复现

~~~bash
python usage_assets/official_frontier_20261015/run.py
python usage_assets/official_frontier_20261015/run.py --report-only
~~~

可用--algorithm ppo或bptt分开运行；--periods和--seeds指定子集，
--budget-multiplier 2进行有限预算对比。每个checkpoint名称唯一，不覆盖此前30次官方训练。
权重在trained_models/official_frontier_*；原始数据、初始真值、曲线和完整配置在本目录。
eval JSON与NPZ可用既有official_report_20261015.verify重新计算RMSE/失败/精度成功，
SHA256标识固定权重。训练源码及参数出处：
[官方PPO](https://github.com/learnsyslab/crazyflow_experiments/blob/bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3/crazyflow_experiments/figure8/train_ppo.py)
与 [官方BPTT](https://github.com/learnsyslab/crazyflow_experiments/blob/bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3/crazyflow_experiments/figure8/train_bptt.py)。
"""
    (DIRECTORY / "README.md").write_text(text)
    print(f"Frontier report: {len(records)} independent runs; {len(selected)} selected configs.")


def main() -> None:
    """Run the finite local frontier protocol without modifying earlier experiment artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("ppo", "bptt", "both"), default="both")
    parser.add_argument("--periods", nargs="+", type=float, default=list(PERIODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--budget-multiplier", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    cli = parser.parse_args()
    official.bootstrap(Path("/tmp/crazyflow_experiments"))
    if cli.report_only:
        report()
        return
    if cli.budget_multiplier < 1:
        raise ValueError("Incomplete official budgets are not allowed")
    official.write_json(
        DIRECTORY / f"protocol_{cli.algorithm}_budget{cli.budget_multiplier}.json",
        {
            "study": "local speed-specific extension",
            "periods": cli.periods,
            "seeds": cli.seeds,
            "budgets": BUDGETS,
            "budget_multiplier": cli.budget_multiplier,
            "validation_seed": 9001,
            "test_position_seed": 14001,
            "test_velocity_seed": 14002,
            "eval_worlds": 64,
            "eval_seconds": 20,
            "official_commit": official.OFFICIAL_COMMIT,
        },
    )
    algorithms = ("ppo", "bptt") if cli.algorithm == "both" else (cli.algorithm,)
    for algorithm in algorithms:
        train(algorithm, cli.periods, cli.budget_multiplier, cli.seeds)
    if cli.algorithm == "both":
        report()


if __name__ == "__main__":
    main()
