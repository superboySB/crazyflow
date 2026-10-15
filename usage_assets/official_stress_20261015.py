"""Evaluate selected frozen policies with real initial velocity and attitude perturbations.

Run inside Docker:
    python usage_assets/official_stress_20261015.py
Velocity-only seed 11001: reference velocity plus U(-0.5,0.5) m/s per axis.
Attitude-only seed 11002: independent roll/pitch U(-15,15) degrees, yaw unchanged.
Perturbations are inserted after all official reset stages and before observation wrappers.
"""

from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
import official_policy_20261015 as experiment
from official_report_20261015 import compact, verify
from scipy.spatial.transform import Rotation as R


def inject_reset(env: Any, scenario: str, evaluation_seed: int) -> Any:
    """Replace the innermost reset so observations reflect the final injected true state."""
    wrappers = importlib.import_module("crazyflow_experiments.core.envs.wrappers")
    original_reset = env.unwrapped.reset
    target_device = env.unwrapped.data.states.vel.device

    def reset(raw: Any, *, seed: int | None = None, options: dict | None = None) -> Any:
        raw, (obs, info) = original_reset(raw, seed=seed, options=options)
        states = raw.data.states
        if scenario == "velocity":
            reference_vel = (
                raw.target_vel[:, None] if hasattr(raw, "target_vel") else jp.zeros_like(states.vel)
            )
            delta = jax.random.uniform(
                jax.device_put(jax.random.PRNGKey(evaluation_seed), target_device),
                states.vel.shape,
                minval=-0.5,
                maxval=0.5,
            )
            states = states.replace(vel=reference_vel + delta)
        else:
            angles = jax.random.uniform(
                jax.device_put(jax.random.PRNGKey(evaluation_seed), target_device),
                (*states.quat.shape[:-1], 3),
                minval=-np.pi / 12,
                maxval=np.pi / 12,
            )
            angles = angles.at[..., 2].set(0.0)
            quaternion = (R.from_quat(states.quat) * R.from_euler("xyz", angles)).as_quat()
            states = states.replace(quat=quaternion)
        raw = raw.replace(data=raw.data.replace(states=states))
        # Position, goals and local reference samples are unchanged. Both of these state fields
        # must be refreshed before QuatToMatrix/NormalizeObs/Flatten process the observation.
        obs = dict(obs)
        obs["vel"] = states.vel[:, 0]
        obs["quat"] = states.quat[:, 0]
        return raw, (obs, info)

    return wrappers.Wrapper.recursive_replace(env, reset=reset)


def main() -> None:
    """Run all selected policies without modifying or selecting their saved parameters."""
    experiment.bootstrap(Path("/tmp/crazyflow_experiments"))
    selected = json.loads(
        (experiment.ASSETS / "official_policy_20261015_selected.json").read_text()
    )
    all_results = {}
    for name, record in selected.items():
        algorithm = record["algorithm"]
        module_name = f"figure8.train_{algorithm}" if algorithm != "td3" else "learn2fly.train"
        module = importlib.import_module(f"crazyflow_experiments.{module_name}")
        args = module.Args(**record["config"])
        results = {}
        for scenario, evaluation_seed in (("velocity", 11001), ("attitude", 11002)):
            print(f"Stress {name} / {scenario}, seed={evaluation_seed}", flush=True)
            existing_path = experiment.ASSETS / f"official_stress_20261015_{name}_{scenario}.json"
            if existing_path.exists():
                existing = json.loads(existing_path.read_text())
                if existing["checkpoint_sha256"] != record["checkpoint_sha256"]:
                    raise ValueError("Saved stress result belongs to another checkpoint")
                results[scenario] = existing
                continue
            if algorithm == "td3":
                env = module.make_reach_pos_envs(64, args)
                agent = module.load_actor(experiment.ROOT / record["checkpoint"])
                seconds, frequency = 7.0, 100
            else:
                ppo = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
                env = ppo.make_jitted_envs(
                    64,
                    module.reward_coefs(args),
                    args.jax_device,
                    reset_rotor=True,
                    reset_random=False,
                    dr_scale=0.0,
                )
                if algorithm == "ppo":
                    agent = module.load_agent(args, env, experiment.ROOT / record["checkpoint"])
                else:
                    agent = module.Agent.create(
                        key=jax.random.PRNGKey(0),
                        obs_dim=env.single_observation_space.shape[0],
                        act_dim=env.single_action_space.shape[0],
                        hidden_size=args.hidden_size,
                        init_logstd=args.init_logstd,
                    )
                    payload = pickle.loads((experiment.ROOT / record["checkpoint"]).read_bytes())
                    agent = agent.replace(
                        actor_states=agent.actor_states.replace(params=payload["actor"])
                    )
                seconds, frequency = 20.0, 50
            baseline, _ = env.reset(env, seed=evaluation_seed)
            initial_reference = np.asarray(
                baseline.unwrapped.target_pos
                if hasattr(baseline.unwrapped, "target_pos")
                else baseline.unwrapped.goal_pos
            )
            baseline_vel = np.asarray(baseline.unwrapped.data.states.vel)
            env = inject_reset(env, scenario, evaluation_seed)
            initial, _ = env.reset(env, seed=evaluation_seed)
            states = initial.unwrapped.data.states
            reference = np.asarray(
                initial.unwrapped.target_pos
                if hasattr(initial.unwrapped, "target_pos")
                else initial.unwrapped.goal_pos
            )
            if not np.array_equal(initial_reference, reference):
                raise ValueError("Perturbation changed the reference")
            initial_vel = np.asarray(states.vel)
            initial_quat = np.asarray(states.quat)
            if scenario == "velocity":
                reference_vel = (
                    np.asarray(initial.unwrapped.target_vel)[:, None]
                    if hasattr(initial.unwrapped, "target_vel")
                    else np.zeros_like(initial_vel)
                )
                delta_vel = initial_vel - reference_vel
                if np.max(np.abs(delta_vel)) < 0.45 or np.max(np.abs(delta_vel)) > 0.500001:
                    raise ValueError("Velocity injection did not take effect")
            else:
                delta_vel = initial_vel - baseline_vel
                if not np.array_equal(initial_vel, baseline_vel):
                    raise ValueError("Attitude-only test changed the velocity")
                angles = R.from_quat(initial_quat).as_euler("xyz", degrees=True)
                if (
                    np.max(np.abs(angles[..., :2])) < 13
                    or np.max(np.abs(angles[..., :2])) > 15.0001
                ):
                    raise ValueError("Attitude injection did not take effect")
            initial_state = {
                "initial_pos": np.asarray(states.pos),
                "initial_vel": initial_vel,
                "initial_quat": initial_quat,
                "initial_velocity_delta": delta_vel,
                "initial_reference": reference,
            }
            raw = experiment.rollout(env, agent, evaluation_seed, seconds)
            stem = f"official_stress_20261015_{name}_{scenario}"
            raw_path = experiment.ASSETS / f"{stem}.npz"
            np.savez_compressed(raw_path, **raw)
            metric = experiment.metrics(raw, frequency) | {
                "seed": evaluation_seed,
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "reference_max_change_m": float(np.max(np.abs(reference - initial_reference))),
                "initial_velocity_delta_abs_max_mps": float(np.max(np.abs(delta_vel))),
            }
            if scenario == "attitude":
                metric["initial_roll_pitch_abs_max_deg"] = float(np.max(np.abs(angles[..., :2])))
            compact(raw_path, frequency)
            verify(raw_path, metric)
            with np.load(raw_path) as content:
                compacted = {k: content[k] for k in content.files}
            np.savez_compressed(raw_path, **compacted, **initial_state)
            experiment.write_json(experiment.ASSETS / f"{stem}.json", metric)
            results[scenario] = metric
            print(
                json.dumps({k: v for k, v in metric.items() if k != "per_world_rmse_m"}), flush=True
            )
        all_results[name] = results
        experiment.write_json(
            experiment.ASSETS / "official_stress_20261015_results.json", all_results
        )
    text = """# 2026-10-15 真正初速与姿态 holdout

这里只加载已按 validation seed9001 选出的冻结权重，不重新训练，也不按新测试结果重新挑权重。
每项 64 worlds；velocity-only 固定 seed11001，attitude-only 固定 seed11002。
官方 reset 全部结束后显式修改 SimData，并更新最内层原始观测；上层官方 wrappers 随后从
真实扰动状态生成首个策略观测。因此没有原 velocity reset 覆盖扰动的问题。

速度任务：Figure8 初速=参考速度+每轴 U(-0.5,0.5)m/s；到点初速=每轴 U(-0.5,0.5)m/s。
姿态任务：独立 roll/pitch 各 U(-15°,15°)，yaw 保持0，速度保持官方 reset 值。
两项独立运行，不叠加动力学随机化；Figure8 从 nominal 参考位置起步，TD3 保持官方随机到点初始位置。
每项保存64个初始真实状态；assert 已验证扰动生效、reference完全未改、attitude-only速度未改。
全统计由每world每step误差/mask复算。到点提前成功终止属于成功，不属于越界失败。

| 冻结策略 | 真正扰动 | 全程RMSE mm | 无越界 | 全时长 | 到点成功 | 轨迹全程<5cm成功 |
|---|---|---:|---:|---:|---:|---:|
"""
    for name, results in all_results.items():
        for scenario, metric in results.items():
            text += (
                f"| {name} | {scenario} | {metric['tracking_rmse_m'] * 1000:.2f} | "
                f"{metric['alive_without_crash_rate']:.2%} | "
                f"{metric['full_horizon_completion_rate']:.2%} | "
                f"{metric['goal_success_rate']:.2%} | "
                f"{metric['success_rate_rmse_5cm']:.2%} |\n"
            )
    text += """
到点全程 RMSE 包含随机初始位置误差，不能当稳态悬停精度。到点看 goal_success，
轨迹看完整时长和精度。结果限定这些冻结策略、指定初始分布、标称物理与精确观测，
不代表任意初始状态或实飞泛化保证。

复现：在研究 Docker 内运行 python usage_assets/official_stress_20261015.py。
原始 official_stress_20261015_* 场景 NPZ 保存完整误差/mask、world0真值轨迹，以及全部64world
初始位置/速度/四元数/参考/实际速度扰动；JSON保存权重SHA256和各world误差。
"""
    (experiment.ASSETS / "official_stress_20261015_results.md").write_text(text)


if __name__ == "__main__":
    main()
