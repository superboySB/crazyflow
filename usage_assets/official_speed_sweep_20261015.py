"""Sweep target speed with two frozen, validation-selected 5.5 s Figure8 policies.

Only trajectory_time changes. Preview samples/spacing, model, physical parameters and
policy stay fixed.
Runs 64 worlds, seed12001, 20 seconds per period. No weights are selected on this test.
"""

from __future__ import annotations

import importlib
import json
import pickle
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import numpy as np
import official_policy_20261015 as experiment
from official_report_20261015 import compact, verify

PERIODS = (5.5, 4.5, 3.5, 2.5, 1.5)
SEED = 12001


def rollout(env: Any, agent: Any) -> dict[str, np.ndarray]:
    """Record first episodes and policy-tick actuator saturation without autoreset contamination."""

    @jax.jit
    def execute(params: Any) -> Any:
        fresh, (obs, _) = env.reset(env, seed=SEED)

        def step(carry: Any, _: Any) -> Any:
            state, observation, active = carry
            action = agent.mean_action(params, observation)
            state, (observation, reward, terminated, truncated, _) = state.step(state, action)
            raw = state.unwrapped
            finite = jp.isfinite(raw.data.states.pos[:, 0]).all(axis=-1)
            valid = active & ~(terminated | truncated) & finite
            # ZeroYaw in the official wrapper fixes the yaw channel; it has no control authority.
            applied = action.at[..., 2].set(0.0)
            rpm = raw.data.controls.rotor_vel[:, 0]
            control_params = raw.data.controls.force_torque.params
            curve = control_params["rpm2thrust"]
            motor_force = curve[..., 0] + curve[..., 1] * rpm + curve[..., 2] * rpm**2
            minimum, maximum = control_params["thrust_min"], control_params["thrust_max"]
            margin = 0.01 * (maximum - minimum)
            motor_saturated = (motor_force <= minimum + margin) | (motor_force >= maximum - margin)
            sample = (
                raw.data.states.pos[:, 0],
                raw.target_pos,
                applied,
                valid,
                active & (terminated | ~finite),
                active & truncated,
                reward,
                jp.zeros_like(active),
                raw.data.states.vel[:, 0],
                raw.data.states.quat[:, 0],
                motor_force,
                motor_saturated,
            )
            return (state, observation, valid), sample

        _, data = jax.lax.scan(step, (fresh, obs, jp.ones(64, dtype=bool)), xs=None, length=1000)
        return data

    result = jax.tree.map(np.asarray, execute(agent.actor_states.params))
    env.close()
    keys = (
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
        "motor_force",
        "motor_saturated",
    )
    return dict(zip(keys, result))


def reference_peaks(period: float) -> dict[str, float]:
    """Evaluate the official analytic trajectory derivatives on a dense complete-cycle grid."""
    trajectories = importlib.import_module("crazyflow_experiments.control.trajectories")
    curve = trajectories.Figure8(
        scaling=jp.array([2.0, 0.0, 1.0]),
        cycle_time=period,
        duration=period,
        origin=jp.array([0.0, 0.0, 1.25]),
        axes="xzy",
    )
    times = jp.linspace(0.0, period, 10001)
    velocity, acceleration = np.asarray(curve.vel(times)), np.asarray(curve.acc(times))
    return {
        "reference_peak_speed_mps": float(np.linalg.norm(velocity, axis=-1).max()),
        "reference_peak_acceleration_mps2": float(np.linalg.norm(acceleration, axis=-1).max()),
        "reference_derivative_grid_samples": len(times),
    }


def main() -> None:
    """Evaluate fixed checkpoints at five periods, keeping every success and failure."""
    experiment.bootstrap(Path("/tmp/crazyflow_experiments"))
    selected = json.loads(
        (experiment.ASSETS / "official_policy_20261015_selected.json").read_text()
    )
    results = {}
    for algorithm in ("ppo", "bptt"):
        record = selected[f"{algorithm}_fast"]
        module = importlib.import_module(f"crazyflow_experiments.figure8.train_{algorithm}")
        ppo = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
        training_args = module.Args(**record["config"])
        config = module.reward_coefs(training_args)
        group = []
        for period in PERIODS:
            print(f"Speed sweep {algorithm.upper()} T={period}s", flush=True)
            saved_path = (
                experiment.ASSETS / f"official_speed_sweep_20261015_{algorithm}_T{period:g}s.json"
            )
            if saved_path.exists():
                existing = json.loads(saved_path.read_text())
                if existing["checkpoint_sha256"] != record["checkpoint_sha256"]:
                    raise ValueError("Saved speed result belongs to another checkpoint")
                group.append(existing)
                continue
            coefs = {**config, "trajectory_time": period}
            env = ppo.make_jitted_envs(
                64,
                coefs,
                training_args.jax_device,
                reset_rotor=True,
                reset_random=False,
                dr_scale=0.0,
            )
            model_path = experiment.ROOT / record["checkpoint"]
            if algorithm == "ppo":
                agent = module.load_agent(training_args, env, model_path)
            else:
                agent = module.Agent.create(
                    key=jax.random.PRNGKey(0),
                    obs_dim=env.single_observation_space.shape[0],
                    act_dim=env.single_action_space.shape[0],
                    hidden_size=training_args.hidden_size,
                    init_logstd=training_args.init_logstd,
                )
                payload = pickle.loads(model_path.read_bytes())
                agent = agent.replace(
                    actor_states=agent.actor_states.replace(params=payload["actor"])
                )
            agent = jax.device_put(agent, jax.devices(training_args.jax_device)[0])
            initial, _ = env.reset(env, seed=SEED)
            initial_state = {
                "initial_pos": np.asarray(initial.unwrapped.data.states.pos),
                "initial_vel": np.asarray(initial.unwrapped.data.states.vel),
                "initial_quat": np.asarray(initial.unwrapped.data.states.quat),
                "initial_reference": np.asarray(initial.unwrapped.target_pos),
            }
            raw = rollout(env, agent)
            metric = experiment.metrics(raw, 50)
            active = raw["active"]
            valid_actions = active[..., None]
            effective_action = raw["action"][..., [0, 1, 3]]
            saturation = np.abs(effective_action) >= 0.99
            fraction = (saturation & valid_actions).sum() / max(active.sum() * 3, 1)
            actuator_fraction = (raw["motor_saturated"] & valid_actions).sum() / max(
                active.sum() * 4, 1
            )
            peak_actions = np.max(np.where(valid_actions, np.abs(raw["action"]), 0), axis=(0, 1))
            metric |= reference_peaks(period) | {
                "algorithm": algorithm,
                "trajectory_time": period,
                "seed": SEED,
                "checkpoint": record["checkpoint"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "training_actual_transitions": record["actual_transitions"],
                "training_trajectory_time": training_args.trajectory_time,
                "n_samples_unchanged": training_args.n_samples,
                "samples_dt_unchanged": training_args.samples_dt,
                "preview_horizon_seconds": (training_args.n_samples - 1) * training_args.samples_dt,
                "physics": "nominal first_principles",
                "drone": "cf21B_500",
                "normalized_effective_action_saturation_rate": float(fraction),
                "normalized_action_abs_max_per_axis_rpyt": peak_actions.tolist(),
                "policy_tick_motor_force_near_bound_rate": float(actuator_fraction),
                "worlds_touching_floor_rate": float(
                    ((raw["pos"][..., 2] <= 0.025) & active).any(axis=0).mean()
                ),
            }
            stem = f"official_speed_sweep_20261015_{algorithm}_T{period:g}s"
            raw_path = experiment.ASSETS / f"{stem}.npz"
            np.savez_compressed(raw_path, **raw)
            compact(raw_path, 50)
            verify(raw_path, metric)
            with np.load(raw_path) as data:
                reduced = {k: data[k] for k in data.files}
            np.savez_compressed(
                raw_path,
                **reduced,
                **initial_state,
                full_effective_action_saturated=saturation,
                full_motor_force_near_bound=raw["motor_saturated"],
                motor_force_world0=raw["motor_force"][:, :1],
            )
            experiment.write_json(experiment.ASSETS / f"{stem}.json", metric)
            group.append(metric)
            print(
                json.dumps({k: v for k, v in metric.items() if k != "per_world_rmse_m"}), flush=True
            )
        results[algorithm] = group
        experiment.write_json(
            experiment.ASSETS / "official_speed_sweep_20261015_results.json", results
        )

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for algorithm, group in results.items():
        x = [m["reference_peak_speed_mps"] for m in group]
        axes[0].plot(x, [m["tracking_rmse_m"] * 1000 for m in group], "o-", label=algorithm.upper())
        axes[1].plot(
            x,
            [m["full_horizon_completion_rate"] * 100 for m in group],
            "o-",
            label=algorithm.upper(),
        )
        axes[2].plot(
            x,
            [m["policy_tick_motor_force_near_bound_rate"] * 100 for m in group],
            "o-",
            label=algorithm.upper(),
        )
    for ax in axes:
        ax.set_xlabel("Peak reference speed (m/s)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set(ylabel="First-episode tracking RMSE (mm)", yscale="log")
    axes[1].set_ylabel("Full 20 s completion (%)")
    axes[2].set_ylabel("Motor-force samples within 1% of bound (%)")
    figure.savefig(experiment.ASSETS / "official_speed_sweep_20261015.png", dpi=160)
    plt.close(figure)

    text = """# 2026-10-15 名义动力学下的冻结策略速度扫描

仅测试已用独立validation seed9001选出的 PPO/BPTT 5.5s checkpoint，
不重新训练、不按本扫描挑权重。所有场景为原始 cf21B_500、名义first_principles参数，
每项64world、固定新test seed12001、评估20s、控制50Hz/物理500Hz。
只把官方周期参考的 trajectory_time 改成5.5/4.5/3.5/2.5/1.5s，
保持actor、n_samples、samples_dt、动作范围和preview不变。初始位置/速度按新参考正确初始化，
本扫描不替代另外的真正初速/姿态扰动测试。

峰值参考速度/加速度来自官方 Figure8.vel/acc 解析导数，对完整一周期取10001点网格峰值。
策略动作饱和=有效首episode样本中 roll/pitch/thrust 三个归一化输出 |a|≥0.99 的比例；
yaw由官方ZeroYaw固定为0，不混入分母。另记录每个控制tick最后一个物理子步的四电机
命令推力位于控制器thrust_min/max的1%范围内的比例，作为实际分配器接近限幅的指标；
这不是全部500Hz子步的计数。近地world定义为有效样本中任意高度≤2.5cm，
只是高度代理，不是MuJoCo几何接触查询。失败后的autoreset动作不计入。

| Model | T s | vmax | amax | RMSE mm | 20s | <5cm | Action sat. | Motor sat. | Near floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
"""
    for algorithm, group in results.items():
        for metric in group:
            text += (
                f"| {algorithm.upper()} | {metric['trajectory_time']:g} | "
                f"{metric['reference_peak_speed_mps']:.3f} | "
                f"{metric['reference_peak_acceleration_mps2']:.3f} | "
                f"{metric['tracking_rmse_m'] * 1000:.2f} | "
                f"{metric['full_horizon_completion_rate']:.2%} | "
                f"{metric['success_rate_rmse_5cm']:.2%} | "
                f"{metric['normalized_effective_action_saturation_rate']:.2%} | "
                f"{metric['policy_tick_motor_force_near_bound_rate']:.2%} | "
                f"{metric['worlds_touching_floor_rate']:.2%} |\n"
            )
    text += """
[最快64/64精度通过的PPO 4.5s实际回放](official_speed_sweep_20261015_ppo_T4.5s.mp4)
与 [同权重2.5s失败边界](official_speed_sweep_20261015_ppo_T2.5s.mp4)
固定seed12001/world0，字幕明确 trained5.5s/test4.5s 或 test2.5s，保持原验证集选出的
checkpoint。回放与扫描原始world0的最大位置差分别0.071/0.043mm。

~~~bash
MUJOCO_GL=egl python usage_assets/official_replay_20261015.py \
  --checkpoint trained_models/official_ppo_5_5s_budget2_20261015_seed44.ckpt \
  --trajectory-time 4.5
MUJOCO_GL=egl python usage_assets/official_replay_20261015.py \
  --checkpoint trained_models/official_ppo_5_5s_budget2_20261015_seed44.ckpt \
  --trajectory-time 2.5
~~~

![名义物理参数下的速度扫描](official_speed_sweep_20261015.png)

所有world和失败均保留。RMSE基于首episode终止前有效样本，提前失败会缩短样本时长；
需同时看完整20s、<5cm成功率和近地比例。本扫描测量的是两套固定5.5s策略在新速度参考下的
泛化边界，不能等同于专门重训后的能力或该机型的绝对物理速度极限。

复现：python usage_assets/official_speed_sweep_20261015.py。
JSON保存固定checkpoint SHA、preview参数、逐轴动作峰值、每world误差和全统计。
NPZ保存全world逐step误差/mask/动作与电机饱和bool、全部初始真值，以及world0实际位置、
参考、速度、姿态、动作和电机推力。无动力学随机化，因此不含DR1.0不物理惯量组合。
"""
    (experiment.ASSETS / "official_speed_sweep_20261015_results.md").write_text(text)


if __name__ == "__main__":
    main()
