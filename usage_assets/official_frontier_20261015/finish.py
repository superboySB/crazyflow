"""Verify and document the complete finite dedicated-speed study."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from run import DIRECTORY, official, report, verify


def main() -> None:
    """Recompute all first-episode metrics and freeze provenance, counts and conclusions."""
    official.bootstrap(Path("/tmp/crazyflow_experiments"))
    report()
    records = json.loads((DIRECTORY / "training_records.json").read_text())
    scenarios = []
    for path in sorted(DIRECTORY.glob("official_frontier_*.json")):
        metric = json.loads(path.read_text())
        if "worlds" not in metric:
            continue
        verify(path.with_suffix(".npz"), metric)
        with np.load(path.with_suffix(".npz")) as raw:
            active = raw["full_active"]
            saturated = raw["full_effective_action_saturated"]
            rate = (saturated & active[..., None]).sum() / (active.sum() * 3)
            if abs(rate - metric["normalized_effective_action_saturation_rate"]) > 1e-12:
                raise ValueError(f"Saturation definition mismatch: {path.name}")
            delta = raw["initial_vel"][:, 0] - raw["initial_reference_vel"]
            if not np.isclose(np.abs(delta).max(), metric["initial_velocity_delta_abs_max_mps"]):
                raise ValueError(f"Initial velocity audit mismatch: {path.name}")
        scenarios.append(path.name)
    for record in records:
        checkpoint = official.ROOT / record["checkpoint"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash differs from training record")
    hyperparameter_audit = []
    excluded = {"seed", "total_timesteps", "num_iterations", "trajectory_time"}
    for record in records:
        module = importlib.import_module(
            f"crazyflow_experiments.figure8.train_{record['algorithm']}"
        )
        import dataclasses

        original = json.loads(json.dumps(dataclasses.asdict(module.Args.create("5_5s"))))
        differences = [
            key
            for key, value in original.items()
            if key not in excluded and record["config"][key] != value
        ]
        if differences:
            raise ValueError(f"Unexpected official hyperparameter changes: {differences}")
        hyperparameter_audit.append(
            {
                "checkpoint": record["checkpoint"],
                "preset": "5_5s",
                "unexpected_changes": differences,
                "permitted_changes": sorted(excluded),
            }
        )
    official.write_json(DIRECTORY / "hyperparameter_audit.json", hyperparameter_audit)
    assert len(records) == 19 and len(scenarios) == 33
    selected = json.loads((DIRECTORY / "selected_checkpoints.json").read_text())
    baseline = next(
        r
        for r in records
        if r["algorithm"] == "ppo"
        and r["reference_period_s"] == 3.5
        and r["seed"] == 43
        and r["budget_multiplier"] == 1
    )
    extended = next(
        r
        for r in records
        if r["algorithm"] == "ppo"
        and r["reference_period_s"] == 3.5
        and r["seed"] == 43
        and r["budget_multiplier"] == 2
    )
    extension = {
        "chosen_by": json.loads((DIRECTORY / "budget_extension_plan.json").read_text()),
        "baseline_actual_transitions": baseline["actual_transitions"],
        "extended_actual_transitions": extended["actual_transitions"],
        "baseline_validation_rmse_m": baseline["validation"]["tracking_rmse_m"],
        "extended_validation_rmse_m": extended["validation"]["tracking_rmse_m"],
        "baseline_late_learning": baseline["late_learning"],
        "extended_late_learning": extended["late_learning"],
        "independent_from_scratch": True,
    }
    official.write_json(DIRECTORY / "budget_extension_result.json", extension)
    module = importlib.import_module("crazyflow_experiments.figure8.train_ppo")
    args = module.Args.create("5_5s", jax_device="cpu")
    env = module.make_jitted_envs(
        1, module.reward_coefs(args), "cpu", reset_rotor=True, reset_random=False, dr_scale=0.0
    )
    params = env.unwrapped.data.params
    controls = env.unwrapped.data.controls.force_torque.params
    mass = float(np.asarray(params.mass).reshape(-1)[0])
    maximum = float(np.asarray(controls["thrust_max"]))
    gravity = float(np.linalg.norm(np.asarray(params.gravity_vec)))
    physics = {
        "drone": "cf21B_500",
        "mass_kg": mass,
        "gravity_mps2": gravity,
        "motor_thrust_max_N": maximum,
        "command_collective_max_N": 4 * maximum,
        "upright_no_drag_max_up_acceleration_mps2": 4 * maximum / mass - gravity,
        "scope": "Command envelope diagnostic; ignores drag, rotor lag and attitude coupling. "
        "Not a global physical speed limit or proof of policy optimality.",
        "per_period": {
            str(t): {
                "reference_peak_up_acceleration_mps2": 2 * (2 * np.pi / t) ** 2,
                "no_drag_vertical_force_required_N": mass * (gravity + 2 * (2 * np.pi / t) ** 2),
            }
            for t in (3.5, 3.0, 2.5)
        },
    }
    env.close()
    official.write_json(DIRECTORY / "nominal_control_envelope.json", physics)
    official.write_json(
        DIRECTORY / "artifact_verification.json",
        {
            "passed": True,
            "training_runs": len(records),
            "baseline_runs": 18,
            "finite_budget_comparison_runs": 1,
            "actual_transitions_total": sum(r["actual_transitions"] for r in records),
            "selected_period_algorithm_configurations": len(selected),
            "evaluation_scenarios": len(scenarios),
            "worlds_per_scenario": 64,
            "checks": [
                "checkpoint SHA256",
                "global/per-world first-episode RMSE",
                "completion/crash/5cm precision",
                "effective action saturation",
                "actual initial velocity perturbation",
            ],
            "scenarios": scenarios,
        },
    )
    text = """
## 官方完整训练配方

采用固定commit的Args.create("5_5s")。PPO：1024env×32steps，32minibatches、15epochs，
actor/critic LR=0.0003/0.004且anneal_lr=true，gamma=0.96、GAEλ=0.94、clip=0.5，
entropy=0.007、value coef=0.7、hidden_size=32、init_logstd=-1；参考preview为11×0.06s。
BPTT：16env×64steps，actor LR=0.01且不退火，gamma=1、hidden_size=12，
init_logstd=-1，参考preview为12×0.04s。两者rpy_coef=0.1，DR=0；
动作/动作差分奖励系数随完整config逐项保存。
基础PPO20M请求对应610次完整update、19,988,480transitions；BPTT1.6M请求对应
1,562次完整update、1,599,488transitions。40M PPO为1,220次update、39,976,960transitions。
hyperparameter_audit.json逐run核对官方preset：除seed、预算/派生迭代数和参考周期外无参数改动。
参数出处是上方固定版本的官方PPO/BPTT源码；每个训练JSON完整config可直接构造Args重跑。

## 有限延训与结果边界

18个基础run结束后，只根据训练奖励最后10%相对前10%的改善幅度选择一个延训对照，
没有用测试集选择延训对象。PPO3.5s seed43的改善幅度最大（+0.02784），因此额外从头跑
40M预算，实际39,976,960步；validation误差由31.38mm降到13.58mm，按相同验证规则选中。
末段数值和全部曲线见budget_extension_result.json；有限延训也不证明全局最优。

专门训练已把原冻结5.5s策略的3.5s迁移失败推进为稳定名义跟踪：两个算法全部三个种子
validation均64/64完整且<5cm。位置holdout中，选中的PPO3.5s为19.63mm、
64/64完整且<5cm；BPTT3.5s为52.25mm、61/64完整且<5cm。
加真实初速扰动后，PPO为65.48mm、61/64完整、58/64<5cm；BPTT为132.50mm、
49/64完整、47/64<5cm。BPTT名义初始化更准，PPO对这些位置/速度扰动更稳健。
这也说明名义初始化的8.56mm不能当作任意初态的泛化保证。

3.0s BPTT validation为33.20mm、63/64<5cm，位置holdout为38.27mm、55/64；
PPO分别48.81mm/41和52.24mm/38。2.5s所有基础run仍飞完整20s，
但validation约129–162mm、全部0/64<5cm，轨迹精度未通过；不能用存活包装成功。

名义控制包络另见nominal_control_envelope.json：机体质量43.4g，单电机命令上限0.2N，
总0.8N，忽略阻力/姿态耦合时最大向上加速度约8.62m/s²。
参考3.0/2.5s的峰值向上加速度约8.77/12.63m/s²，提供性能下降的动力学线索；
它不是这台机型所有轨迹/参数/控制方法的绝对速度上限，也不能代替完整可达性分析。

## 实际MuJoCo回放

[最快全部validation通过且误差最低的BPTT3.5s](best_dedicated_speed.mp4)：
选择规则仅使用验证集，固定独立seed14001/world0，完整位置扰动，不挑最好world。
[位置holdout更稳健的PPO3.5s](ppo_T3.5s_position_holdout.mp4)同样固定seed/world；
两个视频都由冻结策略逐步驱动实际仿真状态，未用参考轨迹冒充飞行状态。
各自_replay.json记录检查点哈希和与原始评估状态的最大差异，_replay_states.npz保留真状态。
视频是仿真回放，未启动实飞。

## 一键完整复现与核验

~~~bash
bash usage_assets/official_frontier_20261015/reproduce.sh
~~~

此命令覆盖完整18run、固定40M对照、验证/两个新holdout、图表、33场景重算和两个视频。
已存在且哈希一致的checkpoint跳过训练。单独核验用finish.py；不混入旧30run/126场景统计。
artifact_verification.json记录19run总实际步数和33项完整场景。
"""
    with (DIRECTORY / "README.md").open("a") as handle:
        handle.write(text)
    print(
        json.dumps(
            {
                "verified_training_runs": len(records),
                "verified_scenarios": len(scenarios),
                "actual_transitions": sum(r["actual_transitions"] for r in records),
                "extended_tail10_reward_delta": extended["late_learning"]["last_10pct_minus_prior"],
            }
        )
    )


if __name__ == "__main__":
    main()
