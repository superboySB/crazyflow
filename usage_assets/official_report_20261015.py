"""Validate, compact and summarize official policy evidence after all training runs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import official_policy_20261015 as experiment

ASSETS = experiment.ASSETS


def compact(path: Path, frequency: int) -> None:
    """Retain world 0 and every world's full-resolution errors and first-episode masks."""
    with np.load(path) as data:
        raw = {k: data[k] for k in data.files}
    if "full_error_sq" in raw:
        if "full_reward" not in raw and not np.any(raw["full_error_sq"][~raw["full_active"]]):
            return
        raw["full_error_sq"] = np.where(raw["full_active"], raw["full_error_sq"], 0.0)
        raw.pop("full_reward", None)
        temporary = path.with_name(f".{path.stem}.tmp.npz")
        np.savez_compressed(temporary, **raw)
        temporary.replace(path)
        return
    out = {
        "schema_version": np.array(2),
        "frequency_hz": np.array(frequency),
        "world_indices": np.array([0]),
        "full_error_sq": np.where(
            raw["active"], np.sum((raw["pos"] - raw["target"]) ** 2, axis=-1), 0.0
        ).astype(np.float32),
        "full_active": raw["active"],
        "full_terminated": raw["terminated"],
        "full_truncated": raw["truncated"],
        "full_success": raw.get("success", np.zeros_like(raw["active"])),
    }
    for key in ("pos", "target", "action", "active", "vel", "quat"):
        if key in raw:
            out[key] = raw[key][:, :1]
    if "vel" not in out:
        out["velocity_estimate_finite_difference"] = (
            np.gradient(out["pos"], axis=0) * frequency
        ).astype(np.float32)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **out)
    temporary.replace(path)


def verify(path: Path, metric: dict) -> None:
    """Recompute RMSE and crashes from complete saved evidence."""
    with np.load(path) as data:
        compacted = "full_error_sq" in data
        errors = (
            data["full_error_sq"]
            if compacted
            else np.sum((data["pos"] - data["target"]) ** 2, axis=-1)
        )
        active = data["full_active"] if compacted else data["active"]
        terminated = data["full_terminated"] if compacted else data["terminated"]
        measured = np.sqrt((errors * active).sum() / max(active.sum(), 1))
        if not np.isclose(measured, metric["tracking_rmse_m"], rtol=1e-6, atol=1e-8):
            raise ValueError(f"RMSE does not reproduce: {path}")
        if not np.isclose(terminated.any(axis=0).mean(), metric["crash_rate"]):
            raise ValueError(f"Crash rate does not reproduce: {path}")
        count = active.sum(axis=0)
        survived = (~terminated.any(axis=0)) & (count >= active.shape[0] - 1)
        if not np.isclose(survived.mean(), metric["full_horizon_completion_rate"]):
            raise ValueError(f"Full horizon completion does not reproduce: {path}")
        per_world = np.sqrt((errors * active).sum(axis=0) / np.maximum(count, 1))
        if not np.allclose(per_world, metric["per_world_rmse_m"], rtol=1e-6, atol=1e-8):
            raise ValueError(f"Per-world RMSE does not reproduce: {path}")
        if not np.isclose((survived & (per_world < 0.05)).mean(), metric["success_rate_rmse_5cm"]):
            raise ValueError(f"Tracking success does not reproduce: {path}")
        success = data["full_success"] if compacted else data.get("success", np.zeros_like(active))
        if "goal_success_rate" in metric and not np.isclose(
            success.any(axis=0).mean(), metric["goal_success_rate"]
        ):
            raise ValueError(f"Goal success does not reproduce: {path}")


def select(records: list[dict], algorithm: str, preset: str) -> dict:
    """Select checkpoints using validation only."""
    group = [r for r in records if r["algorithm"] == algorithm and r["preset"] == preset]
    if algorithm == "td3":
        return min(
            group,
            key=lambda r: (
                -r["evaluation"]["validation_hover"]["goal_success_rate"],
                r["evaluation"]["validation_hover"]["mean_first_episode_seconds"],
            ),
        )
    return min(
        group,
        key=lambda r: (
            -r["evaluation"]["validation_nominal"]["full_horizon_completion_rate"],
            r["evaluation"]["validation_nominal"]["tracking_rmse_m"],
        ),
    )


def plots(records: list[dict], selected: dict) -> None:
    """Show all seed/budget curves and the measured transfer boundary."""
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {42: "#3168b3", 43: "#e58e26", 44: "#269d79"}
    for ax, algorithm, preset in zip(
        axes.flat[:3], ("bptt", "ppo", "td3"), ("10s", "5_5s", "rotor_reach")
    ):
        for r in records:
            if r["algorithm"] != algorithm or r["preset"] != preset:
                continue
            with np.load(ASSETS / f"{Path(r['checkpoint']).stem}_training.npz") as data:
                key = "episode_reward" if algorithm == "td3" else "mean_step_reward"
                values = data[key]
                steps = np.linspace(0, r["actual_transitions"], len(values))
                if algorithm == "td3":
                    valid = data["episode_count"] > 0
                    values, steps = values[valid], steps[valid]
                    window = max(1, len(values) // 100)
                    values = np.convolve(values, np.ones(window) / window, mode="valid")
                    steps = steps[window - 1 :]
            ax.plot(
                steps / 1e6,
                values,
                color=colors[r["seed"]],
                linewidth=1,
                linestyle="-" if r["budget_multiplier"] == 1 else "--",
                label=f"seed {r['seed']}, {r['budget_multiplier']}x",
                alpha=0.85,
            )
        ax.set(
            title=f"{algorithm.upper()} | {preset} | all trained seeds",
            xlabel="Environment transitions (millions)",
            ylabel="Episode return" if algorithm == "td3" else "Mean step reward",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)
    ax = axes.flat[3]
    for budget in (1, 2):
        group = [r for r in records if r["algorithm"] == "td3" and r["budget_multiplier"] == budget]
        means = [
            np.mean(
                [r["evaluation"][f"holdout_figure8_{period}s"]["tracking_rmse_m"] for r in group]
            )
            * 1000
            for period in ("15", "5.5", "3.5")
        ]
        ax.plot([15, 5.5, 3.5], means, "o-", label=f"TD3 {budget}x")
    ax.set(
        xlabel="Untrained figure-8 period (s)",
        ylabel="Tracking RMSE (mm)",
        title="Rotor-level reach policy: transfer to faster trajectories",
        yscale="log",
    )
    ax.set_xticks([3.5, 5.5, 15])
    ax.grid(alpha=0.2)
    ax.legend()
    figure.savefig(ASSETS / "official_policy_20261015_learning.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for row, label in enumerate(("ppo_fast", "td3")):
        r = selected[label]
        scenario = "holdout_initial_state" if label == "ppo_fast" else "holdout_figure8_15s"
        with np.load(ASSETS / f"{Path(r['checkpoint']).stem}_{scenario}.npz") as data:
            active = data["active"][:, 0]
            actual, target = data["pos"][:, 0][active], data["target"][:, 0][active]
            values = np.where(data["full_active"], np.sqrt(data["full_error_sq"]) * 1000, np.nan)
        vertical = 2 if label == "ppo_fast" else 1
        axes[row, 0].plot(
            target[:, 0], target[:, vertical], color="#e58e26", linewidth=2, label="Reference"
        )
        axes[row, 0].plot(
            actual[:, 0],
            actual[:, vertical],
            color="#3168b3",
            linewidth=1.2,
            label="Flown path, fixed world 0",
        )
        axes[row, 0].set_aspect("equal")
        axes[row, 0].set(
            title=f"{r['algorithm'].upper()} | {scenario}",
            xlabel="x (m)",
            ylabel="z (m)" if vertical == 2 else "y (m)",
        )
        axes[row, 0].legend(fontsize=8)
        keep_time = np.isfinite(values).any(axis=1)
        times = (np.arange(len(values)) / r["evaluation"][scenario]["frequency_hz"])[keep_time]
        values = values[keep_time]
        axes[row, 1].plot(times, np.nanmedian(values, axis=1), label="Median", color="#3168b3")
        axes[row, 1].fill_between(
            times,
            np.nanquantile(values, 0.05, axis=1),
            np.nanquantile(values, 0.95, axis=1),
            alpha=0.2,
            color="#3168b3",
            label="5th-95th percentile, 64 worlds",
        )
        axes[row, 1].set(
            title="Held-out per-step position error", xlabel="Time (s)", ylabel="Error (mm)"
        )
        axes[row, 1].legend(fontsize=8)
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    figure.savefig(ASSETS / "official_policy_20261015_tracking.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """Validate all training runs, compact evidence and produce a self-contained report."""
    records = []
    for path in sorted(ASSETS.glob("official_*_budget*_20261015_seed*.json")):
        record = json.loads(path.read_text())
        if "algorithm" not in record:
            continue
        record["drone"] = "cf2x_L250" if record["algorithm"] == "td3" else "cf21B_500"
        record["policy_frequency_hz"] = 100 if record["algorithm"] == "td3" else 50
        record["physics_frequency_hz"] = 500
        for scenario, metric in record["evaluation"].items():
            if "survival_rate" in metric:
                metric["full_horizon_completion_rate"] = metric.pop("survival_rate")
            metric["alive_without_crash_rate"] = 1.0 - metric["crash_rate"]
            raw = ASSETS / f"{Path(record['checkpoint']).stem}_{scenario}.npz"
            verify(raw, metric)
            compact(raw, metric["frequency_hz"])
            verify(raw, metric)
        experiment.write_json(path, record)
        records.append(record)
    groups = defaultdict(list)
    for r in records:
        groups[(r["algorithm"], r["preset"], r["budget_multiplier"])].append(r)
    for (algorithm, preset, budget), group in groups.items():
        experiment.write_json(
            ASSETS / f"official_{algorithm}_{preset}_budget{budget}_20261015_results.json", group
        )
    selected = {
        name: select(records, algorithm, preset)
        for name, algorithm, preset in (
            ("ppo_slow", "ppo", "10s"),
            ("ppo_fast", "ppo", "5_5s"),
            ("bptt_slow", "bptt", "10s"),
            ("bptt_fast", "bptt", "5_5s"),
            ("td3", "td3", "rotor_reach"),
        )
    }
    experiment.write_json(ASSETS / "official_policy_20261015_selected.json", selected)
    plots(records, selected)
    text = """# 2026-10-15 官方策略完整训练与能力边界

这是本机 Docker 中的仿真训练和评估，没有实飞。
外部源码固定为 [bfc6414](https://github.com/learnsyslab/crazyflow_experiments/tree/bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3)。
官方仓库没有提供策略 checkpoint，以下权重全部重新训练。

## 方法与完整预算

每个配置独立训练 seeds 42/43/44，分别使用完整官方预算和 2 倍预算。
2 倍预算从头训练，保留原超参数及其学习率退火规则；不是从只保存 actor 的 checkpoint 接续训练。
validation seed 9001 用于选择演示 checkpoint，test seed 10001 用于报告 holdout，每项 64 worlds。

Figure8 是 first_principles + attitude（姿态加集体推力），TD3 是 first_principles + rotor_vel，
直接输出四电机命令。Figure8机型为 cf21B_500（brushless），TD3为 cf2x_L250（brushed），
不能把不同机型成绩当作控制接口的公平消融。
标准四旋翼为 6 自由度、4 输入的欠驱动系统；没有模拟损失旋翼，
也不代表固定翼飞机。物理积分 500 Hz；Figure8 控制 50 Hz、评估 20 s；
TD3 控制 100 Hz，到点最多 7 s，未见八字任务分别 15/5.5/3.5 s。

只计算每个 world 首个 episode，排除 SAME_STEP autoreset 后状态，保存全部失败种子。
alive_without_crash_rate 是没有越界的比例，成功到点提前结束也计入；
full_horizon_completion_rate 是完整时长比例，到点成功提前结束时可以是 0。
到点成功必须位置误差 <5 cm 且速度 <5 cm/s，以 goal_success_rate 判断；
轨迹精度成功定义为完整存活且全程 RMSE <5 cm，另外记录末秒误差 <10 cm。

Figure8 nominal 与训练使用相同参考轨迹，holdout 检查不同初态和物理参数，不声称未见轨迹。
初态扰动实际只有位置 ±0.1 m；官方随机 reset 虽抽取速度 ±0.5 m/s，但随后 reset_velocity
会覆盖为精确参考速度。已实际检查 nominal/holdout 初速与参考的最大差均为 0，不能声称速度
holdout。DR 0.2 是官方 ranges 的 0.2 倍：
质量/惯量/电机动态 ±4%，推力/力矩曲线 ±2%，常值外力每轴 ±0.008 N、
常值力矩 ±4e-5 Nm，白噪声标准差 0.01 N/2e-4 Nm，动作延迟最多 8 ms。
DR 1.0 为上述 5 倍；两种 DR 同时包含初态扰动，均未用于训练。
TD3 只训练固定点到达，八字是未见任务，不能以这些迁移成绩代替专门高速跟踪训练。

参数缩放均为各 world、各电机/系数独立的均匀分布 U(1-r,1+r)；惯量逐轴缩放用
D J D 合同变换并同步更新逆矩阵，保持对称正定。常值外力/力矩为每轴均匀分布 U(-w,w)，
噪声为每个 500 Hz 物理 tick 独立的无截断零均值 Gaussian。延迟按 2 ms 物理 tick
在 0..max_ticks 离散均匀抽样；scale0.2 最大 4 ticks，scale1 最大 20 ticks。
scale1 的质量/惯量/电机系数缩放在 [0.8,1.2]，推力/力矩曲线在 [0.9,1.1]，
不会仅因这些正缩放而产生负质量或非正定惯量。但各曲线系数独立扰动，没有强制整个 RPM
范围的推力非负/单调或耦合物理一致性，也没有实测校准这些参数组合；Gaussian 外扰无硬上限。
因此强扰动只是公开定义的压力分布，不能把它称为所有物理机型的认证或真实故障分布。

首次训练计时含编译，后续相同形状复用编译。论文的训练秒数排除了 JIT 预热，不能直接比较。
保存每次实际 transitions、seed、完整超参数、源码 commit 和 checkpoint SHA256。

| 算法/任务 | 预算 | 实际步数/seed | 验证 RMSE mm（均值±std） | 验证完整时长/到点成功 |
|---|---:|---:|---:|---:|
"""
    for (algorithm, preset, budget), group in sorted(groups.items()):
        validation = "validation_hover" if algorithm == "td3" else "validation_nominal"
        values = np.array([r["evaluation"][validation]["tracking_rmse_m"] for r in group]) * 1000
        key = "goal_success_rate" if algorithm == "td3" else "full_horizon_completion_rate"
        rate = np.mean([r["evaluation"][validation][key] for r in group])
        text += (
            f"| {algorithm.upper()}/{preset} | {budget}x | {group[0]['actual_transitions']:,} | "
            f"{values.mean():.2f} ± {values.std(ddof=1):.2f} | {rate:.2%} |\n"
        )
    text += """
到点误差包含初始随机偏移，不是悬停稳态精度；到点任务主要看到点成功率与时间。
失败种子在以上均值中保留，没有仅用幸存者误差美化结果。

## 独立 validation 选择的策略

PPO/BPTT 按完整时长优先、nominal RMSE 最小选择；TD3 按验证到点成功率优先、平均到点时间最短选择。
视频固定 test world0，没有按测试画面挑 world。不同控制接口的数据不互相替代。

| 演示 | seed/预算 | validation | test 初态/到点 | test DR 0.2 | DR 1.0 边界 |
|---|---|---|---|---|---|
"""
    for name, r in selected.items():
        e = r["evaluation"]
        if r["algorithm"] == "td3":
            val, test = e["validation_hover"], e["holdout_hover"]
            text += (
                f"| {name} | {r['seed']} / {r['budget_multiplier']}x | "
                f"到点 {val['goal_success_rate']:.2%}, {val['mean_first_episode_seconds']:.3f}s | "
                f"到点 {test['goal_success_rate']:.2%}, {test['mean_first_episode_seconds']:.3f}s "
                "| 见未见任务表 | 见未见任务表 |\n"
            )
        else:
            val, test, dr, edge = [
                e[k]
                for k in (
                    "validation_nominal",
                    "holdout_initial_state",
                    "holdout_dynamics_0.2",
                    "boundary_dynamics_1.0",
                )
            ]
            cells = [f"{val['tracking_rmse_m'] * 1000:.2f}mm"] + [
                f"{v['tracking_rmse_m'] * 1000:.2f}mm / {v['full_horizon_completion_rate']:.2%}"
                for v in (test, dr, edge)
            ]
            text += (
                f"| {name} | {r['seed']} / {r['budget_multiplier']}x | "
                + " | ".join(cells)
                + " |\n"
            )
    text += """
## 直接电机 TD3 的未见轨迹边界

所有三种子平均，失败照常计入；到点策略的跨任务迁移边界不能当成 Crazyflow 的全局能力上限。

| 训练预算 | 未见八字周期 | RMSE (mm) | 完整时长存活 | 全程 <5cm 成功 |
|---|---:|---:|---:|---:|
"""
    for budget in (1, 2):
        for period in ("15", "5.5", "3.5"):
            entries = [
                r["evaluation"][f"holdout_figure8_{period}s"]
                for r in groups[("td3", "rotor_reach", budget)]
            ]
            text += (
                f"| {budget}x | {period}s | "
                f"{np.mean([e['tracking_rmse_m'] for e in entries]) * 1000:.2f} | "
                f"{np.mean([e['full_horizon_completion_rate'] for e in entries]):.2%} | "
                f"{np.mean([e['success_rate_rmse_5cm'] for e in entries]):.2%} |\n"
            )
    text += """
## DR 参数物理性审计

按本次相同 seed10001、64 worlds、cf21B_500 重新抽样参数：DR0.2 的惯量全部满足
正定及刚体三角不等式 I1+I2≥I3；DR1.0 虽保持正定，却有 **19/64** 违反三角不等式，
这些惯量组合不能由物理刚体实现。因此 DR1.0 的总失败率包含不物理压力样本，不能解释成
真实硬件失效概率或机型物理极限。质量在两档都为正。以原 rotor 动作区间
6962.07–21302.28 RPM 的 256 点网格检查，两档均未出现负推力或负推力斜率；
该有限采样检查不替代完整曲线/硬件物理认证。

[审计数据](official_dr_physical_audit_20261015.json) 保存范围、惯性条件与world索引，
每档 NPZ 保存实际质量/惯量/推力系数。用 official_dr_audit_20261015.py 可重新计算。

## 真正初速度与姿态 holdout

为补足官方reset速度覆盖，另外对上述5个冻结权重运行
[真正扰动报告](official_stress_20261015_results.md)：
每项64 worlds，速度独立seed11001，姿态独立seed11002；显式在全部reset后注入并刷新观测。
PPO慢/快、BPTT快两项均64/64完整存活且全程误差<5cm；BPTT慢的速度项完整存活64/64，
但精度成功62/64。所选rotor TD3速度项62/64到点、2个超时且无越界，姿态项64/64到点。
这些是新增初态分布的独立测试，不是 nominal 完美初始化下的泛化声明。

## 收敛证据与限制

慢 BPTT 的 seed44 在官方预算失稳，翻倍后仍失稳；PPO 慢配置翻倍会变差。快 BPTT 加倍改善
标称精度，但物理压力下仍会失败。TD3 加倍没有解决 3.5 s 未见轨迹，也没有稳定提升全部速度。
因此保留全部预算/种子，按独立验证集挑权重，不自动展示最长训练或最后一轮。
tracking_rmse_m 对所有world终止前有效样本加权，
提前失败会缩短有效时长，所以必须同时阅读失败率/完整时长；没有用幸存者误差替代主指标。

本机毫米级结果依赖精确状态观测、指定轨迹和标称仿真，不代表实飞已经达到该精度。
专门训练高速 rotor tracking、视觉导航、旋翼失效恢复和实机部署均未由这些实验验证。
[论文](https://arxiv.org/html/2606.01478v1) 的慢轨迹 PPO/BPTT 预算为 2M/100k，
快轨迹为 10M/800k，TD3 为 20M，
以及其真实飞行结果，属于不同实验。本地源码包括
[Figure8 配置](https://github.com/learnsyslab/crazyflow_experiments/blob/bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3/crazyflow_experiments/figure8/train_ppo.py)
和 [TD3 配置](https://github.com/learnsyslab/crazyflow_experiments/blob/bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3/crazyflow_experiments/learn2fly/train.py)。

![完整预算与全部种子曲线](official_policy_20261015_learning.png)
![固定world0实际轨迹和全部world误差带](official_policy_20261015_tracking.png)

## 名义物理下的速度泛化扫描与RNG审计

[冻结策略速度扫描](official_speed_sweep_20261015_results.md) 固定新test seed12001，每项64world，
不改preview或权重，只加快参考周期。PPO/BPTT在4.5s（峰速1.975m/s、峰加速度4.143m/s²）
均64/64全程误差<5cm，RMSE分别16.39/6.22mm。3.5s（峰速2.539m/s）时，
PPO完整60/64、精度成功58/64；BPTT两者26/64。2.5/1.5s两套冻结策略均全部提前失败。
这比不物理DR1.0更适合描述该冻结策略的速度迁移边界，仍不是专门重训后的机型物理极限。

另核查官方到点任务RNG：goal先使用独立split子key，再由sim reset将保留key拆成position、
velocity、angular-velocity子key。1024world、seed13001的实际pos-vel相关为0.001625；
仅用于检查key复用的扩大goal范围诊断，pos-goal相关为-0.000869，goal-pos误差方差明显非零。
默认官方TD3任务goal仍固定[0,0,1.5]，这项RNG诊断没有评估策略，不声称随机3D goal泛化。
[审计数据](official_rng_audit_20261015.json) 与 official_rng_audit_20261015.py 可复算。

## 固定 test world0 的实际 MuJoCo 回放

最直观的高速演示为 [PPO 4.5s周期](official_speed_sweep_20261015_ppo_T4.5s.mp4)，
这是5.5s训练权重在扫描中最快仍64/64全程<5cm通过的参考周期；
[同权重2.5s失败短片](official_speed_sweep_20261015_ppo_T2.5s.mp4) 保留实际失败。
两段固定新test seed12001/world0，回放与扫描raw的最大差均<0.1mm，未按新测试挑权重。

- [PPO快轨迹，20M预算](official_ppo_5_5s_budget2_20261015_seed44_holdout_initial_state.mp4)
- [BPTT快轨迹，1.6M预算](official_bptt_5_5s_budget2_20261015_seed44_holdout_initial_state.mp4)
- [直接电机TD3到点，40M预算](official_td3_rotor_reach_budget2_20261015_seed42_holdout_hover.mp4)
- [同一TD3的15s未见轨迹](official_td3_rotor_reach_budget2_20261015_seed42_holdout_figure8_15s.mp4)
- [3.5s迁移失败](official_td3_rotor_reach_budget2_20261015_seed42_holdout_figure8_3.5s.mp4)

全部来自冻结权重重新模拟，固定test seed10001/world0。对应 sidecar 记录结束原因、
权重SHA256以及与原批量评估的最大位置差（五段均<0.4mm）。

## 复现和证据

在 note.md 的研究 Docker 内、项目根目录运行：

~~~bash
python usage_assets/official_policy_20261015.py --algorithm bptt
python usage_assets/official_policy_20261015.py --algorithm ppo
python usage_assets/official_policy_20261015.py --algorithm td3
python usage_assets/official_policy_20261015.py --algorithm bptt --budget-multiplier 2
python usage_assets/official_policy_20261015.py --algorithm ppo --budget-multiplier 2
python usage_assets/official_policy_20261015.py --algorithm td3 --budget-multiplier 2
python usage_assets/official_report_20261015.py
~~~

依赖 fire/flax/jax/optax/wandb/NumPy/Matplotlib 均在现有容器内，W&B 上传禁用。
外部 checkout 默认 /tmp/crazyflow_experiments，不存在则自动 clone 官方并固定 commit；
eval-only 从 trained_models/official_*.ckpt 重载冻结策略，保留原训练元数据。
选择结果及 SHA256 见 official_policy_20261015_selected.json。

每个 seed 的 JSON 保存完整指标和每world RMSE，training.npz 保存每次迭代曲线。
场景 NPZ 的 full_error_sq/full_active/full_terminated/full_truncated/full_success 保留
64 worlds 每一步的误差与首episode mask，可复算全统计；pos/target/action 保存固定 world0。
旧批次未记录真值速度，velocity_estimate_finite_difference 明确为数值差分估计；
新批次的 vel/quat 和视频 sidecar 保留仿真真值状态。到点成功始终由原环境真值速度条件决定。

视频通过 official_replay_20261015.py 重载冻结权重、重新模拟并用 MuJoCo EGL 渲染。
sidecar JSON 保存 checkpoint SHA256、结束原因和与原评估最大位置差，
独立的 replay_states.npz 保存回放实际状态，不覆盖原批量评估 NPZ。

~~~bash
MUJOCO_GL=egl python usage_assets/official_replay_20261015.py \\
  --checkpoint trained_models/official_ppo_5_5s_budget2_20261015_seed44.ckpt \\
  --scenario holdout_initial_state
MUJOCO_GL=egl python usage_assets/official_replay_20261015.py \\
  --checkpoint trained_models/official_td3_rotor_reach_budget2_20261015_seed42.ckpt \\
  --scenario holdout_hover
MUJOCO_GL=egl python usage_assets/official_replay_20261015.py \\
  --checkpoint trained_models/official_td3_rotor_reach_budget2_20261015_seed42.ckpt \\
  --scenario holdout_figure8_3.5s
~~~
"""
    (ASSETS / "official_policy_20261015_results.md").write_text(text)
    print(f"Validated {len(records)} complete runs.")
    print({name: r["checkpoint"] for name, r in selected.items()})


if __name__ == "__main__":
    main()
