"""Plot Chinese bid figures directly from saved training and simulation records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE / "charts"
plt.rcParams.update(
    {
        "font.family": "Noto Sans CJK SC",
        "font.size": 11,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def neural_figures() -> None:
    """Display the recorded loss, held-out geometry, and actual execution."""
    data = np.load(HERE / "neural_planner_trajectory.npz")
    result = json.loads((HERE / "neural_planner_results.json").read_text())
    predictions, labels = data["test_predictions"], data["test_labels"]
    np.testing.assert_allclose(
        np.sqrt(np.mean((predictions - labels) ** 2)), result["test_waypoint_rmse_m"]
    )
    query_x = np.linspace(-1, 1, 601)
    path_x = np.linspace(-1, 1, predictions.shape[1])
    distances = []
    for obstacle, prediction in zip(data["test_obstacles"], predictions, strict=True):
        query_y = np.interp(query_x, path_x, prediction)
        distances.append(
            np.min(np.hypot(query_x - obstacle[0], query_y - obstacle[1]) - obstacle[2])
        )
    np.testing.assert_allclose(np.min(distances), result["test_center_clearance_min_m"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    axes[0].semilogy(np.arange(1, len(data["losses"]) + 1), data["losses"], color="#23639b")
    axes[0].set(
        title="监督学习训练：2,048 个场景，400 轮", xlabel="训练轮次", ylabel="路径点均方误差（m²）"
    )
    axes[1].hist(distances, bins=18, color="#287ca6", edgecolor="white")
    axes[1].axvline(0, color="#b23b33", linestyle="--", label="圆柱表面")
    axes[1].set(
        title="独立测试：256 个单障碍场景",
        xlabel="规划中心线到障碍表面的最小采样距离（m）",
        ylabel="场景数量",
    )
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(OUT / "neural_training_zh.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    path, actual, obstacle = data["planned"], data["actual"], data["obstacle"]
    ax.plot(path[:, 0], path[:, 1], "--", color="#e58a21", linewidth=2.5, label="网络输出路径")
    ax.plot(actual[:, 0], actual[:, 1], color="#23639b", linewidth=2, label="动力学闭环实际轨迹")
    ax.add_patch(Circle(obstacle[:2], obstacle[2], color="#b23b33", alpha=0.75, label="圆柱障碍"))
    ax.scatter(path[0, 0], path[0, 1], s=60, color="#3f9062", zorder=4, label="起点")
    ax.scatter(path[-1, 0], path[-1, 1], s=100, marker="*", color="#884a9a", zorder=4, label="目标")
    ax.set(
        title="端到端路径输出与实际绕障执行：独立测试第 0 场景",
        xlabel="水平位置 x（m）",
        ylabel="水平位置 y（m）",
        aspect="equal",
    )
    ax.legend(ncol=2, loc="upper right")
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "neural_execution_zh.png", dpi=160)
    plt.close(fig)


def rl_figure() -> None:
    """Plot saved PPO history and the independently replayed 32-world evaluation."""
    checkpoint = torch.load(HERE / "rl_cpu_medium.pt", map_location="cpu", weights_only=False)
    history = checkpoint["history"]
    behavior = np.load(HERE / "presentation/rl_replay_trace.npz")
    metrics = json.loads((HERE / "presentation/rl_replay_metrics.json").read_text())
    assert bool(behavior["per_env_success"].all())
    steps = np.asarray(history["timesteps"]) / 1e4
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    axes[0].plot(steps, history["mean_distance"], color="#d27529")
    axes[0].set(title="PPO 训练采样批次误差", xlabel="交互槽位（万）", ylabel="平均位置误差（m）")
    axes[1].plot(steps, history["mean_reward"], color="#29947c")
    axes[1].set(title="训练环境原始奖励", xlabel="交互槽位（万）", ylabel="每步平均奖励")
    axes[2].bar(np.arange(1, 33), behavior["per_env_last_rmse"], color="#277ca7")
    axes[2].axhline(0.10, linestyle="--", color="#b23b33", label="末秒误差阈值 0.10 m")
    axes[2].set(
        title="独立 32 环境：32/32 达标",
        xlabel="独立测试环境编号",
        ylabel="末秒位置 RMSE（m）",
        ylim=(0, 0.12),
    )
    axes[2].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        "行为克隆初始化 + PPO｜38,400 教师样本｜523,264 有效交互｜"
        f"独立测试末秒 RMSE {metrics['last_second_rmse']:.5f} m",
        fontsize=13,
    )
    fig.savefig(OUT / "rl_training_zh.png", dpi=160)
    plt.close(fig)


def swarm_figure() -> None:
    """Plot four actual paths and measured distances without changing the records."""
    data = np.load(HERE / "swarm_data.npz")
    metrics = json.loads((HERE / "swarm_metrics.json").read_text())
    colors = ["#d65440", "#2c9cba", "#008a70", "#9d79bd"]
    actual, desired = data["actual"], data["desired"]
    centers = (np.asarray(metrics["obstacles"]) - 4) * metrics["cell_m"]
    offsets = np.maximum(np.abs(actual[:, :, None, :2] - centers[None, None]) - 0.16, 0)
    clearance = np.linalg.norm(offsets, axis=-1).min(axis=(1, 2))
    np.testing.assert_allclose(
        clearance.min(), metrics["min_horizontal_obstacle_surface_distance_m"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for center in centers:
        axes[0].add_patch(Rectangle(center - 0.16, 0.32, 0.32, color="#ae5754", alpha=0.8))
    for index, color in enumerate(colors):
        axes[0].plot(desired[:, index, 0], desired[:, index, 1], "--", color=color, alpha=0.5)
        axes[0].plot(
            actual[:, index, 0],
            actual[:, index, 1],
            color=color,
            label=f"无人机 {index + 1} 实际轨迹",
        )
        axes[0].scatter(*actual[0, index, :2], color=color, marker="o", s=45)
        axes[0].scatter(*desired[-1, index, :2], color=color, marker="*", s=100)
    axes[0].set(
        title="四机交叉任务：虚线规划，实线执行",
        xlabel="位置 x（m）",
        ylabel="位置 y（m）",
        aspect="equal",
    )
    axes[0].legend(fontsize=9)
    axes[1].plot(
        data["time"], data["pair_distances"].min(axis=1), label="最小机心间距", color="#3b6fb6"
    )
    axes[1].plot(data["time"], clearance, label="最小机心到障碍表面水平距离", color="#ce6845")
    axes[1].set(title="100 Hz 状态记录：机间与障碍间距", xlabel="时间（s）", ylabel="距离（m）")
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(OUT / "swarm_metrics_zh.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Build figures and record the hashes of the underlying experiment files."""
    OUT.mkdir(exist_ok=True)
    neural_figures()
    rl_figure()
    swarm_figure()
    inputs = [
        "neural_planner_trajectory.npz",
        "neural_planner_results.json",
        "swarm_data.npz",
        "swarm_metrics.json",
        "rl_cpu_medium.pt",
        "presentation/rl_replay_trace.npz",
        "presentation/rl_replay_metrics.json",
    ]
    manifest = {
        "method": "Chinese scientific plots drawn directly from unchanged recorded experiment data",
        "inputs": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in inputs},
        "outputs": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in OUT.glob("*.png")
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
