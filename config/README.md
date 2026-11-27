# ToF PPO 配置

`training.yaml` 是本分支唯一的任务/训练入口：`training` 控制 PPO，`environment` 控制 CrazyRL 任务、传感器、随机化和 Crazyflow 物理。运行 `./crazyrl train --config config/training.yaml`，有效配置冻结在 policy.npz 中。字段由 `crazyflow/tof/config.py` 验证。

`environment.model_params_path: null` 使用官方 cf21B_500 first_principles 参数。需要采用实测结果时填容器路径 `/outputs/measurements/physical.toml`，测量要求见[辨识手册](../docs/identification.md)。`vehicle.toml` 和 `controller.toml` 保留为 dani 辨识工具的默认模板；本任务不通过它们覆盖 CrazyRL PD 控制器。无人机 XML/STL 只保留 CF21B；`config/dynamics/` 中的四种官方动力学表和 `config/control/` 的 Mellinger 参数仍供辨识与官方加载器使用，默认 CF21B 数值不变。

训练使用原生 batched SimData 和 functional step；不为每个世界复制 MuJoCo 场景。ToF 在 JAX 中计算解析几何的 image-plane depth。MuJoCo 可视化显示训练物理产生的闭环轨迹，Isaac 是独立评测物理。
