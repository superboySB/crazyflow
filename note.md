# 从训练到 sim2sim

完整步骤见 [README](README.md)。以下从项目根目录运行，每个代码块只执行一条命令。

构建并检查 Crazyflow 镜像：

```bash
./crazyrl setup --device gpu
```

先跑短 PPO 和双格式导出：

```bash
./crazyrl train --smoke --device gpu --config config/training.yaml --output outputs/smoke
```

没有实测数据时沿用官方 CF21B **0.0434 kg**，跳过真实辨识；需要采集数据时先创建模板，再按 [辨识手册](docs/identification.md) 操作：

```bash
./crazyrl identify init --directory outputs/measurements
```

测量本机完整 PPO 容量：

```bash
./crazyrl capacity --config config/training.yaml --output outputs/capacity
```

使用显式配置与输出目录正式训练：

```bash
./crazyrl train --device gpu --config config/training.yaml --output outputs/my-tof
```

默认 `rsl_aligned`、1,024 环境、32 minibatches、1,000 更新，对应实测 A 配方；本机约 11.24 分钟是探索预算，不保证超过保留模型。2,048 环境短测最快约 434k transitions/s，完整训练也更快，但本次未带来策略质量收益。

每次保存同时产生 `policy.npz` 和 `policy.pt`。训练是 JAX/Flax/Optax，Torch 只在 CPU 上序列化；恢复训练使用 `checkpoint.msgpack`。

用同一目录的 NPZ 做新地图评测：

```bash
./crazyrl evaluate --checkpoint outputs/my-tof/policy.npz --episodes 256 --seed 14001 --map-seed 14001 --output outputs/my-evaluation
```

打开实时 MuJoCo 窗口：

```bash
./crazyrl play --checkpoint outputs/my-tof/policy.npz
```

`R` 换初态/目标，`Backspace` 精确重开，空格暂停，`C` 切换视角，`T` 切换干净/带噪声测距。面板显示前向 8×8 和左、右、后、下 ToF，单位 m；回合结束自动继续，Esc 关闭。

构建第二个独立 Isaac 镜像：

```bash
./crazyrl sim2sim build
```

用同目录的 PT 与自动加载的官方 baseline 公平比较：

```bash
./crazyrl sim2sim compare --policy outputs/my-tof/policy.pt --num-envs 32 --seed 42 --output outputs/sim2sim/my-tof-seed42
```

播放自己的 PT：

```bash
./crazyrl sim2sim play --policy outputs/my-tof/policy.pt --gui --num-envs 8 --max-steps 450
```

播放现成官方权重：

```bash
./crazyrl sim2sim play --policy models/baseline/model_36825.pt --gui --num-envs 8 --max-steps 450
```

查看本地归档的官方历史日志，浏览器访问 `http://127.0.0.1:6006`：

```bash
./crazyrl sim2sim tensorboard --port 6006
```

TensorBoard 是日志展示工具；原 baseline 使用 PyTorch/RSL，本分支训练使用 JAX。新克隆恢复原始事件的方法见 [baseline 说明](models/baseline/README.md)。

直接看仓库保留模型时，用 `models/crazyflow/policy.npz` 做 Crazyflow/MuJoCo，用对应 `models/crazyflow/policy.pt` 做 Isaac。新实验未通过替换门槛，最终保留旧 legacy actor-only 权重；新训练的完整 actor/critic 双导出已验证，但没有作为交付模型替换。

最新 Isaac 配对留出种子 431–435：保留策略 **141/160**、官方 baseline **157/160**。历史种子 42–46 为同一策略 **132/160**、baseline **155/160**；不同种子成绩不能解释成提升。来源和未发布原因见 [模型目录](models/README.md)、[训练研究](reports/training-study/README.md) 和 [任务对应](docs/tof-parity.md)。
