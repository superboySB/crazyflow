# dani-crazyrl：Crazyflow / JAX PPO 的 ToF 避障

本分支用 Crazyflow 训练 CrazyRL baseline 的 ToF 避障任务，PPO、物理和并行 rollout 使用 JAX/Flax/Optax。MuJoCo 显示真实 Crazyflow 闭环轨迹；Isaac Sim 使用独立镜像评测现成策略，不在其中重新训练。

以下命令从项目根目录运行。宿主需要 Docker；GPU 运行另需 NVIDIA 驱动与 NVIDIA Container Toolkit。训练与 Isaac 镜像分别为 `crazyflow:dani-crazyrl`、`crazyflow:dani-crazyrl-eval`，对应 `docker/crazyflow.Dockerfile`、`docker/sim2sim.Dockerfile`。

仓库提供已选 Crazyflow 策略的 NPZ/PT 和官方 baseline PT。保留策略的 PT 为 actor-only 推理导出；新训练同时导出实际 actor/critic 及各自归一化。结构、来源和哈希见 [模型目录](models/README.md)。

## 1. 配置 Crazyflow 镜像

```bash
./crazyrl setup --device gpu
```

此命令构建训练、辨识和 MuJoCo 共用的镜像，并检查实际 JAX 设备。无 GPU 时可明确使用 `--device cpu`。

## 2. 检查训练与双格式导出

```bash
./crazyrl train --smoke --device gpu --config config/training.yaml --output outputs/smoke
```

smoke 执行短 PPO 训练并保存文件，不代表策略已经学会避障。训练始终在 JAX 中完成，Torch 只在 CPU 上序列化 PT。

## 3. 可选：参数辨识

没有实测数据时跳过本步。默认使用官方 CF21B `first_principles / cf21B_500`，质量 **0.0434 kg**，不自动采用示例辨识值。需要实测时先创建数据模板：

```bash
./crazyrl identify init --directory outputs/measurements
```

采集独立的训练与验证数据后，按 [参数辨识手册](docs/identification.md) 运行 flight/rotor/static 等工具。实测物理表通过 YAML 的 `environment.model_params_path` 指定容器内路径，例如 `/outputs/measurements/physical.toml`；默认保持 null。

## 4. 测量 GPU 并行容量

```bash
./crazyrl capacity --config config/training.yaml --output outputs/capacity
```

每档运行真实 rollout 和 PPO 更新，并采样显存；默认保护线为 9.5 GiB。RTX 4070 Ti SUPER 短测 **2,048 环境最快，约 434k transitions/s**；12,288 环境虽能运行，吞吐更低。2,048 环境的完整训练更快，但未观察到策略质量收益，不能按最大容量选默认值。数据与限制见 [训练研究](reports/training-study/README.md)。

## 5. 正式训练

```bash
./crazyrl train --device gpu --config config/training.yaml --output outputs/my-tof
```

当前默认是 `rsl_aligned`、**1,024 环境 / 32 minibatches / 1,000 更新**，对应实际测试过的 A 配方、122.88M PPO transitions。RTX 4070 Ti SUPER 完整调用约 **11.24 分钟**，这是探索预算的实测时间，不保证训练结果超过仓库保留策略；配置以 [`training.yaml`](config/training.yaml) 为准。

每次保存同时生成同一策略的两种导出：

| 文件 | 用途 |
| --- | --- |
| `outputs/my-tof/policy.npz` | 原生 JAX 推理、Crazyflow 评测和 MuJoCo |
| `outputs/my-tof/policy.pt` | 真正的 Torch/RSL 文件，供 Isaac Sim 推理 |
| `outputs/my-tof/checkpoint.msgpack` | 完整 Agent、Optax、RNG 和环境状态，用于恢复 JAX PPO |

新训练的 PT 包含实际可用的 actor、critic 和各自归一化；现有保留模型的 PT 是 actor 与 actor 归一化的推理导出。继续训练使用 `checkpoint.msgpack`，部署 PT 不用于恢复 Optax。配置、曲线和日志也写入此运行目录；`outputs/` 中间产物不进入 Git。

## 6. Crazyflow 留出评测

明确选择自己训练的 NPZ，在新地图上运行：

```bash
./crazyrl evaluate --checkpoint outputs/my-tof/policy.npz --episodes 256 --seed 14001 --map-seed 14001 --output outputs/my-evaluation
```

`--seed` 改变初态与测量随机数，`--map-seed` 另外生成地图；其余物理使用模型冻结配置。评测按每个世界的首个回合统计，失败保留在分母，保存汇总、逐回合 CSV 和轨迹。截短用的 `--max-steps` 不作为完整策略成绩。

## 7. MuJoCo 实时交互

```bash
./crazyrl play --checkpoint outputs/my-tof/policy.npz
```

直接查看仓库保留的现有模型时，明确使用它的路径：

```bash
./crazyrl play --checkpoint models/crazyflow/policy.npz
```

JAX 逐步运行策略与 Crazyflow 物理，回合结束自动继续。右侧显示前方 **8×8 深度数值/热图**，以及左、右、后、下 ToF，单位 m。

| 控制 | 功能 |
| --- | --- |
| `R` / Reset | 同组地图重新采样初态和目标 |
| `Backspace` | 连随机状态一起精确重开当前回合 |
| `Space` / Pause | 暂停与继续 |
| `C` / Camera；`F` | 切换跟随/全景/FPV；回到跟随 |
| `T` | 切换干净测距与 actor 实际收到的带噪声输入 |
| 鼠标左键、右键、滚轮 | 旋转、平移、缩放 |
| `+`、`-`；`Esc` | 调速；关闭 |

导出实际闭环轨迹视频：

```bash
./crazyrl play --checkpoint outputs/my-tof/policy.npz --headless --output outputs/avoidance.mp4
```

MuJoCo 负责显示，不是另一套评测动力学。已保存的轨迹可用 `--evaluation` 精确回放，交互验收见 [现有模型说明](models/crazyflow/README.md)。

## 8. 构建独立 Isaac 镜像

```bash
./crazyrl sim2sim build
```

评测任务、控制/动力学与必要 USD 在 `sim2sim/reference/`，官方权重在 `models/baseline/model_36825.pt`，不依赖其他项目目录。

## 9. 与官方 baseline 配对比较

使用同次训练导出的 PT：

```bash
./crazyrl sim2sim compare --policy outputs/my-tof/policy.pt --num-envs 32 --seed 42 --output outputs/sim2sim/my-tof-seed42
```

`compare` 自动加载官方 baseline，再加载指定策略；仅接受相同初始场景哈希，每个环境各统计首个回合。正式比较应重复多个种子，并保留全部失败。训练物理与预算不同，场景配对不表示两个策略经历逐时刻相同的随机噪声。

## 10. Isaac 模型播放

播放自己训练的 PT：

```bash
./crazyrl sim2sim play --policy outputs/my-tof/policy.pt --gui --num-envs 8 --max-steps 450
```

播放原官方权重：

```bash
./crazyrl sim2sim play --policy models/baseline/model_36825.pt --gui --num-envs 8 --max-steps 450
```

## 11. 查看官方历史 TensorBoard

```bash
./crazyrl sim2sim tensorboard --port 6006
```

访问 `http://127.0.0.1:6006`。它读取 `outputs/sim2sim/tensorboard/` 中的官方历史事件；新克隆需先按 [baseline 归档说明](models/baseline/README.md) 恢复忽略文件。原 baseline 使用 PyTorch/RSL，TensorBoard 是日志工具，并非 TensorFlow 训练；本分支的新训练使用 JAX。

## 结果与详细说明

最新 Isaac 配对留出（种子 431–435）中，保留策略为 **141/160（88.125%）**，官方 baseline 为 **157/160（98.125%）**。新训练及预算续训均未通过替换门槛，没有证明迁移性能提升，最终保留旧 NPZ/PT。

完整训练、不同种子的历史结果和模型选择依据见 [训练研究](reports/training-study/README.md)。

- [任务接口、动作单位、奖励、ToF 与随机化](docs/tof-parity.md)
- [现有 Crazyflow 模型与验收](models/crazyflow/README.md)
- [官方 baseline 来源、配置、全部历史资产和训练日志](models/baseline/README.md)
- [GPU 容量、训练研究与模型选择](reports/training-study/README.md)
- [历史训练与配对评测记录](reports/README.md)

## 项目结构

| 目录 | 内容 |
| --- | --- |
| `crazyflow/tof/` | ToF 任务、JAX PPO、导出和可视化 |
| `config/`、`docker/`、`scripts/` | 配置、两套镜像和命令入口 |
| `models/` | 保留策略、baseline 和来源说明 |
| `sim2sim/` | Isaac 评测入口及冻结的参考任务与资产 |
| `docs/`、`reports/`、`tests/` | 使用手册、可追溯结果和测试 |
| `outputs/` | 本地运行产物与原始归档，Git 忽略 |

## 许可证与来源

Crazyflow 核心沿用 [MIT 许可证](LICENSE)。打包的 CrazyRL 参考文件保留其 [GPLv3 许可证](sim2sim/reference/LICENSE)，来源和文件哈希见 [provenance.json](sim2sim/provenance.json)。参数辨识参考文档的来源与许可证保留在 [docs/crazyflow_experiments_sysid/](docs/crazyflow_experiments_sysid/)。
