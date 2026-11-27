# 发布权重

Git 保存两个固定策略：最佳 Crazyflow 策略的 NPZ/PT 双导出，以及官方 baseline PT。历史中间权重已清理；后续训练的 checkpoint、优化器状态和试验模型写入忽略目录 `outputs/`。

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| `crazyflow/policy.npz` | 本分支按验证成功率选出的第 75 次续训策略，累计 233,472,000 次交互 | `ac8dda90bc960a1e683b63a6bd9697fe0a40e98735dc82e7a09811392b08c206` |
| `crazyflow/policy.pt` | 同一最佳策略的真实 Torch/RSL 推理导出，用于 Isaac Sim | `2d86bd1b3ccd7b38324ec85b4817298ac80cb60049791cfe59524722211e1b8c` |
| `baseline/model_36825.pt` | 官方预训练 baseline，在同一 Isaac 环境中公平比较 | `7c3ecc53bf468989789b1deee0c3efd6856438604aa5b0f19e35693527d15cc3` |

`baseline/agent.yaml`、`baseline/env.yaml` 保存原模型的配置；其中资产位置采用本项目内的路径。选模记录位于 [selection.json](../reports/crazyflow/selection.json)，最终评测位于 [reports/](../reports/)。

每次 JAX/Flax/Optax PPO 保存都在同一训练目录导出 `policy.npz` 和 `policy.pt`。NPZ 供原生 JAX、Crazyflow 评测与 MuJoCo 使用；PT 保存 Torch/RSL 张量，供 Isaac Sim 使用。Torch 只在 CPU 上完成序列化，训练仍由 JAX/Optax 执行。

以下命令从项目根目录运行；双格式导出与推理核验见 [`reports/dual-export.json`](../reports/dual-export.json)。

```bash
# 随仓库保留的最佳策略
./crazyrl play --checkpoint models/crazyflow/policy.npz
./crazyrl sim2sim compare --policy models/crazyflow/policy.pt --num-envs 32 --seed 42
# 自己训练后，选择同一目录中的对应格式
./crazyrl evaluate --checkpoint outputs/my-tof/policy.npz --episodes 256 --seed 9001
./crazyrl play --checkpoint outputs/my-tof/policy.npz
./crazyrl sim2sim play --policy outputs/my-tof/policy.pt --gui --num-envs 8 --max-steps 450
# 官方 baseline 使用现成权重
./crazyrl sim2sim play --policy models/baseline/model_36825.pt --gui --num-envs 8 --max-steps 450
```

现有最佳 PT 保留 actor 与 actor 归一化，是推理导出。后续训练保存实际可用的 critic 和 critic 归一化。恢复 JAX PPO 使用训练目录中的 `checkpoint.msgpack`，其中包含完整 Agent、Optax、RNG 和环境状态；部署 `.pt` 不用于训练恢复。

两个 actor 输入均为 **344 = 10×28 历史特征 + 当前 8×8 前向深度**，历史顺序、坐标系、通道倍率一致，各模型使用自己的归一化统计。输出均为四维原始动作，环境只执行一次 `tanh`，再映射为 roll/pitch 姿态角、yaw 角速度和总推力。434 维特权输入只供 critic 使用，不进入部署 actor。

逐通道物理含义见 [接口对照](../docs/tof-parity.md)，既有权重与 128 个闭环观测包的验收见 [checkpoint-io-audit.json](../reports/sim2sim/checkpoint-io-audit.json)。该记录中 NPZ 的 JAX/Torch 最大原始动作差为 `1.43e−6`。既有 Isaac 配对结果为最佳策略 132/160、官方 baseline 155/160；导出格式变化不构成新的训练或评测成绩。接口一致不表示 Crazyflow 与 Isaac 的动力学逐步相同。
