# 发布权重

Git 只保存两份部署权重。本次历史中间权重已清理；后续训练的 checkpoint、优化器状态和试验模型仅写入被忽略的 `outputs/`，不加入版本控制。

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| `crazyflow/policy.npz` | 本分支按验证成功率选出的第 75 次续训策略，累计 233,472,000 次交互 | `ac8dda90bc960a1e683b63a6bd9697fe0a40e98735dc82e7a09811392b08c206` |
| `baseline/model_36825.pt` | 官方预训练 baseline，在同一 Isaac 环境中公平比较 | `7c3ecc53bf468989789b1deee0c3efd6856438604aa5b0f19e35693527d15cc3` |

`baseline/agent.yaml`、`baseline/env.yaml` 保存原模型的配置；其中资产位置采用本项目内的路径。选模记录位于 [selection.json](../reports/crazyflow/selection.json)，最终评测位于 [reports/](../reports/)。

两个 actor 输入均为 **344 = 10×28 历史特征 + 当前 8×8 前向深度**，历史顺序、坐标系、通道倍率一致，各模型使用自己的归一化统计。输出均为四维原始动作，环境只执行一次 `tanh`，再映射为 roll/pitch 姿态角、yaw 角速度和总推力。434 维特权输入只供 critic 使用，不进入部署 actor。

逐通道物理含义见 [接口对照](../docs/tof-parity.md)，实际权重与 128 个闭环观测包的验收见 [checkpoint-io-audit.json](../reports/sim2sim/checkpoint-io-audit.json)。NPZ 在 JAX 和 Torch 适配器中的最大原始动作差为 `1.43e−6`。接口一致不表示 Crazyflow 与 Isaac 的动力学逐步相同。
