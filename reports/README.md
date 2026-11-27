# 保留的验收记录

这里保留历史选定 Crazyflow 策略、官方 baseline 的结果和已完成的新训练研究。运行命令见[主 README](../README.md)，模型及哈希见 [`models/`](../models/)。两轮新候选均未通过替换门槛，最终保留原 legacy actor-only NPZ/PT；训练在线指标、开发选择与独立测试分别记录。

| 路径 | 内容 |
| --- | --- |
| [`dual-export.json`](dual-export.json) | 同一策略的 NPZ/PT 双导出、数值一致性、实际 PPO 保存和 Isaac 闭环对照 |
| [`repository-cleanup.json`](repository-cleanup.json) | 历史目录与权重整理、双镜像构建、CPU 验收和镜像行为对照 |
| [`crazyflow/`](crazyflow/README.md) | 历史 legacy PPO 的两阶段训练、四候选选择、固定地图验证、新地图测试、GPU 容量及实时 GUI 证据 |
| [`training-study/`](training-study/README.md) | PPO/环境审计、等预算 A/B/C、第一轮留出与未发布结论、另行登记的预算续训 |
| [`training-study/evaluation/`](training-study/evaluation/manifest.json) | 第一轮 181 份精简 CF/开发/留出证据；最新旧模型 141/160、baseline 157/160 |
| [`training-study/round2/`](training-study/round2/training.json) | 245.76M 后续预算、连续曲线、开发门槛失败及 14 份精简评测证据；未运行新留出 |
| [`sim2sim/final/`](sim2sim/final/) | 两个策略的 320 条配对回合、五组相同初始场景、推理一致性和原生 GUI/video 验收 |
| [`sim2sim/checkpoint-io-audit.json`](sim2sim/checkpoint-io-audit.json) | 实际 NPZ/PT 的输入、历史、物理通道、归一化及动作接口核验 |
| [`baseline-reference/`](baseline-reference/) | 官方既有 1,280 回合验收的协议、汇总和逐回合 CSV；与五种子配对评测分别记录 |

上游任务固定到 CrazyRL 修订 `9333442c18fa15799adac7e02988a978aeaf97b9`；文件的上游相对路径、本仓库位置和 SHA256 见 [`sim2sim/provenance.json`](../sim2sim/provenance.json)。运行只读取本仓库保存的任务、模型和 USD 资产。

`rsl_aligned` 对齐了 advantage、解析 KL/学习率、Adam epsilon 和归一化统计停止条件，仍保留 JAX 的 rollout 内冻结统计等差异；它不表示全部 PPO 语义、训练预算、动力学或碰撞几何相同。当前主配置为 1,024 环境、32 minibatches、1,000 更新、半径 0.055 m，0.09 m 的研究候选未带来 Isaac 开发收益。精确公式与边界见[任务对应](../docs/tof-parity.md)。当前配方实测约 11.24 分钟，是探索预算，不保证超过保留模型。

历史种子 42–46 的旧模型 132/160、baseline 155/160，与最新种子 431–435 的 141/160、157/160 使用相同权重、不同场景种子，不能解释为模型提升。历史报告中的并行设置、当时模型清单和测试数量按记录时版本理解；当前配置与三份权重以主 README 和模型目录为准。

目录整理对历史 JSON 中的模型、报告及运行产物路径做了规范化，保留原始数值、时间、模型/源码/场景哈希和退出码。宿主仓库挂载前缀写作 `${REPO}`，容器用户目录写作 `${CONTAINER_HOME}`；报告中的文件位置使用本仓库相对路径。USD 的 `/World/...` 是场景 prim 标识，不是宿主文件路径。历史 `command` 字段是规范化的执行记录，当前可复制运行的命令以主 README 为准。

路径整理没有重新运行历史评测，也没有更新旧镜像 ID 或源码哈希。重复的早期 smoke/转换检查、历史逐种子副本和训练中间模型已清理；最终配对评测保留逐种子记录，以核验初始场景与随机化的一致性。新实验、完整优化器 checkpoint、缓存、TensorBoard 事件和视频写入忽略目录 `outputs/`。
