# 保留的验收记录

这里保留最佳 Crazyflow 策略和官方 baseline 的结果及核验依据。运行命令见[主 README](../README.md)，两份权重及哈希见 [`models/`](../models/)。

| 路径 | 内容 |
| --- | --- |
| [`repository-cleanup.json`](repository-cleanup.json) | 本次目录整理、权重清理、双镜像独立构建、完整 CPU 检查和新旧镜像行为对照 |
| [`crazyflow/`](crazyflow/README.md) | 训练预算与曲线、四候选选择、固定地图验证、新地图测试、GPU 容量及实时 GUI 证据 |
| [`sim2sim/final/`](sim2sim/final/) | 两个策略的 320 条配对回合、五组相同初始场景、推理一致性和原生 GUI/video 验收 |
| [`sim2sim/checkpoint-io-audit.json`](sim2sim/checkpoint-io-audit.json) | 实际 NPZ/PT 的输入、历史、物理通道、归一化及动作接口核验 |
| [`baseline-reference/`](baseline-reference/) | 官方既有 1,280 回合验收的协议、汇总和逐回合 CSV；与本次 160 回合 baseline 单独列出 |

上游任务固定到 CrazyRL 修订 `9333442c18fa15799adac7e02988a978aeaf97b9`；文件的上游相对路径、本仓库位置和 SHA256 见 [`sim2sim/provenance.json`](../sim2sim/provenance.json)。运行只读取本仓库保存的任务、模型和 USD 资产。

目录整理对历史 JSON 中的模型、报告及运行产物路径做了规范化，保留原始数值、时间、模型/源码/场景哈希和退出码。宿主仓库挂载前缀写作 `${REPO}`，容器用户目录写作 `${CONTAINER_HOME}`；报告中的文件位置使用本仓库相对路径。USD 的 `/World/...` 是场景 prim 标识，不是宿主文件路径。历史 `command` 字段是规范化的执行记录，当前可复制运行的命令以主 README 为准。

路径整理没有重新运行历史评测，也没有更新旧镜像 ID 或源码哈希。重复的早期 smoke/转换检查、历史逐种子副本和训练中间模型已清理；最终配对评测保留逐种子记录，以核验初始场景与随机化的一致性。新实验、完整优化器 checkpoint、缓存、TensorBoard 事件和视频写入忽略目录 `outputs/`。
