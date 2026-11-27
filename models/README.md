# 保留的模型

Git 保存两个策略、三个权重文件；训练中间模型与完整优化器 checkpoint 写入忽略目录 `outputs/`。

| 文件 | 用途 | 说明 |
| --- | --- | --- |
| `crazyflow/policy.npz` | Crazyflow/JAX 评测与 MuJoCo | [现有 Crazyflow 模型](crazyflow/README.md) |
| `crazyflow/policy.pt` | 同一 actor 的 Torch/RSL 推理，供 Isaac Sim | [双导出验收](../reports/dual-export.json) |
| `baseline/model_36825.pt` | 官方现成 CrazyRL baseline | [来源、配置、训练日志与全部资产](baseline/README.md) |

保留的 Crazyflow 策略使用历史 `legacy` PPO，其 PT 包含 actor 和观测归一化，用于推理。新训练的 PT 包含实际 actor/critic 及各自归一化；恢复 JAX/Optax 训练使用运行目录的 `checkpoint.msgpack`。训练来源见各子目录，实验及选模依据见 [训练研究](../reports/training-study/README.md)。
