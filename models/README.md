# 保留的模型

Git 保存两个策略、三个权重文件；训练中间模型与完整优化器 checkpoint 写入忽略目录 `outputs/`。

| 文件 | 用途 | 说明 |
| --- | --- | --- |
| `crazyflow/policy.npz` | Crazyflow/JAX 评测与 MuJoCo | [现有 Crazyflow 模型](crazyflow/README.md) |
| `crazyflow/policy.pt` | 同一 actor 的 Torch/RSL 推理，供 Isaac Sim | [双导出验收](../reports/dual-export.json) |
| `baseline/model_36825.pt` | 官方现成 CrazyRL baseline | [来源、配置、训练日志与全部资产](baseline/README.md) |

最终保留的是原 legacy actor 与归一化推理权重，NPZ/PT 哈希未变。新 `rsl_aligned` 训练及预算续训均未通过替换门槛；实际训练的完整 actor/critic 和两套归一化双导出已验证，但未作为交付模型保留。恢复 JAX/Optax 训练使用运行目录的 `checkpoint.msgpack`，部署 PT 不含 Optax 状态。

最新 Isaac 配对留出种子 431–435：保留策略 **141/160**、baseline **157/160**。同一策略的历史种子 42–46 为 **132/160**、baseline **155/160**；种子不同，不代表模型提升。模型来源见各子目录，训练预算、锁定和最终不发布新候选的依据见 [训练研究](../reports/training-study/README.md)。
