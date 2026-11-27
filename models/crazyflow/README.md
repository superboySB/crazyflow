# 当前保留的 Crazyflow 策略

这里最终保留的是历史训练中选出的 **legacy Stage C 第 75 次更新**。`policy.npz` 与 `policy.pt` 包含同一个 actor；PT 由原 NPZ 导出，没有重新训练。权重身份由下表 SHA256 固定。

| 文件 | 用途 | SHA256 |
| --- | --- | --- |
| `policy.npz` | JAX 评测、Crazyflow 闭环与 MuJoCo 可视化 | `ac8dda90bc960a1e683b63a6bd9697fe0a40e98735dc82e7a09811392b08c206` |
| `policy.pt` | Torch/RSL 推理，供 Isaac Sim | `2d86bd1b3ccd7b38324ec85b4817298ac80cb60049791cfe59524722211e1b8c` |

该历史模型只保留 actor、观测归一化与部署所需信息；没有保留已训练 critic 或完整优化器状态。PT 不补造这些参数，也不是恢复 Optax 训练的 checkpoint。新训练每次保存同时导出 NPZ/PT，其真实 critic 和独立归一化已用实际 A 模型验收，但该新候选未被交付替换。恢复 JAX 训练使用运行目录的 `checkpoint.msgpack`。格式和数值验收见[原保留策略双导出](../../reports/dual-export.json)与[实际新模型导出](../../reports/training-study/export-parity.json)。

## 训练来源与预算

训练使用 JAX / Flax / Optax PPO，通过 Crazyflow 原生 `first_principles` 动力学和 RK4 采样，不对物理轨迹做反向传播。以下是产生当前策略的历史训练链，数据来自[训练记录](../../reports/crazyflow/results.json)。

| 阶段 | 并行环境 × rollout 步数 × 更新次数 | 实际执行 transitions | 训练墙钟时间 |
| --- | --- | ---: | ---: |
| Stage A 预训练 | 1,024 × 120 × 1,000 | 122,880,000 | 669.09 秒 |
| Stage C 修正后的续训 | 12,288 × 120 × 100 | 147,456,000 | 941.50 秒 |

Stage A 早于圆柱轴向射线求交修正，存在下视 ToF 在特定平行姿态下误报零距离的问题。Stage C 使用修正后的传感器几何，以 warm-start 继承 Stage A 的 Agent、Adam 和归一化状态，重新创建环境与课程；这与本次 A→E 连同环境/RNG 一起恢复的 exact-resume 不同。四个候选的验证和最终测试也都使用修正后的几何。中途停止的诊断运行不属于当前模型训练链。

两个完整阶段实际执行 **270,336,000 transitions**。选中的是 Stage C 的第 75 次更新，因此该模型累计经历 **233,472,000 transitions**：Stage A 的 122,880,000，加上 Stage C 前 75 次的 110,592,000。Stage C 最后 25 次更新已经执行，但未进入所选模型。

这条历史链使用 `legacy` PPO：每次更新分 32 个 minibatch、训练 5 个 epoch，按 minibatch 归一化 advantage，以采样的近似 KL 调节学习率，观测归一化统计在整个训练过程中持续更新。它复现了 CrazyRL 的核心状态、动作、奖励和 domain randomization 约定，**不能称为与 RSL 完全相同的 PPO 实现**。Crazyflow 的电机/刚体动力学及解析碰撞判定也与 Isaac Sim 的 PhysX 模型不同，详见[任务与实现差异](../../docs/tof-parity.md)。

新的 `rsl_aligned` 对照及预算续训均未通过替换门槛，本目录权重保持不变。该研究的 A/B/C 实验编号与上述历史 Stage A/C 无关，预算及验收过程见[训练研究](../../reports/training-study/README.md)。

## 选择与已有成绩

仅比较修正后 Stage C 的第 25、50、75、100 次更新：固定地图种子 42、验证种子 9001，每个候选完整评测 256 个环境的首个 episode，包含所有失败。预先确定的规则是成功数最多，其次平均 return 最高；第 75 次更新以 255/256 成功、平均 return 27.0345 入选，见[选择记录](../../reports/crazyflow/selection.json)。

锁定模型后，使用新地图种子与初态种子 14001，仍为 256 个首个 episode，得到 **255/256 成功、1 次碰撞**，平均 return 24.8927。这是 Crazyflow 测试成绩。独立 Isaac Sim 配对评测使用 5 个共享地图种子、每个策略 160 个 episode，本模型 **132/160（82.5%）**，官方 baseline **155/160（96.875%）**；没有达到 baseline，见[配对报告](../../reports/sim2sim/final/summary.json)。两种仿真器的成绩不能混作同一测试。

最近一次 Isaac 配对留出使用另一组种子 **431–435**，本模型 **141/160（88.125%）**，官方 baseline **157/160（98.125%）**，见[新留出报告](../../reports/training-study/evaluation/heldout/summary.json)。两组中的本模型、baseline 权重均未改变，种子 42–46 与 431–435 的成绩差异不是训练提升；160 回合来自每组 5 张共享地图，不能当作 160 张独立地图。第二轮因开发门槛未通过，没有运行 441–445。

## 显式加载当前模型

从仓库根目录启动 Crazyflow 实时闭环，MuJoCo 负责显示：

```bash
./crazyrl play --checkpoint models/crazyflow/policy.npz
```

复查已公开的新地图测试协议：

```bash
./crazyrl evaluate --checkpoint models/crazyflow/policy.npz --episodes 256 --seed 14001 --map-seed 14001 --output outputs/crazyflow-best-check
```

在独立 Isaac Sim 镜像中加载同一 actor，并自动配对官方 baseline；下面是种子 42 的单次比较，不等同于历史五种子汇总：

```bash
./crazyrl sim2sim compare --policy models/crazyflow/policy.pt --num-envs 32 --seed 42 --output outputs/sim2sim/crazyflow-best-seed42
```

镜像构建、训练和完整使用顺序见[主 README](../../README.md)，官方 baseline 的来源、原始训练日志与截图见[baseline 说明](../baseline/README.md)。
