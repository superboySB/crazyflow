# 可复现训练权重

全部类别的权重、SHA256及选中/完成预算索引见[manifest_20261015.json](manifest_20261015.json)；下文49个仅指official算法族，Gym/Racing最终权重见文末。

本目录包含49个已跑完整预算、从头训练的官方算法族checkpoint：原官方配置研究30个，
本地专门高速扩展19个；总实际 **530,285,056 transitions**。权重合计694,845 bytes。
这些是本地训练结果，未从官方仓库下载预训练策略。

[严格JSON清单](official_manifest_20261015.json)逐项保存全部49个权重的SHA256、完整训练配置、
实际预算、固定源码版本、验证选择角色和冻结测试记录路径；不含pickle中的参数数组。
未选中、延长后退化和精度失败的完整训练也全部保留。论文参数是可靠起点，有限训练和验证
无法证明全局最优；应同时查看成功率、完整时长、误差及学习曲线。

## 建议查看的冻结策略

选择只使用validation seed9001；测试成绩用于报告，不用于换权重或挑选视频world。
原五项在各算法/任务的三个seed和1x/2x预算中选择：轨迹先最大化完整时长比例再最小化RMSE，
TD3先最大化到点成功率再最短平均首episode时长。高速扩展也按同一轨迹规则逐周期选择。

| 用途 | 请求预算 → 实际transitions | 验证结果 | checkpoint |
|---|---:|---|---|
| PPO 10s轨迹 | 2M → 1,998,848 | 1.01mm，全64通过 | [official_ppo_10s_budget1_20261015_seed44.ckpt](official_ppo_10s_budget1_20261015_seed44.ckpt) |
| PPO 5.5s轨迹 | 20M → 19,988,480 | 7.41mm，全64通过 | [official_ppo_5_5s_budget2_20261015_seed44.ckpt](official_ppo_5_5s_budget2_20261015_seed44.ckpt) |
| BPTT 10s轨迹 | 200k → 199,680 | 2.36mm，全64通过 | [official_bptt_10s_budget2_20261015_seed43.ckpt](official_bptt_10s_budget2_20261015_seed43.ckpt) |
| BPTT 5.5s轨迹 | 1.6M → 1,599,488 | 1.48mm，全64通过 | [official_bptt_5_5s_budget2_20261015_seed44.ckpt](official_bptt_5_5s_budget2_20261015_seed44.ckpt) |
| TD3直接电机到点 | 40M → 39,997,440 | 到点64/64，平均1.017s | [official_td3_rotor_reach_budget2_20261015_seed42.ckpt](official_td3_rotor_reach_budget2_20261015_seed42.ckpt) |
| BPTT专门3.5s：最佳验证精度 | 1.6M → 1,599,488 | 8.56mm，全64通过 | [official_frontier_bptt_T3.5s_budget1_seed43.ckpt](official_frontier_bptt_T3.5s_budget1_seed43.ckpt) |
| PPO专门3.5s：位置holdout更稳健 | 40M → 39,976,960 | 13.58mm，全64通过 | [official_frontier_ppo_T3.5s_budget2_seed43.ckpt](official_frontier_ppo_T3.5s_budget2_seed43.ckpt) |

这里轨迹“全64通过”表示全部64world完成请求20s且各自全程RMSE<5cm。
Figure8的nominal验证使用精确参考初始化，不能用上述毫米数代表任意初态精度。
专门3.5s位置holdout中，PPO为19.63mm、64/64通过，BPTT为52.25mm、61/64通过；
再加真实初速±0.5m/s后分别58/64和47/64通过。
“PPO更稳健”是相同验证规则选出的PPO在这些测试上的观察，不是按测试重新选择的策略。
PPO40M对照来自18个基础run中最大训练尾段改善，延训选择没有使用测试集。

直接电机TD3只训练固定目标[0,0,1.5]，独立测试到点63/64（98.44%）。
其未见八字任务是迁移测试，不能据此声称随机三维goal或专门高速rotor跟踪已经训练成功。

## 控制接口与能力范围

PPO/BPTT Figure8使用cf21B_500、first_principles动力学、四维attitude指令
（roll/pitch/yaw/collective thrust），物理500Hz、策略50Hz；低层控制及电机动力学仍由仿真执行。
TD3使用cf2x_L250、first_principles动力学、直接四电机rotor_vel命令，物理500Hz、策略100Hz。
不同机型和任务不能当作attitude与rotor接口的公平消融。

这两类都是标准六自由度、四输入的欠驱动四旋翼；没有验证固定翼、损失旋翼恢复、视觉导航
或真实硬件性能。所有视频均为实际MuJoCo仿真状态回放，未启动实飞。
强DR1压力中有不物理惯量样本，不能以其失败率解释真实机型极限；详见物理性审计。

## 报告、原始数据与最佳演示

- [原官方配置研究：30个训练、126项基础评估、全部失败及收敛证据](../usage_assets/official_policy_20261015_results.md)
- [真实初速/姿态压力测试](../usage_assets/official_stress_20261015_results.md)
- [冻结5.5s权重的速度迁移扫描](../usage_assets/official_speed_sweep_20261015_results.md)
- [专门更快周期训练：19个训练、33项评估、完整配方及全部种子](../usage_assets/official_frontier_20261015/README.md)
- [BPTT专训3.5s视频](../usage_assets/official_frontier_20261015/best_dedicated_speed.mp4)
- [PPO专训3.5s视频](../usage_assets/official_frontier_20261015/ppo_T3.5s_position_holdout.mp4)
- [PPO冻结策略4.5s迁移视频](../usage_assets/official_speed_sweep_20261015_ppo_T4.5s.mp4)

视频固定事先指定的独立测试seed/world0，没有挑最好world。专训视频旁的_replay.json、
_replay_states.npz保存检查点哈希和实际状态，并与原测试轨迹逐步核对。

## 推理与完整复现

在研究Docker内，从/workspace/crazyflow运行。脚本读取checkpoint旁对应研究记录的完整配置，
以确定观测顺序、网络、preview、控制接口及频率；不应只加载actor后任意拼接观测。
外部官方训练源码自动固定到commit bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3。

~~~bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

# 直接电机TD3的固定点到达推理回放
python usage_assets/official_replay_20261015.py \
  --checkpoint trained_models/official_td3_rotor_reach_budget2_20261015_seed42.ckpt \
  --scenario holdout_hover

# 验证选中的最快全通过专训BPTT，固定seed14001/world0
python usage_assets/official_frontier_20261015/replay.py

# 同一固定holdout下的专训PPO
python usage_assets/official_frontier_20261015/replay.py \
  --checkpoint trained_models/official_frontier_ppo_T3.5s_budget2_seed43.ckpt \
  --output usage_assets/official_frontier_20261015/ppo_T3.5s_position_holdout.mp4

# 全部18个专训基础run、一个40M对照、独立测试/核验/视频
bash usage_assets/official_frontier_20261015/reproduce.sh

# 原官方配置研究的两档完整预算，各seeds42/43/44
for budget in 1 2; do
  for algorithm in ppo bptt td3; do
    python usage_assets/official_policy_20261015.py \
      --algorithm "$algorithm" --budget-multiplier "$budget"
  done
done
python usage_assets/official_report_20261015.py
~~~

原30次研究按官方preset完整训练：慢轨迹PPO2M/4M、BPTT100k/200k，
快轨迹PPO10M/20M、BPTT800k/1.6M，TD3 20M/40M；均用seeds42/43/44。
专训基础为PPO20M/BPTT1.6M × T3.5/3.0/2.5s × seeds42/43/44，
另PPO3.5s seed43独立40M；只改参考周期，保留官方5_5s其余参数与preview。
预算取完整更新，manifest记录每个run的真实步数。

本目录的Gymnasium策略使用另一套so_rpy/attitude环境和训练配方，
其状态、选模及完成情况请查看[Gym训练汇总](../usage_assets/fulltrain_20261015/summary.json)。
Gym/racing训练不计入本页49个official权重清单，最终选择及报告由各自实验汇总说明。

参数与出处：[官方实验源码](https://github.com/learnsyslab/crazyflow_experiments/tree/bfc6414da1f4405e4e41e3a3d0656bb5f38e09e3)、
[论文](https://arxiv.org/html/2606.01478v1)。论文实飞成绩与本地仿真结果分别报告。

## Gym与Racing最终冻结权重

Gym21次完整200M训练和四个历史baseline复评已完成；下表仅按独立validation选择。
每个新增Gym run另有同名_last.pt，完整保存恢复训练所需状态。
控制接口为cf2x_L250 / so_rpy / attitude，不能冒充直接电机策略。

| 任务 | 冻结checkpoint | SHA256前16位 |
|---|---|---|
| 三维到点 | [gym_reach_pos_s19_g0.99_200000000_calibrated.pt](gym_reach_pos_s19_g0.99_200000000_calibrated.pt) | d89ab91c33d3b162 |
| 默认三维速度 | [gym_reach_vel_s7_g0.99_200000000.pt](gym_reach_vel_s7_g0.99_200000000.pt) | 528c5429fd0f7bea |
| 近地悬停 | [gym_landing_s43_g0.99_200000000_calibrated.pt](gym_landing_s43_g0.99_200000000_calibrated.pt) | 6eb44776c04cd123 |
| 八字跟踪 | [gym_figure8_s43_g0.99_200000000.pt](gym_figure8_s43_g0.99_200000000.pt) | 04b92f71488a5b5b |

Racing验证选中权重：[racing_s42_15000000_20261015.ckpt](racing_s42_15000000_20261015.ckpt)，实际预算：14,999,552 transitions；训练实现：truncations-and-not-terminations。
[完整Gym成绩/回放/配置](../usage_assets/fulltrain_20261015/README.md)，[完整赛道成绩/速度扫描/复现](../usage_assets/racing_fulltrain_20261015.md)。
baseline_20261009_*是历史权重，已按新协议复评；legacy_timeout是明确保留的对照，不能自动视为当前严格实现。模型选择、失败和最终预算均由结果JSON核验。
