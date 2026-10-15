# 2026-10-15 真正初速与姿态 holdout

这里只加载已按 validation seed9001 选出的冻结权重，不重新训练，也不按新测试结果重新挑权重。
每项 64 worlds；velocity-only 固定 seed11001，attitude-only 固定 seed11002。
官方 reset 全部结束后显式修改 SimData，并更新最内层原始观测；上层官方 wrappers 随后从
真实扰动状态生成首个策略观测。因此没有原 velocity reset 覆盖扰动的问题。

速度任务：Figure8 初速=参考速度+每轴 U(-0.5,0.5)m/s；到点初速=每轴 U(-0.5,0.5)m/s。
姿态任务：独立 roll/pitch 各 U(-15°,15°)，yaw 保持0，速度保持官方 reset 值。
两项独立运行，不叠加动力学随机化；Figure8 从 nominal 参考位置起步，TD3 保持官方随机到点初始位置。
每项保存64个初始真实状态；assert 已验证扰动生效、reference完全未改、attitude-only速度未改。
全统计由每world每step误差/mask复算。到点提前成功终止属于成功，不属于越界失败。

| 冻结策略 | 真正扰动 | 全程RMSE mm | 无越界 | 全时长 | 到点成功 | 轨迹全程<5cm成功 |
|---|---|---:|---:|---:|---:|---:|
| ppo_slow | velocity | 6.38 | 100.00% | 100.00% | 0.00% | 100.00% |
| ppo_slow | attitude | 1.93 | 100.00% | 100.00% | 0.00% | 100.00% |
| ppo_fast | velocity | 10.93 | 100.00% | 100.00% | 0.00% | 100.00% |
| ppo_fast | attitude | 7.81 | 100.00% | 100.00% | 0.00% | 100.00% |
| bptt_slow | velocity | 22.12 | 100.00% | 100.00% | 0.00% | 96.88% |
| bptt_slow | attitude | 3.19 | 100.00% | 100.00% | 0.00% | 100.00% |
| bptt_fast | velocity | 8.22 | 100.00% | 100.00% | 0.00% | 100.00% |
| bptt_fast | attitude | 2.53 | 100.00% | 100.00% | 0.00% | 100.00% |
| td3 | velocity | 258.94 | 100.00% | 3.12% | 96.88% | 0.00% |
| td3 | attitude | 289.66 | 100.00% | 0.00% | 100.00% | 0.00% |

到点全程 RMSE 包含随机初始位置误差，不能当稳态悬停精度。到点看 goal_success，
轨迹看完整时长和精度。结果限定这些冻结策略、指定初始分布、标称物理与精确观测，
不代表任意初始状态或实飞泛化保证。

复现：在研究 Docker 内运行 python usage_assets/official_stress_20261015.py。
原始 official_stress_20261015_* 场景 NPZ 保存完整误差/mask、world0真值轨迹，以及全部64world
初始位置/速度/四元数/参考/实际速度扰动；JSON保存权重SHA256和各world误差。
