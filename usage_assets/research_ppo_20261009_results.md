# Docker 四 Gym PPO 增量复验（2026-10-09）

原始配置、全部训练 history、优化日志、11 项评估指标、checkpoint SHA256 见 `research_ppo_20261009_results.json`。所有训练与评估在 `dzp-crazyflow-research-20261009` 内使用 CUDA 执行，seed 7。

4096 worlds × 64 steps = 262,144 transitions/update。预算只取完整 update：10,000,000 实际为 9,961,472；20,000,000 实际为 19,922,944。ReachPos/Landing 另有 4,096,000 教师 BC samples，PPO 内 imitation coefficient 10；ReachVel/Figure8 使用原配置，无教师 BC。

| 任务 | PPO transitions / updates | BC samples | PPO 秒 / 全进程秒 | RMSE | 末秒 RMSE | survival | success |
|---|---|---|---|---|---|---|---|
| DroneReachPos-v0 | 9,961,472 / 38 | 4,096,000 | 18.18 / 38 | 0.141301 m | 0.041216 m | 100.00% | 100.00% |
| DroneReachVel-v0 | 19,922,944 / 76 | 0 | 44.49 / 56 | 0.181523 m/s | 0.073598 m/s | 58.20% | 57.42% |
| DroneLanding-v0 | 9,961,472 / 38 | 4,096,000 | 22.86 / 41 | 0.465744 m | 0.046088 m | 100.00% | 100.00% |
| DroneFigureEightTrajectory-v0 | 19,922,944 / 76 | 0 | 28.05 / 38 | 0.040200 m | 0.044990 m | 100.00% | 100.00% |

评估 256 worlds × 500 steps（10 s），deterministic mean action，seed 10007。success 按每个 world 的末 1 s RMSE 判断，并要求整个 500 步未发生 terminated；阈值 ReachPos/ReachVel/Landing <0.10（m 或 m/s），Figure8 <0.15 m；Landing 还要求末秒速度 RMSE <0.10 m/s。全局末秒 RMSE 不能直接换算成成功率。官方 NEXT_STEP autoreset 会在失败后重置并继续；episode_return 是固定 500 步奖励和，可能跨 reset，并非单个未中断 episode 的回报；survival 仍会记录任意一次 terminated。

PPO 秒从正式 PPO rollout 开始计时，包含在该循环内发生的首次 step JIT 编译；全进程秒还含先前的 BC、环境初始化、评估和绘图。本轮共用 GPU，速度数字不用于硬件性能排名。最初没有完成 episode 时，episode_return 的 NaN 占位在 JSON 中写为 null；reward/error/KL/clip、网络权重与最终评估指标全部检查 finite。

| 任务 | 首 update error | 末 update error | 末 5 updates 平均 error | error 减少 | 末 KL |
|---|---|---|---|---|---|
| DroneReachPos-v0 | 0.368256 | 0.059112 | 0.061878 | 83.95% | 0.0000 |
| DroneReachVel-v0 | 1.459221 | 0.112719 | 0.112036 | 92.28% | 0.0000 |
| DroneLanding-v0 | 1.042321 | 0.059659 | 0.113346 | 94.28% | 0.0001 |
| DroneFigureEightTrajectory-v0 | 0.613936 | 0.034690 | 0.040479 | 94.35% | 0.0000 |

## ReachVel 水平目标对照

使用同一个 ReachVel checkpoint，无额外训练，只将 target vz 固定为 0，vx/vy 仍在 [-1, 1] m/s 内采样；其余评估口径相同。

RMSE 0.125064 m/s，末秒 0.049551 m/s；survival 100.00%，success 97.27%。

完整 3D 速度任务的向下速度目标受 floor/terminated 约束；水平子集不能代替官方默认 3D 任务成绩。

![DroneReachPos-v0](research_20261009_reach_pos.png)

![DroneReachVel-v0](research_20261009_reach_vel.png)

![DroneLanding-v0](research_20261009_landing.png)

![DroneFigureEightTrajectory-v0](research_20261009_figure8.png)

![ReachVel horizontal](research_20261009_reach_vel_horizontal.png)
