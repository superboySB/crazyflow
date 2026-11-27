# 本机验收入口

完整安装、参数辨识、PPO、MuJoCo 与 Isaac 对比步骤见 [README](README.md)。

本分支有两个独立镜像：`docker/crazyflow.Dockerfile` 构建 `crazyflow:dani-crazyrl`，`docker/sim2sim.Dockerfile` 构建 `crazyflow:dani-crazyrl-eval`。训练使用 JAX/Flax/Optax PPO；每次保存同时导出 `policy.npz` 与真正的 Torch/RSL `policy.pt`，Torch 只负责 CPU 序列化。前者用于 Crazyflow/MuJoCo，后者用于 Isaac Sim。

```bash
./crazyrl setup --device gpu
./crazyrl train --smoke --device gpu
./crazyrl capacity
./crazyrl train --device gpu --output outputs/my-tof
./crazyrl evaluate --checkpoint outputs/my-tof/policy.npz --episodes 256 --seed 9001
./crazyrl play --checkpoint outputs/my-tof/policy.npz
./crazyrl play --checkpoint outputs/my-tof/policy.npz --headless --output outputs/avoidance.mp4
./crazyrl sim2sim build
./crazyrl sim2sim compare --policy outputs/my-tof/policy.pt \
  --num-envs 32 --seed 42 --output outputs/sim2sim/seed42
./crazyrl sim2sim play --policy outputs/my-tof/policy.pt --gui --num-envs 8 --max-steps 450
./crazyrl sim2sim tensorboard
```

直接验收随仓库保留的最佳策略和官方 baseline：

```bash
./crazyrl play --checkpoint models/crazyflow/policy.npz
./crazyrl evaluate --checkpoint models/crazyflow/policy.npz \
  --episodes 256 --seed 14001 --map-seed 14001 --output outputs/best-unseen-map-test
./crazyrl sim2sim compare --policy models/crazyflow/policy.pt \
  --num-envs 32 --seed 42 --output outputs/sim2sim/best-seed42
./crazyrl sim2sim play --policy models/baseline/model_36825.pt --gui --num-envs 8 --max-steps 450
```

`models/crazyflow/policy.npz` 与 `policy.pt` 是同一选定策略的两种导出；官方 baseline 为 `models/baseline/model_36825.pt`。冻结的评测任务与 USD 在 `sim2sim/reference/`。现有最佳 PT 保存 actor 和 actor 归一化；新训练导出包含实际可用的 critic 与 critic 归一化。恢复 JAX PPO 使用 `outputs/my-tof/checkpoint.msgpack`，部署 `.pt` 不恢复 Optax、RNG 或环境状态。中间模型、缓存、日志和视频都写入忽略目录 `outputs/`。

`--seed` 更换初态；新地图测试另加 `--map-seed`。新训练默认读取 `config/training.yaml`，按实际训练量和显存选择更新数；本机正式续训为 12,288 环境、100 更新。

MuJoCo `play` 是持续交互窗口：`R` 重置初态/目标，`Backspace` 精确重开当前回合，空格暂停，`C` 切换视角，`T` 切换干净/带噪声测距；右侧显示前向 8×8 深度和左、右、后、下 ToF，单位 m。可点击 Reset/Pause/Camera，鼠标旋转、平移、缩放。回合结束自动继续，按 Esc 或关闭窗口退出。

```bash
./crazyrl play --checkpoint models/crazyflow/policy.npz --self-test --gui-frames 20 \
  --screenshot outputs/tof-live-check.png --gui-report outputs/tof-live-check.json
```

交付策略与 baseline 的 actor 都是 344 维输入、4 维输出；完整物理通道、单位、历史排列及归一化对照见 [docs/tof-parity.md](docs/tof-parity.md)，既有权重和闭环输入检查见 [接口审计](reports/sim2sim/checkpoint-io-audit.json)。roll/pitch 输出是姿态角目标，yaw 输出才是角速度目标。

双格式导出核验见 [reports/dual-export.json](reports/dual-export.json)。导出没有重新训练最佳策略，既有 Isaac 成绩仍为新策略 132/160、官方 baseline 155/160。

验收的实际结果、GPU 容量及最终模型路径集中在 [reports/crazyflow/README.md](reports/crazyflow/README.md)。旧 Euler 试训属于积分诊断，正式模型使用配置中冻结的官方 RK4；不能把两者结果混为同一模型。
