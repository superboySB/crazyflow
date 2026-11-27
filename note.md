# 本机验收入口

完整安装、参数辨识、PPO、MuJoCo 与 Isaac 对比步骤见 [README](README.md)。

本分支有两个独立镜像：`docker/crazyflow.Dockerfile` 构建 `crazyflow:dani-crazyrl`，`docker/sim2sim.Dockerfile` 构建 `crazyflow:dani-crazyrl-eval`。训练、评测和可视化使用本仓库文件。Git 只保留 `models/crazyflow/policy.npz` 与官方 `models/baseline/model_36825.pt` 两份权重；中间模型、缓存、日志和视频写入忽略目录 `outputs/`。

```bash
./crazyrl setup --device gpu
./crazyrl train --smoke --device gpu
./crazyrl capacity
./crazyrl train --device gpu
./crazyrl evaluate --example --episodes 256 --seed 9001
./crazyrl play --example
./crazyrl play --example --headless --output outputs/avoidance.mp4
./crazyrl sim2sim build
./crazyrl sim2sim compare --policy models/crazyflow/policy.npz \
  --num-envs 32 --seed 42 --output outputs/sim2sim/best-seed42
./crazyrl sim2sim compare --policy outputs/你的训练目录/policy.npz \
  --num-envs 32 --seed 42 --output outputs/sim2sim/seed42
./crazyrl sim2sim play --policy baseline --gui --num-envs 8 --max-steps 450
./crazyrl sim2sim tensorboard
```

`--example` 读取 `models/crazyflow/policy.npz`，适合直接验收。官方 baseline 别名读取 `models/baseline/model_36825.pt`；冻结的评测任务与 USD 在 `sim2sim/reference/`。`--seed` 更换初态；新地图测试另加 `--map-seed 14001`。新训练默认读取 `config/training.yaml`，按实际训练量和显存选择更新数；本机正式续训为 12,288 环境、100 更新。

`play --example` 是持续交互窗口：`R` 重置初态/目标，`Backspace` 精确重开当前回合，空格暂停，`C` 切换视角，`T` 切换干净/带噪声测距；右侧显示前向 8×8 深度和左、右、后、下 ToF，单位 m。可点击 Reset/Pause/Camera，鼠标旋转、平移、缩放。回合结束自动继续，按 Esc 或关闭窗口退出。

```bash
./crazyrl play --example --self-test --gui-frames 20 \
  --screenshot outputs/tof-live-check.png --gui-report outputs/tof-live-check.json
```

交付 NPZ 与 baseline Torch 的 actor 都是 344 维输入、4 维输出；完整物理通道、单位、历史排列及归一化对照见 [docs/tof-parity.md](docs/tof-parity.md)，实际权重和闭环输入检查见 [接口审计](reports/sim2sim/checkpoint-io-audit.json)。roll/pitch 输出是姿态角目标，yaw 输出才是角速度目标。

验收的实际结果、GPU 容量及最终模型路径集中在 [reports/crazyflow/README.md](reports/crazyflow/README.md)。旧 Euler 试训属于积分诊断，正式模型使用配置中冻结的官方 RK4；不能把两者结果混为同一模型。
