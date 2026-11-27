# 本机验收入口

完整安装、参数辨识、PPO、MuJoCo 与 Isaac 对比步骤见 [README](README.md)。

本分支训练镜像为 `crazyflow:dani-crazyrl`，评测镜像为 `crazyflow:dani-crazyrl-eval`。训练、评测和可视化都使用本目录文件；大模型和中间结果由 `.gitignore` 排除。

```bash
./crazyrl setup --device gpu
./crazyrl train --smoke --device gpu
./crazyrl capacity
./crazyrl train --device gpu
./crazyrl evaluate --example --episodes 256 --seed 9001
./crazyrl play --example
./crazyrl play --example --headless --output outputs/avoidance.mp4
./crazyrl sim2sim compare --policy outputs/你的训练目录/policy.npz \
  --num-envs 32 --seed 42 --output outputs/sim2sim/seed42
./crazyrl sim2sim play --policy baseline --gui --num-envs 8 --max-steps 450
./crazyrl sim2sim tensorboard
```

`--example` 读取随分支保留的最终模型，适合直接验收。`--seed` 更换初态；新地图测试另加 `--map-seed 14001`。新训练默认读取 `config/training.yaml`，按实际训练量和显存选择更新数；本机正式续训为 12,288 环境、100 更新。

`play --example` 是持续交互窗口：`R` 重置初态/目标，`Backspace` 精确重开当前回合，空格暂停，`C` 切换视角，`T` 切换干净/带噪声测距；右侧显示前向 8×8 深度和左、右、后、下 ToF，单位 m。可点击 Reset/Pause/Camera，鼠标旋转、平移、缩放。回合结束自动继续，按 Esc 或关闭窗口退出。

```bash
./crazyrl play --example --self-test --gui-frames 20 \
  --screenshot outputs/tof-live-check.png --gui-report outputs/tof-live-check.json
```

交付 NPZ 与 baseline Torch 的 actor 都是 344 维输入、4 维输出；完整物理通道、单位、历史排列及归一化对照见 [docs/tof-parity.md](docs/tof-parity.md)，实际权重和闭环输入检查见 [接口审计](sim2sim/results/contract-20261127/checkpoint-io-audit.json)。roll/pitch 输出是姿态角目标，yaw 输出才是角速度目标。

验收的实际结果、GPU 容量及最终模型路径集中在 [examples/tof/README.md](examples/tof/README.md)。旧 Euler 试训属于积分诊断，正式模型使用配置中冻结的官方 RK4；不能把两者结果混为同一模型。
