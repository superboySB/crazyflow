# 快速操作

在项目根目录运行。首次使用先按 [README](README.md) 构建 Crazyflow 和 Isaac 镜像；配置、辨识、容量测试及交互按键也以该指南为准。以下命令使用同一运行目录 `outputs/my-tof/`。

训练（默认 1,024 环境、1,000 次 PPO 更新）：

```bash
./crazyrl train --device gpu --config config/training.yaml --output outputs/my-tof
```

用 NPZ 做 Crazyflow 新地图评测：

```bash
./crazyrl evaluate --checkpoint outputs/my-tof/policy.npz --episodes 256 --seed 14001 --map-seed 14001 --output outputs/my-evaluation
```

打开 MuJoCo 实时交互：

```bash
./crazyrl play --checkpoint outputs/my-tof/policy.npz
```

用同次训练导出的 PT 与官方 baseline 做 Isaac 配对比较：

```bash
./crazyrl sim2sim compare --policy outputs/my-tof/policy.pt --num-envs 32 --seed 42 --output outputs/sim2sim/my-tof-seed42
```

直接查看仓库保留策略：

```bash
./crazyrl play --checkpoint models/crazyflow/policy.npz
```

直接播放官方 baseline：

```bash
./crazyrl sim2sim play --policy models/baseline/model_36825.pt --gui --num-envs 8 --max-steps 450
```

每次训练保存 `policy.npz`、`policy.pt` 和完整恢复文件 `checkpoint.msgpack`。部署格式与模型来源见 [模型说明](models/README.md)，硬件吞吐和选模结果见 [训练研究](reports/training-study/README.md)。
