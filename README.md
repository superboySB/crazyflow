# dani-crazyrl：Crazyflow / JAX PPO 的 ToF 避障

本分支在 Crazyflow 上训练 CrazyRL `baseline/static_goals` 避障任务。训练、物理推进、ToF 测距、rollout、GAE 和 PPO 更新均在同一 JAX 设备上批量执行，使用 Flax + Optax；训练器沿用 `research` 分支验证过的官方 [crazyflow_experiments](https://github.com/learnsyslab/crazyflow_experiments/tree/203ad4e31b9380fec985f17008bb7a68ad5cf768) 的 functional/JIT/scan 方式，适配 CrazyRL 的网络与任务接口。梯度只来自 PPO 的网络损失。气球任务及其专用训练代码已删除。

Isaac Sim 仅用于最后的 sim2sim 对比，使用现成的 CrazyRL 权重，不在 Isaac 中重新训练。MuJoCo 用于显示 Crazyflow 实际闭环飞行轨迹和 ToF。所有入口在项目根目录运行，宿主只需 Docker；GPU 需要 NVIDIA 驱动和 NVIDIA Container Toolkit；宿主调度器使用 Python 3 标准库。

## 1. 安装、独立 Docker 与检查

```bash
./crazyrl setup --device gpu
./crazyrl train --smoke --device gpu
```

训练镜像是 **`crazyflow:dani-crazyrl`**，评测镜像是 **`crazyflow:dani-crazyrl-eval`**，均独立于 `dani` 的镜像。依赖固定在 `requirements-{cpu,gpu}.lock`；`setup` 运行真实 JAX GPU 运算。省略 `--device` 时自动选择 GPU，失败则使用 CPU；强制 GPU 失败会退出。首次 JIT 编译较慢，`training.jsonl` 分别记录首轮和后续耗时。

训练源码和配置只读挂载进容器，实验写入 `outputs/`。改代码后可直接重跑；依赖变更后需重新 `setup`。训练默认关闭 JAX 显存预分配，分配器限制为整卡的 55%，并提供独立的多环境容量测量。`--smoke` 只检查完整训练/导出流程，不更新“最近正式模型”指针。

```bash
./crazyrl identify init
./crazyrl identify check
./crazyrl identify drag-check
```

`check/drag-check` 是带真值的合成软件检查，输出不会冒充实测参数。验收记录及训练结果见文末。

## 2. 参数辨识：复用 dani，默认无需实测

默认使用 Crazyflow 官方 `first_principles / cf21B_500`：质量 **0.0434 kg**，惯量、四电机 RPM 推力/反扭矩曲线、转子惯量、旋翼动态和阻力均来自 [`config/dynamics/first_principles/params.toml`](config/dynamics/first_principles/params.toml)。不把辨识示例或 CrazyRL 的 Isaac 机体数值自动覆盖为本机参数。

更换机体或载荷后，按[辨识手册](docs/identification.md)独立采集训练/验证 CSV，再使用原有工具：

```bash
./crazyrl identify init --directory outputs/measurements
./crazyrl identify inertia --suspended-mass 0.0434 --spacing 0.08 --length 0.4 --period 1.0
./crazyrl identify flight --negligible-drag
./crazyrl identify rotor --negligible-drag
./crazyrl identify static
# 另一种简化闭环诊断，质量需填写真实称量值：
./crazyrl identify drag-init
./crazyrl identify drag --mass 0.0434
```

`flight/rotor` 导出 `physical.toml`。需要采用真实辨识结果时，在训练 YAML 的 `environment.model_params_path` 填 **`/outputs/measurements/physical.toml`**。本次训练保持默认值。简化模型的 `closed-loop-so-rpy-rotor-drag.toml` 不能作为该物理表。CrazyRL 的 PD 控制器与 Crazyflow 的物理模型分别配置；`config/controller.toml` 是保留的 Mellinger 辨识模板，并非本任务的 PD 参数入口。

## 3. state、action、reward 与随机化

任务实现是 [`crazyflow/tof/env.py`](crazyflow/tof/env.py)，全部参数见 [`config/training.yaml`](config/training.yaml)。参考版本和文件哈希保存在 [`sim2sim/provenance.json`](sim2sim/provenance.json)。

| 项目 | 与 CrazyRL baseline 相同的接口 |
| --- | --- |
| Actor | 每帧 28 维 × 10 帧历史 + 当前 8×8 前向深度，共 **344** 维 |
| 每帧内容 | body 线速度、旋转矩阵、gyro、目标方向、高度差、期望速度、上一动作、左/右/后/下单点 ToF；顺序、倍率和噪声按参考实现 |
| Critic | 每帧 37 维 × 10 + 64 深度，共 **434** 维；增加特权信息，actor 不可访问 |
| 深度 | 8×8 针孔 image-plane depth，focal=20、aperture=20.955，最大 4 m；ToF 作为深度相机变体 |
| 策略 | 15 Hz；物理 150 Hz，decimation=10；动作延迟 4 个物理 tick |
| Actor/Critic 网络 | ELU MLP `[256,256,128,64]`；独立经验归一化 |
| 动作 | 原始 4D Gaussian；环境 `tanh` 后 roll/pitch ±π/6，yaw rate ±π/1.5，推力映射到参考总推力上限 |
| 回合 | 最长 80 s；接触、成功或数值失稳终止；超时单独记录并正确 bootstrap |
| 成功 | 目标距离 <0.5 m 且速度 <0.5 m/s，持续时间按配置；评测不把超时剔除 |

原始 reward 逐项重写，保留单位和限幅，不额外乘 dt：距离进展（上限 0.01）×8、yaw 方向×0.3、动作模长×0.3、动作差×0.5、速度方向 Huber×0.15、超速指数罚×0.5、高度 Huber×0.25、碰撞 −80、ESDF 安全距离指数罚×0.1、到达 +80。速度匹配系数默认 0。精确公式和参考的非直观处理在代码及[环境对应说明](docs/tof-parity.md)中列明。

保留参考的观测乘性噪声、深度无效点、OU 风、质量/惯量控制模型随机化、推力初始效率 `[0.85,1]`、退化率 `[0.10,0.15]`、最低效率 0.7、速度 `[0.5,1.5]`、目标采样噪声和障碍尺寸随机化。训练与最终对比使用各自冻结的配置。Crazyflow 继续使用自己的 RPM 电机/刚体模型，使用官方 RK4 积分以避免 150 Hz Euler 引入的数值振荡；其解析几何碰撞与 Isaac PhysX 碰撞不是逐步相同的物理结果，最终迁移效果由独立 Isaac 评测衡量。

## 4. GPU 并行容量与正式 PPO 训练

```bash
./crazyrl capacity
# 也可限定测试范围；每档执行完整 rollout 和 PPO，而非仅分配环境：
./crazyrl capacity --envs 1024 4096 8192 12288 14336 --output outputs/my-capacity
```

容量脚本每 0.25 s 采样 NVIDIA 计算进程显存，超过 9.5 GiB 即停止；JAX 分配器另有 55% 整卡限制。请先结束其它 CUDA 计算。结果包括每档环境数、显存采样峰值、完整更新耗时、实际命令和源码哈希，保存在 `outputs/tof-capacity/capacity.json`。首轮编译不作为稳态吞吐。

```bash
./crazyrl train --device gpu
# 修改总更新数、并行环境数或独立配置：
./crazyrl train --device gpu --iterations 1000 --num-envs 12288 --output outputs/my-tof
./crazyrl train --device gpu --config config/training.yaml
# 延续同一批环境、随机状态和 Adam 状态；iterations 是目标总次数：
./crazyrl train --device gpu --resume outputs/my-tof/checkpoint.msgpack \
  --iterations 2000 --num-envs 12288 --output outputs/my-tof-continued
# 增加并行数时沿用策略、优化器与归一化，但重新创建环境：
./crazyrl train --device gpu --warm-start outputs/my-tof/checkpoint.msgpack \
  --iterations 100 --num-envs 12288 --output outputs/my-tof-large
```

本机已测得 12,288 环境可运行；默认使用此并行数。14,336 环境触发显存保护。容量和每秒交互数是两个指标，详见[本机结果](examples/tof/README.md)。

PPO 默认每环境收集 120 步，5 个 epoch、32 个 minibatch；学习率 `5e-4` 按 KL 自适应，γ=0.99、λ=0.95、clip=0.2、entropy=0.005、value=1、梯度范数上限 1。用 `num_envs×120` 计算每次更新的 transitions；改变并行数不等于增加训练更新数。最初使用随机 rollout 校准观测归一化，之后只做 PPO。

`--resume` 的 `iterations` 是该运行的总更新目标；`--warm-start` 从 0 重新计数，执行额外的指定次数，并记录父检查点的训练量。

每次运行保存 `config.json`、`training.jsonl`、`training.png`、可独立推理的 `policy.npz` 和含优化器/RNG/环境状态的 `checkpoint.msgpack`。`policy.npz` 内包含网络、归一化、有效环境配置和源码哈希，不依赖后来修改的本地配置。GPU 正式训练全程由同样的显存 watchdog 监测，采样峰值写入 `gpu-memory.json`；超过限制会停止并保留此前成功 checkpoint。正式成功结束才更新 `outputs/.latest-policy`。日志的训练成功率来自带探索的在线回合，不能替代下面的确定性留出评测。

## 5. 留出评测与 MuJoCo 小飞机可视化

```bash
./crazyrl evaluate --episodes 256 --seed 9001
./crazyrl evaluate --checkpoint outputs/my-tof/policy.npz \
  --episodes 256 --seed 9001 --output outputs/my-evaluation
# 独立新障碍地图：训练模型的物理及其余任务参数保持冻结
./crazyrl evaluate --example --episodes 256 --seed 14001 --map-seed 14001 \
  --output outputs/my-unseen-map-test
./crazyrl play
# 仓库中保留本次选出的最终模型，可直接看示例：
./crazyrl play --example
./crazyrl play --checkpoint outputs/my-tof/policy.npz --seed 9001
./crazyrl play --headless --checkpoint outputs/my-tof/policy.npz \
  --output outputs/avoidance.mp4
# 精确回放已评测的某条飞行：
./crazyrl play --evaluation outputs/my-evaluation --episode 0
```

`evaluate/play --example` 使用 `examples/tof/policy.npz`，不改最近正式训练指针。`--seed` 只改变初态与观测随机数；默认保留模型中的地图。检查泛化时另外指定 `--map-seed` 生成新地图。评测使用确定性动作均值，每个并行世界只统计首个完整回合，保存 `summary.json`、`episodes.csv` 和 `trajectories.npz`；成功、碰撞、超时、数值失稳均保留在分母。`--max-steps` 是调试截短，剩余回合按超时计，不能用来制造成功率。

MuJoCo 画面显示官方 CF21B 网格、随机障碍、目标、轨迹和仿真 ToF 射线；支持 follow/overview/fpv 视角及 PNG/GIF/MP4。GUI 用鼠标控制 MuJoCo 相机、空格暂停；`--headless` 使用 EGL，CPU 使用 OSMesa。这里显示的是 **Crazyflow 闭环产生的轨迹**，不是第二套 MuJoCo 动力学控制实验；独立物理迁移评测在 Isaac 进行。

## 6. 独立 Isaac Sim：同场景对比新模型和现成 baseline

本目录的 `sim2sim/reference/` 仅保留 baseline 任务、必要控制/动力学/传感器/地图与评测源码；必要 USD 随 `sim2sim/reference/assets/` 保存，原始 baseline 权重与参数随 `sim2sim/artifacts/assets/replementation/rl/` 保存；TensorBoard 历史事件也已复制到本机同目录，作为训练产物由 git 忽略。没有搬入 CrazyRL 的训练器和其它任务。导入器记录逐文件 SHA256，当前工作目录已包含运行所需文件，baseline 权重、USD 与参数随分支保留；重新导入历史 TensorBoard 或刷新副本时才需要参考项目：

```bash
python3 sim2sim/stage_reference.py --source ~/projects/CrazyRL
./crazyrl sim2sim build
./crazyrl sim2sim evaluate --policy baseline --num-envs 8 --seed 20261127 \
  --output outputs/sim2sim/baseline-new
./crazyrl sim2sim compare --policy outputs/my-tof/policy.npz --num-envs 32 \
  --seed 42 --output outputs/sim2sim/paired-seed42
```

`compare` 依次加载同一个 Isaac 环境中的现成 `model_36825.pt` 和新 NPZ，使用相同种子、地图、初始位姿、目标和速度，**校验初始场景哈希相同才接受比较**；每环境各一个回合，避免策略结束时间不同使后续随机重置失去配对。更充分的比较应对 42–46 等多个种子重复运行，报告全部回合。评测保留源项目动力学、资产和 reward；没有在评测时调参帮助候选模型。这里比较两个已有策略的迁移表现，训练物理与训练预算不同，新模型的实际预算在结果中完整记录。

```bash
./crazyrl sim2sim play --policy baseline --gui --num-envs 8 --max-steps 450
./crazyrl sim2sim play --policy outputs/my-tof/policy.npz --gui --num-envs 8 --max-steps 450
./crazyrl sim2sim play --policy baseline --video --num-envs 8 --max-steps 450
./crazyrl sim2sim tensorboard --port 6006
```

TensorBoard 打开 `http://127.0.0.1:6006`，展示复制的 CrazyRL 历史训练日志。新 Crazyflow 模型的训练曲线直接用 `training.png/jsonl`。Isaac 的 shader/cache、运行日志、视频和评测中间产物均在忽略目录。Isaac 容器需要本机支持的 NVIDIA GPU；不与正在训练的 JAX 进程同时争用显存。

## 7. 验收与版本边界

本次完成两阶段 PPO 训练，共 270,336,000 次交互；交付第 75 次续训检查点，累计 233,472,000 次交互。12,288 环境的完整 100 次续训采样峰值为 **9,310 MiB**，经过三次课程换图。新模型与 baseline 的结果如下：

| 环境 / 模型 | 成功 | 碰撞 |
| --- | ---: | ---: |
| Crazyflow 固定地图、新初态 / 新模型 | 255/256（99.609%） | 1 |
| Crazyflow 全新地图 / 新模型 | 255/256（99.609%） | 1 |
| Isaac 同场景 / 原 baseline | 155/160（96.875%） | 5 |
| Isaac 同场景 / 新模型 | 132/160（82.5%） | 28 |

Isaac 比较为 seeds 42–46、每种子 32 环境各一个回合；无超时或数值失稳，所有初始场景与随机化审计通过。新模型迁移成功率仍低于 baseline **14.375 个百分点**。完整 320 回合和场景记录见 [`sim2sim/results/final-20261127/`](sim2sim/results/final-20261127/)，模型选择在这些测试之前完成。

原项目 2026-11-27 的留存验收为 seeds 42–46、32 环境×8 回合，共 **1265/1280 成功、15 次接触**，原始协议与每回合结果原样保存在 [`sim2sim/results/reference-20261127/`](sim2sim/results/reference-20261127/)。这是既有 baseline 的结果，不能充当新策略的成绩。

本目录重新运行的独立镜像验收、NPZ 与原始 Torch 权重的数值一致性、配对场景一致性保存在 [`sim2sim/results/local-20261127/`](sim2sim/results/local-20261127/)。完整本次训练、容量、留出与迁移结果见 [`examples/tof/README.md`](examples/tof/README.md)。中间 checkpoint、缓存、TensorBoard 事件及视频均由 `.gitignore` 排除；仓库保留必要的原始 baseline、一份最终 Crazyflow NPZ，以及实现、配置、说明和可核验的精简结果。
