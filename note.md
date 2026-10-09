# Crazyflow Docker 用户手册

本次增量同步于 2026-10-09：`research` 合入官方 `main` 从 `58e8fb4` 到
[`70d09e4`](https://github.com/learnsyslab/crazyflow/commit/70d09e4) 的全部 32 个提交。
Crazyflow 当前版本为 0.3.2；完整提交清单、新接口、实验方法和本次结果见
[`usage.md` 第 15 节](usage.md#15-2026-10-09-官方-main-增量同步)。已有 8 月实验记录和图片保留作历史对照。

开发、依赖安装、测试、实验和 Git 操作均在 Docker 容器内进行；宿主机只负责构建、启动和进入容器。

## 配置

在宿主机进入项目根目录：

```bash
cd /home/dzp/projects/crazyflow
```

构建镜像：

```bash
docker build -f .devcontainer/Dockerfile \
  -t dzp_crazyflow:0.3.2-research-20261009-cuda12.6-py312-racing-rl \
  --progress=plain .
```

Dockerfile 会在构建开始和结束时访问 `https://www.google.com/`；HTTP 错误会直接中止构建。

启动带 GPU 和 X11 图形界面的容器：

```bash
test -f "$XAUTHORITY"

docker run --name dzp-crazyflow-research-20261009 -itd \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=4g \
  -e DISPLAY \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v "$XAUTHORITY:/tmp/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/dzp/projects/crazyflow:/workspace/crazyflow \
  dzp_crazyflow:0.3.2-research-20261009-cuda12.6-py312-racing-rl
```

进入容器：

```bash
docker exec -it dzp-crazyflow-research-20261009 bash
cd /workspace/crazyflow
```

代码通过 bind mount 实时映射到 `/workspace/crazyflow`；镜像内的固定环境位于
`/opt/crazyflow/.pixi/envs/gpu-tests`。GPU 环境固定 Python 3.12，使 CUDA 动态库路径和
PyTorch 2.8 wheel 保持兼容；Gymnasium 升级到 1.4，MuJoCo/MJX 使用锁定的 3.10.0。
依赖和兼容补丁变更后应重建镜像。

## 检查运行环境

查看显卡、CUDA、JAX 和 PyTorch 后端：

```bash
nvidia-smi
nvcc --version

python - <<'PY'
import jax
import jax.numpy as jnp
import torch

x = jax.device_put(jnp.ones((1024, 1024)), jax.devices("gpu")[0])
y = (x @ x).block_until_ready()
print(jax.default_backend(), y.device, y[0, 0])

t = torch.ones((1024, 1024), device="cuda")
u = t @ t
print(torch.__version__, torch.version.cuda, u.device, u[0, 0].item())
PY
```

检查容器网络：

```bash
curl --fail --location --output /dev/null \
  --write-out 'status=%{http_code}\n' \
  https://www.google.com/
```

## Crazyflow 核心用法

Crazyflow 是基于 JAX 的四旋翼仿真器。数组统一采用 `(n_worlds, n_drones, ...)`：`n_worlds` 表示互相独立的并行环境，`n_drones` 表示每个环境中的无人机数量。

最小状态控制示例：

```python
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim

sim = Sim(
    n_worlds=128,
    n_drones=1,
    control=Control.state,
    dynamics="first_principles",
    device="gpu",
)

cmd = np.zeros((sim.n_worlds, sim.n_drones, 16))
cmd[..., :3] = [0.0, 0.0, 0.5]
cmd[..., 12] = 1.0  # xyzw 单位四元数；9:13 为姿态，13:16 为机体角速度

for _ in range(100):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)

print(sim.data.states.pos.shape)
sim.close()
```

五种控制接口如下：

| 控制模式 | 命令内容 | 接口 |
|---|---|---|
| `state` | 16D：位置(3)、速度(3)、加速度(3)、`xyzw` 四元数(4)、机体角速度(3) | `sim.state_control(cmd)` |
| `attitude` | `roll, pitch, yaw, collective_thrust` | `sim.attitude_control(cmd)` |
| `body_rate` | `wx, wy, wz, collective_thrust`，rad/s 与 N | `sim.body_rate_control(cmd)` |
| `force_torque` | `force_z, torque_x, torque_y, torque_z` | `sim.force_torque_control(cmd)` |
| `rotor_vel` | 四个电机转速（RPM） | `sim.rotor_vel_control(cmd)` |

`body_rate`、`force_torque` 和 `rotor_vel` 需要 `first_principles` 动力学。可选动力学包括：

- `first_principles`：包含电机和刚体物理的高保真模型；
- `so_rpy`：直接抽象姿态响应；
- `so_rpy_rotor`：在抽象姿态模型中加入集体推力响应动态；
- `so_rpy_rotor_drag`：额外加入机体系线性阻力。

常用控制示例：

```bash
python examples/control/hover.py          # 定点悬停
python examples/control/spiral.py         # 多机螺旋轨迹
python examples/control/attitude.py       # 姿态控制
python examples/control/force_torque.py   # 力和力矩控制
python examples/control/body_rate.py      # 新增：机体角速度控制
python examples/control/dynamics.py       # 新增：四种动力学模型对照
python examples/control/sampling.py       # GPU 并行采样 MPC
```

## Gymnasium 环境与批量强化学习接口

导入 `crazyflow` 后会注册四个向量环境：

| 环境 ID | 任务 |
|---|---|
| `DroneReachPos-v0` | 到达目标位置 |
| `DroneReachVel-v0` | 跟踪目标速度 |
| `DroneLanding-v0` | 着陆 |
| `DroneFigureEightTrajectory-v0` | 跟踪八字轨迹 |

运行向量环境示例：

```bash
python examples/environments/gymnasium_env.py
python examples/environments/figure8.py
```

代码中可用 `gymnasium.make_vec(..., num_envs=N, device="gpu")` 创建并行环境；`JaxToNumpy` 或 `JaxToTorch` 用于与 NumPy/PyTorch 策略连接。

## 可微仿真、随机化与扰动

仿真步进函数是 JAX PyTree 到 JAX PyTree 的纯函数，可直接使用 `jax.jit`、`jax.vmap` 和 `jax.grad`：

```bash
python examples/jax/gradient.py
python examples/symbolic.py
```

`step_pipeline` 和 `reset_pipeline` 可以插入自定义函数，用于域随机化、外力、传感器估计或动作延迟：

```bash
python examples/plugins/randomize.py
python examples/plugins/disturbance.py
python examples/plugins/action_delay.py
python examples/plugins/estimation.py
```

修改 pipeline 后调用 `sim.build_step_fn()` 或 `sim.build_reset_fn()` 重新编译。

新增的地效、下洗和导数记录示例：

```bash
python examples/plugins/ground_effect.py  # 悬停高度与推力的关系
python examples/plugins/downwash.py       # 多机下洗外力与力矩
python examples/plugins/derivatives.py    # 在 plugins 中显式记录状态导数
python examples/jax/gradient_clipping.py  # 电机转速裁剪及其梯度
python examples/jax/sharding.py           # 可用设备上的 world 分片
```

state 命令从旧 13D 改为 16D；不要把 yaw 标量直接放到四元数位置。
`sim.data.states_deriv` 已移除，导数可用 `sim.dynamics_fn(sim.data)` 获取。
逐 world 插件状态需 `flax.struct.dataclass` 和 `CORE_NDIM_KEY`，保证局部 reset 和分片正确。
完整迁移示例见 [`usage.md`](usage.md)。

## 可视化、相机与接触

MuJoCo 支持交互窗口、离屏 RGB、深度和 RGB-D 渲染。`camera` 可传相机名称（如 `fpv_cam:0`、`track_cam:0`）、相机编号或 `-1` 全局视角。

常用示例：

```bash
python examples/rendering/render.py       # 25 架无人机
python examples/rendering/cameras.py      # 机载 RGB-D 相机
python examples/rendering/raycasting.py   # JAX/MJX 深度射线
python examples/rendering/led_deck.py     # LED 与自定义几何体
python examples/contacts/crash.py         # 碰撞和接触检测
```

把 RGB-D 动画保存到当前目录：

```bash
python -c 'from examples.rendering.cameras import main; main(show_plot=False, save_plot=True)'
```

生成文件为 `cameras.gif`。程序中可按需调用：

```python
rgb = sim.render(mode="rgb_array", camera="fpv_cam:0", width=320, height=240)
depth = sim.render(mode="depth_array", camera="fpv_cam:0", width=320, height=240)
rgb, depth = sim.render(mode="rgbd_tuple", camera="fpv_cam:0", width=320, height=240)
contacts = sim.contacts()
```

镜像还包含 Gaussian Splatting 查看器和 CUDA 相机。示例首次运行会从 Hugging Face 下载场景与无人机 `.ply`：

```bash
python examples/rendering/splat_viewer.py
python examples/rendering/splat_camera.py
python examples/rendering/splat_depth.py
```

## LSY Autonomous Drone Racing

drone racing 项目固定安装在 `/opt/lsy_drone_racing`，包含 Crazyflow 赛道环境、单机/多机控制器、acados MPC 和 PPO 训练代码：

本次兼容补丁保留课程控制器对外的 13D state 动作，在赛道环境内部将 yaw 转成四元数后
提交给 Crazyflow 的 16D state 接口；直接使用 `Sim.state_control()` 时必须传 16D。
参数加载及每 world 随机化也已适配新版接口，外部 racing 固定提交保持不变。

```bash
cd /opt/lsy_drone_racing
```

难度配置位于 `config/`：

| 配置 | 内容 |
|---|---|
| `level0.toml` | 固定赛道和标称动力学 |
| `level1.toml` | 随机惯性参数 |
| `level2.toml` | 随机惯性、障碍物和门框 |
| `level3.toml` | 进一步随机化赛道，需要在线规划 |
| `multi_level*.toml` | 多机 drone racing |

单机 drone racing：

```bash
python scripts/sim.py --config=level0.toml --render=True
```

用命令行覆盖同一控制模式的控制器：

```bash
python scripts/sim.py \
  --config=level0.toml \
  --controller=state_controller.py \
  --render=True
```

姿态、MPC 和 RL 控制器输出 4 维姿态命令，不能直接配合 `level0.toml` 的 13 维 `state` 动作空间。先生成姿态控制配置：

```bash
cp config/level0.toml config/level0_attitude.toml
sed -i 's/control_mode = "state"/control_mode = "attitude"/' \
  config/level0_attitude.toml

python scripts/sim.py \
  --config=level0_attitude.toml \
  --controller=attitude_mpc.py \
  --render=True
```

控制器文件位于 `lsy_drone_racing/control/`：

- `state_controller.py`：状态轨迹控制；
- `attitude_controller.py`：姿态控制；
- `attitude_mpc.py`：基于 acados 的姿态 MPC；
- `attitude_rl.py`：加载 PPO 检查点的策略控制。

多机 drone racing：

```bash
python scripts/multi_sim.py \
  --config=multi_level0.toml \
  --controllers=attitude_controller_multi.py,attitude_mpc_multi.py \
  --render=True
```

课程的 Level 2 批量评测固定运行 20 局，并使用 `config/level2.toml` 中选择的控制器：

```bash
python scripts/evaluate.py
```

成功率不低于 50% 时结果写入 `evaluation.csv`，否则脚本会报错退出。仓库默认的 `state_controller.py` 是基础示例；评测前应先在 `level2.toml` 中换成自己的 Level 2 控制器。

## 训练和评估 PPO 策略

训练器默认使用 1024 个 JAX/CUDA 并行环境，完成约 150 万环境步；训练和评估会自动选择 PyTorch/CUDA。JAX 环境与 PyTorch 策略位于同一 GPU，数据通过 GPU array 转换传递。

完整训练：

```bash
cd /opt/lsy_drone_racing
python -m lsy_drone_racing.control.train_rl \
  --wandb-enabled False \
  --train True \
  --eval 0
```

检查点写入：

```text
/opt/lsy_drone_racing/lsy_drone_racing/control/ppo_drone_racing.ckpt
```

加载检查点并运行一局可视化评估：

```bash
python -m lsy_drone_racing.control.train_rl \
  --wandb-enabled False \
  --train False \
  --eval 1
```

关闭窗口并重复评估五局：

```bash
python -m lsy_drone_racing.control.train_rl \
  --wandb-enabled False \
  --train False \
  --eval 5 \
  --render False
```

`--render` 默认 `True`，用于保留可视化评估。

在 drone racing 控制器中使用该策略：

```bash
python scripts/sim.py \
  --config=level0_attitude.toml \
  --controller=attitude_rl.py \
  --render=True
```

需要保留新训练的检查点时，从宿主机复制出来：

```bash
docker cp \
  dzp-crazyflow-research-20261009:/opt/lsy_drone_racing/lsy_drone_racing/control/ppo_drone_racing.ckpt \
  ./ppo_drone_racing.ckpt
```

## 本次增量功能的复现实验

在容器的 Crazyflow 根目录执行：

```bash
cd /workspace/crazyflow
MPLBACKEND=Agg python usage_assets/research_update_experiments.py --device gpu
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python usage_assets/research_update_experiments.py --mode sharding --device cpu
MPLBACKEND=Agg pytest -q tests
pytest -q -m render tests
pytest -q --markdown-docs --markdown-docs-syntax=superfences \
  --doctest-glob='*.md' crazyflow/ docs/ --ignore=docs/gen_ref_pages.py
```

新功能实验覆盖 16D state、body rate、X500、地效/下洗对照、局部 reset、导数与裁剪梯度，
方法及结果见 [`usage.md` 第 15 节](usage.md#15-2026-10-09-官方-main-增量同步)。
多 GPU 性能需多张 GPU；单 GPU 结果和模拟多 CPU 设备的一致性检查分别记录。

四个 Gymnasium 任务继续使用同一 PPO 训练器。新实验另存 checkpoint 和图片，保留历史结果：

```bash
MPLBACKEND=Agg python usage_assets/train_gymnasium.py \
  --env-id DroneReachPos-v0 --num-envs 4096 --num-steps 64 \
  --total-timesteps 10000000 --eval-envs 256 \
  --checkpoint saves/research_20261009_reach_pos.pt \
  --output usage_assets/research_20261009_reach_pos.png
```

ReachVel 和 FigureEight 用 `--total-timesteps 20000000`，Landing 用 `10000000`；
切换 `--env-id` 并对应修改输出名称即可。评估口径及历史/本次对照见 `usage.md`。

## 本次 Docker 实验结果（2026-10-09）

在 RTX 4070 Ti SUPER 上用 seed 7 训练，评估采用 256 个 world、500 步（10 s）和
确定性动作；成功需全程未终止且末 1 s 误差低于任务阈值。ReachPos/Landing 各完成
9,961,472 个 PPO transitions，并各有 4,096,000 个教师 BC samples；ReachVel/Figure8
各完成 19,922,944 个 PPO transitions。BC 与 PPO 样本分开计数。

| 任务 | 末 1 s RMSE | 成功率 |
|---|---|---|
| ReachPos | 0.041216 m | 100.00% |
| ReachVel（默认三维目标） | 0.073598 m/s | 57.42% |
| Landing | 0.046088 m | 100.00% |
| Figure8 | 0.044990 m | 100.00% |
| ReachVel（同 checkpoint，只评估水平目标） | 0.049551 m/s | 97.27% |

三维速度任务的 survival 为 58.20%；水平目标对照的 survival 为 100%。全局 RMSE
不能直接换算成功率，水平目标结果也不能代替默认三维任务成绩。完整条件、训练曲线和原始指标见
[本轮 Gym PPO 报告](usage_assets/research_ppo_20261009_results.md)。

新接口实验中，16D 圆轨迹的位置 RMSE 为 0.084014 m；0.4 rad/s 的 body-rate
指令测得 0.399999 rad/s。地效、下洗、导数和裁剪梯度对照的全部断言通过，双逻辑 CPU
设备的 world 分片一致性通过。默认 cf21B 控制参数与物理质量不同，1 m 悬停指令的
实际末秒高度为 1.081058 m；详情见[功能实验报告](usage_assets/research_update_results.md)。

Racing 默认 PPO 完成 1,499,136 个 transitions；重载五局平均奖励 703.63，均运行
750 步。同 checkpoint 的姿态控制实赛在 13.34 s 通过全部 4 个门，单机 state 控制为
16.50 s、多机姿态与 MPC 分别为 10.10 s / 13.43 s，均完成 4/4 门。

验证结果：核心测试 **771 passed / 47 skipped / 12 deselected**，文档测试 **103 passed**，
渲染测试 **12 passed**，关键用法片段 **26/26 passed**；Ruff 检查和格式检查通过。
本节记录的构建、验证与实验均在 Docker 内执行，未使用服务器。

## 常用操作

```bash
docker stop dzp-crazyflow-research-20261009
docker start dzp-crazyflow-research-20261009
docker exec -it dzp-crazyflow-research-20261009 bash
docker rm -f dzp-crazyflow-research-20261009
```

Crazyflow 源码和示例在 `/workspace/crazyflow`；镜像内 drone racing 项目在 `/opt/lsy_drone_racing`。修改 Crazyflow Python 代码后无需重建镜像，重新启动对应程序即可；修改系统依赖或 drone racing 安装补丁后需要重建镜像。
