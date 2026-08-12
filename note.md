# Crazyflow Docker 用户手册

## 配置

在宿主机进入项目根目录：

```bash
cd /home/dzp/projects/crazyflow
```

构建镜像：

```bash
docker build -f .devcontainer/Dockerfile \
  -t dzp_crazyflow:0.3.2-cuda12.6-py312-racing-rl \
  --progress=plain .
```

Dockerfile 会在构建开始和结束时访问 `https://www.google.com/`；HTTP 错误会直接中止构建。

启动带 GPU 和 X11 图形界面的容器：

```bash
test -f "$XAUTHORITY"

docker run --name dzp-crazyflow -itd \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size=4g \
  -e DISPLAY \
  -e XAUTHORITY=/tmp/.Xauthority \
  -v "$XAUTHORITY:/tmp/.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/dzp/projects/crazyflow:/workspace/crazyflow \
  dzp_crazyflow:0.3.2-cuda12.6-py312-racing-rl
```

进入容器：

```bash
docker exec -it dzp-crazyflow bash
cd /workspace/crazyflow
```

代码通过 bind mount 实时映射到 `/workspace/crazyflow`；镜像内的固定环境位于 `/opt/crazyflow/.pixi/envs/gpu-tests`。容器将 MuJoCo 与 MJX 固定为 3.10.0，以兼容 Gymnasium 1.3 的交互相机接口，并保持 Crazyflow 核心源码不变。

## 检查运行环境

查看显卡、CUDA 和 JAX 后端：

```bash
nvidia-smi
nvcc --version

python - <<'PY'
import jax
import jax.numpy as jnp

x = jax.device_put(jnp.ones((1024, 1024)), jax.devices("gpu")[0])
y = (x @ x).block_until_ready()
print(jax.default_backend(), y.device, y[0, 0])
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

cmd = np.zeros((sim.n_worlds, sim.n_drones, 13))
cmd[..., :3] = [0.0, 0.0, 0.5]

for _ in range(100):
    sim.state_control(cmd)
    sim.step(sim.freq // sim.control_freq)

print(sim.data.states.pos.shape)
sim.close()
```

四种控制接口如下：

| 控制模式 | 命令内容 | 接口 |
|---|---|---|
| `state` | `x, y, z, vx, vy, vz, ax, ay, az, yaw, roll_rate, pitch_rate, yaw_rate` | `sim.state_control(cmd)` |
| `attitude` | `roll, pitch, yaw, collective_thrust` | `sim.attitude_control(cmd)` |
| `force_torque` | `force_z, torque_x, torque_y, torque_z` | `sim.force_torque_control(cmd)` |
| `rotor_vel` | 四个电机转速（RPM） | `sim.rotor_vel_control(cmd)` |

`force_torque` 和 `rotor_vel` 需要 `first_principles` 动力学。可选动力学包括：

- `first_principles`：包含电机和刚体物理的高保真模型；
- `so_rpy`：直接抽象姿态响应；
- `so_rpy_rotor`：在抽象姿态模型中保留电机动态；
- `so_rpy_rotor_drag`：额外加入旋翼阻力。

常用控制示例：

```bash
python examples/control/hover.py          # 定点悬停
python examples/control/spiral.py         # 多机螺旋轨迹
python examples/control/attitude.py       # 姿态控制
python examples/control/force_torque.py   # 力和力矩控制
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

训练器默认使用 1024 个 JAX/CUDA 并行环境，完成约 150 万环境步；策略网络使用 CPU PyTorch，避免镜像中重复安装一套 PyTorch CUDA 运行库。

完整训练：

```bash
cd /opt/lsy_drone_racing
python lsy_drone_racing/control/train_rl.py \
  --wandb_enabled=False \
  --train=True \
  --eval=0
```

检查点写入：

```text
/opt/lsy_drone_racing/lsy_drone_racing/control/ppo_drone_racing.ckpt
```

加载检查点并运行一局可视化评估：

```bash
python lsy_drone_racing/control/train_rl.py \
  --wandb_enabled=False \
  --train=False \
  --eval=1
```

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
  dzp-crazyflow:/opt/lsy_drone_racing/lsy_drone_racing/control/ppo_drone_racing.ckpt \
  ./ppo_drone_racing.ckpt
```

## 常用操作

```bash
docker stop dzp-crazyflow
docker start dzp-crazyflow
docker exec -it dzp-crazyflow bash
docker rm -f dzp-crazyflow
```

Crazyflow 源码和示例在 `/workspace/crazyflow`；镜像内 drone racing 项目在 `/opt/lsy_drone_racing`。修改 Crazyflow Python 代码后无需重建镜像，重新启动对应程序即可；修改系统依赖或 drone racing 安装补丁后需要重建镜像。
