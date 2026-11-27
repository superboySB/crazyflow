# CrazyRL → Crazyflow 的任务对应

参考仓库固定到 CrazyRL `9333442c18fa15799adac7e02988a978aeaf97b9`，各保留文件的 SHA256 见 [`sim2sim/provenance.json`](../sim2sim/provenance.json)。这里只重写训练环境；Isaac 评测环境继续使用保存的参考源码。Crazyflow 的 functional `SimData`、`Sim.build_step_fn()`、默认 first-principles 动力学和官方 RK4 积分器负责训练物理推进。训练循环使用 JAX/Flax/Optax PPO，环境输入动作和所有输出都有 `jax.lax.stop_gradient`，物理轨迹不参与反向传播。

对照源文件是 CrazyRL 的：

- `e2e_drone/tasks/baseline/quad_baseline_env_cfg.py`：任务、传感器、时间、奖励系数和随机化参数。
- `e2e_drone/tasks/baseline/quad_baseline_env.py`：`_pre_physics_step`、`_apply_action`、`_get_observations`、`_get_rewards`、`_get_dones`、`_reset_idx`、`_regenerate_terrain`、`_update_curriculum`。
- `e2e_drone/utils/controller.py`、`wind_gen.py`、`thrust_uncertainty.py`、`map_generator.py`：控制映射、OU 风、推力衰减和障碍分布。
- 固定 IsaacLab 的 `sensors/ray_caster/patterns/patterns.py` 与 `ray_caster_camera.py`：像素中心、射线方向和 image-plane depth。

本分支对应实现为 [`config.py`](../crazyflow/tof/config.py)、[`env.py`](../crazyflow/tof/env.py)、[`scene.py`](../crazyflow/tof/scene.py)。数值验收位于 [`test_tof_environment.py`](../tests/test_tof_environment.py)，包含手算奖励、动作单位、深度射线、原生批量动力学、无物理梯度、超时 bootstrap、随机化、双地图和参数文件冻结。

## 策略输入与动作

每个 actor 帧为 28 维，排列和倍率如下，历史按“旧 → 新”连接，共 10 帧，最后连接当前 64 维前向深度，合计 344 维。

| 数据 | 维数 | 输入倍率 | 逐元素乘性噪声 |
|---|---:|---:|---|
| body 线速度 | 3 | 25 | N(1, 0.3²) |
| body→world 旋转矩阵，按行展开 | 9 | 127 | 无 |
| body gyro | 3 | 30 | N(1, 0.1²) |
| world XY 目标单位方向、目标绝对 z | 3 | 127 | XY 位移先乘 N(1, 0.2²)，再单位化 |
| 当前 z − 目标 z | 1 | 30 | N(1, 0.1²) |
| 当前期望速度 | 1 | 25 | 无 |
| 上一拍处理后的动作 | 4 | 127 | 无 |
| 左、右、后、下单点 ToF | 4 | 30 | N(1, 0.1²) |

critic 每帧在上述真值帧之后加入：目标相对三维位置（模长截到 5 m）×127、最近障碍相对向量（模长截到 1 m）×127、最近 10 个物理子步的平均风加速度 ×30。37×10+64=434 维。参考代码实际上复用了带 XY 噪声的目标方向，critic 也保留这个细节；其余上述通道不加观测噪声。回合开始时历史清零，再写入首帧。

网络输出原始无界 Gaussian 四维动作。环境先 `tanh`，再将推力通道从 [−1,1] 映射到 [0,1]。最终物理目标是 roll/pitch ±π/6 rad、yaw rate ±π/1.5 rad/s、总推力 `throttle × reference_max_thrust`。最大总推力沿用参考控制器的多项式与 1.358 系数，不另用“悬停附近”的动作定义。PPO 的 log probability 始终对应原始 Gaussian 动作。

策略 15 Hz，物理 150 Hz，每拍 10 个子步。这里选用 Crazyflow 自带的 `Integrator.rk4`，配置与模型元数据明确保存 `integrator: rk4`；旧 Euler 模型仍可按其 `integrator: euler` 重放。选择理由是：原生电机动态与参考高增益 PD 组合在 Euler/150 Hz 下出现数值振荡。相同初始姿态 `[0.2,−0.1,0]`、悬停转速、无噪声条件下，8 s 测试最后一秒的 roll RMS 为 Euler/150 Hz 0.281 rad、RK4/150 Hz 0.00259 rad；Euler/500 Hz 参考为 0.000274 rad。RK4 保持参考 150 Hz 控制与延迟时间，不修改辨识参数。动作 FIFO 延迟 4 个物理子步，即约 26.7 ms。参考初始 FIFO 的默认命令是 `[0, 0, 0.5, 0]`，这里保留该初始化。本实现逐世界重置 FIFO；参考源码中的共享 deque 未在 `_reset_idx` 清空，可能跨回合保留命令。本实现避免这项回合间状态泄漏。

## ToF / 深度

前向相机为 8×8，光心 body `[0.02,0,0]`；其余四个单点相机位于对应方向的 0.02 m 偏置。焦距 20、水平/垂直 aperture 20.955，像素采用 `arange + 0.5` 的中心坐标。配置里 `MP_TOF_FOV=45` 和 `SP_TOF_FOV=10` 未用于实际参考相机；本实现以实际针孔参数生成射线。

保存的是 image-plane depth：用 `[1, −u, −v]` 非单位前向射线时，射线参数就是该深度。Isaac 的实际 ray caster 先以 `max_dist=1e6` 求交，转换到相机 image plane，再把深度截至 4 m，因此本实现也不按射线欧氏长度提前裁剪。没有命中时报告 4 m。环境每次观测先抽一个共享失效概率 `U(0,0.15)`，再独立将前向像素替换成 4 m，最后乘 N(1,0.1²)；所以进入 actor 的加噪深度可能超过 4 m。critic 使用未失效、未加噪的深度。

传感器数值验收覆盖圆柱轴线平行射线的命中/未命中与端盖，以及圆锥母线的线性退化方程和重合表面射线；普通二次求交采用避免相消的求根形式。

`state.depth` 是原始 64 维深度，`env.ray_directions` 是相应射线。MuJoCo 显示实际 Crazyflow 轨迹与测距射线，显示过程中不使用第二套动力学推进飞机。

## 奖励与回合结局

所有奖励按策略拍计算，不另乘 dt。令 `d` 为目标三维距离，`v` 为 body 速度，`s=||v||`，动作 `a` 为经过 `tanh` 与推力映射的四维量，`Hδ(x)` 为 `|x|≤δ` 时 `0.5x²/δ`、否则 `|x|−0.5δ` 的 Huber 形状。

| 项 | 未乘系数的公式 | 默认系数 |
|---|---|---:|
| 接近进展 | `min(d_previous − d, 0.01)`；负值不截断 | 8 |
| yaw 对齐 | `(dot(forward_xy_unit, goal_xy_unit)−1) × clip(d,0,1)`；d<0.2 时为 0 | 0.3 |
| 动作幅度 | `−||a||` | 0.3 |
| 动作变化 | `−||a−a_previous||` | 0.5 |
| 速度方向 | `−H0.2(s × (1−clip(vx/max(s,1e−6),−1,1)))` | 0.15 |
| 超速 | `−min(exp(5×max(s−desired,0))−1,5)` | 0.5 |
| 速度匹配 | `2×exp(−5×|s−desired|)` | 0 |
| 高度误差 | `−H0.3(z−goal_z)` | 0.25 |
| 碰撞 | −1 | 80 |
| 障碍距离 | `−(exp(5×max(0.5−nearest_surface_distance,0))−1)` | 0.1 |
| 到达 | +1 | 80 |

`desired = initial_speed × clip(d,0,1)`，初始速度 U(0.5,1.5) m/s。到达要求 d<0.5 m、s<0.5 m/s，并连续满足当前 hover hold 时长；初始 hold 为 0。碰撞、到达或 body 任意角速度绝对值超过 `3.14×2×20` 导致终止；另处理非有限状态。最长 1200 策略拍，即 80 s。高度不是额外的低空/高空终止阈值，地板和天花板通过碰撞判定。

训练在产生成功奖励的同一步结束回合，并在返回前局部重置结束的世界。参考 DirectRLEnv 的 done/reward 调用顺序可能让成功状态到下一拍才结束。`StepResult.terminal_obs/terminal_critic_obs` 始终保留重置前的数据；超时从该 critic 状态 bootstrap，成功/碰撞等真正终止不 bootstrap。所有重置都会切断 GAE 的回合递推。

## 随机化：保留参考实现真正执行的语义

**质量扰动并未写入参考物理机体。** 参考 `_setup_scene` 将 USD body 的质量固定为 0.049 kg、惯量属性写为 `(1.3615e−5,1.3615e−5,3.257e−5)`。随后 `__init__` 每个世界仅采样一次 controller mass `0.049×U(0.8,1.2)`，令控制器惯量 `[2.16e−5,2.16e−5,4.33e−5]` 同乘该质量倍率；风外力也乘这个 controller mass。本实现按同样方式扰动控制器惯量和风力换算，回合重置不会重采样 controller mass，Crazyflow 的实际机体质量保持官方默认。不能将它解释成“每回合物理质量 ±20%”。

风是每个 150 Hz 子步更新的三轴 OU 加速度：`alpha=exp(−dt/1.0)`、`wind_next=alpha×wind+0.5×sqrt(1−alpha²)×N(0,I)`。重置时为零，外力为 `controller_mass×wind`。critic 的额外风输入仍是风加速度的子步均值。

每回合推力有效率初值 U(0.85,1.0)，衰减因子 U(0.10,0.15)，单步衰减率为 `factor/1200`。**参考在每个物理子步执行 `eta=max(eta×(1−factor/1200),0.7)`，不是每个策略拍更新**，本实现保留这个容易被误读的细节。有效率只缩放总推力，不额外缩放控制力矩；Crazyflow 中作用于原生电机当前推力，因此仍保留原生电机动态。

默认高度随机器关闭。开启观测噪声时采用上表的乘性噪声，不加入原参考没有的 gyro bias、动作白噪声或额外观测延迟。`domain_randomization=false` 关闭控制器质量扰动、风和推力衰减，`observation_noise=false` 单独关闭测量噪声。

## 地图、双地图更新和课程

障碍区为 16×16 m，墙内边界为 ±11 m，地板顶面 z=0.01、天花板底面 z=2.0。地面立柱由 30 次候选尝试的 Bridson Poisson disk 采样，中心最小间隔为 `max_obstacle_size + min_distance`；长/宽 U(0.4,0.8)、高 U(2.4,2.5)，随机 cuboid/cylinder。浮空障碍采用参考的三维分层、0.8 格内 jitter、相邻网格检查和补充随机尝试，含随机姿态 cuboid/cylinder/cone，尺寸 U(0.4,0.8)，中心高 U(0.3,1.7)。保留参考浮空 cuboid 的原型长宽高比例 1:0.5:0.5。请求上限并非最终障碍数量；最小间隔限制会显著减少最终数量。

出生点与目标候选来自 x/y∈[−7,7)、z∈[0.5,1.5) 的 0.2 m 网格，先抽最多 20000 个候选、对外部点抽最多 10000 个，再保留至少 0.20 m 表面间隙的点。出生点不加位置扰动；目标再加入每轴 U(−0.15,0.15)。初始 yaw 为 U(−π,π)。解析表面查询替代参考三角网格奇偶检测与采样点 KD-tree，因此同一个随机种子不承诺跨 PyTorch/NumPy/JAX 生成逐点一致的地图；Isaac 公平评测通过自己的配对场景校验保证同场景。

默认使用两个地图槽，所有新回合进入当前 active 槽；旧回合可继续使用旧槽。参考的“3000”计数实际在 `_reset_idx` 批量调用时递增，**不是 3000 个 PPO update，也不是累计 3000 个单独 episode**。设备端按 `any(done)` 维护同样的批量重置计数。训练器在每个 PPO rollout 后调用 `maybe_regenerate`：到阈值且 inactive 槽无人占用时，CPU 生成新的 inactive 地图、交换 active，并重新编译使用新常量的环境函数。它不覆盖正在飞行的旧地图。课程更新先于 inactive 占用检查；因此槽仍被占用时，本次课程变化仍会生效，这是参考实现的调用顺序。

地图维护检查在 rollout 边界发生，默认 120 拍长度带来最多 120 次 reset-batch 的调度偏移；GPU 内的传感器、物理和 PPO rollout 仍全批量执行，地图生成不会把每个物理步搬到 CPU。地图检查前先应用参考课程：最近 100 个结束回合成功率>0.9 时，hover hold 向 `min(2.0,2.0×success_rate)` 每次最多增加 0.1 s，min_distance 每次除以 1.1、下限 0.4 m；已到下限后改超速系数为 0.6、速度匹配系数为 0.2。`runtime_state()` 保存当前课程参数、地图代数、每次生成的 seed/间隔和 active 槽；完整 checkpoint 还保存环形结果窗口与计数，恢复时重建相同地图。

## 有意保留的模拟器差异

这里的训练物理是 Crazyflow，而非把 Isaac 外力动力学换个库名。默认 `cf21B_500` 来自 [`config/dynamics/first_principles/params.toml`](../config/dynamics/first_principles/params.toml)：质量 0.0434 kg、惯量对角 `[25e−6,28e−6,49e−6]` kg·m²，原生 RPM→推力/反扭矩曲线、四电机转速动态、桨惯性和机体阻力。PD 姿态/角速度目标、控制器惯量及其随机化保持参考定义；分配后的电机推力转换到原生 Crazyflow RPM/电机排列，由官方 pipeline 推进。参考 Isaac 评测继续使用原始 USD、PhysX 和外力/力矩控制，不改动力学。

训练对 cuboid、cylinder、cone 使用解析射线与表面距离，支持浮空障碍三维随机姿态；MuJoCo 的 cone 显示为分面网格。无人机训练碰撞用半径 0.055 m 的质心球，与每个物理子步的场景解析距离比较；参考使用缩放 USD collider 的 PhysX contact force 阈值 0.01 N。解析 ESDF 的最近表面与参考对 200000 个采样表面点的 KD-tree 查询也不会逐数相同。这些几何与电机差异正是需要最后 Isaac sim2sim 验证的部分，不应把 Crazyflow 成功率当成 Isaac 成功率。

可选 `model_params_path` 读取参数辨识导出的 TOML；默认不用它。加载时通过共享物理校验，重算惯量逆矩阵和转速限值，不自动改变参考 PD 控制器。导出策略的 `effective_config()` 无论使用官方默认值还是辨识文件，都会把整份实际物理参数嵌入 `model_params` 并清空路径；完整 checkpoint 的 runtime 同样冻结这些值。因此移动或修改原 TOML、更新本地官方参数表都不会悄悄改变已训练模型的评测物理。数值验收比较默认环境与导出配置重建的环境，在零推力、悬停和饱和动作下推进 100 个原生物理子步，要求完整 SimData 逐位相同。训练元数据另记录源码 SHA256，供版本核验。
