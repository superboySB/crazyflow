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

| 每帧切片（从 0 起，右端不含） | 数据 | 维数 | 输入倍率 | 逐元素乘性噪声 |
|---|---|---:|---:|---|
| `[0:3]` | body 线速度，m/s | 3 | 25 | N(1, 0.3²) |
| `[3:12]` | body→world 旋转矩阵，按行展开 | 9 | 127 | 无 |
| `[12:15]` | body gyro，rad/s | 3 | 30 | N(1, 0.1²) |
| `[15:18]` | world XY 目标单位方向、目标绝对 z（m） | 3 | 127 | XY 位移先乘 N(1, 0.2²)，再单位化 |
| `[18:19]` | 当前 z − 目标 z，m | 1 | 30 | N(1, 0.1²) |
| `[19:20]` | 当前期望速度，m/s | 1 | 25 | 无 |
| `[20:24]` | 上一拍归一化动作：前三项为 tanh 后的量，第四项为 [0,1] throttle | 4 | 127 | 无 |
| `[24:28]` | 左、右、后、下单点 ToF，m | 4 | 30 | N(1, 0.1²) |

最终 actor 输入 `[0:280]` 是 10 个完整帧，最新一帧为 `[252:280]`；`[280:344]` 只有当前前向 8×8 深度，不是 10 帧深度历史。像素按行、再按列展开，索引为 `8×row+column`；图像左上角射线指向 body 的 +y/+z，右下角为 −y/−z。四个辅助 ToF 保留在每个历史帧内部。上一拍动作通道尚未换算为 rad、rad/s 或 N。冻结参考对应 [`quad_baseline_env.py`](../sim2sim/reference/e2e_drone/tasks/baseline/quad_baseline_env.py) 的 `_get_observations`，本分支对应 [`env.py`](../crazyflow/tof/env.py) 的 `_observe`；[`test_tof_observation_contract.py`](../tests/test_tof_observation_contract.py) 用不同通道/历史/像素的哨兵值检查这些语义。

前向 64 个深度通道同样先把以 m 表示的测量乘 **30**，再使用该模型自己的归一化；actor 在失效点替换和乘性噪声之后缩放，critic 对未加噪深度缩放。GUI 显示的 m 数值不能直接作为未经缩放的网络输入。

critic 每帧在上述真值帧之后加入：目标相对三维位置（模长截到 5 m）×127、最近障碍相对向量（模长截到 1 m）×127、最近 10 个物理子步的平均风加速度 ×30。37×10+64=434 维。参考代码实际上复用了带 XY 噪声的目标方向，critic 也保留这个细节；其余上述通道不加观测噪声。回合开始时历史清零，再写入首帧。

网络输出原始无界 Gaussian 四维动作。环境先 `tanh`，再将推力通道从 [−1,1] 映射到 [0,1]。最终物理目标是 roll/pitch ±π/6 rad、yaw rate ±π/1.5 rad/s、总推力 `throttle × reference_max_thrust`。最大总推力沿用参考控制器的多项式与 1.358 系数，不另用“悬停附近”的动作定义。PPO 的 log probability 始终对应原始 Gaussian 动作。

| 原始输出索引 | 经过环境映射后的物理含义 |
|---|---|
| `0` | `tanh(u[0])×π/6`，期望 roll **角度**，rad |
| `1` | `tanh(u[1])×π/6`，期望 pitch **角度**，rad |
| `2` | `tanh(u[2])×π/1.5`，期望 body yaw 角速度，rad/s |
| `3` | `(tanh(u[3])+1)/2×0.775313294`，期望总推力，N |

参考观测的部分注释把 roll/pitch 写成 rate，但实际 [`controller.py`](../sim2sim/reference/e2e_drone/utils/controller.py) 的 `compute_control` 使用姿态角；两边都按实际控制器实现。原始四个输出为零时，推力是 0.5 倍上限，而不是 0 N。策略不直接输出电机 RPM；Crazyflow 后续通过自己的电机分配与动态执行这些目标。

NPZ 和原始 `.pt` 的 actor 使用同样的输入物理约定，但分别携带各自训练得到的归一化统计。在上述单位与倍率之后，NPZ 使用 `(obs−own_mean)/(sqrt(own_var)+0.01)`，原始 RSL `.pt` 使用保存的 `own_std+0.01` 作分母，再经过 ELU MLP 输出原始动作均值。基线文件内保存的 std 与重新计算的 sqrt(var) 有约 `7.63e−6` 的最大浮点差异；不是输入物理含义的变化。JAX [`policy.py`](../crazyflow/tof/policy.py) 的 `act` 与 Isaac [`npz_policy.py`](../sim2sim/npz_policy.py) 的 `NpzPolicy.__call__` 不再做 `tanh`；Isaac 仍由冻结任务 `_pre_physics_step` 映射一次。不能把 baseline 的 mean/var 配到新模型上，也不能把 434 维 critic 输入传给 344 维部署 actor。

相同接口并不意味着两个独立训练模型输出相同。[实际 checkpoint I/O 验收](../reports/sim2sim/checkpoint-io-audit.json) 检查了交付 NPZ、原始 `.pt` 和 128 个闭环飞行观测包，交付模型在 JAX 与 Torch 的最大动作差为 `1.43e−6`；多加一次 `tanh` 则会产生最大约 2.07 的原始动作偏差。同一模型的另一组数值一致性验收见 [`jax-torch-policy-parity.json`](../reports/sim2sim/final/jax-torch-policy-parity.json)。

训练在每个保存点同时写出 `policy.npz`（JAX/MuJoCo）与 `policy.pt`（Torch/RSL、Isaac）。PT 使用转置后的同一组 Dense 权重、相同的均值/方差，以及 `std=exp(clip(logstd,−5,2))`；输入仍为 344 维，输出仍为未经 `tanh` 的四维动作。每次导出先验证两种实现的输出一致，并记录源 NPZ 的 SHA256。新训练保存点的 PT 包含真实 critic 和 434 维归一化统计，已用[实际 A 模型](../reports/training-study/export-parity.json)验收；新候选没有通过替换门槛，因此交付仍为原 actor-only 策略，其配套 PT 不补造 critic。完整 JAX/Optax 续训仍读取 `checkpoint.msgpack`。原保留策略的格式验收见 [`dual-export.json`](../reports/dual-export.json)。

## PPO：历史模型与当前对齐实现

当前 [`training.yaml`](../config/training.yaml) 明确选择 `ppo_variant: rsl_aligned`；缺失该字段的历史配置仍按 `legacy` 解释，完整 resume 会检查两者不可混用。现有已发布策略由 legacy 训练得到，代码对齐和 PT 导出没有改变其权重或历史成绩。

| 内容 | 历史 `legacy` | 当前 `rsl_aligned` |
| --- | --- | --- |
| Advantage 归一化 | 每个 minibatch，population std | 整个 rollout 一次，sample std（`ddof=1`） |
| 自适应 KL | 采样近似 `mean(exp(logratio)−1−logratio)` | 解析 `KL(old Gaussian ∥ new Gaussian)`，含参考的 log 内 `1e−5` |
| 学习率时机和范围 | 当前梯度之后调整，`1e−5…1e−3` | 当前梯度之前调整，`1e−5…1e−2`，仅正 KL 触发增长 |
| Adam epsilon | `1e−5` | `1e−8` |
| 观测统计停止条件 | 默认持续更新 | 默认 count 达到 `100,000,000` 后停止后续更新 |

两种 JAX 实现均在**整段 rollout 及随后的所有梯度 epoch 内冻结 actor/critic 归一化统计**，PPO 更新结束后才合并原始观测统计。停止条件检查的是更新前 count，所以到阈值的一批仍完整计入，越过阈值的幅度小于一个 rollout batch；count 也包含初始校准数据。固定 RSL 实现则在每次返回环境观测时更新统计，缓冲区保存当时已归一化的输入。这是仍保留的训练语义差异。

从随机网络开始时，JAX 先做 `num_steps` 拍随机动作 rollout 校准归一化，不计算策略梯度；resume/warm-start 不重复该步骤。当前 120 拍设置下，研究 A 的额外校准为 122,880 个环境步，B/C 各为 245,760 个；这些步不计入报告的 PPO transitions，发生在训练循环计时起点之前。不能把“PPO 预算相同”解释为所有模拟步及总运行时间完全相同。

网络尺寸与 ELU、horizon=120、5 epochs、γ=0.99、λ=0.95、ratio clip=0.2、entropy=0.005、clipped-value coefficient=1、梯度范数上限 1 已对应。仍保留 JAX 的 logstd 参数化及限幅、正交初始化、每个 epoch 重新 shuffle、按硬件选择 minibatch 数，以及重置前 next-state 的 timeout bootstrap；参考使用直接 std 参数、Torch 默认初始化、跨 epoch 复用排列、保存配置中的 200 minibatches 和当前 value 的 timeout 近似。因此 `rsl_aligned` 不表示整个 PPO 算法及全部超参数逐项相同。详细源码审计与独立数值测试见 [`ppo-parity.json`](../reports/training-study/ppo-parity.json)。[训练研究](../reports/training-study/README.md)已完成等预算对照和单独的预算续训；均未证明迁移提升，最终保留旧策略，默认训练代码仍采用显式 rsl_aligned。

## 物理时间与积分

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

地图维护检查在 rollout 边界发生，默认 120 拍带来至多一个 rollout 的阈值检查延迟；inactive 槽仍被占用时还会继续等待。GPU 内传感器、物理和 PPO rollout 仍全批量执行，地图生成不会把每个物理步搬到 CPU。

达到维护阈值后，先检查最近最多 100 个结束回合的成功率 `s>0.9`。令初始 hold 为 `h0`、当前值为 `h`、上限为 `H`；在默认有效范围 `0≤h0≤H` 下，目标为 `target=min(H,h0+(H−h0)×s)`，仅在 `target>h` 时应用 `h←h+min(0.1,target−h)`。默认 `h0=0、H=2 s`，所以目标才简化为 `min(2,2×s)`，不是每次直接跳到 2 s。同时，间距大于 0.4 m 时应用 `spacing←max(spacing/1.1,0.4)`；若本次检查前已到下限，则改超速系数为 0.6、速度匹配系数为 0.2。`runtime_state()` 保存当前课程参数、地图代数、每次生成的 seed/间隔和 active 槽；完整 checkpoint 还保存环形结果窗口与计数，恢复时重建相同地图。

## 有意保留的模拟器差异

这里的训练物理是 Crazyflow，而非把 Isaac 外力动力学换个库名。默认 `cf21B_500` 来自 [`config/dynamics/first_principles/params.toml`](../config/dynamics/first_principles/params.toml)：质量 0.0434 kg、惯量对角 `[25e−6,28e−6,49e−6]` kg·m²，原生 RPM→推力/反扭矩曲线、四电机转速动态、桨惯性和机体阻力。PD 姿态/角速度目标、控制器惯量及其随机化保持参考定义；分配后的电机推力转换到原生 Crazyflow RPM/电机排列，由官方 pipeline 推进。参考 Isaac 评测继续使用原始 USD、PhysX 和外力/力矩控制，不改动力学。

相同随机化公式也不代表相同物理响应：名义 controller mass=0.049 kg 时，仅风力除以机体质量这一项，Crazyflow 相对 0.049 kg 参考机体的加速度倍率为 `0.049/0.0434≈1.129`。参考推力效率直接缩放即时命令外力；Crazyflow 则按当前 RPM 推力施加效率修正，保留电机动态与 RK4 的瞬态。没有为了匹配分数改写官方 CF21B 参数或 Isaac 评测动力学。

训练对 cuboid、cylinder、cone 使用解析射线与表面距离，支持浮空障碍三维随机姿态；MuJoCo 的 cone 显示为分面网格。当前默认与保留模型的碰撞半径均为 **0.055 m**，按每个物理子步的质心球距离判断。USD 审计识别到唯一启用的 Cube collider，任务 1.5 倍缩放后尺寸为 **0.15×0.15×0.045 m**，水平角点半径约 0.1061 m；参考以 PhysX contact force>0.01 N 判定碰撞。对照 [`training-clearance.yaml`](../config/training-clearance.yaml) 的 **0.09 m** 球仍不能包住水平角点，又大于盒子的垂直半厚度，不是精确碰撞体对齐；本次 C 对照未带来 Isaac 开发收益，也未被交付选用。运行时 contact offset 未测量，不能仅用盒子尺寸推算实际接触阈值。

解析 ESDF 的最近表面与参考对 200000 个采样表面点的 KD-tree 查询也不会逐数相同。物理与碰撞审计见 [`environment-parity.json`](../reports/training-study/environment-parity.json)。这些差异需要独立 Isaac sim2sim 验证，不应把 Crazyflow 成功率当成 Isaac 成功率。

可选 `model_params_path` 读取参数辨识导出的 TOML；默认不用它。加载时通过共享物理校验，重算惯量逆矩阵和转速限值，不自动改变参考 PD 控制器。导出策略的 `effective_config()` 无论使用官方默认值还是辨识文件，都会把整份实际物理参数嵌入 `model_params` 并清空路径；完整 checkpoint 的 runtime 同样冻结这些值。因此移动或修改原 TOML、更新本地官方参数表都不会悄悄改变已训练模型的评测物理。数值验收比较默认环境与导出配置重建的环境，在零推力、悬停和饱和动作下推进 100 个原生物理子步，要求完整 SimData 逐位相同。训练元数据另记录源码 SHA256，供版本核验。
