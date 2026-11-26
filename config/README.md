# 配置入口与测量覆盖

常用配置分为以下四份。修改后运行 `./dani train`；启动器会把输入复制到新结果目录的 `config/`，训练将完整有效配置存入 checkpoint。评测和 GUI 继续读取该策略冻结的参数，不会临时套用后来修改的文件。

| 文件 | 改什么 | 当前默认来源 |
| --- | --- | --- |
| [training.yaml](training.yaml) | 训练规模、种子、时长/频率、网络、气球与接触几何、损失、噪声和随机化 | 本任务的 BPTT 配置；噪声是开发假设，需按实测调整 |
| [vehicle.toml](vehicle.toml) | 第一性原理质量、惯量、力臂、推力/反扭矩曲线、机械 RPM 动态及气动力 | 官方 `cf21B_500` 完整物理默认表，未针对新增载荷重新辨识 |
| [controller.toml](controller.toml) | 官方 Mellinger 姿态/角速度反馈与总推力/力矩到目标 RPM 的分配 | 官方姿态与分配完整默认参数，未自动调增益 |
| [camera.json](camera.json) | 图像尺寸、水平 FOV 或标定 K、相机光心与安装方向 | 640×480、水平 120°、前移 0.03 m 的针孔假设；可替换实测标定 |

四份文件保留物理模型、机载控制器、视觉测量和优化任务各自的含义。例如，改了载荷质量，不会自动重新估计惯量、调整控制增益或标定镜头。

## 1. 训练、任务、损失与噪声

编辑 [training.yaml](training.yaml)，或另存文件后用 `./dani train --settings 文件路径`。完整取值见 YAML 注释：

| 位置 | 作用与单位 |
| --- | --- |
| `training.iterations`、`batch_size.gpu/cpu`、`seed`、`lr` | 优化次数、并行环境数、种子、学习率；启动时按实际设备选并行数 |
| `task.dt`、`substeps`、`steps` | 策略周期 s、每周期物理子步、展开长度；默认 50 Hz 策略、500 Hz 物理/控制器、3 s 飞行 |
| `task.history`、`hidden` | 观测/动作历史长度与网络隐藏宽度 |
| `task.target_diameter_m`、`target_scale_range` | 标称直径 m 与无量纲倍率；默认 0.3 m × log-uniform[0.1,2]，实际直径 3–60 cm |
| `task.target_speed_min/max`、`target_stationary_prob` | 非静止子集的速度模长范围 m/s、静止比例；默认 20% 静止，其余 U[0,1.5] m/s，方向在三维球面均匀采样 |
| `task.target_spawn_margin`、`distance_min/max` | 球面离最低高度的出生裕量 m、初始向前距离 m；前距不是完整三维距离 |
| `task.drone_radius`、`collision_offset` | 接触球半径、机体系球心偏置，m；与官方碰撞球对齐 |
| `task.max_impact_speed`、`min_altitude`、`max_range`、`max_unseen_seconds` | 撞击速度训练偏好 m/s、最低高度 m、活动范围 m、连续失视缓冲 s；首次实际触球就是成功，速度仅作诊断 |
| `task.max_tilt`、`max_yaw` | 动作对应的目标姿态上限，rad |
| `objective.loss_weights` | 距离、接触、速度、过快撞击、FOV、地面、倾角、角速度、动作幅度和平滑性权重 |
| `objective` 中的温度与软阈值 | 可微最小距离/接触/撞击加权温度、地面余量 m、倾角软限 rad；改变优化目标，不改变触球成功定义 |
| `noise.bbox_noise_std`、`dropout_prob` | 归一化 bbox 坐标白噪声、视觉漏检概率；漏检产生无效框 |
| `noise.attitude_noise_std`、`noise.gyro_noise_std`、`noise.gyro_bias_std` | 姿态旋转向量噪声 rad、机体系 gyro 噪声/偏置 rad/s |
| `noise.mass_fraction`、`thrust_fraction`、`response_fraction`、`wind_std` | 质量/推力/旋翼响应相对变化、每回合固定世界系外扰加速度幅度 m/s² |
| `noise.hardware` | 逐包视觉等待、独立 IMU/指令等待与丢包、相关偏置/阵风和额外物理不确定性，见下表 |
| `noise.max_delay_steps`、`action_delay_steps` | 旧整数 FIFO 的视觉/动作最大延迟拍数；仅 `hardware.enabled: false` 时使用，一拍为 `task.dt` |

命令行 `--iterations`、`--batch-size`、`--steps`、`--seed`、`--lr` 优先于 YAML。位置、线速度和目标真值只用于仿真、损失与评测；增加噪声不会把这些量送入策略。

目标在整段飞行中使用同一个 `p(t)=p0+v*t`，没有碰壁反弹或逐拍改方向。为避免向下运动的球在展开末端进入地下，出生时按整段最低球面向上平移飞机和气球，保持初始相对几何；因此需要与设定速度/时长相适应的飞行空间。静态 checkpoint 未提供速度字段时继续按零速度解释。旧自定义文件仍可用米制 `radius_min/max`；新文件优先使用直径与倍率这组配置，避免把倍率误当半径。

### 自主飞行时提供哪些输入

默认 `history: 5` 时 actor 共 65 维，历史按从旧到新排列：

| 索引（从 0 开始） | 内容 | 实机来源 |
| --- | --- | --- |
| 0–8 | 机体到起飞局部参考系的旋转矩阵，逐行展开 | 机载估计姿态；输入姿态与输出 yaw 共用该局部参考，不要求动捕或真北航向 |
| 9–11 | 机体系 gyro / 5 rad/s | IMU；训练叠加乘性误差、白噪声和回合零偏 |
| 12–36 | 5 拍 `[cx, cy, width, height, valid]` | RGB tracker 收到的裁剪 bbox，含保持、延迟和丢帧 |
| 37–56 | 前 5 拍归一化四维已发动作 | 控制端历史，不是真实执行动作或电机反馈 |
| 57–59 | 当前 bbox 中心对应的机体系单位视线 | bbox 与相机标定计算 |
| 60 | bbox 光学对角线的对数 | 框宽高与 K 计算，**不是物体真实尺寸** |
| 61 | 历史光学尺度的因果拟合斜率 | 至少 3 个有效历史框 |
| 62–63 | 最近两拍框中心变化率 | 历史框差分 |
| 64 | 最近两拍对数光学尺度变化率 | 历史框差分 |

特征实现见 [observations.py](../crazyflow/balloon/observations.py)，推理接口见 [policy.py](../crazyflow/balloon/policy.py) 的 `Controller.step`。输出 4 维归一化动作映射为目标 roll、pitch、局部 yaw 和总推力 N；yaw 参与控制。没有输入无人机位置/线速度、目标位置/速度、真实半径、距离或实际电机转速。Flow deck 的速度估计无需送给策略；气球 bbox 仍由前置 RGB 相机与 tracker 提供，Flow deck 本身不提供这个框。

机载姿态是有误差的估计，局部 yaw 也会随时间漂移，不能当成长时间无漂移的绝对航向。部署适配层负责保持姿态和指令的局部参考一致；`Controller.reset()` 只清历史，不会重置固件估计器。当前姿态噪声是估计误差近似，没有实现完整机载状态估计器，也不会自动将 gyro 误差积分成相关 yaw 漂移。未知物体大小与单目距离存在歧义，框变化率也不等于真实相对速度。仿真仍必须知道几何尺寸，才能生成 bbox、计算可微接触目标和评测碰撞；这些真值不进入网络。

### 硬件链路与额外扰动

新训练默认使用 `noise.hardware.enabled: true`。视觉等待 `[0,40]` ms、RPY/推力指令等待 `[0,20]` ms 沿用本项目原有等待范围的开发假设，时序改为下面的包处理；不会再串联旧整数 FIFO。`gyro_scale_std: 0.1` 按用户选择直接采用 CrazyRL 的乘性噪声作为训练先验，**无需等待辨识即可训练，也不是 cf21B 实测标定**。其余新增未知噪声、丢包、漂移、阵风和物理随机化幅度仍为 **0**，可按训练假设或实测逐项调整；正的时间常数或超时值本身不会产生非零扰动。

| `noise.hardware` 字段 | 实际含义 |
| --- | --- |
| `camera_delay_ms` | 每个 50 Hz 视觉包在 `[min,max]` ms 连续均匀采等待，50 Hz 轮询后采用最新到达的捕获值；不插值、不按当前姿态重投影 |
| `camera_packet_loss_prob`、`camera_stale_ms` | 传输丢包时保持上一收到的包；从捕获时刻算的年龄超过 stale 后输出无效框，stale 至少覆盖最大等待；这与 `dropout_prob` 的视觉漏检不同 |
| `bbox_bias_std` | 每回合固定的四维框偏置标准差，使用与 bbox 白噪声相同的归一化单位 |
| `gyro_delay_ms`、`attitude_delay_ms`、两个 `*_packet_loss_prob` | gyro 与姿态分别在回合开始采固定等待，按最近 2 ms 取整；500 Hz 采样，丢包各自保持上一测量 |
| `gyro_scale_std` | 默认 `0.1`：每个 500 Hz gyro 样本、每轴独立乘 `N(1,0.1²)`；无量纲，0.1是标准差，不是 ±10% 硬截断。来自 CrazyRL 的训练先验，非本机实测；原有加性白噪声和零偏继续叠加 |
| `gyro_drift_std`、`attitude_bias_std`、`bias_tau_s` | 相关 gyro 漂移 rad/s、SO(3) 姿态偏置 rad 的稳态标准差及相关时间 s；与原有白噪声、回合 gyro 固定零偏叠加 |
| `attitude_command_delay_ms`、`thrust_command_delay_ms` | 每回合、各 RPY 轴与总推力分别采有效链路等待，500 Hz 入队并按最近 2 ms 取整；实际姿态与推力仍由官方控制器和电机状态推进 |
| `command_jitter_std_ms`、`command_jitter_tau_s` | 每次 50 Hz 发动作时更新相关等待抖动；标准差 ms、相关时间 s，叠加基线后裁剪在相应等待范围内，旧包不会覆盖已接收的新包 |
| `inertia_fraction`、`drag_fraction` | 回合内整块 J、整块阻力矩阵的正倍率均匀半宽；同步重算 `J_inv`，保留原阻尼符号，不改变控制器参数 |
| `motor_thrust_fraction`、`motor_response_fraction` | 各电机独立的推力/反扭矩曲线倍率、升降速响应倍率半宽；同一电机两条力/矩曲线共倍率，升速和降速分别采倍率 |
| `gust_std`、`gust_tau_s` | 相关世界系外扰加速度的稳态标准差 m/s²、相关时间 s；叠加原回合固定外扰，以实际质量换算外力，不是风速 |

相关偏置、指令抖动和阵风使用 OU 更新：`ρ=exp(-Δt/τ)`，`x_next=ρ*x+std*sqrt(1-ρ²)*ε`。std 是稳态标准差，不是每秒扩散系数；指令抖动的 std 指裁剪前值。新增物理 fraction 使用 `U[1-f,1+f]`，在 perturbed 套件中半宽放大 1.5 倍；其余硬件噪声不会统一乘 1.5。

队列和保持的测量值保留 BPTT 梯度；整数延迟、随机丢包、到达选择等离散事件本身不求导，时间戳和调试掩码不进入 actor。旧的兼容 BPTT checkpoint 若没有 `hardware`，会继续使用旧整数 FIFO；已有该节的策略也保持冻结的 `gyro_scale_std`。改 YAML 不会改造旧策略，要使用新设置需重新训练；兼容加载用的类默认值仍为 0。`--nominal` 关闭随机化和额外链路等待，官方电机与刚体动态仍然存在。统计与测量方法见[辨识手册](../docs/identification.md#21-测量噪声与链路时序)。

## 2. 机体与机载控制器

`vehicle.toml` 的 `[cf21B_500]` 包含 12 个官方物理字段；`J_inv` 由 J 计算，不单独配置。质量用 kg，J 用 kg·m²，L 用 m，推力边界为**单电机** N。`rpm2thrust`、`rpm2torque` 系数依次为常数、一次、二次项，输入为机械 RPM，不能用 PWM 或电气 eRPM 代替。

`controller.toml` 的 `[attitude]` 保存姿态反馈、角速度反馈、积分/微分及边界；`[force_torque]` 保存分配用的力臂、曲线、混控与推力边界。它们是控制器自身采用的参数，与飞机物理值分别保存；物理拟合不会自动同步控制器。当前没有控制器增益自动辨识，也没有把总推力 N 自动校准成真实固件驱动。

显式替换物理导出表或控制器表：

```bash
./dani train --model-params outputs/measurements/physical.toml
./dani train --controller-params config/controller.toml
```

默认两份表由官方加载函数导出，逐字段值相同。以后填写实测值时保留独立验证与来源，流程见[实物参数辨识](../docs/identification.md)。不能只按质量比例缩放 J，也不能把 `so_rpy` 闭环系数放进第一性原理表。

## 3. 相机内参与外参

| `camera.json` 字段 | 填写方法 |
| --- | --- |
| `image_width` / `image_height` | tracker 输入图像实际宽、高，像素 |
| `hfov_deg` | **水平**视场角，度；默认 120°，不会作为 MuJoCo 竖直 FOV 直接传入 |
| `camera_intrinsics` | `null` 时由 HFOV 计算居中方形像素 K；实测后填 `[fx, fy, cx, cy]`，像素，它优先于 HFOV |
| `camera_offset` | 光心相对模型质心的 `[前, 左, 上]`，m；默认向前 3 cm |
| `camera_rpy` | 安装 `[roll, pitch, yaw]`，rad；旋转顺序 Rz(yaw) Ry(pitch) Rx(roll) |

零安装角时相机朝机体前方，像素向右、向下。默认 640×480、120° HFOV 对应 `fx=fy≈184.752`、`cx=320`、`cy=240`，竖直 FOV 约 104.822°。载荷改变质心后，应按新质心重测外参。

当前采用去畸变针孔图像。120° 镜头原图若有畸变，应先校正，再在校正图上 track bbox，并提供校正后的 K；OV5647 传感器名称不能确定实际镜头、裁切与内参。用 `./dani train --camera 文件路径` 可选另一份标定。Crazyflow 理想 bbox、native MuJoCo 闭环观测、视频和 GUI 使用 checkpoint 冻结的同一标定；调整本地文件后需重新训练。

## 4. 辨识覆盖矩阵

| 量或配置 | 所需测量 | 当前工具覆盖 | 填写位置 |
| --- | --- | --- | --- |
| 质量、质心、力臂、J | 最终装机称重、几何/平衡、摆测或可信 CAD | `inertia` 计算悬挂轴惯量；其它值人工测量与确认，不自动识别全部 J | `measurements.toml` 后导出物理表 |
| 推力与反扭矩曲线 | 可信机械 RPM、质心比力/角加速度、已测质量/J；或台架推力/扭矩 | `flight --negligible-drag` 或 `static` 拟合公共曲线，报告验证与各路残差 | `physical.toml` 的 `rpm2thrust/rpm2torque` |
| 旋翼升降速 | 校准的目标/实际机械 RPM、时间对齐与独立验证 | `rotor --negligible-drag` 拟合升/降速一次、二次系数 | `physical.toml` 的 `rotor_dyn_coef` |
| 第一性原理阻力 | 平移参考、姿态、可信 RPM/推力模型与方向/速度激励 | 当前不自动拟合该物理模型阻力，默认值为继承假设 | `vehicle.toml` 的 `drag_matrix`，需另行物理验证 |
| 简化闭环阻力与姿态响应 | 离线三维位置、姿态、姿态/总推力指令、实测质量 | `drag` 拟合两项机体系阻力、总推力比例/滞后及二阶姿态系数 | 单独闭环 TOML，**不能覆盖第一性原理 vehicle/controller** |
| 控制器增益及 N→实机驱动映射 | 固件配置、响应/饱和测量、独立完整飞行 | 导出官方模板；没有自动调增益或标定驱动 | `controller.toml` 及独立实机执行接口 |
| K、镜头畸变、相机安装 | 实际镜头/分辨率标定、质心到光心测量 | 读取针孔内外参，不自动标定镜头 | `camera.json` |
| 噪声、丢帧、通信延迟 | 静态/动态重复日志、采集与执行端时间戳 | 可直接采用训练先验；gyro乘性std为0.1，其余新增未知幅度为0，均不冒充实测置信区间 | YAML 的 `noise` / `noise.hardware` |
| 目标采样、碰撞几何、损失权重 | 任务需求、实物尺寸、训练/评测结果 | 用户设置，不属于机体辨识 | YAML 的 `task/objective` |

`./dani identify check` 与 `drag-check` 使用明确标记的合成样例；合成验证不能证明实物已辨识。[统一辨识手册](../docs/identification.md)和[示例数据说明](../examples/identification/README.md)解释所需观测与范围。

## 5. 其它外层资源

`config/dynamics/` 与 `config/control/` 保存原样迁出的官方多机型默认参数，是加载器基准；日常本机配置编辑上述四份文件。`assets/scene.xml`、`assets/drones/*.xml` 及网格保存场景与机体视觉资源。当前 BPTT/native 飞行的质量和惯量来自冻结的 `vehicle.toml`，不能用修改可视 XML 惯量替代；任务接触球由 YAML 任务几何配置。

源码 checkout 与 Docker 自动定位本项目外层目录；wheel 安装时资源随包部署到安装前缀的 `share/crazyflow/`，任意工作目录可运行。通常无需资源环境变量；显式 `CRAZYFLOW_RESOURCE_ROOT` 只供完整自定义资源树使用，须同时包含 `assets/` 和 `config/`。
