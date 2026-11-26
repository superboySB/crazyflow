# cf21B_500：从测量到 BPTT 与实机控制的一份手册

默认仿真直接运行 `./dani train`，不需要实飞日志或先做辨识。增加 RGB 载荷、更换电机/桨后，才按最终装机状态逐项更新。本手册区分**飞机物理、机载控制器、相机/时序和实机执行接口**；一个拟合文件不能同时校准全部环节。

## 1. 先明确推理时到底需要什么

`crazyflow.balloon.policy.Controller` 按 50 Hz 调用，开始新飞行先 `reset()`：

- 输入 `bbox_xyxy=[left,top,right,bottom]`，单位是校正后图像的像素；失去跟踪传 `None`。图像尺寸、裁剪和 K 必须与策略一致。
- 输入 `quaternion_xyzw`，表示机体 FLU（前/左/上）到世界 ENU 的姿态，以及 `angular_velocity_body=[wx,wy,wz]`，单位 rad/s。姿态包含 yaw；不需要位置、线速度、目标三维位置或 RPM。
- 输出 `[roll_des,pitch_des,yaw_des,T_N]`：前三项是 rad，最后是整机总推力 N。**yaw 是角度，不是 yaw-rate。** 默认姿态限幅来自 `task.max_tilt/max_yaw`。

适配器负责传感器坐标转换、采集/执行时间对齐，以及 N→真实固件驱动/目标 RPM 的校准。飞控需匹配绝对 yaw 角模式与实际姿态环频率（仿真为 500 Hz）。姿态和输出 yaw 必须共用航向参考；若使用起飞局部航向，需同时转换输入姿态和输出 yaw。`reset()` 只清历史，不校正航向；六轴 IMU 不能长期校正 yaw 漂移。

策略内部由姿态、gyro、最近 bbox/动作构造因果历史特征。BPTT 的位置、速度、四路 RPM 和控制器积分/上一拍 gyro 状态由仿真推进；它们不是策略额外输入。下文的“训练日志”指**系统辨识的估计数据**，不是 BPTT 要读取的真实数据集。

## 2. 全部参数放在哪里、默认从哪里来

日常编辑四份配置：V = [vehicle.toml](../config/vehicle.toml)，C = [controller.toml](../config/controller.toml)，K = [camera.json](../config/camera.json)，Y = [training.yaml](../config/training.yaml)。V 的 `[cf21B_500]` 有 **12 项物理字段**；C 的 `[attitude]` / `[force_torque]` 有 **11 + 6 项生效控制参数**。

V 来自当前官方 `Dynamics.first_principles / Drone.cf21B_500` 加全局重力，基准表在 `config/dynamics/first_principles/params.toml` 和 `config/dynamics/params.toml`；`J_inv` 由 J 计算。C 来自当前 `attitude2force_torque / force_torque2rotor_vel` 的官方参数加载器，基准在 `config/control/mellinger/params.toml`。位置环的其它参数不参与当前姿态指令路径。默认来源可由[当前加载与控制代码](../crazyflow/balloon/model.py)核对。

下表逐项覆盖当前配置。控制器增益沿用官方固件数值尺度，不能把 `kR/kw/ki_m/kd_omega` 当作未经换算的 SI 力矩增益；相同字段在 V/C 中也分别保存，修改物理表不会自动改控制器。

| 字段 | 当前默认 | 如何测量或核对、当前工具覆盖 | 填写位置 |
| --- | --- | --- | --- |
| `vehicle.gravity_vec` | `[0.0,0.0,-9.81]` | 世界 ENU 的重力方向与单位；通常保留，非本工具拟合项。 | V；通常保留默认，若改须确认世界坐标/重力约定 |
| `vehicle.mass` | `0.0434` | 最终电池、相机、支架、桨全部装好后称重，kg。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `vehicle.J` | `[[2.5e-05,0.0,0.0],[0.0,2.8e-05,0.0],[0.0,0.0,4.9e-05]]` | 质心处惯量，kg·m²；双线摆/可信 CAD，`inertia` 仅计算悬挂轴惯量。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `vehicle.thrust_min` | `0.02136263065537499` | 单电机可持续稳定推力下限，N；核对实际电机/桨/供电与飞控停转阈值。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `vehicle.thrust_max` | `0.2` | 单电机推力上限，N；需覆盖工作电压/RPM 的受控测量，不能从悬停数据证明最大值。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `vehicle.L` | `0.035355` | 量电机相对质心的 X 构型轴向力臂，m；不是斜向半径。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `vehicle.prop_inertia` | `3.893e-08` | 电机转子与桨绕转轴的惯量，kg·m²；单独测量/可信 CAD，当前不自动辨识。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `vehicle.rpm2thrust` | `[0.0,-3.133427287299859e-07,4.407354891648379e-10]` | 常数/一次/二次项，输入机械 RPM、输出单电机 N；`flight` 或 `static` 拟合公共曲线。 | 由拟合导出 `physical.toml` → `--model-params`；不填 `measurements.toml` |
| `vehicle.rpm2torque` | `[0.0,1.65886356219615e-09,2.4693477924534137e-12]` | 同序二次曲线，输出反扭矩大小 N·m；已测 J 的 `flight` 或有扭矩数据的 `static`。 | 由拟合导出 `physical.toml` → `--model-params`；不填 `measurements.toml` |
| `vehicle.rotor_dyn_coef` | `[13.996001897562685,0.00011093207920685363,5.933168530682111,0.00031951312393561264]` | 升速一次/二次、降速一次/二次；用已校准目标/实际机械 RPM 的 `rotor` 拟合。 | 由拟合导出 `physical.toml` → `--model-params`；不填 `measurements.toml` |
| `vehicle.mixing_matrix` | `[[-1.0,-1.0,1.0,1.0],[-1.0,1.0,1.0,-1.0],[-1.0,1.0,-1.0,1.0]]` | 逐路核对电机编号、位置、旋向与力矩符号；由几何/接线确定，不由当前工具自动识别。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `vehicle.drag_matrix` | `[[-0.021643637770852733,0.0,0.0],[0.0,-0.021643637770852733,0.0],[0.0,0.0,-0.02477138502714662]]` | 机体系线性气动力，N·s/m；由参考平移和已校准推力另行验证，当前不自动导出 FP 阻力。 | 人工测量填 `measurements.toml`，合并导出 `physical.toml`；或直接编辑 V |
| `attitude.L` | `0.035355` | 核对控制器采用的轴向力臂与真实几何。 | C 的 `[attitude]` |
| `attitude.thrust2torque` | `0.00593893393599368` | 反扭矩/推力换算，单位 m；核对工作区间 Q/F 与固件采用的近似。 | C 的 `[attitude]` |
| `attitude.mixing_matrix` | `[[-1.0,-1.0,1.0,1.0],[-1.0,1.0,1.0,-1.0],[-1.0,1.0,-1.0,1.0]]` | 核对固件电机编号、旋向和坐标符号。 | C 的 `[attitude]` |
| `attitude.pwm_max` | `65535.0` | 固件 PWM/驱动数值尺度，非转速测量。 | C 的 `[attitude]` |
| `attitude.thrust_max` | `0.2` | 核对单电机推力与 PWM 换算/饱和上限，N。 | C 的 `[attitude]` |
| `attitude.torque_pwm_max` | `[32000.0,32000.0,32000.0]` | 三个轴的力矩指令数值上限；从固件与饱和日志核对。 | C 的 `[attitude]` |
| `attitude.kR` | `[70000.0,70000.0,60000.0]` | 姿态误差反馈增益；读取实际固件，再用独立姿态响应检查。 | C 的 `[attitude]` |
| `attitude.kw` | `[20000.0,20000.0,12000.0]` | 机体角速度误差反馈增益；读取固件并检查 gyro/响应。 | C 的 `[attitude]` |
| `attitude.ki_m` | `[0.0,0.0,500.0]` | 姿态误差积分增益；读取固件并检查稳态偏差/积分行为。 | C 的 `[attitude]` |
| `attitude.kd_omega` | `[200.0,200.0,0.0]` | 角速度误差微分增益；读取固件并检查采样率/滤波。 | C 的 `[attitude]` |
| `attitude.int_err_max` | `[1.0,1.0,1500.0]` | 积分状态限幅；读取固件并检查饱和，不由物理曲线拟合。 | C 的 `[attitude]` |
| `force_torque.L` | `0.035355` | 核对控制器采用的轴向力臂与真实几何。 | C 的 `[force_torque]` |
| `force_torque.rpm2thrust` | `[0.0,-3.133427287299859e-07,4.407354891648379e-10]` | 核对分配器使用的曲线；物理拟合不会自动同步它。 | C 的 `[force_torque]` |
| `force_torque.thrust2torque` | `0.00593893393599368` | 反扭矩/推力换算，单位 m；核对工作区间 Q/F 与固件采用的近似。 | C 的 `[force_torque]` |
| `force_torque.mixing_matrix` | `[[-1.0,-1.0,1.0,1.0],[-1.0,1.0,1.0,-1.0],[-1.0,1.0,-1.0,1.0]]` | 核对固件电机编号、旋向和坐标符号。 | C 的 `[force_torque]` |
| `force_torque.thrust_min` | `0.02136263065537499` | 核对分配器停转/最小单电机推力阈值，N。 | C 的 `[force_torque]` |
| `force_torque.thrust_max` | `0.2` | 核对单电机推力与 PWM 换算/饱和上限，N。 | C 的 `[force_torque]` |
| `camera.image_width` | `640` | tracker 实际处理的去畸变/裁剪/缩放后图像宽，px。 | K |
| `camera.image_height` | `480` | 同一图像的高，px。 | K |
| `camera.hfov_deg` | `120.0` | 水平视场角，度；仅未提供 K 时使用，120° 是当前假设。 | K |
| `camera.camera_intrinsics` | `null` | 实测校正图像的 `[fx,fy,cx,cy]`，px；非 null 时优先于 HFOV。 | K |
| `camera.camera_offset` | `[0.03,0.0,0.0]` | 光心相对新质心的 `[前,左,上]`，m；装载后重测。 | K |
| `camera.camera_rpy` | `[0.0,0.0,0.0]` | 光轴安装 `[roll,pitch,yaw]`，rad，Rz·Ry·Rx；零角朝机体前方。 | K |
| `task.dt / substeps / steps` | `[0.02,10,150]` | 测策略与飞控实际时序；默认 50 Hz 策略、500 Hz 物理/控制、3 s 展开。 | Y 的对应节 |
| `task.max_impact_speed / min_altitude / max_range / max_unseen_seconds` | `[2.5,0.12,12.0,0.5]` | 分别为撞击速度偏好 m/s（不作为接触成功门槛）、最低高度 m、活动范围 m、连续失视终止时长 s。 | Y 的 `task` |
| `task.history / hidden` | `[5,64]` | 历史长度/网络宽度是设计量，不是机体辨识结果。 | Y 的对应节 |
| `task.max_tilt / max_yaw` | `[0.5,0.8]` | 目标姿态限幅，rad；核对实机可执行姿态范围与 yaw 模式。 | Y 的对应节 |
| `task.drone_radius / collision_offset` | `[0.086,[0,0,0.02]]` | 量包含新增载荷的接触包络；默认是官方碰撞球半径/机体系球心，m。 | Y 的对应节 |
| `task.radius_min/max / distance_min/max` | `[0.13,0.18,1.5,3.0]` | 气球尺寸和初始前距，m；按真实任务选择，不由飞行动力学拟合。 | Y 的对应节 |
| `noise.bbox_noise_std / dropout_prob` | `[0.006,0.015]` | tracker 与参考框残差、视觉漏检统计；漏检产生无效框。 | Y 的对应节 |
| `noise.gyro_noise_std / gyro_bias_std` | `[0.015,0.005]` | 静态/重复日志估计 gyro 白噪声和偏置，rad/s。 | Y 的对应节 |
| `noise.attitude_noise_std` | `0.005` | 姿态估计与独立参考的小旋转误差，rad。 | Y 的对应节 |
| `noise.max_delay_steps / action_delay_steps` | `[2,1]` | 旧整数 FIFO 的最大视觉/动作等待拍数；仅 `hardware.enabled=false` 时使用。 | Y 的对应节 |
| `noise.mass_fraction / thrust_fraction / response_fraction` | `[0.05,0.08,0.1]` | 不同装载/电压/重复测量的相对变化；当前 ±5%/±8%/±10% 是假设。 | Y 的对应节 |
| `noise.wind_std` | `0.12` | 未建模外扰加速度幅度，m/s²；按独立响应统计，不能当作已测风速。 | Y 的对应节 |
| `noise.hardware` | 开启新时序；gyro乘性std为 `0.1`，其余新增未知幅度为 `0` | 视觉逐包等待、IMU/动作独立等待与保持、相关噪声和物理不确定性；作为可直接训练的先验，不冒充实测。 | Y 的 `noise.hardware` |
| `training.* / objective.*` | 见 Y 的完整默认值 | 优化次数、批量、学习率和损失权重是训练设计，完整默认值见 Y；无需伪造测量。 | Y 的对应节 |

120° 是**校正针孔图像**的水平 FOV 假设，不是未经校正广角原图的模型。应先测镜头畸变并校正，在校正图上跟踪 bbox，再提供对应 K；不能把原始广角/DoubleSphere 图像直接当作针孔输入。镜头型号或 OV5647 传感器名称不能替代标定。噪声表中的默认值也不是本机实测置信区间。

### 2.1 测量噪声与链路时序

编辑 Y 的 `noise.hardware`；[完整字段与单位](../config/README.md#硬件链路与额外扰动)按实际实现说明。新 YAML 开启该管线，视觉等待 `[0,40]` ms、指令等待 `[0,20]` ms 沿用本项目既有先验；gyro乘性标准差按用户选择设为 `0.1`，直接采用 CrazyRL 的训练范围。**这些是可直接训练的开发先验，无需等待辨识，也不是 cf21B 实测结论**；其余新增未知幅度仍为 0。旧的兼容 BPTT checkpoint 若没有本节，则继续使用旧整数 FIFO；已有该节的策略也保持其冻结值，不会读取本地新 YAML 改变评测，要使用新设置需重新训练。

先记录同一时钟下的图像捕获、检测完成、收包、指令发送与实际执行时间，保留包序号和丢失记录。视觉逐包采等待，50 Hz 接收端只采用已经到达的最新捕获包；传输丢包保持旧包，年龄从捕获时刻算，超过 `camera_stale_ms` 后失效。视觉漏检则传入无效框，两者分别统计。`camera_stale_ms` 至少覆盖最大链路等待。

gyro 与姿态在 500 Hz 分别采样，各自的等待在每回合采一次并取最近 2 ms，丢包保持原测量；噪声在捕获时加入，不对同一旧包反复重采。用静止和重复运动日志估计 gyro 白噪声、回合零偏及慢漂移；姿态误差与独立参考比较，并以 SO(3) 小旋转向量表示。标准差应对应实际采样率和滤波。`gyro_scale_std: 0.1` 使每个 500 Hz 样本、每轴独立乘 `N(1,0.1²)`；0.1是无量纲标准差，不是 ±10% 硬截断，原有加性白噪声和回合零偏保持不变。该设置来自 CrazyRL 的输入噪声定义，作为用户选定先验使用，本机尚未标定；以后若校准比例误差，需多档非零角速度参考，不能只靠静止日志。

对相关 gyro 漂移、姿态偏置、链路抖动和阵风，从相关性估计时间常数，从长期波动估计稳态标准差；它们使用 OU 过程，std 不是谱密度。RPY 各轴和总推力的有效等待分别采样，指令抖动每 50 Hz 更新、500 Hz 执行；按实际同包/分通道执行方式核对这些假设。链路等待与官方控制器、电机升降速造成的动态响应分别处理，不能把同一硬件滞后重复计入二者。

新增 J/阻力和逐电机倍率描述围绕可信物理表的不确定性，不替代质量、惯量和曲线测量：J 整体缩放并同步求逆，阻力乘正倍率保持耗散符号，逐电机曲线与升降速倍率按重复试验差异填写。`gust_std` 是加速度 m/s²，叠加回合固定外扰并按实际质量换算外力；它不是直接测得的风速。物理基准和控制器仍来自 V/C。

相同硬件管线用于 Crazyflow BPTT 与 native MuJoCo 评测。队列、保持值和物理状态都参与 BPTT；延迟取整、到达选择和随机丢包等离散采样事件本身不求导。时间戳、延迟值和真值状态仅供仿真诊断，actor 仍只有实收 bbox、姿态、gyro 和已发动作历史；开启噪声不会引入位置或线速度输入。

## 3. 一份日志够不够，为什么有 train/valid

**一份包含所需观测和充分激励的日志可以用于参数拟合；它本身不提供独立验证，也不能识别上表所有量。** 当前官方 `sys_id_translation`、`sys_id_rotation` 的 `data_validation=None`，低层 `crazyflow.identify fit --validation` 可省略。验证数据只在得到拟合参数之后计算预测误差/作图，不参加参数优化。`identify_physics` 的各 `--*-validation` 也可省略。

`./dani identify flight/rotor/static/drag` 要求两份文件，是本项目为保留独立检验而采用的**入口工程选择，不是官方算法强制**。推荐另一次飞行作为 valid；复制 train 或换个文件名没有验证意义。程序检查路径和数据格式，不能自动证明采集独立。如果目前只有一段，也可通过低层接口先拟合，明确记录“没有独立验证”，之后再补采。

下面始终区分两套数据：物理曲线日志是机械 RPM/IMU/台架量；闭环诊断日志是位置/姿态/指令。二者的 train/valid 都是辨识用文件，默认 BPTT 不读取它们。

## 4. 按顺序测量并导出第一性原理物理表

在项目根目录、完成 `./dani setup` 后执行。所有辨识在容器 CPU 运行，不连接飞机或自动采集数据；默认目录 `outputs/measurements`，可用 `--directory outputs/另一目录` 指定，文件必须位于项目 outputs 内。

### 4.1 建模板，测装机与 IMU/相机

```bash
./dani identify init
```

生成未确认的 `measurements.toml`、官方 `controller.toml` 模板，以及 `flight_train/valid.csv`、`rotor_train/valid.csv`、`static_train/valid.csv` 空表头；重复执行保留已有文件。

固定最终电池、桨、相机、支架、护桨、动捕标记与固件。称质量、测质心和电机力臂，取得质心处 J，逐项填入 `measurements.toml` 的 `[measurements]`。只有核实真实测量后才设 `confirmed_measurements=true`；未测字段继续作为继承假设，不冒充测量。

双线摆可计算某悬挂轴惯量：`J_suspended=m*g*d²*T²/(16π²*L)`，m 为全部悬挂质量，d 为两线**全间距**，L 为单线长度，T 为多周期小角度平均周期。以下数字仅演示 kg/m/s 单位：

```bash
./dani identify inertia \
  --suspended-mass 0.0434 --spacing 0.08 --length 0.4 --period 1.0
```

用 `--fixture-inertia` 扣同轴夹具惯量；有偏心时用 `--offset`、`--drone-mass` 扣 `m_drone*offset²`。分别测三个轴；显著非对称载荷还需更多轴向测量/可信 CAD 得到非对角项。J 必须对称正定，不能按新质量比例直接缩放旧 J。

同时标定 IMU 轴向、偏置、时间及传感器到质心的杆臂 r，并将比力修正到质心：

`f_COM = f_IMU - alpha×r - omega×(omega×r)`。

omega/alpha 是机体 gyro 与滤波求导后的角加速度，分别为 rad/s、rad/s²；欧拉角导数不能替代 gyro。比力单位 m/s²，若原日志用 g 先乘 9.81，水平静止时向上约 +9.81。增加偏心载荷后不能默认为 r=0。另按上表测相机 K/安装姿态、图像尺寸与新质心到光心位置，填 K。

### 4.2 无推力台：采实际 RPM 与质心 IMU

保留原始时间戳、四路有效机械 RPM、电池电压、姿态、gyro/IMU、发送与执行指令、飞控/ESC/磁极数配置。PWM/归一化驱动、电气 eRPM 和机械 RPM 不能改列名互换；转速遥测中的 `65535` 是无效值。没有可信实际 RPM 时，当前工具不能仅凭姿态与角速度识别全物理表，应先保留默认值。

采近水平、低空速的升降/总推力变化和正反 yaw 激励，另采验证段。分别整理 `flight_train.csv`、`flight_valid.csv`：

```text
time,rpm1,rpm2,rpm3,rpm4,specific_force_z,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z
```

`time` 为递增 s，RPM 为机械转/分，`specific_force_z` 是质心处 body-FLU 比力。确认所选片段可忽略阻力、其它外力和气动偏航力矩后，显式执行：

```bash
./dani identify flight --negligible-drag
```

工具用 `m*f_COM,z=ΣF_i` 拟合公共单电机推力曲线，并用已测 `J*alpha+omega×(J*omega)`、扣转子角动量导数后拟合公共 yaw 反扭矩曲线。后者依赖 `prop_inertia`；共享反扭矩常数项不可观测，固定为零并记录。持续悬停/单一推力差可能欠激励，四电机差异大时公共曲线不能代替各路模型。

输出 `physical.toml` / `physical.json`。JSON 区分测量、拟合、继承项，报告训练及留出推力/扭矩误差；检查 RPM 范围、条件数和重复采集的一致性。不能把默认继承的 J、转子惯量、阻力或推力上限称作已辨识。

### 4.3 补旋翼升降速

先测不同电压下的稳定驱动→实际 RPM 映射，得到可信目标机械 RPM，再同步采瞬态并整理 `rotor_train.csv` / `rotor_valid.csv`：

```text
time,motor,target_rpm,actual_rpm
```

motor 为 1..4。按左端目标/实际 RPM 拟合 `dn/dt=c1*(n_target-n)+c2*(n_target²-n²)`，升降速各两项：

```bash
./dani identify rotor --negligible-drag
```

它联合重读 flight 与 rotor 日志后导出 `physical.toml`，不是把未知的上次结果自动补齐。检查验证的一步 RPM 误差。快速电机瞬态可能需要 500 Hz 或更高可信采样；100–200 Hz 姿态/动捕日志只是同步检查起点，不能用插值增加带宽。发送时间和执行时间需分别保留，避免把通信延迟拟进电机动态。

#### （可选）推力/扭矩台交叉检查

有台架时另采 `static_train.csv` / `static_valid.csv`，每行单个电机的稳定值：

```text
motor,rpm,thrust_n,torque_nm
```

推力为单电机 N、反扭矩大小为 N·m；无扭矩测量则删掉该列，不填假值。覆盖各电机的实际 RPM/电压范围并重复测量：

```bash
./dani identify static
```

单独输出 `static-check.toml` / `.json`，不覆盖 flight/rotor 的 `physical.toml`。它可以交叉检查曲线，不能凭悬停附近样本证明最大推力。

### 4.4 补未自动辨识的量，再接入训练

第一性原理 `drag_matrix` 目前没有一键辨识器。可用同步参考位姿平滑得到质心处 `v_world/a_world`，结合可信姿态和实际 RPM 推力曲线，在无风/已知风条件下检查：

`m*Rᵀ*(a_world-g)-[0,0,ΣF] = D*(Rᵀ*v_air_world)`。

用不同方向/速度及留出飞行验证 D 的耗散符号与预测误差。没有可信速度/风参考时，它仍是继承假设；参考位置/速度只用于离线测量。

再核对控制器 C 的全部字段与真实固件、执行器饱和和姿态阶跃响应。当前工具不自动调姿态增益、不测全部 J、不自动辨识转子惯量/风，也不校准 N→固件驱动。换电机后的曲线、Q/F、力臂和混控应分别核对 V 与 C，不能只改物理质量。

保留原配置，优先用显式覆盖文件训练：

```bash
./dani train --model-params outputs/measurements/physical.toml
# 已另行核对实机控制器后，才使用控制器覆盖：
./dani train --model-params outputs/measurements/physical.toml \
  --controller-params outputs/measurements/controller.toml
```

两条是替代选择，无需先后各训练一次。也可将已验证物理值填入 V、控制值填入 C，相机值填入 K，时序/噪声填入 Y，再运行 `./dani train`。启动器把四份配置复制到新训练目录 `config/`，完整生效值冻结进 checkpoint；修改本地文件不会改变旧策略。

## 5. 闭环阻力/姿态诊断：另一套日志和另一种输出

不具备机械 RPM 时，可用**离线参考位置、姿态、姿态角/总推力指令与实测质量**拟合 `so_rpy_rotor_drag` 简化闭环模型。它拟合总推力比例、总推力一阶响应、两项机体系线性阻力，以及二阶姿态响应；不测电机 RPM 曲线或 Mellinger 增益。

```bash
./dani identify drag-init
```

生成 `closed_loop_train.csv` / `closed_loop_valid.csv` 空表头。用动捕或其它验证过的参考系统采集三维位置和姿态，统一 ENU/FLU 与初始航向，记录实际姿态角 rad 和总推力 N，整理为：

```text
time,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,cmd_roll,cmd_pitch,cmd_yaw,cmd_f
```

位置 m，四元数为机体到世界的 xyzw；不要求直接提供速度/加速度，工具对位置离线滤波求导。NPZ 等价字段为 `time[N]`、`pos[N,3]`、`quat[N,4]`、`cmd_rpy[N,3]`、`cmd_f[N]`。姿态+gyro 不能代替参考位置，PWM 或 yaw-rate 也不能代替这里的总推力/角度。

从接近零姿态、零角速度、稳定悬停的片段开始；预处理将最初约 0.1 s 的姿态归零，却不会同时旋转位置与指令，不能靠它补救坐标错误。采多方向、正反向、不同速度平移和可分辨总推力滞后的激励，另留验证飞行，避免未知风/接触力。最低加载条件为 100 点、至少 2 s、平均采样率大于 20 Hz，间隙不超过 3 倍中位周期；满足门槛不等于充分激励。平移/姿态使用 6/8 Hz SVF，需检查原始离群点警告与滤波偏差。

把下例 `0.0434` 换成真实称量的 kg 值：

```bash
./dani identify drag --mass 0.0434
# 其它目录或NPZ可显式提供：
./dani identify drag --mass 0.0434 --directory outputs/my-aircraft \
  --data outputs/my-aircraft/closed_loop_train.npz \
  --validation outputs/my-aircraft/closed_loop_valid.npz
```

两种写法选符合实际文件的一条。输出 `closed-loop-so-rpy-rotor-drag.toml`、同名 JSON 和 `fit.log`。检查两部分优化收敛、留出 RMSE/R²、重复飞行稳定性与阻力符号。D=`diag(d_xy,d_xy,d_z)`，`F_drag_world=R*D*Rᵀ*v_world`，单位 N·s/m，代码用加号所以耗散系数应非正；当前优化未约束其非正，正值不能取绝对值后直接使用。推力滞后是单位 N 的总推力状态，可能吸收执行/通信误差，不是四电机 RPM 动态。

该闭环 TOML **不能传给 `--model-params`，也不能只抄其同名 `drag_matrix` 到 V**。它把控制、延迟与气动合并在简化参数中，直接用于显式含这些环节的第一性原理模型可能重复补偿。它保留为离线诊断/对照，V 的阻力仍需 4.4 的物理验证。

只用一份日志先估计时，可跳过工程包装器，使用同一 Docker 中的低层可选验证接口；以下质量仍需替换为实测值：

```bash
docker run --rm --init --user "$(id -u):$(id -g)" -e JAX_PLATFORMS=cpu \
  -v "$PWD/outputs:/outputs" crazyflow:dani \
  python -m crazyflow.identify fit --dynamics so_rpy_rotor_drag --mass 0.0434 \
  --data /outputs/measurements/closed_loop_train.csv \
  --output /outputs/measurements/closed-loop-one-log.toml
```

这是单日志拟合，不能报告独立验证成绩。低层进程退出 0 不保证优化日志中的 `Training success=True`；`./dani identify drag` 会拒绝未收敛结果。平移求解采用三轴残差展平，平方和目标与逐样本残差范数相同，避免零范数处的 Jacobian 数值问题；没有修改默认物理参数或优化容差。

## 6. 没有硬件：运行两套明确标记的合成例子

```bash
./dani identify check
./dani identify drag-check
```

两者各自在 `outputs/measurements/` 创建唯一子目录，不覆盖实测文件。`check` 生成机械 RPM/IMU、旋翼瞬态与可选台架合成表，检查第一性原理曲线/升降速导出；`drag-check` 用官方 `so_rpy_rotor_drag` 前向模型的已知非零阻力/滞后，生成 15 s、100 Hz、seed 0/1 的两份位姿/指令日志，再拟合与验证。

这两套中的 train/valid 都是**合成估计/验证数据，不是真实实飞，更不是 BPTT 的数据集**。seed 1 改变激励相位，用来检查 seed 0 得到的参数在另一激励下的预测；仍是同一已知模型与参数，不能称为跨物理或硬件迁移验证。CSV 来源在同名 JSON 的 `synthetic`、`known_parameters` 等字段中，NPZ 还内嵌标记；物理清单保留未确认实测的标记，仅合成路径允许导出。

[示例目录](../examples/identification/README.md)保留两份闭环 CSV 和来源 JSON，可直接查看列名、单位与真值。默认的低层 `identify synthetic` 仍是无阻力/滞后的 `so_rpy` 数据，不能冒充这份匹配例子。没有任何真实测量时，应保留官方默认来源，不能把合成例子的质量、J 或曲线称为实测。

[官方系统辨识 README 原文](crazyflow_experiments_sysid/README.md)逐字保留固定版本的上游文档，来源版本和 SHA256 见同目录的 [provenance.json](crazyflow_experiments_sysid/provenance.json)。其中的相对路径、安装说明和命令属于上游仓库的独立 ROS2/Pixi 环境，不是本项目 Docker 的直接命令；本项目入口以 `./dani identify --help` 为准。上游 `lol_sysid.py` 的变量 `rpms` 实际来自归一化电机指令，且含 Jzz 比例假设，不能直接改名当作机械 RPM 或独立惯量测量。本手册按当前本地配置、加载函数和辨识函数说明。

## 本地技术资料

- [Crazyflie/Crazyflow：BPTT 关键问题与辨识方案](Crazyflie与Crazyflow_BPTT关键问题与辨识方案.pdf)
- [视觉可微分仿真与实机控制](苏黎世穿缝_可微分仿真与实机控制.pdf)

资料保留原文，当前的文件路径、相机假设和启动方式以本手册及 `config/` 为准。
