# cf21B_500 离线参数辨识

默认执行 `./crazyrl train` 使用官方参数，不需要实飞日志。本手册说明测量、拟合与独立验证流程。辨识工具不连接飞机、不采集数据、不自动把拟合结果用于训练。

本任务使用 Crazyflow first_principles 的 RPM 电机和刚体模型，以及参考 CrazyRL 的姿态/角速度 PD。物理参数、控制器、传感器噪声是独立的配置；`config/controller.toml` 为原有 Mellinger 模板，不能作为此任务 PD 的参数文件。质量、惯量、推力曲线、转子响应和阻力的默认来源在 `config/dynamics/first_principles/params.toml`。单位为 SI，机械转速为 RPM，机体轴 FLU、世界轴 ENU，四元数 xyzw。

**一份包含所需观测和充分激励的日志可以用于参数拟合；它本身不提供独立验证，也不能识别本文涉及的全部物理量。** 当前官方 `sys_id_translation`、`sys_id_rotation` 的 `data_validation=None`，低层 `crazyflow.identify fit --validation` 可省略。验证数据只在得到拟合参数之后计算预测误差/作图，不参加参数优化。`identify_physics` 的各 `--*-validation` 也可省略。

`./crazyrl identify flight/rotor/static/drag` 要求两份文件，是本项目为保留独立检验而采用的**入口工程选择，不是官方算法强制**。推荐另一次飞行作为 valid；复制 train 或换个文件名没有验证意义。程序检查路径和数据格式，不能自动证明采集独立。如果目前只有一段，也可通过低层接口先拟合，明确记录“没有独立验证”，之后再补采。

物理曲线日志记录机械 RPM/IMU/台架量；闭环诊断日志记录位置/姿态/指令。二者的 train/valid 都是辨识用文件，默认 PPO 不读取它们。

## 1. 按顺序测量并导出第一性原理物理表

在项目根目录、完成 `./crazyrl setup` 后执行。所有辨识在容器 CPU 运行，不连接飞机或自动采集数据；默认目录 `outputs/measurements`，可用 `--directory outputs/另一目录` 指定，文件必须位于项目 outputs 内。

### 1.1 建模板，测装机与 IMU/相机

```bash
./crazyrl identify init
```

生成未确认的 `measurements.toml`、官方 `controller.toml` 模板，以及 `flight_train/valid.csv`、`rotor_train/valid.csv`、`static_train/valid.csv` 空表头；重复执行保留已有文件。

固定最终电池、桨、相机、支架、护桨、动捕标记与固件。称质量、测质心和电机力臂，取得质心处 J，逐项填入 `measurements.toml` 的 `[measurements]`。只有核实真实测量后才设 `confirmed_measurements=true`；未测字段继续作为继承假设，不冒充测量。

双线摆可计算某悬挂轴惯量：`J_suspended=m*g*d²*T²/(16π²*L)`，m 为全部悬挂质量，d 为两线**全间距**，L 为单线长度，T 为多周期小角度平均周期。以下数字仅演示 kg/m/s 单位：

```bash
./crazyrl identify inertia \
  --suspended-mass 0.0434 --spacing 0.08 --length 0.4 --period 1.0
```

用 `--fixture-inertia` 扣同轴夹具惯量；有偏心时用 `--offset`、`--drone-mass` 扣 `m_drone*offset²`。分别测三个轴；显著非对称载荷还需更多轴向测量/可信 CAD 得到非对角项。J 必须对称正定，不能按新质量比例直接缩放旧 J。

同时标定 IMU 轴向、偏置、时间及传感器到质心的杆臂 r，并将比力修正到质心：

`f_COM = f_IMU - alpha×r - omega×(omega×r)`。

omega/alpha 是机体 gyro 与滤波求导后的角加速度，分别为 rad/s、rad/s²；欧拉角导数不能替代 gyro。比力单位 m/s²，若原日志用 g 先乘 9.81，水平静止时向上约 +9.81。增加偏心载荷后不能默认为 r=0。另外测量相机内参矩阵、安装姿态、图像尺寸及新质心到光心的位移，并单独记录标定结果。

### 1.2 无推力台：采实际 RPM 与质心 IMU

保留原始时间戳、四路有效机械 RPM、电池电压、姿态、gyro/IMU、发送与执行指令、飞控/ESC/磁极数配置。PWM/归一化驱动、电气 eRPM 和机械 RPM 不能改列名互换；转速遥测中的 `65535` 是无效值。没有可信实际 RPM 时，当前工具不能仅凭姿态与角速度识别全物理表，应先保留默认值。

采近水平、低空速的升降/总推力变化和正反 yaw 激励，另采验证段。分别整理 `flight_train.csv`、`flight_valid.csv`：

```text
time,rpm1,rpm2,rpm3,rpm4,specific_force_z,omega_x,omega_y,omega_z,alpha_x,alpha_y,alpha_z
```

`time` 为递增 s，RPM 为机械转/分，`specific_force_z` 是质心处 body-FLU 比力。确认所选片段可忽略阻力、其它外力和气动偏航力矩后，显式执行：

```bash
./crazyrl identify flight --negligible-drag
```

工具用 `m*f_COM,z=ΣF_i` 拟合公共单电机推力曲线，并用已测 `J*alpha+omega×(J*omega)`、扣转子角动量导数后拟合公共 yaw 反扭矩曲线。后者依赖 `prop_inertia`；共享反扭矩常数项不可观测，固定为零并记录。持续悬停/单一推力差可能欠激励，四电机差异大时公共曲线不能代替各路模型。

输出 `physical.toml` / `physical.json`。JSON 区分测量、拟合、继承项，报告训练及留出推力/扭矩误差；检查 RPM 范围、条件数和重复采集的一致性。不能把默认继承的 J、转子惯量、阻力或推力上限称作已辨识。

### 1.3 补旋翼升降速

先测不同电压下的稳定驱动→实际 RPM 映射，得到可信目标机械 RPM，再同步采瞬态并整理 `rotor_train.csv` / `rotor_valid.csv`：

```text
time,motor,target_rpm,actual_rpm
```

motor 为 1..4。按左端目标/实际 RPM 拟合 `dn/dt=c1*(n_target-n)+c2*(n_target²-n²)`，升降速各两项：

```bash
./crazyrl identify rotor --negligible-drag
```

它联合重读 flight 与 rotor 日志后导出 `physical.toml`，不是把未知的上次结果自动补齐。检查验证的一步 RPM 误差。快速电机瞬态可能需要 500 Hz 或更高可信采样；100–200 Hz 姿态/动捕日志只是同步检查起点，不能用插值增加带宽。发送时间和执行时间需分别保留，避免把通信延迟拟进电机动态。

#### （可选）推力/扭矩台交叉检查

有台架时另采 `static_train.csv` / `static_valid.csv`，每行单个电机的稳定值：

```text
motor,rpm,thrust_n,torque_nm
```

推力为单电机 N、反扭矩大小为 N·m；无扭矩测量则删掉该列，不填假值。覆盖各电机的实际 RPM/电压范围并重复测量：

```bash
./crazyrl identify static
```

单独输出 `static-check.toml` / `.json`，不覆盖 flight/rotor 的 `physical.toml`。它可以交叉检查曲线，不能凭悬停附近样本证明最大推力。

### 1.4 补未自动辨识的量，再接入训练

第一性原理 `drag_matrix` 目前没有一键辨识器。可用同步参考位姿平滑得到质心处 `v_world/a_world`，结合可信姿态和实际 RPM 推力曲线，在无风/已知风条件下检查：

`m*Rᵀ*(a_world-g)-[0,0,ΣF] = D*(Rᵀ*v_air_world)`。

用不同方向/速度及留出飞行验证 D 的耗散符号与预测误差。没有可信速度/风参考时，它仍是继承假设；参考位置/速度只用于离线测量。

再核对控制器参数的全部字段与真实固件、执行器饱和和姿态阶跃响应。当前工具不自动调姿态增益、不测全部 J、不自动辨识转子惯量/风，也不校准 N→固件驱动。更换电机后，应分别核对第一性原理物理参数与控制器参数中的曲线、反扭矩与推力的比值、电机力臂和混控，不能只改物理质量。

在 `config/training.yaml` 的 `environment.model_params_path` 填 `/outputs/measurements/physical.toml`，再运行 `./crazyrl train`。默认保持为 null，使用官方 cf21B_500 参数。当前参考 PD 控制器保留 CrazyRL 数值，不读取 Mellinger 模板；物理辨识不会自动重调控制器。

## 2. 闭环阻力/姿态诊断：另一套日志和另一种输出

不具备机械 RPM 时，可用**离线参考位置、姿态、姿态角/总推力指令与实测质量**拟合 `so_rpy_rotor_drag` 简化闭环模型。它拟合总推力比例、总推力一阶响应、两项机体系线性阻力，以及二阶姿态响应；不测电机 RPM 曲线或 Mellinger 增益。

```bash
./crazyrl identify drag-init
```

生成 `closed_loop_train.csv` / `closed_loop_valid.csv` 空表头。用动捕或其它验证过的参考系统采集三维位置和姿态，统一 ENU/FLU 与初始航向，记录实际姿态角 rad 和总推力 N，整理为：

```text
time,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,cmd_roll,cmd_pitch,cmd_yaw,cmd_f
```

位置 m，四元数为机体到世界的 xyzw；不要求直接提供速度/加速度，工具对位置离线滤波求导。NPZ 等价字段为 `time[N]`、`pos[N,3]`、`quat[N,4]`、`cmd_rpy[N,3]`、`cmd_f[N]`。姿态+gyro 不能代替参考位置，PWM 或 yaw-rate 也不能代替这里的总推力/角度。

从接近零姿态、零角速度、稳定悬停的片段开始；预处理将最初约 0.1 s 的姿态归零，却不会同时旋转位置与指令，不能靠它补救坐标错误。采多方向、正反向、不同速度平移和可分辨总推力滞后的激励，另留验证飞行，避免未知风/接触力。最低加载条件为 100 点、至少 2 s、平均采样率大于 20 Hz，间隙不超过 3 倍中位周期；满足门槛不等于充分激励。平移/姿态使用 6/8 Hz SVF，需检查原始离群点警告与滤波偏差。

把下例 `0.0434` 换成真实称量的 kg 值：

```bash
./crazyrl identify drag --mass 0.0434
```

使用其它测量目录或 NPZ 时，显式提供数据路径：

```bash
./crazyrl identify drag --mass 0.0434 --directory outputs/my-aircraft \
  --data outputs/my-aircraft/closed_loop_train.npz \
  --validation outputs/my-aircraft/closed_loop_valid.npz
```

两种写法选符合实际文件的一条。输出 `closed-loop-so-rpy-rotor-drag.toml`、同名 JSON 和 `fit.log`。检查两部分优化收敛、留出 RMSE/R²、重复飞行稳定性与阻力符号。D=`diag(d_xy,d_xy,d_z)`，`F_drag_world=R*D*Rᵀ*v_world`，单位 N·s/m，代码用加号所以耗散系数应非正；当前优化未约束其非正，正值不能取绝对值后直接使用。推力滞后是单位 N 的总推力状态，可能吸收执行/通信误差，不是四电机 RPM 动态。

该闭环 TOML **不能填入 `environment.model_params_path`，也不能只抄同名 `drag_matrix`**。它把控制、延迟与气动合并在简化参数中，直接用于显式含这些环节的第一性原理模型可能重复补偿。它保留为离线诊断/对照；第一性原理模型的阻力仍需第 1.4 节的物理验证。

只用一份日志先估计时，可跳过工程包装器，使用同一 Docker 中的低层可选验证接口；以下质量仍需替换为实测值：

```bash
docker run --rm --init --user "$(id -u):$(id -g)" -e JAX_PLATFORMS=cpu \
  -v "$PWD/outputs:/outputs" crazyflow:dani-crazyrl \
  python -m crazyflow.identify fit --dynamics so_rpy_rotor_drag --mass 0.0434 \
  --data /outputs/measurements/closed_loop_train.csv \
  --output /outputs/measurements/closed-loop-one-log.toml
```

这是单日志拟合，不能报告独立验证成绩。低层进程退出 0 不保证优化日志中的 `Training success=True`；`./crazyrl identify drag` 会拒绝未收敛结果。平移求解采用三轴残差展平，平方和目标与逐样本残差范数相同，避免零范数处的 Jacobian 数值问题。

## 3. 没有硬件：运行两套明确标记的合成例子

检查第一性原理曲线和旋翼响应导出：

```bash
./crazyrl identify check
```

检查简化闭环模型的拟合与验证：

```bash
./crazyrl identify drag-check
```

两者各自在 `outputs/measurements/` 创建唯一子目录，不覆盖实测文件。`check` 生成机械 RPM/IMU、旋翼瞬态与可选台架合成表，检查第一性原理曲线/升降速导出；`drag-check` 用官方 `so_rpy_rotor_drag` 前向模型的已知非零阻力/滞后，生成 15 s、100 Hz、seed 0/1 的两份位姿/指令日志，再拟合与验证。

这两套中的 train/valid 都是**合成估计/验证数据，不是真实实飞，更不是 PPO 的训练数据集**。seed 1 改变激励相位，用来检查 seed 0 得到的参数在另一激励下的预测；仍是同一已知模型与参数，不能称为跨物理或硬件迁移验证。CSV 来源在同名 JSON 的 `synthetic`、`known_parameters` 等字段中，NPZ 还内嵌标记；物理清单保留未确认实测的标记，仅合成路径允许导出。

[示例目录](../examples/identification/README.md)保留两份闭环 CSV 和来源 JSON，可直接查看列名、单位与真值。默认的低层 `identify synthetic` 仍是无阻力/滞后的 `so_rpy` 数据，不能冒充这份匹配例子。没有任何真实测量时，应保留官方默认来源，不能把合成例子的质量、J 或曲线称为实测。

[官方系统辨识 README 原文](crazyflow_experiments_sysid/README.md)逐字保留固定版本的上游文档，来源版本和 SHA256 见同目录的 [provenance.json](crazyflow_experiments_sysid/provenance.json)。其中的相对路径、安装说明和命令属于上游仓库的独立 ROS2/Pixi 环境，不是本项目 Docker 的直接命令；本项目入口以 `./crazyrl identify --help` 为准。上游 `lol_sysid.py` 的变量 `rpms` 实际来自归一化电机指令，且含 Jzz 比例假设，不能直接改名当作机械 RPM 或独立惯量测量。本手册的参数与单位说明对应本项目的加载器和辨识接口。
