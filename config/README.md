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
| `task.radius_min/max`、`distance_min/max` | 气球半径与初始目标距离，m |
| `task.drone_radius`、`collision_offset` | 接触球半径、机体系球心偏置，m；与官方碰撞球对齐 |
| `task.max_impact_speed`、`min_altitude`、`max_range`、`max_unseen_seconds` | 撞击速度训练偏好 m/s、最低高度 m、活动范围 m、连续失视缓冲 s；首次实际触球就是成功，速度仅作诊断 |
| `task.max_tilt`、`max_yaw` | 动作对应的目标姿态上限，rad |
| `objective.loss_weights` | 距离、接触、速度、过快撞击、FOV、地面、倾角、角速度、动作幅度和平滑性权重 |
| `objective` 中的温度与软阈值 | 可微最小距离/接触/撞击加权温度、地面余量 m、倾角软限 rad；改变优化目标，不改变触球成功定义 |
| `noise.bbox_noise_std`、`dropout_prob`、`max_delay_steps` | 归一化 bbox 坐标噪声、丢帧概率、最大相机延迟拍数；一拍为 `task.dt` |
| `noise.attitude_noise_std`、`noise.gyro_noise_std`、`noise.gyro_bias_std` | 姿态旋转向量噪声 rad、机体系 gyro 噪声/偏置 rad/s |
| `noise.action_delay_steps`、`mass_fraction`、`thrust_fraction`、`response_fraction`、`wind_std` | 动作延迟拍数、质量/推力/旋翼响应相对变化、世界系外扰加速度幅度 m/s² |

命令行 `--iterations`、`--batch-size`、`--steps`、`--seed`、`--lr` 优先于 YAML。位置、线速度和目标真值只用于仿真、损失与评测；增加噪声不会把这些量送入策略。

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
| 噪声、丢帧、通信延迟 | 静态/动态重复日志、采集与执行端时间戳 | 手工统计后填写；默认不是实测置信区间 | YAML 的 `noise` |
| 目标采样、碰撞几何、损失权重 | 任务需求、实物尺寸、训练/评测结果 | 用户设置，不属于机体辨识 | YAML 的 `task/objective` |

`./dani identify check` 与 `drag-check` 使用明确标记的合成样例；合成验证不能证明实物已辨识。[统一辨识手册](../docs/identification.md)和[示例数据说明](../examples/identification/README.md)解释所需观测与范围。

## 5. 其它外层资源

`config/dynamics/` 与 `config/control/` 保存原样迁出的官方多机型默认参数，是加载器基准；日常本机配置编辑上述四份文件。`assets/scene.xml`、`assets/drones/*.xml` 及网格保存场景与机体视觉资源。当前 BPTT/native 飞行的质量和惯量来自冻结的 `vehicle.toml`，不能用修改可视 XML 惯量替代；任务接触球由 YAML 任务几何配置。

源码 checkout 与 Docker 自动定位本项目外层目录；wheel 安装时资源随包部署到安装前缀的 `share/crazyflow/`，任意工作目录可运行。通常无需资源环境变量；显式 `CRAZYFLOW_RESOURCE_ROOT` 只供完整自定义资源树使用，须同时包含 `assets/` 和 `config/`。
