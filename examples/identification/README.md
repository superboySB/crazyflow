# 参数辨识示例数据

这些文件是系统辨识的估计/验证日志，**不是 PPO 训练数据集**。默认 PPO 从物理仿真采样，无需读取本目录。

本目录保留两段由官方 `cf21B_500` 的 `so_rpy_rotor_drag` 前向模型生成的合成飞行日志，用于检查已知非零阻力、总推力滞后与闭环姿态响应的拟合。每段 15 s、100 Hz；它们来自仿真，不能作为实机测量或真实飞机的精度报告。

| 文件 | 内容与用途 |
| --- | --- |
| [synthetic-train-drag.csv](synthetic-train-drag.csv) / [来源 JSON](synthetic-train-drag.json) | 种子 0，用于拟合；JSON 保存生成模型、积分方法与 `known_parameters` |
| [synthetic-valid-drag.csv](synthetic-valid-drag.csv) / [来源 JSON](synthetic-valid-drag.json) | 种子 1，使用另一组激励相位进行合成验证；不复制训练轨迹 |

两份 CSV 使用同一列结构：

```text
time,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,cmd_roll,cmd_pitch,cmd_yaw,cmd_f
```

时间为 s，世界 ENU 位置为 m；四元数为机体 FLU 到世界的 `xyzw`，姿态指令为 rad，`cmd_f` 为整机总推力指令 N。这些列不含机械 RPM，不能当作第一性原理电机辨识日志。

先在项目根目录执行一次 `./crazyrl setup`。之后无需宿主 Python，也无需保留终端变量：

```bash
./crazyrl identify check
./crazyrl identify drag-check
```

两个命令各自在 `outputs/measurements/` 下创建唯一目录，保留输入日志、拟合结果与检查输出，不修改已有测量文件。它们是**两套不同观测/模型的检查**：

- `check` 的 train/valid 是机械 RPM、IMU/旋翼/可选台架合成表，用于第一性原理曲线辨识管线。
- `drag-check` 的 train/valid 是同一官方闭环模型、相同已知参数下的两组不同相位激励；seed 0 估计参数，seed 1 检查另一激励下的预测。两段都不是实飞机录，也不是跨模型验证。

官方底层 `sys_id_translation/sys_id_rotation` 的验证数据可选，低层 `identify fit --validation` 也可省略：一份充分激励的日志可以先拟合。这里的入口和示例主动保留两份用于独立检验，是工程选择，不是官方算法强制；复制同一文件不能提供独立验证。

| 命令 | 合成数据及用途 | 不能据此声称的结果 |
| --- | --- | --- |
| `./crazyrl identify check` | 机械 RPM、质心比力/角加速度、旋翼瞬态与可选台架表，检查第一性原理曲线与旋翼拟合链路 | 已完成实机测量、已测得真实 J、阻力或控制增益 |
| `./crazyrl identify drag-check` | 固定种子 0/1、100 Hz 的 `so_rpy_rotor_drag` 位姿及姿态/总推力指令，含已知非零阻力和总推力滞后，检查同模型拟合与验证 | 已恢复实物阻力、真实电机时常或实飞机体动力学 |

闭环合成 CSV 的同名 JSON 记录 `synthetic=true`、生成模型、种子、采样、积分方法和 `known_parameters`；NPZ 还内嵌合成及来源标记。它们不是机械 RPM 日志。第一性原理合成测量文件保留未确认实测的标记，仅在合成检查路径允许导出。不同随机种子的验证仍是合成验证。即使拟合文件里出现 `measured_mass_kg` 字段，合成检查中的质量也只是假定值，不能解释为经过称量。

示例用于检查数据列名、单位、滤波、拟合及导出，不用于评价真实飞机误差。匹配模型的合成检查可将拟合值与来源 JSON 中已知的阻力及滞后比较；误差仍受滤波、离散和激励影响。低层 `crazyflow.identify synthetic` 的旧默认模型仍是 `so_rpy`，不含这两个效应，不能冒充非零阻力样例。

要直接拟合本目录两份样例，先复制到入口允许的输出目录，再指定数据文件：

```bash
mkdir -p outputs/drag-example
cp examples/identification/synthetic-*-drag.csv examples/identification/synthetic-*-drag.json outputs/drag-example/
./crazyrl identify drag --directory outputs/drag-example --mass 0.0434 \
  --data outputs/drag-example/synthetic-train-drag.csv \
  --validation outputs/drag-example/synthetic-valid-drag.csv
```

这里 `0.0434` 是生成模型的已知质量，不是称重结果。输出仍标记合成来源；保留 `fit.log` 并检查两部分优化收敛状态及独立合成验证误差。平移优化采用逐轴残差向量求最小二乘，平方和目标与原先逐样本残差范数相同，同时避开零范数处的不可微数值问题。

准备真实日志时先生成空表头：

```bash
./crazyrl identify init
./crazyrl identify drag-init
```

`init` 对应机械 RPM/IMU/台架物理日志，`drag-init` 对应参考位置/姿态及姿态/总推力指令。两套字段不能混用；模板只有列名和未确认的默认示例，不是实测数据。独立采集训练与验证飞行，并一同保留飞机与载荷清单、采集时间、固件/控制器、电压、坐标系、时间对齐及单位说明。

实际操作统一见[cf21B_500 测量与辨识手册](../../docs/identification.md)。离线位置和速度只用于测量与辨识，不会送入气球策略。闭环 `so_rpy` 参数文件不能作为当前第一性原理训练的物理表，也不能只按同名字段复制阻力系数。
