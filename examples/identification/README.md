# 参数辨识示例数据

这些文件用于系统辨识，**不是 ToF PPO 训练数据集**。PPO 从 Crazyflow 仿真在线采样，不读取本目录的 CSV。

两段日志由官方 `cf21B_500` 的 `so_rpy_rotor_drag` 前向模型生成，每段 15 s、100 Hz，包含已知非零阻力、总推力滞后与闭环姿态响应。它们均为**合成数据，不是实机测量或真实飞机的精度报告**。

| 文件 | 内容与用途 |
| --- | --- |
| [synthetic-train-drag.csv](synthetic-train-drag.csv) / [来源 JSON](synthetic-train-drag.json) | 种子 0，用于拟合 |
| [synthetic-valid-drag.csv](synthetic-valid-drag.csv) / [来源 JSON](synthetic-valid-drag.json) | 种子 1，使用另一组激励相位独立验证，不复制训练轨迹 |

两份 CSV 使用同一列结构：

```text
time,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,cmd_roll,cmd_pitch,cmd_yaw,cmd_f
```

时间单位为 s，世界 ENU 位置为 m；四元数为机体 FLU 到世界的 `xyzw`，姿态指令为 rad，`cmd_f` 为整机总推力指令 N。这些列不含机械 RPM，不能当作第一性原理电机辨识日志。同名 JSON 保存 `synthetic=true`、生成模型、种子、采样、积分方法和 `known_parameters`；生成的 NPZ 也保留合成来源标记。

先按[主 README](../../README.md) 构建镜像，以下命令均从项目根目录执行，无需宿主 Python。

## 合成检查

```bash
./crazyrl identify check
```

```bash
./crazyrl identify drag-check
```

两个命令在 `outputs/measurements/` 下创建唯一目录，保存输入、拟合结果和检查输出，不修改已有测量文件。它们使用不同观测与模型：

| 检查 | 数据与用途 |
| --- | --- |
| `check` | 机械 RPM、IMU/角加速度、旋翼瞬态及可选台架合成表，检查第一性原理曲线与旋翼拟合 |
| `drag-check` | 同一闭环模型、相同参数的两组相位激励，检查阻力/总推力滞后拟合与独立合成验证 |

独立 train/valid 是示例的工程选择；官方低层 `sys_id_translation/sys_id_rotation` 和 `identify fit --validation` 允许省略验证数据。复制同一文件不构成独立验证，不同种子的合成验证也不属于实机或跨模型验证。合成检查中的质量是假定值，即使输出字段名为 `measured_mass_kg`，也不表示经过称量。

## 拟合本目录样例

先复制到入口允许的输出目录：

```bash
mkdir -p outputs/drag-example
```

```bash
cp examples/identification/synthetic-*-drag.csv examples/identification/synthetic-*-drag.json outputs/drag-example/
```

使用独立训练与验证日志：

```bash
./crazyrl identify drag --directory outputs/drag-example --mass 0.0434 \
  --data outputs/drag-example/synthetic-train-drag.csv \
  --validation outputs/drag-example/synthetic-valid-drag.csv
```

`0.0434` 是生成模型的已知质量，不是称重结果。输出仍标记合成来源；检查 `fit.log` 中平移/转动优化的收敛状态和验证误差。拟合值可与来源 JSON 的已知参数比较，但误差受滤波、离散和激励影响，不能据此评价真实飞机。

## 准备真实测量

分别生成两类日志模板：

```bash
./crazyrl identify init
```

```bash
./crazyrl identify drag-init
```

`init` 对应机械 RPM/IMU/台架物理日志，`drag-init` 对应位置/姿态及姿态/总推力指令。字段不能混用；模板只有列名和未确认的默认示例，不是实测数据。训练与验证飞行应独立采集，并记录飞机、载荷、采集时间、固件/控制器、电压、坐标系、时间对齐和单位。

完整操作见[测量与辨识手册](../../docs/identification.md)。闭环 `so_rpy` 参数文件不能直接用作当前 ToF PPO 的 `first_principles` 物理表，也不能只按同名字段复制阻力系数；默认 PPO 仍使用官方物理参数。
