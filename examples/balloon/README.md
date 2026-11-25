# BPTT 撞气球示例

这里保存一份实际训练得到的 `cf21B_500` 策略及训练曲线。它使用官方第一性原理动力学做纯 BPTT；策略只接收 bounding box 历史、姿态、机体角速度和先前指令，不接收位置或线速度。训练共完成 1500 次更新，`policy.npz` 选取固定验证集损失最低的第 **1150** 次参数，而非第 1500 次参数。

| 文件 | 内容 |
| --- | --- |
| [policy.npz](policy.npz) | 可直接加载的最优策略；内嵌完整物理、控制器、相机及观测配置 |
| [training.jsonl](training.jsonl) | 原始训练标量日志，含训练/验证损失、梯度范数及耗时，共 61 条记录，最后为第 1500 次 |
| [training.png](training.png) | 从原始 JSONL 绘制的离线曲线，无需运行 TensorBoard 服务 |
| [evaluation-summary.json](evaluation-summary.json) | 原生 MuJoCo 三场景评测汇总、冻结配置、相机与统计口径 |
| [provenance.json](provenance.json) | 原始本地文件路径、SHA256 及最优/最后参数核对结果 |
| [gpu-capacity.json](gpu-capacity.json) | 默认完整展开的独立 GPU 容量短测；每档 3 次更新，不是示例策略的训练日志 |
| [gui.png](gui.png) | 用本示例策略实际运行 `play --example --headless` 的 GUI 自检截图 |

初始验证损失为 **12.43505**，最优为 **1.31182**；第 1500 次为 **1.39257**。JSONL 在第 1 次、每 25 次及最后一次记录标量，损失是可微优化目标，不能当作接触成功率。日志最后的 `seconds=758.7963` 从初始验证结束后计时，包含首次更新编译，不含容器启动及初始验证。

相机为 640×480、水平视场 120° 的无畸变针孔模型，前置 0.03 m；具体 K 和安装旋转已冻结在 checkpoint。控制周期 0.02 s，物理/姿态控制为 500 Hz，每次轨迹包含 150 个策略控制步。评测种子为 `20261125`，每个场景均保留全部 256 次试验作为分母：

| 场景 | 实际首次接触 | 接触率 | 其余结果 |
| --- | --- | --- | --- |
| nominal | 256 / 256 | 100% | 无 |
| noisy | 255 / 256 | 99.61% | 超时 1 |
| perturbed | 244 / 256 | 95.31% | 失视 3、超时 9 |

成功按每个物理子步内扫掠球体的首次接触计算；超过 2.5 m/s 仅作速度诊断，仍算接触。这份评测没有高速接触。MuJoCo 的 `mj_step` 真正积分刚体，控制器、旋翼响应和力/力矩模型与训练共享；这些结果属于仿真验证，未验证实机或真实视觉检测器。

在项目根目录执行：

```bash
./dani setup
./dani evaluate --example
./dani play
```

`--example` 把本目录策略复制到新的 `outputs/example-policy-*` 后执行评测，不覆盖已有训练。只想直接看示例策略，可运行 `./dani play --example`；没有桌面时用 `./dani play --example --headless` 保存 GUI 自检截图。视频、完整轨迹和新评测结果均写入 `outputs/`，本目录仅保留轻量示例。
