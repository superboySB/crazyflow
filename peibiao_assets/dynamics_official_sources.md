# Crazyflow 上游公开实飞资料

以下资料由 Crazyflow 上游研究团队公开，作为上游系统能力和实飞研究的参考依据。其研究对象、测试设备、训练权重及实验组织均属于上游研究，不构成本项目或投标人的实机测试记录。

## 视觉策略实飞公开视频

[![Crazyflow 上游视觉策略实飞视频的公开视频封面](https://i.ytimg.com/vi/QhOUePHSmyA/maxresdefault.jpg)](https://www.youtube.com/watch?v=QhOUePHSmyA)

图：Crazyflow 上游公开视频的原站封面；点击图片访问视频。该视频由 [Crazyflow 官方主页](https://learnsyslab.github.io/crazyflow/)作为“End-to-end vision-based flight in Crazyflow”展示，主页说明该视频展示 Crazyflow 中训练的策略在硬件上的视觉飞行。封面用于识别公开视频，不作为本项目运行结果。

## 动力学与策略迁移论文

[Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX](https://arxiv.org/html/2606.01478v1)，Schuck 等，2026，arXiv:2606.01478v1，Sim-to-Real、Trajectory Tracking Benchmark 及 Table 1、Figure 4。

论文 Table 1 报告：Crazyflie 2.1，姿态控制、原理模型、控制频率 100 Hz，圆形轨迹周期 9 s 时，仿真与实机位置差 RMSE 为 19.0 ± 2.6 mm；周期 3.5 s 时为 59.6 ± 1.3 mm。数值为 5 次实验的均值和标准差，适用范围限于论文所列平台、模型、轨迹、参数和实验条件。

论文 Figure 4 报告 Crazyflie 2.1 Brushless 上 PPO 和 BPTT 策略的实飞轨迹跟踪；其结果属于论文作者的独立研究，具体任务与统计方法以论文为准。

## 图片来源与使用范围

视频封面通过 YouTube 原站地址引用，视频和封面权利属于相应权利人；本文件未确认图片或视频在投标材料印刷、离线复制、再分发或其他商用场景中的授权。相关使用方式应按权利人授权及招标文件要求核验。
