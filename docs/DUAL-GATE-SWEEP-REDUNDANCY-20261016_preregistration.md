# DUAL-GATE-SWEEP + DUAL-REDUNDANCY 预注册（2026-10-16，运行前）

> **勘误（2026-10-16，提交后即补）**：本文件提交（`e47fb2c`，10:29）时，
> 并行工作流已在 10:21–10:24 运行同一对探针并于 10:25:58 提交实现与结论
> （`d71fecc`，`docs/DUAL-WHY-NULL-20261016_conclusion.md`）——本文件因此
> **不构成前瞻性预注册**，仅保留为判读带对照记录。权威读数以 d71fecc 的
> 结论为准；其数字与本文件判读带的对照：描述子 R²≈0.01（落在
> RESIDUAL-SIGNAL 支的 R² 条件）但残差 fg/bg=1.19 < 1.5（不满足其任务
> 对齐条件）——即"独特但不对题"的中间带读数；门控扫描 0014=INERT、
> 0012 出现 k=2 峰值（+0.40 点，满足 RISE 支）但峰值仍 ≤ 单流。

PI 约束（2026-10-16）：**不换路线**——继续在双流门控注入路线上寻找双指标增益
路径。本对探针是该路线内的下一级冻结诊断（不训练检测模型），回答两个问题：
门控开度是否标定不足（函数级）；卫星流尚未重建的 UAV 独有信息还有多少、
是否对齐结构（信息级）。脚本沿用并行工作流已写就的实现
（`scripts/ue/probe_gate_sweep.py`、`scripts/ue/probe_bev_redundancy.py`，
本预注册先于任何运行提交）。

## GATE-SWEEP（函数级）

- 输入锚点：五个双流臂的 `last_model_epoch100.pth`（0009 单流不适用；其身份
  清单沿用 `probes/DUAL-AB-LADDER-20261016/identity_*.json` 机制）。
- 操作：对每臂全部融合门控乘 k∈{0, 0.25, 0.5, 1, 2, 4, 8}，另加 k=1 通道置换
  sham（seed 44）。val 前 60 场逐场 COCO AP（与拉通工具同一评估实现）。
- 主读数：per-scene 平均 AP 随 k 的曲线形状（segm/bbox 双指标同报）。
- 预注册判读带（运行前写死）：
  - **RISE**（支持重标定重训臂）：任一 k∈{2,4,8} 的均值比 k=1 高 ≥ +0.30 点
    （双指标分别报告，任一触发即登记）；
  - **INERT**（通路功能惰性）：k∈{0.25,0.5,2,4} 全部与 k=1 差 < 0.15 点；
  - **WIRED-AROUND**（网络绕开有害信号）：k=0 比 k=1 高 ≥ 0.30 点；
  - sham 校验：置换门在 k=1 下不得高于真实 k=1（高即门控通道身份无意义，
    读数全部降级）。
- 判读边界：N=60 场、单 checkpoint、eval 前向；曲线形状结论不直接等价于
  重训收益，RISE 只作为重标定重训臂的立项依据。

## REDUNDANCY（信息级）

- 输入锚点：EXP-0009 单流 P3（64²，冻结前向）+ EXP-0014 的定标 BEV
  （`_uav_bev` 含 bev_scale，64² 网格对齐）。
- 操作：train 60 场 harvest，ridge（λ∈{0.1,1,10,100} 选 val 最优）与 2×512 MLP
  读出（seed 44，6 epoch）预测 262 个 BEV 通道；val 40 场评分 R²。分组：
  descriptor 0–255 / zmean / zstd / vcount / zres / missing / visible。
- 主读数：MLP overall R² 与 descriptor 组 R²；ridge 作线性下界同报。
- 预注册判读带：
  - **REDUNDANT**：descriptor 组 R² ≥ 0.90（融合输入在信息层冗余）；
  - **RESIDUAL-SIGNAL**：descriptor 组 R² ≤ 0.70 且残差 fg/bg 能量比 ≥ 1.5
    （存在卫星流未覆盖且对齐建筑结构的 UAV 独有信号——路线内继续注入的
    信息学依据）；
  - 中间带如实报告，不强行二值。
- 判读边界：读出为探针局部回归器（非检测训练）；R² 是"可线性/浅非线性重建"
  的下界口径；fg 对齐用 512 GT 并集 area-降采样到 64²。

## 与下一步臂的接线（预注册映射，均不换路线）

- GATE-SWEEP=RISE → 立项"门控重标定重训"臂（如 gate 初始化/缩放修正，
  单变量对相应基臂）；
- REDUNDANCY=RESIDUAL-SIGNAL 且 GATE-SWEEP 非 INERT → 注入形式加强臂
  （LoRA×仅P2 组合 / image_embedding 注入点）的信息学依据成立；
- 双双 INERT/REDUNDANT → 在不换路线前提下，问题定位到训练动力学，
  立项"融合延迟解冻"臂（gate 前 N epoch 锁零）。

两探针均不读 test；工件落 `/data2/.../probes/DUAL-GATE-SWEEP-20261016/` 与
`/data2/.../probes/DUAL-REDUNDANCY-20261016/`；结论前独立子代理审
实现+数字。
