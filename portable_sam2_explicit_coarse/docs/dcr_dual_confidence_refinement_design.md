# DCR：双置信度解码细化模块（A4）

## 论文中的模块表述

**Dual-Confidence Refinement (DCR，双置信度解码细化)** 位于 SAM2 mask decoder 的
native mask-logit 输出端。它把“哪些位置值得检查”和“检查后是否应改写”分成两个
不同的置信度问题：原始预测的不确定性只用于定位候选点；独立学习的纠错置信度决定
残差改写的强度。这样避免 A3/UDPR-v1 将所有低置信点一视同仁地写入残差，进而扰动
原本正确但置信度较低的像素。

对每个 RoI，记原生 decoder logit 为 `z`，upscaled decoder feature 为 `U`，选中的
mask token 为 `t`。DCR 以最小 `|z_p|` 的固定 `K=64` 个 native-grid 点作为候选集
`S`（训练与推理完全一致）。对 `p in S`，共享特征为：

`h_p = MLP([U_p, t, z_p, x_p, y_p, |z_p|])`。

两个轻量头分别预测有界残差与纠错置信度：

`delta_p = delta_max tanh(W_delta h_p)`，`q_p = sigmoid(W_gate h_p)`，
`z'_p = z_p + q_p delta_p`。

其中 `q_p` 不是阈值门，也不声称产生稀疏写入；它是连续的、保守的残差幅度授权。
未选中的点严格保持原始 `z`。推理期不读取 GT、不新增 prompt、不重跑 decoder，也
不改变检测器的 bbox、类别或排序分数；其预期贡献仅是 segmentation/boundary。

## 训练监督与初始条件

训练中，从既有 full-image mask assignment 构造 detached 的纠错标签：

`e_p = 1[(z_p >= 0) != y_p]`。

它仅监督 `q_p`，不参与 Top-K 选择，也绝不会在推理期出现。首个 A4 配方为：

`L = L_refined + 1.0 L_gate + 0.05 L_keep`，

其中 `L_refined` 是沿用 A3 的 selected-point 正/负独立归一化 BCE，`L_gate` 是对
error/correct 两类点分别归一化后等权的 `BCEWithLogits(g,e)`（`q=sigmoid(g)`，AMP-safe），`L_keep` 是原本正确候选点的
`mean(|q delta|)`。三项均按全局 DDP 计数做局部梯度缩放，避免不同 rank 的候选类别
比例改变目标。

为严格继承 A0 的初始前向，`delta_head` 的权重和偏置为零；`gate_head` 权重为零、偏置
为 `logit(0.1)`。因而初始化时 `delta=0` 且 `z'=z`，同时 residual head 仍获得
`q=0.1` 缩放后的梯度，gate head 获得 `L_gate` 梯度；第二个优化步开始共享 trunk 也有
有效梯度。

## 与 A3 及已有方法的边界

- **A3 / UDPR-v1**：仅以不确定性选 K 个点，直接写 `z'=z+delta`。A4 唯一新增的是
  `q`、其训练期纠错监督和 correct-pixel keep 约束；K、输入、残差上界、PBM 底座与
  单阶段训练协议均不变。
- **PointRend**：PointRend 在多轮/细粒度渲染中选择并细化点；DCR 不执行 subdivision
  或 iterative rendering，而是在既有 SAM2 decoder 的固定 native grid 上作一次固定预算
  的尾部改写。
- **质量路由/提示选择**：DCR 不评估或选择 prompt quality；它只对已经产生的 mask logit
  做点级纠错授权。

因此，DCR 可作为一个独立的 decoder-tail refinement component 表述，但不能仅凭结构
宣称创新成立：必须由机制和效用两层证据共同支持。

## A4 实施契约

- arm：`a4`；tag 后缀：`_a4_pbm_udprcgk64`；architecture ID 后缀：`_udprcgk64`。
- 共同底座：A0/PBM、P2 off、`K=64`、hidden=128、`delta_logit_max=2.0`、完整网络
  单阶段训练、`DECODER_TAIL_LR_MULT=1.0`。
- 显式配置：`DECODER_TAIL_MODE=confidence_gated`、`GATE_INIT_PROB=0.1`、
  `GATE_LOSS_WEIGHT=1.0`、`KEEP_LOSS_WEIGHT=0.05`。
- A3/v1 的 `decoder_tail_refiner_cfg` **不写** `mode` 及任何 gate 字段，仍使用原始
  `_udprk64` ID；不得以 A4 代码重写其 checkpoint/配置语义。
- `vhr10_p2v2_dev.sh`、`vhr10_p2v2_full600.sh`、`vhr10_p2v2_matrix300.sh` 均通过同一
  arm 定义接线。A4 不自动插入当前的 matrix300 串行队列，避免改变正在运行的
  P→PB→A0→A3-v1 比较。

## 观测、判定与停止条件

日志字段见 `DEBUG_FIELDS.md` 的 `TAIL/*` 表。A4 的核心机制检查是：

1. `gate_on_error > gate_on_correct`，说明授权确实区分 pre-tail 错误与正确候选点；
2. `applied_delta_correct_abs < applied_delta_error_abs`，且相较 A3 的
   `destroy_fraction` 降低、`net_flip_fraction` 改善；
3. `selected_count`、`selected_bce`、`gate_bce` 均有限，且初始化前向逐位等于 A0；
4. 最终以同协议完整验证的 A4 同时相对 A0 和 A3 的 paired bootstrap 检验效用。

若 gate 在 error/correct 上无分离，或 correct-pixel 的实际改写不降、destroy/net flip
不改善，则停止扩展该模块；若机制成立但 A4 仍未优于 A0/A3，则只能把它记录为阴性
消融，不能作为论文主贡献。

## 当前状态

本文档定义的是待运行的 A4，不包含任何精度收益声明。正在运行的 matrix300 A3 是
UDPR-v1，对其参数、架构 ID 和行为零改动。
