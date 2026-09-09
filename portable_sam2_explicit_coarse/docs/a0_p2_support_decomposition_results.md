# A0 P2 支持域可达收益分解 Oracle

日期：2026-09-09。目标不是宣称 GT Oracle 可训练，而是定位 A2/R1 未能改善 coarse 的
结构性瓶颈：P2 的 raw-boundary S4 支持域本身是否保留了可兑现的 dense 路径收益。

## 决策前提

- 冻结 A0 已证明：完整 GT coarse 可经既有 dense canvas 消费（不是 point）带来稳定收益。
- A2/R1 第一门失败：自然 P2 residual/canvas 非零，但 refined coarse 内容改善的 CI 跨零；
  715 个匹配 ROI 中有 443 个被 `max_search_coverage=0.5` 整 ROI reject。

## 方法

冻结 A0 epoch36，NWPU VHR-10 全验证 130 图。支持域几何从 A2 epoch63 checkpoint 的
保存配置固定读取：raw-only 形态学边界、`boundary=2`、`search=4`、阈值 0.5、coverage
reject 0.5。每个同类且 proposal IoU≥0.5 的匹配 ROI，只将 raw 二值标签错误的 coarse
像素写到目标符号 `±8`；正确像素、点挖掘、proposal、类别与分数、未匹配 ROI 一律保持 A0
原值。干预仅进入生产 `_shape_prior_to_prompt_mask` 的 dense 消费者。

| 臂 | 允许写入错误的空间域 |
|---|---|
| raw | 无干预 |
| accepted S4 | A2 当前 raw-support，coverage 超 0.5 的 ROI 整体关闭 |
| ungated S4 | 同一 raw S4，仅取消 coverage reject |
| full ROI | 匹配 ROI 的所有错误像素 |

完整运行要求 A0 unhooked standard 与既有 production inference records/mAP 精确一致；
raw hook 与 standard final-mask hash 相同；四臂 detector hash 相同。统计使用十类 COCO，
并以相同 130 图执行清单做 500 次 paired image-bootstrap（seed44）。

## 结果

| 臂 | segm/mAP | 相对 raw | 95% paired CI | P(Δ>0) |
|---|---:|---:|---:|---:|
| raw | 0.666929 | — | — | — |
| accepted S4 | 0.668838 | +0.001909 | [+0.000009, +0.003221] | 0.974 |
| ungated S4 | 0.672712 | +0.005782 | [+0.002985, +0.007898] | 1.000 |
| full ROI | 0.674430 | +0.007501 | [+0.004103, +0.010134] | 1.000 |

`ungated − accepted = +0.003874`，由同一重采样的 paired deltas 相减得到 CI
`[+0.001670,+0.005927]`；`full ROI − ungated = +0.001718`，CI
`[+0.000533,+0.002981]`。

728 个 matched ROI 中，accepted S4 覆盖 133,594/298,451（44.8%）错误像素；ungated S4
覆盖 283,435/298,451（95.0%）。accepted/ungated support 像素为 641,870/1,543,813。
190 个未匹配 ROI 从未写入。

外部工件：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_p2_support_decomposition_full_v2/`。

## 裁决与下一步

这给出一个新的、可执行的突破口：**在“已知正确的纠错方向”这个 Oracle 条件下，coverage
超阈值后的整 ROI 硬拒绝是可达收益的高杠杆限制。** 只要保留同一个 raw S4，解除硬拒绝
就可恢复 full-ROI error-only Oracle 的约 77% mAP 增益（0.005782/0.007501）。这并不单独
证明它是自然 A2 P2 的唯一实际根因：GT 在此 Oracle 中替代了最难的 correction direction。

下一候选不应再扩大 P2 residual cap、加入新 sparse points，或从 P2 另造 dense 信息源；应设计
`P2-conditioned adaptive correction support`：以连续/预算受限的 support 取代 whole-ROI hard
reject，保持 raw coarse 是唯一语义源、P2 仍只预测局部残差。训练前必须再通过以下门：预测的
support 在带外 raw 错误上相对 raw-only selector 有稳定 image-bootstrap recall/precision 提升，
并且部署形态的 signed correction 能提高 coarse IoU 且不增加正确像素破坏。

以上是空间域 Oracle，不是新模块的精度声明；GT 在推理时不可用，也未证明 P2 feature 已经能
识别这些新增位置。
