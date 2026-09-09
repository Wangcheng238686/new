# P2 第一门：自然 coarse→canvas 内容审计（A2/R1）

日期：2026-09-09。此文档记录第一门的实际裁决，而不是训练曲线的二次解读。

## 问题与方法

冻结 NWPU dev100 A2/R1 的 `best_model_epoch63.pth`，在 NWPU VHR-10 全验证集 130
图上观察**自然**的 P2 输出：`coarse_outputs.raw_logits` 与
`coarse_outputs.refined_logits`。GT 只用于同类、proposal IoU≥0.5 的事后匹配和指标，
不进入模型、不替换任何 tensor、不改变 detector proposal/score/class。

对每个匹配 ROI，分别统计：

- ROI-local coarse 的阈值 IoU/Dice；
- 两者经生产 `_shape_prior_to_prompt_mask` 渲染后、限制在原 proposal support 的
  canvas IoU/Dice；
- 连续值 canvas 是否实际改变。

不确定性按 **130 张图像**配对 bootstrap 1000 次（seed 44），从不按 ROI 独立重采样。
探针同时执行 unhooked 标准推理，要求 detector 和最终 mask hash 与仅观察的 hooked
推理完全一致。

几何口径与生产/训练严格对齐：coarse GT crop 为 `floor(x1,y1):ceil(x2,y2)` 后 nearest
resize（`get_coarse_targets` 同一规则）；canvas support 是
`paste_roi_to_full_canvas` 相同的缩放后 `floor:ceil` 粘贴区域，而非像素中心近似。

## 不变量与覆盖

- checkpoint strict load：missing/unexpected 均为 0；十类 NWPU data contract；
- 标准与 observed 的 detector SHA256 均为
  `ad68da7a…b8290a9`，最终输出 SHA256 均为 `fdcd4fbb…644a23b`；
- 715 个同类 IoU 匹配 ROI，覆盖 130/130 图；P2 search support 有效 272 个、拒绝
  443 个。这是实际 forward 的拒绝策略，不是剔除样本后的子集结论。

工件（仓库外）：
`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/p2_first_gate_a2_epoch63_final_v2/`。

## 结果

| 指标 | raw | refined | Δ refined−raw | 95% paired-image CI |
|---|---:|---:|---:|---:|
| coarse IoU | 0.834221 | 0.834275 | +0.000054 | [−0.000048, +0.000172] |
| coarse Dice | 0.909619 | 0.909651 | +0.000032 | [−0.000030, +0.000102] |
| support canvas IoU | 0.871106 | 0.871315 | +0.000209 | [+0.000123, +0.000298] |
| support canvas Dice | 0.931113 | 0.931233 | +0.000119 | [+0.000070, +0.000171] |

自然 coarse 平均绝对 residual 为 0.00748；生产 canvas support 内平均绝对变化
为 0.00674，19.82% support 像素连续值变化。P2 特征、residual-head 输入与输出均非零；
因此这不是 P2 未构建、feature 空置或 hook 自身的死通路。

## 裁决

第一门规则预注册为：**coarse IoU 与生产 support-canvas IoU 的 paired-image 95% CI
下界都必须 >0，且 canvas 连续值确实改变。**

**A2/R1 第一门失败。** canvas 端有极小但稳定的正向变化，说明 renderer/PE 输入并非
断路；但它没有来自 P2 的稳定 coarse-content 改善（coarse IoU CI 跨 0）。因此不能把
当前 A2 的 P2 作为“先改善 coarse、再改善 dense”的有效模块，也不应以它启动重训或进入
主消融矩阵。

这不是“P2 永远无效”的普遍否定：结论仅约束此 checkpoint、此 R1 objective、此
raw-only search support 和此数据协议。下一设计门应针对**如何让 P2 对 coarse 内容形成可
验证的纠错**，而不是再扩大 dense renderer 或再试 sparse-point 挖掘；冻结 A0 Oracle 已表明
现有 dense 消费端对更准 canvas 有可达空间。
