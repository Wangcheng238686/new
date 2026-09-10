# A0 提示几何解耦审计

2026-09-09，冻结 NWPU A0 epoch36，完整 validation-130；不训练、不改生产模型。

## 审计协议

1. 从既有 raw GT/DT 记录统计框外前景、最终 mask 的框外恢复、mask IoU，按框
   覆盖/类别/尺寸分组；同类 bbox IoU>=0.5 匹配，proposal 可重复匹配同一 GT。
2. B：只把 PromptEncoder 的 box token 输入换成匹配 GT 框；检测输出、ROI、
   点和 dense canvas 全不变。这是受限 Oracle，不等同重训收益。
3. S：只把编码后 dense 支持域的框中心固定、宽高乘1.1。只干预匹配行，
   未匹配原样保留；不产生新 canvas 内容，不改变 PE 输入。
4. 全部臂复用 standard/canonical/identity 校验，detector hash 恒等；保存实际
   干预记录及 points/dense/pre-support embedding 不变量供独立审核。
5. 500次图像配对 bootstrap（seed44、10类）评估全量 mAP，配合机制统计。
   此验证集已用于选权重/开发，不能声称独立测试或跨种子稳定性。

R（GT ROI 采样及重建提示）暂不自动执行：只有 B/S 不足以解释结果且统计支持
几何相关误差时才继续评估其必要性；R 是内容与映射联合干预，不是单变量证明。

## 当前状态

统计与 B/S 完整运行已完成；代码/全量records/RLE独立审核通过。保存的500个
bootstrap原始差值由主代理逐项复算一致；原独立审查者在最后CI复算前额度耗尽，
故此项不标为“独立CI审核通过”。

### 全量结果

工件：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/a0_geometry_full_20260909/`。

| 臂 | segm/mAP | Δ vs raw |
|---|---:|---:|
| raw | 0.6669291021 | 0 |
| B / box_gt | 0.6698729894 | +0.0029438873 |
| S / support_expand | 0.6652999180 | -0.0016291841 |

标准输出精确复现canonical、strict load零缺失、raw hook恒等。三臂918 proposals /
728 matched、69 PE/support调用一致；B改728个PE框且support不变；S改728个支持框、
其中523个支持网格变化，PE框不变。points/canvas/pre-support dense/ROI框hash全部
符合预注册不变量。逐records复查非mask字段全相等，未匹配mask逐位相等；B/S分别
705/514 masks改变。manifest代码SHA与当前源码相等。两个500次bootstrap均完成：
B Δ95% CI [-0.001200526,+0.006535647]；S [-0.004550307,+0.000987256]。
均为130图10类、seed44，bootstrap点估计精确复现推理。

独立子代理已完成代码、CPU smoke和完整records审核，未见阻断。
首次smoke因误拒raw的全true validity失败；守卫已修为仅拒非全true，新目录v2
smoke通过后才跑本次full，原失败输出未覆盖。

独立审查者随后直接解码RLE完成逐记录复算，保存于上述根目录
`independent_mask_audit.json`：728匹配proposal的平均maskIoU raw/B/S
=0.804324/0.808147/0.803856；以原检测框为统一边界，框外GT前景181958像素，
恢复99498/108664/101963（54.6818%/59.7193%/56.0366%）；总漏分
228161/208833/223396。S的局部漏分减少并没有转化成整体IoU/mAP收益。
独立验证4臂GT/images相等、standard=raw、918非mask字段相等、190未匹配记录
全相等，且B的GT框、S的扩张公式逐行正确。主代理独立于输出payload重算两个
500元素`deltas`的均值/2.5、97.5分位数/正差比例，精确复现文件的CI和点估计；
独立审查者的该最后步骤未返回，原因见上。

## 裁决与下一步

两臂未确立全量mAP收益，当前**不直接启动提示几何模块训练**。
B对maskIoU和框外恢复的正向点估计是继续探索box-token校准的线索，不是总体
统计显著收益；S的点估计下降也不能称显著恶化。不能把两项CI跨零当作等效性证明。

R未执行，不据此否定重新采样ROI的联合路线。当前先将B保留为候选信息，S固定
1.1扩张不晋级；如果继续投入，优先解释B的实例收益/损害分布并在另一个预先固定
权重上复核，而不是立刻扩大为ROI+points+dense一起改的R或启动长训。
单checkpoint、多轮开发验证集、matched-only干预、GT不可部署等限制始终保留。

独立统计工件：`/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/a0_geometry_stats_20260909.json`。
728 matched proposals /717 unique GT；平均 GT 框外比例3.388%，框外前景 pooled
恢复率54.68%，框外漏分仅占全部漏分36.14%。框外比例>10%的62个proposal
平均maskIoU0.7143，1–5%组336个的均值0.8265；相关系数-0.2117，分组非单调。
每GT仅最高分预测敏感性：框外3.174%、恢复52.42%。仅支持做因果干预，不能
把框外GT直接算成模型无法预测的像素，也不据此宣布模块有收益。

## 已发现的解释风险

工作区 `box_prompt_dependence_vs_contribution.md` 把 mask ROI crop 写作 decoder
image embedding，与当前 full-image 架构不符：crop 进入 coarse，decoder 则接收
SAM2 原生全图 embedding/high-res。推理 prompt 框来自 predict_bbox 后的检测结果，
不是未经回归的 RPN proposal。不能以该文档的双路径机制解释支撑本审计。
训练 sampled proposal 与推理回归框存在差异，但它本身不是已证实的 bug。
