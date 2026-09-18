# MHBEV-DIAG-20260915 — B1 frozen post-RPN diagnosis

**状态：PASS（诊断有效）；正式结论：POST_RPN_ROI_BOX；不授权训练。**

## 预注册口径

B1 继承 B0 的同一单流 checkpoint、UE val 与 1024² 部署预测，只读取 B0 固定的
120 个 target：small、final `missed/partial_or_merge`，且 RPN 有 IoU≥0.5 proposal。
每个 target 所在场景运行两次正常 `model.predict(rescale=False)`；不注入 UAV、sham
或 proposals，输出 bbox/scores/labels/masks 均作 typed hash parity。

对每个 target：

- `post_rpn_box_lost_or_mislocalized`：没有任一最终 bbox 与 GT 的 IoU≥0.5；
- `mask_only_failure`：有最终 bbox IoU≥0.5，但任一最终 mask 对 GT 的 IoU 都 `<.5`。

判定门为：box-lost 比例 `>=.60` 且 image-paired bootstrap（完整 259 scene universe，
10,000 次、seed 20260916）的 `mask_only - box_lost` 95% CI 上界 `<0`，才可判定
`POST_RPN_ROI_BOX`。该名称涵盖 ROI 分类抑制、NMS 与 box refinement 偏移，不能狭义
归因于“分类错误”。

## 完整性与不变量

| 项目 | 结果 |
|---|---:|
| B0 lineage | checkpoint、root/split/1024²、259/5,937 与 B0 invariants 全部一致 |
| target scenes / targets | 94 / 120（完整） |
| B0 target 身份与 B1 重算 final failure | 120 / 120 一致 |
| 两次部署预测逐字段 typed hash | 94 / 94 一致 |
| bootstrap scene universe / 有效 draws | 259 / 10,000 |
| detector 的 UAV 或 sham 干预 | 无 |

## 结果与落判

| B1 outcome | target 数 | 比例 |
|---|---:|---:|
| post-RPN box lost / mislocalized | 101 | **84.17%** |
| box reached but mask-only failure | 19 | 15.83% |

`mask_only - box_lost` 的 bootstrap 差为 `-0.68587`，95% CI 为
`[-0.81482, -0.55556]`。因此 box-lost 比例超过 `.60`，CI 上界也严格低于零，
满足预注册的 **POST_RPN_ROI_BOX** 门。

按类别继承的 B0 correct-geometry ≥2-view 覆盖：box-lost 为 88/101（87.13%），
mask-only 为 19/19（100%）；对应的 XY+80m 与 scene-permuted 数字均归档于
`summary.json`。这些只是 UAV 几何可查询性，不能证明 UAV 内容或任何未来模块会改善
post-RPN 路径。

## 可信解释与边界

B0 表明小目标问题一半发生在 RPN 前；B1 进一步表明“RPN 已到达”的另一半中，主要
不是纯 mask 轮廓质量，而是最终 box 未到达/被抑制/偏移的 post-RPN 失败。因此，若另立
下一模型假设，最有证据支持的目标是 **UAV 辅助 ROI classification / box refinement /
proposal-to-final 保留**，而不是先做 ROI-local mask-only BEV，也不是回到全局 P2
proposal 注入。

这仍不是 UAV 增益的因果证据，且 B1 只是一条冻结诊断：不得据此训练。任何后续模型
臂必须另立项、先做与 post-RPN path 对齐的冻结干预探针，包含 correct/geometry-sham/
scene-permuted UAV 对照、zero-gate parity 以及单独的热启动门。

## 归档

- [full summary](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B1-fullval/summary.json)
- [targets CSV](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B1-fullval/targets.csv)
- [bootstrap artifact](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B1-fullval/bootstrap.json)
- [run log](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B1-fullval/run.log)
- [independent review](reviews/MHBEV-DIAG-20260915_B1_postrpn_probe_review.md)
