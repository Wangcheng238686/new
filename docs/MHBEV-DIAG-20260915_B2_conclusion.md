# MHBEV-DIAG-20260915 — B2 frozen post-RPN mechanism diagnosis

**状态：PASS（诊断有效）；正式结论：ROI_SCORE；不授权训练。**

## 预注册口径

B2 继承 B1 固定的 101 个 `post_rpn_box_lost_or_mislocalized` small target，跨 84 个
UE val scene。对每个 target scene，先后两次运行部署正常 `model.predict(rescale=False)`，
中间在新 data sample 上只读复现部署顺序：原始 FPN 供 RPN，独立的
`extra_pe(FPN)` 供 ROI bbox forward，随后只读 decode、未运行 NMS。

实时锁定的部署 bbox 合同为 `Shared2FCBBoxHead + DeltaXYWHBBoxCoder`；ROI test config
为 `score_thr=.05`、NMS IoU `.5`、`max_per_img=100`。P\* 是 1,000 个 RPN proposal
中与 GT IoU 最大者，且必须 `>=.5`。

三类互斥且穷尽：

1. `bbox_refinement_lost`：没有 decoded pre-NMS box 达 GT IoU `.5`；
2. `roi_score_rejected`：存在 decoded 对齐框，但全部对齐框 foreground score `<=.05`；
3. `nms_or_rank_loss`：存在 decoded 对齐框且至少一个 score `>.05`，但最终框仍未到达。

门：最大类别须 `>=.60`，且其比例减其余两类比例的完整 259-scene image-paired
bootstrap（10,000 次、seed 20260917）95% CI 下界 `>0`。

## 完整性与不变量

| 项目 | 结果 |
|---|---:|
| target scenes / targets | 84 / 101（完整） |
| B1 target 身份、B1 final failure | 101 / 101 一致 |
| P\* pre-RPN IoU `>=.5` | 101 / 101 |
| 两次正常预测逐字段 typed hash | 84 / 84 一致 |
| RPN proposals / scene | 1,000 |
| bootstrap universe / 有效 draws | 259 scenes / 10,000 |
| UAV 或 sham 注入 detector | 无 |

## 结果与落判

| B2 mechanism | target 数 | 比例 |
|---|---:|---:|
| ROI score rejected | 87 | **86.14%** |
| bbox refinement lost | 13 | 12.87% |
| NMS/rank loss | 1 | 0.99% |

P\* 的平均 pre-RPN IoU 为 `0.6840`，decoded 后为 `0.6154`；所有 decoded box 的最大
IoU 平均为 `0.6701`。这说明在绝大多数 target 中，ROI 前仍存在几何上可对齐的候选，
并非主要由 box refinement 推离 GT。87 个 ROI-score-rejected target 的对齐 decoded
box 均未超过部署前景分数阈值 `.05`，于是不会进入最终检测结果。

`ROI_SCORE − 其余两类` bootstrap 差为 `0.72198`，95% CI
`[0.57009, 0.85841]`；ROI-score 比例超过 `.60` 且 CI 下界大于零。因此
**ROI_SCORE 是预注册门下唯一合法结论。**

## 可信解释与边界

当前小目标故障链条可以被精确表述为：

```text
227 个 final-failure small
├─ 107：没有 IoU≥.5 RPN proposal
└─ 120：已有 proposal
   ├─ 19：框到达但 mask-only failure
   └─ 101：最终框未到达
      ├─ 87：ROI score 未过 .05  ← 当前主要、最可操作的断点
      ├─ 13：refinement 后无对齐框
      └─ 1：NMS/rank loss
```

这不是 UAV 增益证据。它只说明：如果另立下一条模型假设，最有证据支持的目标应是
**候选级、ROI-local 的前景评分/保留**，而不是重启失败的全局 P2 residual、优先做
mask-only BEV、或把问题误说成 NMS/box refinement。后续仍须先做新的 ROI-score-specific
冻结干预探针，含 zero-gate parity 与 correct/geometry-sham/scene-permuted UAV 对照；
不得直接热启动或全量训练。

## 归档

- [full summary](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B2-fullval/summary.json)
- [targets CSV](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B2-fullval/targets.csv)
- [bootstrap artifact](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B2-fullval/bootstrap.json)
- [run log](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B2-fullval/run.log)
- [independent review](reviews/MHBEV-DIAG-20260915_B2_postrpn_mechanism_review.md)
