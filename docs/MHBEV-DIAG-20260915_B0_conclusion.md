# MHBEV-DIAG-20260915 — B0 frozen small-failure diagnosis

**状态：PASS（诊断有效）；正式结论：AMBIGUOUS；不授权任何训练或 A2/A3。**

## 口径

本冻结、只读诊断以单流锚点
`ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth` 扫描官方 UE
val 的 259 scenes / 5,937 GT instances。small 定义为同模型 1024-raster GT mask
面积 `<32²`。RPN proposal 来自与部署预测同源的
`extract_feat → rpn_head.predict(rescale=False)`，未写入 data sample、未修改 ROI；
最终 failure 则来自正常 `model.predict` 的 mask Hungarian matching（IoU 0.5）。

预注册主指标为 final-failure small 的 RPN recall@0.5。分流规则原文：

- `recall@.5 < .50`、与 matched-good small 相差至少 `.15`，且 image-paired
  bootstrap 差值 95% CI 上界 `<0`：才可优先 proposal/objectness；
- `recall@.5 >= .75`：才可优先 ROI / instance separation / mask；
- 其余为 `AMBIGUOUS`，不得据此启动训练。

## 完整性与不变量

| 项目 | 结果 |
|---|---:|
| scenes / GT instances | 259 / 5,937（完整） |
| 正常预测 → RPN 只读 → 正常预测的逐字段 hash | 259 / 259 相同 |
| 重复 RPN readout boxes/scores hash | 259 / 259 相同 |
| 每景 RPN proposals | 1,000；未受 geometry sham 影响 |
| bootstrap | 10,000 image-paired draws，seed 20260915，10,000 有效 |

## 结果

small 共 332 个：177 missed、50 partial-or-merge、66 matched-boundary-error、39
matched-good。预注册的 final-failure small 集合是 227（177+50）。

| RPN proposal recall | 值 |
|---|---:|
| small 全集 @ IoU .3 | 0.7651 |
| small 全集 @ IoU .5 | 0.6717 |
| final-failure small @ IoU .5 | **0.5286**（120 / 227） |
| matched-good small @ IoU .5 | **1.0000**（39 / 39） |
| failure − good @ IoU .5 bootstrap 95% CI | **[-0.5402, -0.4055]** |

在 227 个最终失败的小目标中，107 个（47.1%）没有 IoU≥0.5 的 RPN proposal，
而 120 个（52.9%）已有 proposal 却仍在最终结果失败。这证明 proposal 召回和
ROI/mask 质量两个瓶颈都存在；但 failure recall `0.5286` 既不低于预注册 proposal
门的 `0.50`，也未到 ROI/mask 门的 `0.75`。

失败 small GT center 的 ≥2-view UAV 几何可查询率为 correct **202/227 = 88.99%**、
`XY+80m` **152/227 = 66.96%**、scene-permuted **172/227 = 75.77%**。这是几何
可查询性与 sham 对照证据，**不能**解释为 UAV 内容有语义增益。

## 落判与下一步

**AMBIGUOUS 是唯一符合预注册规则的结论。** Bootstrap 的显著负差仅证明 final-failure
small 比 matched-good small 更难被 RPN 覆盖；它不能替代 `.50/.75` 的预注册门，不能
事后把本实验称为 proposal 分支胜出。

因此：

1. 保持 A1 失败后的停线：不启动 A2/A3，也不启动任何 proposal 或 ROI/mask 训练；
2. 未来若继续，须另立 B1 冻结探针，针对已有 RPN proposal 的 120 个 failure small
   再分解 proposal rank/score、ROI 分类、box refinement 与 mask separation，或以
   对称的 oracle intervention 比较两条路径；预先给出新的决策门；
3. B0 不能支持“UAV 无效”或“UAV 已证明改善某一路径”的表述。

## 归档

- [full summary](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B0-fullval/summary.json)
- [per-instance CSV](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B0-fullval/instances.csv)
- [bootstrap artifact](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B0-fullval/bootstrap.json)
- [run log](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B0-fullval/run.log)
- [independent review](reviews/MHBEV-DIAG-20260915_B0_rpn_probe_review.md)
