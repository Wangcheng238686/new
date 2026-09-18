# MHBEV-DIAG-20260915-C1 — frozen UAV dense-RPN preselection conclusion

**状态：完整性 PASS；正式结论：`NO_FROZEN_SIGNAL`；关闭 C1。**

## 问题与预注册门

C1 独立于已关闭的 B3 ROI-score 路径。它针对 B0 固定的 107 个 small final-failure
RPN-miss target：部署 post-NMS proposals 没有 IoU>=.5 框。C1 不训练或修改 detector/RPN/UAV
encoder；仅从冻结 RPN 的 dense、pre-score-filter/pre-`nms_pre`/pre-NMS decoded anchors 中，
取与 GT 最对齐的 raw candidate（要求 IoU>=.5），与 raw objectness 匹配的真背景 candidate
比较 frozen UAV descriptor 的线性 readout 排序。

预注册门见 `EXPERIMENT_GOVERNANCE.md` 附录 F：在完整 128-scene train、107 target/259-val
评估中，共同有效 coverage `>=.80`、correct win `>=.60`，且 correct 相对 `XY+80m` 和
scene-permuted 的 image-paired bootstrap 95% CI 下界均 `>+.05`，才可判
`FROZEN_UAV_RPN_PRESELECTION_SIGNAL`。

## 完整性与不变量

| 项目 | 结果 |
|---|---:|
| frozen readout train scenes / pairs | 128 / 105 positive + 105 negative |
| B0 RPN-miss target scenes / targets | 85 / 107 |
| dense layout records | 1,065 = 213 scenes × 5 FPN levels |
| val ledger | 44 included / 40 descriptor-invalid / 23 raw-geometry-unavailable |
| detector typed-hash parity | PASS |
| post-NMS RPN boxes/scores pre/post hash | PASS |
| dense anchor/delta/score/decode cardinality | PASS |
| B0 target lineage / replay, condition pair SHA, detector/UAV hashes | PASS |
| fixed per-pair telemetry | 44 columns, PASS |
| bootstrap universe / draws | 259 scenes / 10,000 (seed 20260919) |

所有完整性与不变量成立；本结果不是 `INVALID`。B0 107 个 target、其 final-failure
类别及 post-NMS RPN IoU 被逐条精确重放；最高 replay IoU 为 `0.4932717`，仍严格低于
部署 `.5` 判定线。

## 结果与落判

| 指标 | 结果 |
|---|---:|
| dense raw geometry available | 84 / 107 = 0.78505 |
| score-match eligible | 84 / 107 = 0.78505 |
| 三条件共同有效 coverage | **44 / 107 = 0.41121** |
| correct win rate | 25 / 44 = 0.56818 |
| XY+80m win rate | 21 / 44 = 0.47727 |
| scene-permuted win rate | 24 / 44 = 0.54545 |
| correct − XY+80m 95% CI | `[-0.11364, 0.28205]` |
| correct − scene-permuted 95% CI | `[-0.15385, 0.18919]` |

共同有效 coverage 未达到 `.80`；correct win 也未达到 `.60`；两个 correct-sham CI 均跨过
零。因此 C1 的唯一合法落判是 **`NO_FROZEN_SIGNAL`**。

## 能证与不能证

能证：在当前 frozen UAV SAM2、z=0、3×3 decoded dense-anchor 查询及 B0 RPN-miss slice 上，
没有证据表明 UAV 内容可稳定、特异地将潜在 raw RPN 正 candidate 排在 objectness-matched
hard negative 之前。23 / 107 target 在 dense raw decode 中都不存在 IoU>=.5 candidate，
另有 40 / 107 无三条件共同 descriptor，二者均被保留在固定分母而未静默删除。

不能证：这不是端到端 proposal recall、bbox/segm AP 的负结果，也不等于 UAV 对所有小目标
任务无价值。它只否定本轮具体的 dense-RPN preselection scoring 假设；不能把 raw dense
candidate 的 oracle readout 误写成真实 RPN recall 改善。

## 准入与后续

**禁止**根据 C1 启动 UAV→P2/RPN zero-gate intervention、热启动或训练；不得通过调整
readout C、FPN 层、网格、阈值、几何高度或 target 子集挽救该预注册臂。

B3（ROI score）与 C1（dense RPN preselection）均已以正确/两种 sham 的完整冻结证据
关账为负。在不另立、且有不同可证伪机制与预注册门的新假设前，不应继续把当前 UAV
SAM2 z=0 局部查询作为小目标增益路径推进。

## 工件

- [full summary](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/C1-fullval/summary.json)
- [per-pair telemetry](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/C1-fullval/val_pairs.csv)
- [dense layout audit](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/C1-fullval/dense_layout.csv)
- [bootstrap artifact](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/C1-fullval/bootstrap.json)
- [run log](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/C1-fullval/run.log)
- [independent implementation and result review](reviews/MHBEV-DIAG-20260915_C1_rpn_preselection_result_review.md)
