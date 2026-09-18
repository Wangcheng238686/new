# MHBEV-DIAG-20260915-B3 — frozen UAV ROI-score readout conclusion

**状态：完整性 PASS；正式结论：`NO_FROZEN_SIGNAL`；关闭 B3。**

## 问题与预注册门

B2 已在同一单流 checkpoint 上定位 87 个 `roi_score_rejected` 小目标：decoded
pre-NMS 候选框可以与 GT 对齐，但前景分数未过部署阈值 `.05`。B3 不训练、也不修改
detector 或 frozen UAV encoder；它只检验：正确几何采样的 UAV 局部 descriptor 是否能让
固定线性 readout 在 score-matched positive / hard-negative pair 上优于两个阴性对照。

固定门来自 `EXPERIMENT_GOVERNANCE.md` 附录 E：完整 128-scene train、87 target / 259-val
评估下，三条件共同有效 coverage `>= .80`、correct win rate `>= .60`，且
`correct - XY+80m` 与 `correct - scene-permuted` 的 image-paired bootstrap 95% CI
下界均 `> +.05`，才可判 `FROZEN_UAV_ROI_SCORE_SIGNAL`。

## 完整性与不变量

| 项目 | 结果 |
|---|---:|
| frozen readout train scenes / pairs | 128 / 127 |
| B2 ROI-score target scenes / targets | 74 / 87 |
| val target ledger | 45 included / 41 descriptor-invalid / 1 score-match rejected |
| detector typed-hash parity | PASS |
| UAV encoder parameter hash | PASS |
| B2 P* lineage、pair identity、split、no-intervention | PASS |
| E3 fixed 41-column per-pair telemetry | PASS |
| bootstrap universe / draws | 259 scenes / 10,000 (seed 20260918) |

所有完整性、不变量及 B2 谱系条件均成立；本结果不是 `INVALID`。

## 结果与落判

| 指标 | 结果 |
|---|---:|
| 三条件共同有效 coverage | **45 / 87 = 0.51724** |
| correct win rate | 29 / 45 = 0.64444 |
| XY+80m win rate | 27 / 45 = 0.60000 |
| scene-permuted win rate | 25 / 45 = 0.55556 |
| correct − XY+80m 95% CI | `[-0.14286, 0.23810]` |
| correct − scene-permuted 95% CI | `[-0.10204, 0.26190]` |

因此 coverage 未达到 `.80`，且两条 correct-sham CI 都跨过零、下界均不大于 `+.05`。
尽管 correct 的条件内 win rate 是 0.64444，单独这一项不能越过预注册门，合法结论只能是
**`NO_FROZEN_SIGNAL`**。

## 能证与不能证

能证：在本次严格冻结、score-matched、三条件共同支持的 B2 ROI-score target 切片中，
当前 UAV ROI descriptor 没有提供足以区别于几何偏移和 scene-permuted sham 的可靠信号。

不能证：这不等于 UAV 对所有小目标任务无用，也不等于端到端模型绝无可能从 UAV 获益。
尤其不能把它外推为 bbox/segm AP 的负结论。B3 的覆盖短板也表明：当前 z=0 的 3×3
ROI 查询在该 failure slice 上不具备充分共同几何支持。

## 准入与后续

**禁止**基于 B3 启动 ROI zero-gate intervention、ROI 热启动或任何 ROI-UAV detector
训练；不得通过调 C、FPN 层、网格、阈值或选择更有利子集“挽救”B3。

若继续寻求 UAV 对小目标的增益，必须另立项并预注册新假设。现有失败树中独立的
107 / 227 final-failure small 在 RPN 阶段没有 IoU>=.5 proposal，故下一条可审计路线应是
**UAV 对小目标 proposal recall 的单独冻结诊断**，而非把 B3 的 ROI-score 负结果改写为
召回正证据。

## 工件

- [full summary](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B3-fullval/summary.json)
- [per-pair telemetry](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B3-fullval/val_pairs.csv)
- [bootstrap artifact](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B3-fullval/bootstrap.json)
- [run log](../portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B3-fullval/run.log)
- [independent implementation and result review](reviews/MHBEV-DIAG-20260915_B3_uav_roi_score_readout_review.md)
