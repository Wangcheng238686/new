# MHBEV-ROI-20260914 — A1 hot-start implementation review

## A1-HotStart-S1 两 seed last 标准推理结果审计（2026-09-14）

**复核对象。**
`outputs/MHBEV-ROI-20260914/A1-HotStart-S1/s44_last_standard/` 与
`s45_last_standard/` 的 `metrics.json`、`run_manifest.json`，比较冻结单流锚点
`outputs/ue_single_c5v2_e86_val/metrics.json`。

**完整性与可比性：通过。** 两个 A1 checkpoint 均以 strict load、model weights、
UE val、1024²、batch=1、同一 bbox/segm `scores` 与 `roi_local_bbox_paste` 口径完成
标准推理；各自处理 **259 images / 5,937 GT**。A1 s44/s45 预测实例数为
6,224/6,171，基线为 6,223；这不是漏样。两套 manifest 也确认 checkpoint 是各 seed
的 `last_checkpoint.pth`（epoch 28），而非事后选择的 best checkpoint。

| metric (last standard inference) | s44 Δ vs baseline | s45 Δ vs baseline | two-seed mean Δ | screen requirement |
|---|---:|---:|---:|---:|
| segm/mAP_s | +0.001097 | −0.001243 | **−0.000073** | >= +0.015, same direction |
| bbox/mAP_s | +0.000589 | −0.004432 | **−0.001922** | >= 0, same direction |
| segm/mAP | −0.000163 | −0.000881 | −0.000522 | >= −0.002 |
| bbox/mAP | −0.000090 | +0.000414 | +0.000162 | >= −0.002 |

**Gate audit.** The global mAP guardrails happen to hold, but the primary
small-mask gate fails by approximately 0.01507 absolute mAP_s below its
required mean uplift; s44/s45 also have opposite directions. The small-bbox
mean is negative and directions likewise disagree. Therefore the A1→A2
screen gate fails on multiple independent pre-registered conditions.

The supplied last-standard artifacts do not themselves contain the per-epoch
`mAP_s`/`bbox mAP_s` trace needed to independently recompute the protocol’s
last-five-validation-epoch screening average. That is an archive omission to
record; it cannot rescue the arm: the mandated last-checkpoint standard
inference selection is already non-positive in mean and directionally
inconsistent, so there is no valid positive A1 result on which to advance.

**VERDICT: FAIL — 关闭 A1-HotStart-S1，停止 A2 与全量训练。** Archive both
checkpoints, manifests, prediction JSONs, this gate decision, and the missing
last-five trace as a negative result. Do not tune/restart A1 or implement A2
under `MHBEV-ROI-20260914`; any materially revised hypothesis requires a new
pre-registered frozen probe rather than post-hoc adjustment of the threshold
or checkpoint selection.

---

## v2 telemetry 最终复核（2026-09-14）

**复核对象。** `residual_abs_max`、`candidate_hard_mask_zero` 的新接线及报告的
v2 one-batch forward/backward/optimizer-step 工件。

**通过。** `A1P2GeometryResidual` 现在从真实 residual 计算
`residual_abs_max`；模型 monitor 如实映射它，而不再以 mean 伪充 max。Trainer 的
`_DUAL_STREAM_MONITOR_KEYS` 已加入 `candidate_hard_mask_zero`，其累积/FP64 DDP
reduce 与其他标量字段同一口径；epoch 日志也明确打印 `hardmask_zero`。这满足
candidate-valid、two-view 与 hard-mask 的同步遥测要求。

报告的 v2 one-batch 数值与实现和已冻结探针的几何量级相容：valid **83.15%**、
two-view **98.53%**、residual mean/max **0.008444/0.088501**、`gamma=0`、注入比
`0`、hard-mask **100%**。mean/max 不再相同，且 hard-mask 字段是 DDP epoch 可归档
字段；一 batch 已完成 forward/backward/optimizer step，无 metrics 声称。

**VERDICT: PASS — 可重启 seed 44、45 的 A1 热启动筛查。** 此项解除上一节仅关于
telemetry 保真的 HOLD。保持原有约束：两 seed 分开运行、30 epochs、last checkpoint
标准推理与 last-5 validation-epoch 门；P3/embedding 路径在 A1 中不存在，日志中的
相应缺失字段不能被解释为异常。该结论仍不授权全量训练。

---

## 首 batch 修复复核（2026-09-14）

**复核对象。** `A1P2GeometryResidual` 的 telemetry mask reshape 修复、
`A1P2RSPrompterAnchor.get_dual_stream_debug_stats()` 的旧 trainer schema 映射，及
报告的一 batch forward/backward/optimizer-step 无指标 smoke。

**已修复并证实的部分。** `support` 原先是 `[B,H*W]`，却被按可广播的二维 mask
使用；现在明确 reshape 为 `[B,1,H,W]` 再 `expand_as(residual)`，与 P2 residual
的 `[B,C,H,W]` 形状一致。该检查不会改变 candidate、delta 或梯度，仅修复 telemetry
读取路径。报告的一 batch 完成 forward、backward 和 optimizer step 也足以证实此前
首 batch runtime 崩溃已消除、零 gamma 的梯度链可执行。

**仍有两个阻塞的遥测保真问题。**

1. `get_dual_stream_debug_stats()` 把 `bev_abs_max` 赋值为
   `residual_abs_mean`。这会把日志字段 `bev[mean/max]` 伪报为相同数值，而不是
   “A1 P2 residual 的最大绝对值”。必须在 residual 模块中真实计算并输出
   `residual_abs_max`，再如实映射；不能以 mean 充当 max。
2. 模块已经计算 `candidate_hard_mask_zero`，但旧 schema 映射把该字段丢弃，trainer
   也不会在 epoch 遥测中归档它。Appendix A 要求 candidate-valid、two-view 和
   hard-mask telemetry 同步交付。需扩充 A1 专用/通用 monitor key 并做 rank-reduced
   记录，或提供同等的每 epoch 聚合字段；仅在模块的瞬态 `last_stats` 保存不足以
   形成筛查归档。

旧 schema 中 P3/embedding 的 `nan` 是 A1 不存在这些路径的预期缺失，不应伪造为
0；但 A1 实际存在的 P2 字段必须准确。当前映射确实使 valid、two-view、residual
mean、gamma_p2 与 P2 injected ratio 不再缺失，却尚未满足上述两个遥测契约。

**VERDICT: HOLD — 暂不重启两 seed 筛查。** 一 batch smoke 是必要但不足的运行
证据。先按既定字段语义修正 `residual_abs_max` 和 hard-mask 的 epoch/DDP 归档，并
重跑同一 batch smoke，确认没有 NaN/错误字段，再重新复核后启动 seed 44/45；这不
更改 A1 模型或筛查阈值。

---

## VERDICT: PASS — 可启动 A1 两 seed 热启动筛查；不授权全量训练

**Scope.** Independent review of `A1P2GeometryResidual`,
`A1P2RSPrompterAnchor`, `configs/ue_a1_p2_z0_hotstart.py`,
`--train-a1-only`/`--init-from` guards in `train_rsprompter_fusion.py`, and
`scripts/ue/run_ue_a1_p2_hotstart.sh`, against the pre-registered
`A1-HotStart-S1` protocol. The R2 frozen gate is already PASS.

## Findings

- **A1 scope is isolated.** `A1P2RSPrompterAnchor.extract_feat()` takes only
  the finest frozen UAV FPN and replaces only `x[0]` (P2). It does not touch
  P3, SAM image embedding, ROI retrieval, or mask prompt paths. The configured
  P2 is 256²; `GeometrySparseBEV._ground_points()` fixes world Z to zero and
  the hot-start config fixes the existing 360-m contract.
- **Geometry contract is preserved.** `A1P2GeometryResidual` uses W2C/K in a
  FP32, autocast-disabled projection block. Its validity rule is the deployed
  512-pixel in-frame condition; `support = valid.sum(1) >= 2` and the fused
  UAV feature is multiplied by that support before projection. Therefore every
  `<2-view`, including all-invalid, P2 residual position is exactly zero.
- **Anchor/UAV freezing is effective.** At setup, `--train-a1-only` executes
  `model.requires_grad_(False)` then re-enables only `a1_p2`. Before every
  training epoch it executes `core.eval(); core.a1_p2.train();
  core.drone_encoder.eval()`, so frozen anchor modules remain eval as well as
  non-trainable, and the UAV encoder is both `requires_grad_(False)` and
  called under `torch.no_grad()`.
- **Zero-gate learning is viable.** `a1_p2.gamma` is initialized to zero while
  the 256→512 `a1_p2.proj` is conventionally random-initialized. Thus step zero
  is exactly anchor-equivalent, while `dL/dgamma` sees a nonzero projected
  residual; `proj` receives gradients after gamma departs zero. This avoids the
  prohibited serial all-zero initialization.
- **Checkpoint safety guard is adequate.** Cross-architecture init is explicit.
  In A1 mode the loader raises on any source `unexpected` or `shape_mismatch`,
  and rejects a target missing key unless it is under `drone_encoder.*` or
  `a1_p2.*`. This prevents silently changing anchor weights.
- **Harness evidence.** The supplied harness facts—`loaded=679`,
  `unexpected=0`, `shape_mismatch=0`, and exactly two trainable tensors under
  `a1_p2`—are consistent with the code and the pre-registration. The two
  tensors are the projection weight and scalar gamma; the optimizer’s name
  routing puts those and only those trainable tensors in `drone_guidance`.
  Preserve the concrete harness log path in each run’s archive/manifest; it
  was not present as a separately named file under the reviewed output tree.

## Launch constraints

- Invoke the wrapper once per pre-registered seed: `A1_SEED=44` and
  `A1_SEED=45`, using the fixed single-stream anchor via `--init-from`.
- Keep the wrapper’s fixed screen protocol: 30 epochs, 128 train scenes,
  complete 259-scene validation, batch 1/accumulation 8, and evaluate the
  **last** checkpoint for each seed.
- Record candidate-valid/two-view/hard-mask, gamma and residual/base-norm, and
  small proposal/recall/miss-merge telemetry. Stop and archive on NaN, a stuck
  zero gate, calibration mismatch, or a sham anomaly as required by Appendix A.
- After both runs, apply the pre-registered last-five-validation-epoch two-seed
  screen gate. Do not begin A2 or any full training unless that gate passes;
  full training additionally needs the separate commit/EXP-ID/full-arm harness
  authorization.
