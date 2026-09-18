# MHBEV-DIAG-20260918-D2 — frozen RPN rank-profile full-result audit

**Review scope (independent, read-only; 2026-09-18):** Appendix H of
`EXPERIMENT_GOVERNANCE.md`; the current D2 implementation; B0 and D1 fullval
lineage; and all artifacts under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260918/D2-fullval`.
No code, model, data, or experiment artifact was changed by this audit.

**VERDICT: PASS (result audit); registered result:
`D2_BROAD_RANK_DEFICIT`.**

This is a credible negative for the registered per-level proposal-cap boundary
hypothesis. It does not authorize candidate-cap configuration validation,
intervention, hot start, detector training, UAV work, or an AP/compute claim.

## Completeness and provenance

- `exit_code` is `0`; the log terminates with a full summary; manifest,
  summary, and standalone bootstrap artifact agree on the registered arm and
  decision.
- The full run covers all 85 D1 target-bearing validation scenes, all 107 B0
  targets, and all 59 D1 score/`nms_pre` main targets. The unique D2 target
  key set is exactly equal to both D1's 107 target keys and the B0 small,
  failed-final, post-NMS-RPN-miss filter.
- For every target, B0, D1, and D2 deployed-final replay IoUs agree at the
  registered `1e-5` tolerance; D1 and D2 raw/pre-NMS/unlimited/final values
  also agree, and D2's recomputed stage category equals D1's archived
  category. Thus no lineage or category drift is hidden in the rank profile.

## Invariants and controls

All six reported typed/read-only invariants pass across 85 telemetry records:
normal-prediction parity, deployed-RPN parity, staged-final/live equality,
repeat feature/logit/anchor/decoded-rank hashes, fail-closed D1 rank/stage
replay, and no proposal write. The asserted live RPN contract is exactly
`nms_pre=1000`, `max_per_img=1000`, `min_bbox_size=0`, score threshold `0`,
NMS type `nms`, IoU `.7`.

The negative control passes `23/23`: every raw-geometry target has no raw
IoU≥.5 candidate/rank. The selected positive separation control passes
`25/25`: every D1 NMS/max-per-image target has minimum positive rank at most
1,000. These controls confirm that the rank readout separates the registered
stage populations rather than simply returning generic large ranks.

## Independent profile and bootstrap recomputation

| Main-set outcome (59 targets) | Recomputed value |
|---|---:|
| recovered by cap 1,250 | 10 / 59 = 0.16949 |
| recovered by cap 1,500 | 14 / 59 = 0.23729 |
| recovered by cap 2,000 | 20 / 59 = 0.33898 |
| recovered by cap 4,000 | 28 / 59 = 0.47458 |
| minimum-positive-rank median | 4,163 |
| minimum-positive-rank p90 | 31,278.4 |
| positive-bearing levels | L0: 55; L1: 28; L2: 8; L3/L4: 0 |

Level counts are not mutually exclusive because a target can have aligned raw
candidates on multiple levels. Reconstructed 85-scene paired bootstrap inputs
sum to 20 cap-2,000 recoveries and 59 main targets. Re-running the fixed
10,000-draw NumPy bootstrap with seed `20260921` gives exactly the archived
CI `[0.22033051, 0.46000000]`, with zero redraws for an all-zero denominator.

## Gate application and authorization

Appendix H permits `D2_BOUNDARY_RECOVERABLE` only if the complete/invariant
prerequisites pass, cap-2,000 recovery is at least `.70`, and its paired CI
lower bound is at least `.50`. Here recovery is `.33898` and CI lower bound is
`.22033`; both fail. `D2_BROAD_RANK_DEFICIT` is therefore the only legal
decision.

The conclusion is narrow but decisive for the next decision: most of the 59
correct raw anchors are not just near the per-level top-1,000 cutoff; their
rank distribution is broad and often far below it. Although cap 4,000 has a
higher descriptive recovery rate, it is an accompanying profile metric and
cannot be substituted for the preregistered cap-2,000 gate.

Accordingly, **do not run a candidate-cap config-only proposal-recall
validation, do not change deployed `nms_pre`, and do not train any detector or
UAV branch on the strength of D2.** Appendix H directs the next work back to a
separately registered frozen satellite-side anchor-geometry or objectness
hypothesis. That future probe must use a different falsifiable mechanism and
new gates; it cannot tune cap values to rescue D2.
