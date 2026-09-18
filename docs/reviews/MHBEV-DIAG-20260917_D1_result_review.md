# MHBEV-DIAG-20260917-D1 — RPN stage-attribution full-result audit

**Review scope (independent, read-only; 2026-09-17):** Appendix G of
`EXPERIMENT_GOVERNANCE.md`; the current D1 probe implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_rpn_stage_attribution.py`;
B0 full-validation lineage; and all D1-fullval artifacts under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/D1-fullval`.
No code, model, data, or experimental artifact was modified by this audit.

**VERDICT: PASS (result audit); registered result: `D1_MIXED`.**

The full run is complete and its mixed conclusion is supported by independently
recomputed lineage, category, invariant, and bootstrap checks. This PASS
certifies the diagnostic result only; it does not authorize an intervention,
hot start, detector training, or an AP/proposal-recall claim.

## Completeness, lineage, and stage contracts

- `exit_code` is `0`, the log ends with a written full summary, and the run
  manifest agrees with `decision=D1_MIXED`.
- The run covers all 259 validation scenes in the bootstrap universe, all 85
  target-bearing scenes, and all 107 expected B0 targets. `targets.csv` has
  exactly 107 unique `(scene_id, gt_index)` records. Independently reapplying
  the B0 filter (`small`, failed final state, no post-NMS RPN IoU `.5` recall)
  yields exactly the same key set.
- Every target's archived deployed replay IoU equals B0
  `max_rpn_box_iou` exactly at stored float precision (maximum absolute
  difference `0.0`); every staged-final target IoU also equals that replay
  exactly. Thus B0 lineage and deployed-final replay did not drift.
- `dense_layout.csv` has 425 rows = 85 scenes × 5 levels, covers levels 0--4,
  and every row has equal prior/delta/score/decoded cardinality. The resolved
  live contract is the registered one: `nms_pre=1000`, `max_per_img=1000`,
  `min_bbox_size=0`, resolved RPN threshold `0`, NMS type `nms`, IoU `.7`.

## Invariants and telemetry

All nine summary invariants are true: normal-prediction parity, deployed-RPN
pre/post parity, staged-final equals live RPN, raw/pre-NMS/final repeats,
dense cardinality, fail-closed B0 lineage, and no proposal write.

The 85 per-scene parity records, 85 repeat records, and 85 typed stage records
are all present and internally passing. They archive feature and per-level
anchor digests, raw/pre-NMS/NMS-unlimited/final hashes, stage configurations,
and the unlimited cap. In every scene the unlimited cap is non-binding; the
smallest `(bound - survivor_count)` is 2,650. This substantiates that the
unlimited cell is not accidentally max-per-image truncated. Stage maximum
IoUs are nested for all 107 targets and every target's category agrees with
the four Appendix-G mutually exclusive rules.

## Category result and bootstrap recomputation

| Stage | Count | Fraction |
|---|---:|---:|
| raw geometry unavailable | 23 | 0.21495 |
| score / `nms_pre` rejected | 59 | 0.55140 |
| NMS suppressed | 2 | 0.01869 |
| `max_per_img` truncated | 23 | 0.21495 |

The counts sum to 107. The companion alignment rates recompute to raw
`84/107 = .78505`, pre-NMS `25/107 = .23364`, and NMS-unlimited `23/107 =
.21495`.

I reconstructed the full 259-scene, zero-filled table from `summary.json` and
independently reran the prescribed 10,000 image-paired bootstrap with NumPy
seed `20260920`. It exactly reproduces the archived observed major category
`score_nms_pre_rejected` and major-minus-rest 95% percentile CI
`[-0.09090909, 0.29204221]` (all 10,000 draws valid).

## Gate application and interpretation

Appendix G allows a unique `D1_<STAGE>` conclusion only when the largest class
is at least `.60` and its fixed-major minus all other stages bootstrap CI lower
bound is strictly positive, after completeness/invariants pass. Although the
largest class is score/`nms_pre` rejection, it is `59/107 = .55140` and its CI
lower bound is `-.09091`. It fails both signal-facing gates. `D1_MIXED` is
therefore the only legal conclusion; this is not `INVALID`, because all
completeness and invariants pass.

The credible conclusion is a distributional one: this B0 small-object RPN
failure slice is split between raw-geometry absence, per-level score/
`nms_pre` loss, and final 1,000-proposal truncation, with only two NMS losses.
It does **not** establish that improving any one component will improve
proposal recall or AP.

## Authorization boundary and next diagnostic

Because D1 did not produce a unique `D1_<STAGE>` result, it authorizes **no
satellite RPN intervention, hot start, or training**. In particular, the
largest 59-target score/`nms_pre` cell cannot be promoted to an intervention
target simply because it is numerically largest.

The appropriate next step is a separately preregistered frozen **D2 RPN rank
profile / stage-separation probe**, not a training run: on the fixed 59
score/`nms_pre` targets, identify the max-IoU raw positive's FPN level, raw
objectness rank, rank margin around the live per-level top-1000 boundary, and
counterfactual retention at a pre-registered small set of `nms_pre` caps;
compare those profiles with matched raw-geometry and max-per-image slices.
It must retain D1's live-final/B0 replay and typed-hash controls, use fixed
scene bootstrap, and explicitly state that counterfactual retention is a
diagnostic—not an AP or proposal-recall intervention. Only a new registered
probe with a passed gate could define an intervention candidate; even then,
training remains separately gated.
