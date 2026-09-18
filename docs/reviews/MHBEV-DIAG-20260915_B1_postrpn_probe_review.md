# MHBEV-DIAG-20260915-B1 — Post-RPN frozen diagnosis implementation review

**Review scope (independent, read-only):** Appendix C in
`EXPERIMENT_GOVERNANCE.md`; implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_small_postrpn_bottleneck.py`;
and smoke artifacts under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B1-smoke-r2/`.

**VERDICT: PASS — the complete 94-target-scene B1 frozen, read-only scan is
authorized.** This authorization is only for the Appendix-C diagnostic.  It
does not authorize any training arm.

## Lineage, target, and outcome definition

1. The implementation hard-requires B0's completed 259-scene/5,937-GT
   archive, all recorded B0 invariants, the identical checkpoint, and now also
   the same data root, `val` split, and 1024-square input contract.  It derives
   the target exclusively from B0 rows with `small`, final `missed` or
   `partial_or_merge`, and RPN IoU@.5 present; it hard-requires exactly 120
   unique `(scene_id, gt_index)` rows.  The archived B0 data independently
   gives 120 targets in 94 target-containing scenes.
2. For each target, the script recomputes the normal-deployment final mask
   outcome using the B0 Hungarian convention and rejects any target drift.  It
   also rejects a target with maximum final mask IoU >= .5, matching Appendix
   C's more explicit final-mask-failure contract.  In the verified B0 archive,
   all 120 target rows meet this condition.
3. Final box reach is correctly the maximum xyxy IoU against *all* deployed
   final `pred_instances.bboxes`, without label filtering.  A target is
   therefore assigned to exactly one of the prescribed exhaustive categories:
   box reach >= .5 plus mask failure (`mask_only_failure`), or box reach < .5
   (`post_rpn_box_lost_or_mislocalized`).  The latter is not overstated as a
   classification-only error.

## Normal-predict parity and UAV/sham boundary

1. Each target-containing scene makes two fresh, normal
   `model.predict(..., rescale=False)` calls.  The `bboxes`, `scores`,
   `labels`, and `masks` digests include dtype and shape as well as bytes.  The
   smoke's one processed scene has `prediction_parity_bitwise: true`.
2. B1 contains no RPN readout, proposal injection, UAV forward, or geometry
   sham forward.  It merely inherits the three B0 raw view counts and reports
   their >=2-view coverage after B1 regrouping.  Added lineage checks reject a
   B0 inherited view count outside `[0,5]`.  Thus B1 cannot change detector
   output, proposals, or categories through a sham.
3. The smoke recomputed both B0 target rows in scene `002007`: both retain
   `partial_or_merge`, max final mask IoU below .5, and their final box reaches
   are below .5.  This is correct target-drift and final-box behavior, not a
   scientific result.

## Bootstrap correction and smoke status

The prior implementation was held because it resampled only the 94
target-containing scenes.  The revision constructs `bootstrap_per_scene` over
all 259 `dataset.ids`, filling the 165 non-target scenes with zero target
counts.  It records `bootstrap_scene_universe: 259` in the summary.

The smoke records exactly that universe, `target_scenes_expected: 94`,
`target_scenes_processed: 1`, and `targets_processed: 2`.  Its 100 bootstrap
draws have 66 valid target-containing replicates.  That is consistent with
resampling 259 scenes from a population with one nonzero scene (expected
valid probability `1 - (258/259)^259`, approximately 0.633), and inconsistent
with mistakenly resampling the one processed target scene alone.

The smoke decision is correctly `INVALID` **only because full target coverage
is false**: `data.complete: false` and `full_target_coverage: false`.  Its
processed-target membership, normal-predict parity, B0-failure recomputation,
and no-UAV/sham-intervention invariants are all true.  The smoke uses 100
bootstrap samples strictly for wiring validation; the complete run must use
the pre-registered 10,000 samples and seed 20260916.

## Required result-side checks

Before any conclusion, the complete artifact must show all of the following:

- 94/94 target scenes, 120/120 targets, `data.complete: true`, and
  `bootstrap_scene_universe: 259`;
- all listed invariants true and 10,000 bootstrap samples with seed 20260916;
- category counts summing exactly to 120; and
- the Appendix-C rule applied exactly: `ROI_LOCAL_MASK` only if mask-only is
  at least .60 with CI lower bound >0; `POST_RPN_ROI_BOX` only if box-lost is
  at least .60 with CI upper bound <0; otherwise `MIXED`.

Any non-`INVALID` B1 decision remains a localization of the baseline failure
stage, not evidence that a UAV intervention will causally improve it.

---

## Full-val result audit (B1-fullval)

**VERDICT: PASS — the pre-registered B1 result is valid, and
`POST_RPN_ROI_BOX` is the legally required Appendix-C decision.**

**Artifact scope.** Reviewed and independently recomputed
`summary.json`, `targets.csv`, `bootstrap.json`, and `run_manifest.json` under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B1-fullval/`.

### Reproduced data, lineage, and invariants

- The summary and manifest agree on the B0 anchor checkpoint, the completed
  B0 lineage, a 259-image bootstrap universe, 94/94 target-containing scenes,
  and 120/120 targets.  `data.complete` is true.
- Refiltering B0's complete `instances.csv` with the Appendix-C target rule
  reproduces exactly the 120 B1 `(scene_id, gt_index)` pairs; B1 has 120
  unique rows across 94 scenes.  The B1 target CSV has 121 lines including its
  header.
- All four B1 result-side requirements are true: exact processed membership,
  full target coverage, bitwise two-normal-predict parity, and B0 final-mask
  failure recomputation.  The declared UAV/sham intervention is `none`.
- Every target's final mask IoU is below .5 (range `0.0` to `0.48251748`).
  The 19 `mask_only_failure` rows each have final box IoU >= .5; the 101
  `post_rpn_box_lost_or_mislocalized` rows each have final box IoU < .5.
  Thus the categories are mutually exclusive and exhaustive, with no outcome
  row violating its definition.

### Independently reproduced statistics

- Category counts are `19/120 = 0.15833333` mask-only and
  `101/120 = 0.84166667` post-RPN box-lost.
- Reconstructing the ordered 259-image bootstrap population from B0's official
  scene ids, filling non-target scenes with zero counts, and using NumPy's
  `default_rng(20260916)` reproduces all 10,000 valid draws, mean
  `mask_only - box_lost = -0.6858658720`, and percentile CI
  `[-0.8148225957, -0.5555555556]` exactly.  The standalone bootstrap JSON is
  byte-for-value consistent with the summary object.
- The inherited coverage numerator/denominator reports also reproduce from the
  target CSV: mask-only correct/XY+80m/permuted = `19/19`, `15/19`, `17/19`;
  post-RPN box-lost = `88/101`, `70/101`, `81/101`.  These remain geometry
  query-availability statistics, not semantic-effect measurements.

### Gate execution and permitted next step

The post-RPN box-lost fraction is `0.84166667 >= 0.60`, while the bootstrap CI
upper bound is `-0.55555556 < 0`.  Therefore the `POST_RPN_ROI_BOX` condition
is satisfied.  The result is not a selection artifact: its input target set,
selection rule, 10,000-draw seed, and threshold were pre-registered in
Appendix C.

Allowed: formulate a new, separately pre-registered **frozen diagnostic** for
the post-RPN ROI/box path—for example, one that distinguishes score/ranking or
NMS suppression from box-refinement displacement—before proposing an
intervention.

Forbidden: use B1 as evidence that UAV content improves a target; relabel the
post-RPN result as a mask-separation result; revive the failed A1 P2 residual
route or enter A2/A3, hot-start, or full training.  B1 localizes a baseline
failure stage only and its own pre-registration expressly authorizes no model
arm.
