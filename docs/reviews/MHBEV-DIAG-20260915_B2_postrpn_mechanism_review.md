# MHBEV-DIAG-20260915-B2 — post-RPN mechanism frozen-diagnostic implementation review

**Review scope (independent, read-only):** Appendix D of
`EXPERIMENT_GOVERNANCE.md`; implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_small_postrpn_mechanism.py`;
the archived B0/B1 full-val lineage.  No B2 full-val artifact existed at
review time.

**VERDICT: HOLD — do not start B2 full-val until the lineage and deployment
contract checks below are made fail-closed and this review is refreshed.**
This is an implementation hold, not evidence for or against any mechanism.

## Addendum — 2026-09-15 re-review after DeltaXYWH correction

**VERDICT remains HOLD.**  Correcting the actual anchor to
`Shared2FCBBoxHead + DeltaXYWHBBoxCoder`, removing the nonexistent SABL
confidence factor, and adding the proposed lineage, 1,000-proposal,
threshold-boundary, and raw-margin checks are all correct changes.  However,
the revised readout still misses a deployed operation: in
`RSPrompterAnchorRoIPromptHead.predict`, `extra_pe` is interpolated and added
to every FPN level immediately before `predict_bbox` (the normal path in
`rsprompter/models.py`, around lines 1115--1132).  B2 calls
`_bbox_forward` directly on the pre-`extra_pe` feature tuple returned from
`model.extract_feat`.  Its ROI logits and DeltaXYWH boxes therefore do not
measure the normal deployment path.

Before even a smoke run, B2 must reproduce that `extra_pe` feature transform
exactly on a local feature tuple before its read-only `_bbox_forward` call;
RPN must continue to use the original pre-PE tuple, matching deployment.
Also remove the two remaining `confidence` references in Appendix D, and
fail-closed assert the loaded bbox-head/coder types are
`Shared2FCBBoxHead`/`DeltaXYWHBBoxCoder`.  Then re-request review.

## Addendum — 2026-09-15 second re-review

**VERDICT remains HOLD.**  The latest revision now locks the correct
`Shared2FCBBoxHead` and `DeltaXYWHBBoxCoder`, removes the stale confidence
contract, and accurately reproduces the `extra_pe` expression.  But it
applies that expression before `rpn_head.predict`.  Deployment instead uses
the original FPN tuple for RPN (`RSPrompterAnchor.predict`, lines 214--218),
then applies `extra_pe` only inside `roi_head.predict` before
`predict_bbox`.  This version changes the RPN proposal set and invalidates
the B0 P* lineage.  Preserve `x_base` for `rpn_head.predict`; create
`x_roi = add_extra_pe(x_base)` only for `_bbox_forward`.  Re-review is
required after that one ordering correction.

## Final implementation re-review — 2026-09-15

**VERDICT: PASS — the bounded B2 smoke is authorized.**  The readout now
preserves the actual deployment order: `x_base` feeds `rpn_head.predict`,
while a separately constructed `x_roi` applies the exact `extra_pe` and
bilinear interpolation expression before `_bbox_forward`.  Thus the passive
readout neither changes the RPN/P* population nor omits the ROI positional
feature used by normal prediction.

The current implementation also fail-closes on B0/B1 checkpoint and complete
data contracts, the B1 94-scene/120-row lineage, exact 101 target membership,
the actual `Shared2FCBBoxHead + DeltaXYWHBBoxCoder` types, and exactly 1,000
RPN proposals per processed scene.  It uses the deployed raw-score boundary
(`<= score_thr` rejected), stores raw-score margins, and has no remaining
SABL/confidence claim.  Its Delta decode is the same
`bbox_coder.decode(rois[:, 1:], bbox_pred, img_shape)` operation used in the
loaded head's deployment `predict_by_feat` path; with one foreground class,
`softmax[:, 0]` is the correct raw foreground score.

Smoke requirements: use the fixed B1 lineage/checkpoint, a deliberately
incomplete target-scene cap, and 100 bootstrap draws only for wiring.  The
result must be `INVALID` solely because coverage is incomplete; all lineage,
P*, parity, and B1-final-failure checks must pass.  A successful smoke does
not authorize full-val by itself: submit its artifacts for the result-side
review required before the 94-scene, 10,000-draw scan.

## Smoke result audit — B2-smoke-r5

**VERDICT: PASS — the complete B2 scan is authorized.**  Reviewed
`summary.json`, `targets.csv`, `bootstrap.json`, and `run_manifest.json`
under `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B2-smoke-r5/`.

- The smoke is correctly `INVALID` **only** because it deliberately processes
  1/84 target-containing scenes and 2/101 targets.  Its full 259-scene
  bootstrap universe is preserved; therefore partial coverage is not silently
  promoted into a mechanism conclusion.
- The other mandatory checks passed: exact B1 membership for the processed
  targets, two normal-predict typed-hash parity, P* pre-IoU >= .5, recomputed
  B1 final-box/mask failure, no UAV/sham intervention, and exactly 1,000 RPN
  proposals for scene `002007`.
- The archived contract is now explicit in both summary and manifest:
  `Shared2FCBBoxHead`, `DeltaXYWHBBoxCoder`, live `score_thr=0.05`, NMS IoU
  `0.5`, and `max_per_img=100`.  Target CSV rows include both raw-score
  margins required to audit the score-rejection category.

**Authorization:** run the pre-registered complete scan over **84 target
scenes / 101 targets**, with the fixed B0/B1 checkpoint lineage and complete
259-scene bootstrap, 10,000 draws, seed `20260917`.  It must be independently
result-audited before any conclusion or follow-on diagnostic.  This
authorization never permits training.

## Full-val result audit — B2-fullval

**VERDICT: PASS — `ROI_SCORE` is the required Appendix-D decision.**  Reviewed
and independently recomputed `summary.json`, `targets.csv`, `bootstrap.json`,
and `run_manifest.json` under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B2-fullval/`.

### Reproduced scope and invariants

- The archive has the fixed B0/B1 checkpoint lineage, the explicitly recorded
  `Shared2FCBBoxHead + DeltaXYWHBBoxCoder` contract, and the live deployed
  RCNN settings: raw `score_thr=0.05`, NMS IoU `0.5`, `max_per_img=100`.
- Coverage is complete: 84/84 target-containing scenes and 101/101 unique
  B1 post-RPN-box-lost targets; the bootstrap population is all 259 official
  validation scenes.  All declared result-side invariants pass, including
  normal-predict typed-hash parity, P* pre-IoU >= .5, B1 final-failure
  recomputation, and no UAV/sham intervention.
- The categories are mutually exclusive and exhaustive: 13 refinement-lost,
  87 ROI-score-rejected, and 1 NMS-or-rank loss (total 101).  Each of the 13
  refinement rows has max decoded IoU < .5; every score-rejected row has an
  aligned decoded box and aligned best raw score <= its recorded threshold;
  the sole NMS/rank row has an aligned decoded box and raw score > threshold.
  All final box IoUs remain < .5 (maximum `0.49810442`) and all final mask
  IoUs remain < .5 (maximum `0.47115386`), as B1 requires.

### Independently reproduced gate

`roi_score_rejected = 87/101 = 0.86138614`, exceeding the .60 gate.  Rebuilding
the ordered 259-scene population from B0 scene IDs, zero-filling the 175
non-target scenes, and using `numpy.default_rng(20260917)` reproduces the
reported 10,000/10,000 valid draws exactly.  For the fixed observed winner
ROI-score rejection, category-minus-others has mean `0.72197516` and 95%
percentile CI `[0.57009346, 0.85840708]`; its lower bound is strictly above
zero.  `bootstrap.json` is byte-for-value consistent with the summary.

The Appendix-D gate therefore legally yields **`ROI_SCORE`**.  This is a
localization of where the single-stream detector loses these already-proposed
small objects: most retain a decoded aligned box, but its raw ROI foreground
score is below deployment threshold.  It is not evidence that UAV content
will improve the score, nor authorization for A2/A3, hot-start, or training.

**Permitted next step:** separately pre-register a frozen, read-only or
zero-gate **ROI-score-specific** diagnostic/intervention probe, with its own
parity/sham controls and an outcome gate.  **Forbidden:** reinterpret this as
an NMS-only or box-refinement result, or start any training from B2 alone.

## What is correct

- The readout uses fresh data samples and preserves the prescribed order:
  normal `predict`, read-only `extract_feat -> rpn_head.predict ->
  roi_head._bbox_forward`, then another normal `predict`.  It checks that no
  readout sample receives `proposals`; typed normal-predict hashes cover
  boxes, scores, labels, and masks.
- `bbox_coder.decode` is the same SABL decoding operation used by
  `SABLHead._predict_by_feat_single` before NMS.  Foreground `softmax[:, 0]`
  is correct for this one-foreground-class-plus-background head.  The
  deployment `multiclass_nms` applies `score_thr` to that raw class score
  before multiplying SABL's location-confidence factor, so the intended
  score/NMS/confidence partition is sound in principle.
- Given valid lineage, the three branches are mutually exclusive and
  exhaustive: no decoded IoU>=.5; decoded-aligned but raw score below the
  live threshold; or decoded-aligned and threshold-passing but absent from
  final output.  The last is properly broad enough to include NMS,
  max-per-image ranking, and SABL confidence-factor loss.
- The bootstrap population is correctly built over all 259 official val
  scenes, zero-filling non-target scenes.  It uses a fixed winner from the
  observed sample for the reported winner-minus-others CI, as Appendix D
  requires.
- The available B1 archive is compatible with the expected lineage: its
  target CSV has 120 rows, exactly 101 `post_rpn_box_lost_or_mislocalized`
  rows in 94 scenes; its checkpoint is the B0 checkpoint; its completed
  full-val invariants are true.  B0 records 259 scenes, 5,937 GT instances,
  and exactly 1,000 RPN proposals per scene.

## Required fixes before full-val

1. **Harden the B0/B1 lineage checks.**  B2 currently checks B1 completion,
   decision, and invariants, but only checks B0 root/split.  It must also
   assert the B0 checkpoint equals `--checkpoint`, B0's completed
   259-scene/5,937-instance/1024-square contract and all B0 invariants, and
   B1's 94 target-scene / 120 target contract.  Appendix D makes all of these
   prerequisites, not merely informational metadata.
2. **Make P*'s 1,000-proposal contract explicit.**  The code only rejects a
   readout with more than 1,000 proposals.  Appendix D defines P* over the
   deployment's 1,000 RPN proposals, and the archived B0 run proves this is
   the active deployment cardinality.  Require exactly 1,000 in B2 and
   persist/assert it per target scene.
3. **Make score-factor telemetry auditable.**  Persist the SABL decoded
   location confidence for the aligned boxes (and preferably the best
   threshold-passing aligned effective score), together with an explicit
   `raw_fg_score - score_thr` margin.  The current CSV stores raw score and
   threshold, so the margin can be reconstructed, but omits the returned
   confidence altogether.  Without it, a claimed
   `nms_rank_or_confidence_loss` sub-mechanism cannot be independently
   checked against the deployed post-threshold factor.
4. **Match the executable threshold boundary.**  MMDetection's deployed
   `multiclass_nms` retains raw scores only when `score > score_thr`.
   Categorize equality as `roi_score_rejected` (or assert equality cannot
   occur); the current `<` condition assigns equality to the third category.

After these changes, rerun the smoke first.  The smoke must remain `INVALID`
only for incomplete coverage while all revised lineage/P*/parity checks pass;
then request a renewed independent review before the 94-scene B2 full-val
scan.  Any later non-`INVALID` result still only localizes a single-stream
failure mechanism and does not authorize UAV training.
