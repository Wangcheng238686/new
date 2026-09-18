# MHBEV-DIAG-20260915-C1 — dense-RPN preselection frozen-readout result audit

**Review scope (independent, read-only; 2026-09-17):** Appendix F of
`EXPERIMENT_GOVERNANCE.md`; the frozen-probe implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_uav_rpn_preselection_signal.py`;
B0 full-validation lineage; and C1 full-validation artifacts under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/C1-fullval`.
No source, checkpoint, or C1 output was changed by this review.

**VERDICT: PASS (result audit); registered result: `NO_FROZEN_SIGNAL`.**

The execution and the negative decision are supported by the archived data.
This PASS certifies the probe result, **not** the UAV-to-RPN direction.  It
does not authorize an intervention, hot start, detector/UAV training, proposal
recall claim, or AP claim.  Per F.1/F.2, only a registered
`FROZEN_UAV_RPN_PRESELECTION_SIGNAL` could have authorized a separately
registered *zero-gate frozen intervention*, and that gate was not met here.

## Independent checks

- **B0 lineage and post-NMS replay:** independently regenerated the Appendix-F
  filter over B0 `instances.csv`: exactly 107 unique `(scene_id, gt_index)`
  rows, each small, final-failure `missed` or `partial_or_merge`, and with B0
  `rpn_recall_iou05=False`.  Its key set is exactly the C1 107-row validation
  ledger.  For all 107 rows, archived C1 `post_nms_replay_max_iou` equals its
  B0 `max_rpn_box_iou` exactly at the CSV float representation (maximum
  absolute difference `0.0`); C1's live fail-closed replay check also rejects
  any replay at or above `.5`.
- **Full-data cardinality:** summary records 128/128 fixed anchor-train
  scenes, 85/85 target-containing validation scenes, the full 259-scene
  bootstrap universe, and 107/107 target ledger rows.  `invalid_reasons` is
  empty and the manifest agrees with the summary's `NO_FROZEN_SIGNAL`.
- **Dense preselection contract:** `dense_layout.csv` has 1,065 rows =
  `(128 + 85) × 5` FPN-level records, covers levels 0--4 for every processed
  scene, and every row satisfies `prior_count == delta_count == score_count ==
  decoded_count`.  The level counts are 393,216 / 98,304 / 24,576 / 6,144 /
  1,536, consistent with the deployed dense layout.
- **Fixed telemetry schema and ledger:** `val_pairs.csv` has exactly 44
  predeclared fields and 107 rows.  The mutually exclusive ledger recomputes
  to `included=44`, `descriptor_invalid=40`, `score_match_rejected=0`, and
  `raw_geometry_unavailable=23` (total 107).  Thus exclusions were not
  silently dropped.  There are 84 raw-geometry-available targets (`78.50%`);
  their raw positive IoU range is `[0.50140, 0.98014]`.  Every included row
  has all three conditions' 9-point supports and continuous scores/win bits;
  its minimum recorded valid-point count is 9.
- **Identity and non-intervention invariants:** recomputed the canonical
  SHA-256 record from the archived scene/GT, dense candidate indices, decoded
  boxes, raw objectness, and labels for all 84 raw-geometry-available rows;
  every recomputed SHA equals the base SHA.  All three condition SHAs equal
  that same base SHA wherever a pair exists.  The archived run reports all
  nine invariants true: train/val disjointness, normal-predict parity,
  post-NMS-RPN parity, dense cardinality, condition identity, unchanged
  detector hash, unchanged encoder hash, no detector/UAV/sham intervention,
  and fail-closed B0 lineage.  The implementation locates the post-NMS RPN
  and normal-predict comparisons both before and after dense/UAV/sham work;
  no data sample proposal is written by the dense reader.
- **Statistics independently recomputed:** reconstructed the 259-scene
  zero-filled per-scene table from `summary.json`, used image-paired resamples
  of all 259 scenes for 10,000 draws with NumPy RNG seed `20260919`, and got
  exactly the archived CIs: correct minus `XY+80m` `[-0.11363636,
  0.28205128]`; correct minus scene-permuted `[-0.15384615, 0.18918919]`.
  The direct CSV sums are correct/XY/permuted wins `25/21/24` over 44 common
  pairs, i.e. `0.56818/0.47727/0.54545`.

## Gate application

Appendix F requires complete data and invariants, common-valid coverage
`>= .80`, correct win `>= .60`, and both correct-minus-sham 95% CI lower
bounds `> +.05`.

All completeness and invariant prerequisites pass.  The outcome nevertheless
fails every signal-facing gate: coverage is `44 / 107 = 0.41121`, correct win
is `25 / 44 = 0.56818`, and the two CI lower bounds are `-0.11364` and
`-0.15385`.  The code therefore correctly emits `NO_FROZEN_SIGNAL` rather
than `INVALID` or a positive signal.

## Interpretation and authorization

The result is a credible negative for this exact frozen descriptor/readout
contract: z=0, 3x3 decoded-dense-box sampling, two-view-per-point support,
and the fixed 128-scene linear readout do not establish a sham-specific dense
RPN preselection signal on the B0 RPN-miss targets.  It does **not** establish
that UAV information can never help proposal recall, nor does it reverse the
separate B3 ROI-score negative conclusion.  In accordance with Appendix F,
do not proceed to UAV→P2/RPN intervention, hot start, or training from C1.

## Conclusion-document audit — 2026-09-17

**VERDICT: PASS — `docs/MHBEV-DIAG-20260915_C1_conclusion.md` is faithful to
Appendix F and the complete C1 artifact bundle.**

Checked the conclusion line by line against `summary.json`, `val_pairs.csv`,
`dense_layout.csv`, `bootstrap.json`, and this review:

- Its stated population and method preserve the crucial distinction between
  B0 post-NMS RPN misses and C1's dense, decoded, pre-filter/pre-NMS oracle
  candidates.  It does not call C1 a deployed proposal-recall experiment.
- The train/validation cardinalities, 1,065 dense-layout count, 44-column
  telemetry schema, ledger values, maximum replay IoU (`0.4932716787`),
  raw-geometry and score-match counts, common-valid denominator, win rates,
  and both bootstrap intervals exactly match the archived values.
- It states all three signal gates and correctly applies all three failures;
  it does not relabel the otherwise complete run as `INVALID`.
- Its scope language is appropriately narrow: the negative result is limited
  to the registered frozen z=0 / 3x3 dense-RPN preselection readout, explicitly
  excludes end-to-end proposal-recall and AP conclusions, and preserves the
  separate B3 conclusion.
- The authorization text is stricter than, and compatible with, Appendix F:
  no C1-based zero-gate intervention, hot start, or training is authorized.
  The linked fullval artifacts and independent review provide all required
  archive routes.

No correction or caveat is required.  This document audit does not alter the
registered `NO_FROZEN_SIGNAL` result.
