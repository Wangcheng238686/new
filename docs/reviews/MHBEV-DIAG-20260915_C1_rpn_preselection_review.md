# MHBEV-DIAG-20260915-C1 — dense-RPN preselection frozen-readout implementation review

**Review scope (independent, read-only):** Appendix F of
`EXPERIMENT_GOVERNANCE.md`; implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_uav_rpn_preselection_signal.py`;
and its declared B0 full-val lineage.  No code or experiment output was
modified by this review.

**VERDICT: HOLD — do not run C1 smoke until the blocking audit/telemetry gaps
below are repaired and this review is refreshed.**  This is an implementation
hold, not evidence for or against UAV content.

## What is correct

- `dense_raw_rpn` uses the frozen detector's `extract_feat` followed by
  `rpn_head.forward`, then decodes every FPN level's grid priors with the
  live RPN bbox coder.  It does not call `rpn_head.predict`, score filtering,
  `nms_pre`, or NMS, and it uses fresh data samples while rejecting written
  proposals.  This is the correct conceptual location for a dense,
  preselection readout rather than a deployment-proposal replacement.
- Positives are maximum-IoU decoded raw anchors with IoU at least .5;
  negatives are true background (`max IoU < .1`) anchors matched only on raw
  sigmoid objectness.  Pair selection never reads UAV features.  Raw geometry
  absence, score-match rejection, descriptor invalidity, and inclusion are
  represented as distinct val statuses and the fixed schema preserves null
  fields.
- The B0 filter, target count/uniqueness check, checkpoint identity check,
  post-NMS replay IoU check, 128-train/259-val split assertion, fixed sklearn
  fit, frozen encoder/model hashes, correct/XY/permuted descriptor paths, and
  259-scene zero-filled bootstrap all follow the intended design.  There is
  no evident train-to-val fitting leakage.

## Required fixes before smoke

1. **Implement the pre/post post-NMS RPN hash invariant.**  Appendix F.2
   explicitly requires the post-NMS RPN readout before and after dense-RPN,
   UAV encoding, and sham evaluation to be identical.  The code calls
   `read_rpn` only before that work (lines 201--207); detector normal-predict
   parity is necessary but does not establish this separate RPN-readout
   invariant.  Read boxes *and scores* both before and after, compute typed
   hashes (including shape/dtype), fail closed on change, persist per-scene
   comparison, and expose a summary invariant.

2. **Make the raw-RPN flattening contract fail-closed.**  Before concatenation,
   assert for every FPN level that the number of flattened class scores, bbox
   deltas, decoded boxes, and generated priors are equal; assert a nonzero
   total and persist per-level/count total.  The current reshape will usually
   be correct, but without these checks a changed anchor multiplicity or
   output layout can silently misassociate prior/delta/objectness while still
   producing finite arrays.  This check is essential to substantiate the
   claimed pre-score/pre-NMS dense-anchor contract.

3. **Report all pre-registered companion-accounting fields explicitly.**
   The ledger is useful but summary metrics omit raw-geometry-available count
   and rate, score-match-included count/rate, and train positive/negative
   counts (class balance).  Add these under `metrics`, retaining the fixed
   denominator of 107; do not infer them only informally from CSV statuses.

4. **Make three-condition pair identity a computed invariant.**  The
   immutable raw-anchor record is written once, which structurally helps, but
   summary currently has no `condition_pair_identity_fixed` assertion.
   Persist/reuse the same SHA for all conditions and calculate/fail-close an
   invariant that conditions use unchanged dense indices, raw objectness,
   decoded boxes, and labels.  It must be auditable rather than a claim.

## Re-review smoke requirement

After these bounded changes, run an intentionally incomplete smoke using all
128 fixed train scenes, one C1 target-containing val scene, and 100 bootstrap
draws.  It must be `INVALID` only for incomplete validation target coverage;
lineage, dense layout counts, B0 replay, post-NMS RPN hash parity,
normal-predict parity, detector/UAV hashes, fixed pair identity, fixed schema,
and ledger must all pass.  Then request result-side review before the complete
107-target/259-scene scan.  Neither smoke nor C1 PASS authorizes training.

## Re-review after four requested repairs — 2026-09-15

The substantive repairs are correctly present: dense per-level cardinality is
checked and archived, pre/post `read_rpn` boxes and scores are compared per
scene, explicit availability/score-match/class-balance metrics are recorded,
and condition SHA values are computed before the decision branch.

**VERDICT remains HOLD — two bounded wiring errors prevent any smoke.**

1. Lines 202 and 244 call `array_hash`, but the imports include only
   `output_digest` from `probe_small_postrpn_bottleneck`.  `array_hash` is
   not otherwise defined or imported, so the first train scene will raise a
   `NameError`.  Import the existing helper explicitly (or define an
   equivalent typed-array hash locally).

2. `empty_val_row` writes `correct_pair_identity_sha256`,
   `xy_pair_identity_sha256`, and `perm_pair_identity_sha256` (line 152), but
   none of these fields occurs in `VAL_FIELDS` (lines 46--55).  Python's
   default `csv.DictWriter` extras action is `raise`, so writing any val row
   will fail with an extra-fields `ValueError`.  Add the three fixed SHA
   columns to the schema; do not drop them, because they are the evidence for
   the new condition-identity invariant.

After these two wiring corrections and a compile check, re-request review.

## Final implementation re-review — 2026-09-15

**VERDICT: PASS — authorize the C1 smoke with all 128 fixed train scenes, one
C1 target-containing val scene, and 100 bootstrap draws.**

The two execution blockers are fixed in the current source: `array_hash` is
explicitly imported from the established RPN diagnostic helper, and all three
condition pair-SHA fields are predeclared in `VAL_FIELDS` before
`DictWriter` receives any row.  The CLI now has exactly one definition each
for train/val scene caps and bootstrap parameters.  The post-NMS RPN parity
is measured before and after dense/UAV/sham work for both train and val;
dense per-level cardinalities are asserted and archived; availability,
score-match, and class-balance metrics are explicit; and condition SHA
identity is evaluated before deciding the arm.

The base interpreter available to this reviewer lacks sklearn, so it cannot
execute this project probe's import-time `--help`; the project-specified
environment's successful `--help`, compile, and diff checks are the relevant
execution checks.  This does not change the source-level authorization.

Smoke must be `INVALID` solely for
`validation_target_coverage_incomplete`.  It must also retain all B0 replay,
dense-layout, post-NMS-RPN parity, detector/UAV hash, condition-identity,
fixed-schema, and ledger invariants.  A result-side audit is required before
the complete 107-target scan.  This authorization never permits an
intervention or training.

## Smoke result audit — C1-smoke — 2026-09-15

**VERDICT: PASS — authorize the complete C1 scan over 128 fixed train scenes,
85 target-containing val scenes / 107 B0 targets, and 10,000 bootstrap draws
(seed 20260919).**

Independently inspected `exit_code`, summary, bootstrap, manifest, fixed
val schema, train pairs, and dense-layout archive under
`outputs/MHBEV-DIAG-20260915/C1-smoke`:

- Exit code is 0.  The fit used 128/128 fixed train scenes and yielded 105
  positive/105 negative pairs.  The val smoke covers 1/85 target scene and
  its one B0 target.
- Decision is correctly `INVALID` for exactly
  `validation_target_coverage_incomplete`; all nine recorded invariants pass,
  including B0 post-NMS replay, normal-predict parity, post-NMS RPN parity,
  detector/UAV hashes, dense anchor cardinality, and condition identity.
- `dense_layout.csv` has five FPN levels for each of the 129 processed scenes;
  all prior/delta/score/decoded cardinalities agree.  The val row's replay
  IoU exactly equals its B0 record (`.49269623`) while its dense raw positive
  reaches `.92218518`, demonstrating the intended preselection distinction.
- The fixed val schema has 44 fields, including all three condition SHA,
  support, score, and win columns.  The one target is raw-geometry available
  and score-matched, but descriptor-invalid; undefined scores/wins are
  explicit null/empty cells.  Its three condition SHA values equal the
  immutable base pair SHA.

The complete scan remains probe-only and requires a new independent
result-side audit before any conclusion.  C1 PASS, if obtained, would at
most authorize a separately registered RPN zero-gate frozen intervention;
it never authorizes training.
