# MHBEV-DIAG-20260915-B3 — UAV ROI-score frozen-readout implementation review

**Review scope (independent, read-only):** Appendix E of
`EXPERIMENT_GOVERNANCE.md`; implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_uav_roi_score_signal.py`; and
the B2 archived lineage interface.  No code or experiment artifact was
modified by this review.

**VERDICT: HOLD — do not start the B3 smoke until the required audit fields
and fail-closed checks below are implemented and this review is refreshed.**
This is not evidence for or against a UAV signal.

## What is correctly specified and implemented

- The linear readout itself has no val-label leakage: `StandardScaler` and
  the fixed `LogisticRegression(C=1, class_weight="balanced", max_iter=1000,
  random_state=20260918)` are fitted only on the fixed 128-scene official
  train subset.  The code hard-fails if that subset and the 259-scene val set
  overlap.  It does not fit the scaler or classifier on val descriptors, and
  it exposes no tuning parameter that selects a better val readout.
- The positive/negative construction follows the pre-registered decoded-box
  definition: maximum GT IoU positive (at least .5), background negative with
  maximum IoU below .1, selected only by nearest *satellite* foreground
  score.  UAV values are not inspected when selecting either candidate.  The
  val `abs(score_pos-score_neg)<=.02` rule is applied before evaluation.
- The frozen descriptor uses the required decoded-box 3x3 grid, z=0/360m
  mapping, W2C column-vector FP32 projection under disabled autocast,
  finest-resolution frozen UAV FPN, bilinear sampling, and a strict
  two-or-more-view requirement for all nine query points.  The encoder is
  built in eval mode, set `requires_grad_(False)`, called under inference
  mode, and parameter-hashed before/after.
- `correct`, `XY+80m`, and the visual-content-only next-val-scene permutation
  share the previously chosen candidate indices.  The scene-permuted branch
  deliberately retains the current scene K/E, as Appendix E requires.
- The bootstrap correctly reconstructs the 259-scene population, zero filling
  non-target scenes, and uses the prescribed 10,000 draws/seed by default.
  Its statistic is pair-weighted, image-resampled `correct-sham` win-rate
  difference, which matches Appendix E.

## Blocking issues

1. **The required three-condition telemetry is discarded, so the output is
   not independently auditable.**  `uav_descriptor` returns support metadata,
   but lines 177--178 discard it.  Lines 181--183 write only win bits.  Thus
   `val_pairs.csv` lacks each condition's 9-point support/min-support and
   the three continuous readout probabilities.  Appendix E.3 explicitly
   requires per-pair three-condition support, three readout scores, common
   validity, and win bits.  Add those fields (including score values for both
   positive and negative, or an unambiguous pairwise score difference) before
   a smoke.

2. **Condition/pair identity is asserted, not verified.**  The summary sets
   `condition_pair_identity_fixed: True` unconditionally (line 200).  It
   neither persists nor hashes the positive/negative decoded boxes, raw
   foreground scores, candidate labels, and candidate indices per condition.
   The normal-predict hash only surrounds `bbox_readout` (lines 165--167),
   before UAV descriptor execution and the three sham evaluations.  Appendix
   E.2 requires bitwise identity of pair identity, satellite score, decoded
   boxes, and label between correct and shams, as well as detector typed-hash
   parity before/after each readout scene.  Make a fail-closed immutable pair
   record/hash, attach it to all three cells, persist the individual fields,
   and run normal prediction both before and after the complete descriptor /
   sham sequence.  The invariant must be computed from those records, not
   declared true.

3. **The score-match accounting is incomplete.**  A target with gap above
   `.02` silently receives no diagnostic row explaining exclusion beyond the
   bare `score_gap`; the summary reports only common pairs divided by 87.
   Persist counts for all 87 targets partitioned into: candidate drift,
   score-match rejected, descriptor-invalid by condition, and common-valid;
   report the score-gap distribution for the eligible population.  This is
   necessary to distinguish a coverage failure from loss of score-matched
   evaluation support and to satisfy Appendix E.2/E.3's “report included
   count and score gap” contract.  Keep the already pre-registered gate
   denominator explicit; do not change it after smoke.

4. **B2 lineage is too weakly pinned to the actual target record.**  It
   verifies the B2 checkpoint, counts, and broad decision, but it recomputes
   a candidate pair without checking the B2 row's declared `pstar_index`,
   raw score/IoU fields, or that the B2 `roi_score_rejected` predicate still
   holds.  Re-evaluation should be deterministic, but a B3 smoke must
   fail-closed on candidate drift rather than merely `candidate_pair is not
   None`.  At minimum persist and compare the fixed B2 row identity and its
   target classification fields; clearly distinguish B2 P* from B3's
   pre-registered maximum-decoded-IoU positive when they differ.

## Non-blocking clarifications

- Training pairs are not constrained to a .02 satellite-score gap.  This is
  not leakage and does not invalidate a held-out, score-matched val win-rate:
  satellite score is not an input feature and the held-out gate is fixed.
  Still report the train score-gap distribution and positive/negative counts
  to make the conditional-targetness claim appropriately bounded.
- A PASS would establish only frozen, conditionally readable UAV targetness
  on the post-selected B2 target slice.  It would authorize solely the
  Appendix-E ROI-score zero-gate frozen intervention, never detector training
  or an AP claim.

## Required re-review smoke scope

After the four blocking fixes, run only a deliberately incomplete fixed
train/target-scene smoke with a small bootstrap count.  It must be
`INVALID` **solely** for incomplete coverage while lineage, split, encoder
hash, post-sequence detector parity, fixed pair identity, all three condition
records, and common-support accounting pass.  Submit the resulting archive
for re-review before the 74-target-scene / 87-target complete scan.

## Re-review — 2026-09-15

The four original blocking items are now correctly addressed:

- every eligible val row persists each condition's positive/negative support,
  continuous readout scores, win bit, and common-valid status;
- the immutable satellite-side pair record includes indices, decoded boxes,
  raw scores, labels, and SHA-256; descriptor/sham work now occurs between
  normal-predict hashes;
- the ledger exhaustively partitions all seen B2 targets into score-match
  rejected, descriptor-invalid, and included, and reports score-gap summary;
- B2 P* pre-RPN IoU, decoded IoU, raw foreground score, threshold/category,
  and B3 maximum-decoded-IoU positive are separately checked fail-closed.

**VERDICT remains HOLD for one narrow, executable-completeness fix.**
Appendix E requires the readout fit on the fixed **128** train scenes.  Yet
the decision condition at lines 266--268 checks full val only, not
`train_n == len(train)`.  Consequently a command with
`--max-train-scenes=32 --max-val-scenes=0` can legally emit a non-`INVALID`
signal/no-signal decision from an unregistered 32-scene fit.  In addition,
the requested `32 train + 1 target-val` smoke would be incomplete in *both*
the fit and evaluation, so its `INVALID` cannot truthfully be attributed
solely to evaluation coverage.

Before smoke, explicitly record `train_complete`, `val_complete`, and an
`invalid_reasons` list; require both complete before any non-`INVALID`
decision.  Then either (a) use all 128 train scenes plus one target val scene
when the desired smoke claim is “INVALID solely for incomplete val coverage”,
or (b) keep the 32+1 wiring smoke but label it `INVALID` for the two declared
incompleteness reasons and never use it for the stronger sole-reason claim.
After this bounded change, the 32+1 version is authorized as a wiring smoke
under option (b); submit its artifacts for result-side review before full
coverage.

## Final smoke authorization — 2026-09-15

**VERDICT: PASS — authorize the B3 incomplete smoke using all 128 fixed train
scenes, one B2 target-containing val scene, and 100 bootstrap draws.**

The completion repair is correct: `train_complete` and `val_complete` are
recorded independently, `complete` requires both, and only `complete` runs
may emit `FROZEN_UAV_ROI_SCORE_SIGNAL` or `NO_FROZEN_SIGNAL`.  The B2 target
set is now fail-closed on exactly 87 unique `(scene, GT)` records across
exactly 74 scenes, preventing dictionary overwrite from weakening the stated
87-target contract.

For this authorized 128-train + 1-target-scene smoke, the expected terminal
state is `decision=INVALID` with precisely
`invalid_reasons=["validation_target_coverage_incomplete"]`.  All lineage,
full-train split, encoder hash, post-descriptor/sham detector parity,
condition identity, per-condition support/readout fields, and ledger checks
must pass.  Any other invalid reason, exception, or missing artifact is a
HOLD and requires re-review.  This authorization is for wiring only and does
not authorize a full B3 scan or any training.

## Zero-common-pair re-review — 2026-09-15

**VERDICT: PASS — re-authorize the same 128-train + 1-target-scene smoke.**

The selected smoke scene has zero common-valid pairs, so every image-paired
bootstrap resample has a zero denominator.  The revised bootstrap now emits
`samples_valid=0` and null CI endpoints instead of calling percentile on an
empty array.  This is the correct representation of an undefined statistic;
it is not a zero effect or a passing CI.

The decision path remains fail-closed.  The incomplete smoke is still
`INVALID` solely because `validation_target_coverage_incomplete`; zero pairs
does not add a spurious invalid reason.  On a future complete scan, zero
common pairs necessarily fails coverage and is therefore correctly classified
as `NO_FROZEN_SIGNAL`, never as a signal.  No gate, feature contract, or
control was relaxed.

## Smoke-r4 result audit — 2026-09-15

**VERDICT: HOLD — do not authorize the complete scan from this artifact yet.**

The executable result is otherwise correct and independently reproduced:

- `exit_code` is 0; it fitted on all 128 fixed train scenes and wrote 127
  train pairs.
- It processed one of the 74 B3 target scenes and its two B2 target rows.
  The decision is `INVALID` with exactly
  `validation_target_coverage_incomplete`; all six declared invariants are
  true.
- Both val rows are score-matched, `descriptor_invalid`, and non-common;
  their immutable pair SHA-256 values recompute exactly from their persisted
  satellite-side fields.  The ledger is 2/2 descriptor-invalid, with zero
  common pairs.  The bootstrap correctly records 0/100 valid draws and null
  CI endpoints.

However, `val_pairs.csv` has **no readout-score or win columns at all**.  The
current dynamic CSV schema constructs fields only from keys present in a row;
because this deliberately chosen smoke has no common-valid pair, the code
never inserts the three conditions' positive/negative readout-score and win
keys.  Appendix E.3 requires the per-pair three-condition support, readout
scores, common-valid flag, and win bits.  For an invalid descriptor those
readout fields are necessarily undefined, but they must still be serialized
as explicit null/empty values under a fixed schema.  The same applies to
descriptor support fields for score-match-rejected rows.

Required bounded repair: initialize every val output row with all
three-condition support, positive/negative readout-score, and win fields set
to `None`, then overwrite only values that are defined; write a fixed,
predeclared CSV column list.  Rerun the same smoke.  If it retains the above
counts/invariants and now exposes the null-valued E3 fields, it may be
authorized for the 74-scene/87-target complete scan.  This is an archive
schema repair, not a change to the readout, controls, or decision rule.

## Fixed-schema re-review — 2026-09-15

**VERDICT: PASS — authorize B3-smoke-r5 with the same 128-train, one
target-scene val, 100-bootstrap scope.**

`VAL_PAIR_FIELDS` now predeclares the immutable identity fields and all three
conditions' positive/negative support, positive/negative readout score, and
win fields.  Every val record initializes these condition fields to `None`
before score-match filtering; descriptor and readout steps overwrite only
defined values.  The fixed `DictWriter` schema consequently preserves the
E3 contract for score-match-rejected, descriptor-invalid, and included rows.
This is correctly an archive-format repair: it does not alter candidate
selection, descriptor computation, classifier fit, sham construction,
bootstrap, or decision gating.

For r5, repeat the r4 checks plus verify that all fixed E3 columns exist and
are null for values undefined by the two descriptor-invalid pairs.  A passing
r5 may authorize the complete 74-target-scene / 87-target / 10,000-draw B3
scan, subject to its own independent result review.

## Smoke-r5 result audit — 2026-09-15

**VERDICT: PASS — authorize the complete B3 scan over all 128 train scenes,
74 target-containing validation scenes / 87 targets, and 10,000 bootstrap
draws (seed 20260918).**

Independently inspected `exit_code`, summary, bootstrap, manifest, and both
CSVs under `outputs/MHBEV-DIAG-20260915/B3-smoke-r5`:

- Process exit is 0.  The fit used exactly 128 train scenes and emitted 127
  train pairs.  The val run used one of 74 target scenes containing exactly
  two B2 targets.
- The terminal decision is correctly `INVALID` for the sole declared reason
  `validation_target_coverage_incomplete`; it is not promoted into a signal
  or no-signal result.  All six recorded invariants are true.
- The ledger is exhaustive: both targets are score-matched but
  descriptor-invalid, yielding zero common pairs.  Hence bootstrap has
  0/100 valid draws and null CI endpoints, as required for an undefined
  statistic.
- `val_pairs.csv` has the required fixed **41 columns**.  All three
  conditions' support fields are present; all nine undefined readout-score /
  win values are explicitly empty CSV cells (serialized `None`) for each
  descriptor-invalid row.  This satisfies the E3 per-pair schema rather than
  silently dropping columns.

The complete scan remains probe-only.  Its result, including the fixed
87-target ledger, common-coverage denominator, two sham CIs, and all
invariants, must receive a separate result-side audit before any follow-on
zero-gate intervention; it does not authorize training.

## Full-val result audit — B3-fullval — 2026-09-15

**VERDICT: PASS (result integrity); required decision: `NO_FROZEN_SIGNAL`.**
**No ROI intervention and no training are authorized.**

Independently inspected `exit_code`, manifest, summary, bootstrap, train/val
CSVs, and the B2 target archive under
`outputs/MHBEV-DIAG-20260915/B3-fullval`.

- The run exited 0 and is complete: 128/128 fixed train scenes, 74/74 B3
  target-containing val scenes, and all 87/87 B2 ROI-score target identities.
  Each B3 `(scene, GT)` record and its recorded B2 P* index exactly match the
  B2 archive.  The 127 train pairs and 41-column E3 val schema are present.
- All six invariants pass.  The val ledger is exhaustive: 45 included,
  41 descriptor-invalid, 1 score-match-rejected (87 total).  Pair identities
  recompute from their persisted boxes/scores/labels; readout scores and wins
  are defined exactly for the 45 common-valid rows and null otherwise.
- Common coverage is `45/87 = 0.51724138`, below the pre-registered `.80`
  gate.  Among the included pairs, correct/XY/permuted win rates are
  `29/45 = .644444`, `27/45 = .600000`, and `25/45 = .555556`.
- Rebuilding the 259-scene zero-filled bootstrap population with
  `numpy.default_rng(20260918)` exactly reproduces all 10,000 valid draws and
  both reported percentile CIs: correct-minus-XY
  `[-.14285714, +.23809524]`, correct-minus-permuted
  `[-.10204082, +.26190476]`.  Neither lower bound exceeds `+.05`.

Thus the correct win rate alone clears its companion threshold, but coverage
and both sham-separation requirements fail.  Appendix E consequently requires
`NO_FROZEN_SIGNAL`; it does not support a claim that frozen UAV content adds
independent ROI targetness under this contract.  Per Appendix E.2, even a
PASS would only have authorized a separate ROI-score zero-gate frozen
intervention; this negative result authorizes neither that intervention nor
any hot-start/full detector training.  Archive this negative result and close
the B3 direction rather than retuning the readout, FPN level, grid, or gate.

## Conclusion-document audit — 2026-09-15

**VERDICT: PASS — `docs/MHBEV-DIAG-20260915_B3_conclusion.md` faithfully
executes the Appendix-E decision and may close B3.**

The document reproduces the complete scope, ledger, coverage, all three win
rates, both independently recomputed CIs, and the operative gate without
selecting a favorable submetric.  It correctly states `NO_FROZEN_SIGNAL`,
forbids the ROI zero-gate intervention and all ROI-UAV training, and limits
the negative conclusion to the frozen z=0/3x3/score-matched ROI descriptor
contract.  Its proposed RPN-recall direction is explicitly a separate future
hypothesis, not an inference that B3 proves a recall gain.  No overreach or
post-hoc gate change was found.
