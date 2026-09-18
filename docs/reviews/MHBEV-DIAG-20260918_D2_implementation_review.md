# MHBEV-DIAG-20260918-D2 — RPN rank-profile implementation review

**Review scope (independent, read-only; 2026-09-18):** Appendix H of
`EXPERIMENT_GOVERNANCE.md`; candidate implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_rpn_rank_profile.py`; and the
declared D1-fullval/B0 lineage. No code, experiment output, or model was
modified by this review.

**VERDICT: FAIL — do not run a D2 capped smoke yet.**

The central per-level rank implementation is conceptually well aligned with
the deployed one-class RPN: it recomputes sigmoid scores, uses descending
`torch.sort`, produces one-based rank via inverse permutation, uses the
correct per-level caps, and never writes proposals. Its complete-run bootstrap
is correctly scoped to the 85 target-bearing D1 scenes, with capped smokes
leaving the full-population CI undefined. `py_compile` and project-environment
CLI parsing pass. The failures below are provenance and contract gaps that
would let a wrong D1/B0 population or stage record be relabeled as a valid D2
rank result.

## Blocking fixes

1. **Direct B0 key lineage is absent.** Appendix H requires the D1 107-target
   key set to be equal to B0. The script accepts a D1 CSV with the right row
   count and category counts, but never loads the B0 directory declared in
   D1's lineage, regenerates the B0 small/RPN-miss filter, or compares the two
   complete unique key sets. `by_scene` can also overwrite a duplicate key
   before the later `len(out)==107` completeness check. Load B0 directly,
   fail-close on its checkpoint/259-scene/invariants, assert 107 unique keys,
   and compare B0, D1, and D2 key sets before processing any image.

2. **D1 stage-IoU/category replay is incomplete.** The current loop compares
   live final RPN IoU only with `deployed_replay_max_iou` from the D1 row and
   tests category-implied rank inequalities. It does not compare D1's stored
   `raw_max_iou`, `pre_nms_max_iou`, `nms_unlimited_max_iou`, and
   `final_max_iou`, nor independently verify that the D1 category follows
   those four values. This violates H.1's fail-closed D1 category/stage-IoU
   replay. Read the B0 replay and D1 per-stage values, assert the live final
   against both B0 and D1, re-establish the expected category from the D1
   stage-IoU tuple, and record the comparisons per target. Rank separation is
   an additional D2 contract, not a substitute for it.

3. **Normal-prediction parity is not typed.** `output_digest` is inherited
   from the B0 helper, which hashes only contiguous bytes and does not bind
   shape/dtype. Appendix H explicitly asks for typed normal-prediction and
   deployed-RPN hashes. Replace it with local shape+dtype+bytes digests for
   bbox, score, label, and mask fields (the RPN `pair_hash` is already typed),
   archive those values, and make failure an `INVALID` reason.

4. **The required bootstrap artifact is not written.** D2 writes the
   bootstrap only inside `summary.json`; it does not create `bootstrap.json`,
   unlike B0/C1/D1 and the registered artifact convention. Write a distinct
   bootstrap artifact (`null` is acceptable only for an intentionally
   incomplete smoke) and include it in the run archive. This is required for
   independent result-side recomputation.

## Required smaller hardening

- Persist a per-target immutable lineage record: B0 final replay, all four D1
  stage IoUs, recomputed category, raw candidate/rank contract, and the
  positive candidate level/rank list or digest. Current output preserves only
  derived rank statistics and category, making stage replay un-auditable after
  the run.
- Add a `--device` argument or record the hard-coded CUDA device explicitly.
  This is not a scientific-validity gate, but avoids a silent GPU-placement
  discrepancy with the project wrappers.
- Persist the 23/25 control *expected denominators* along with their passed
  counts. Per-target throws currently protect correctness, but an explicit
  denominator makes a later audit able to distinguish missing from passing
  controls.

## What may be retained

- The fixed category-count requirement (`23/59/2/23`), exact current live RPN
  configuration assertion, strict checkpoint load, 259-scene validation split,
  raw-geometry negative control, selected nms/max positive separation
  control, repeated feature/logit/anchor/decoded-rank hashes, and no-proposal
  check all fit Appendix H's intended read-only diagnostic.
- `cap=2000` is correctly the only decision cap; `1250/1500/4000` are stored
  as profile telemetry. The 85-scene paired bootstrap redraw logic correctly
  handles all-zero denominators, provided it is written as an artifact on the
  complete scan.

## Re-review and smoke condition

After the four blocking repairs, re-request implementation review. The capped
smoke must process a deterministic nonempty subset of D1 target scenes and
be `INVALID` **only** for `validation_target_coverage_incomplete`; all B0/D1
key and stage-IoU lineage, typed normal/RPN/readout repeat hashes, rank/control
contracts, no-write condition, and output schemas must pass. Only a result
audit of that smoke may authorize the fixed 59-target/85-scene full scan.
Neither smoke nor a future D2 pass authorizes training.

---

## Re-review of the B0/D1 replay revision — 2026-09-18

**VERDICT: HOLD — do not run the capped smoke until the two remaining
per-target lineage checks below are added.**

This supersedes the prior four blocking findings in part: direct B0 loading,
unique D1 key checks, stage-IoU readout, typed normal digest, a device
argument, control denominators, and a standalone `bootstrap.json` are now
present and correct. The script compiles and its CLI parses in the project
environment.

### Repairs verified

- B0 is loaded directly, constrained to complete 259-scene/5,937-instance
  lineage and the checkpoint, and its 107 filtered keys are compared with the
  D1 keys after an explicit duplicate-key rejection. This repairs the former
  key-set overwrite hazard.
- D2 invokes D1's vetted `staged_readout` and now recomputes all four
  raw/pre-NMS/unlimited/final IoUs for each target, comparing them to the D1
  target row at `1e-5`; it also requires exact typed staged-final/live RPN
  equality per scene.
- The imported post-RPN `output_digest` is genuinely typed (dtype + shape +
  bytes), unlike the older B0 helper. Normal prediction, deployed RPN, rank
  readout, and staged final therefore have appropriate parity records.
- Controls have declared expected denominators (23 raw, 25 selected), the
  CUDA device is explicit/configurable, and `bootstrap.json` is written
  (including `null` on an intentionally incomplete smoke).

### Remaining blockers

1. **B0 final-IoU numeric replay is still indirect.** `b0targets` is a set of
   keys only. The live replay is compared with D1's
   `deployed_replay_max_iou`, but not with B0's original
   `max_rpn_box_iou`. Appendix H requires B0 final RPN IoU to be replayed,
   not merely D1's copy. Retain a B0 row mapping, compare live replay against
   both B0 and D1 values at the fixed tolerance, and archive both values per
   target.

2. **The D1 category is not recomputed from the four replayed stage IoUs.**
   Observed values are compared to D1 values, and rank inequalities cover raw,
   main, and a combined selected-control group. But an archived
   `nms_suppressed` tuple can be relabeled `max_per_img_truncated` (or vice
   versa) while satisfying the shared `min_positive_rank <= 1000` control.
   Implement the D1 `category(raw, pre, unlimited, final)` function locally,
   require it to equal `r["category"]`, and archive expected/observed stage
   IoUs plus the recomputed category in `D2 targets.csv`. This also makes the
   asserted D1 stage replay independently result-auditable.

After these bounded additions, request another re-review. The intended smoke
remains `INVALID` only for incomplete validation coverage; its B0 numeric
replay, four-stage category recomputation, typed parity/repeat logs, controls,
and no-write contract must all pass. No D2 cap result may be interpreted or
used to change detector configuration before that smoke review.

---

## Third re-review — direct B0 numeric replay and category reconstruction — 2026-09-18

**VERDICT: PASS — authorize a deliberately incomplete D2 capped smoke.**

The two remaining HOLD conditions are now implemented correctly and
fail-closed; the script compiles in the project environment.

- B0 is now retained as a keyed row map, not only a key set. For every target,
  the live deployed final-RPN IoU is compared independently against both
  B0 `max_rpn_box_iou` and D1 `deployed_replay_max_iou` at `1e-5`. The D2
  target CSV archives all three replay values.
- D2 recomputes raw, pre-NMS, NMS-unlimited, and final maximum IoUs from the
  live `staged_readout`; each is compared to the D1 target row. It then calls
  the same D1 `category` function on the observed four-tuple and requires that
  category to equal the archived D1 category. This closes the former ambiguity
  between `nms_suppressed` and `max_per_img_truncated`. Expected and observed
  stage IoUs plus both D1/recomputed categories are archived per target.

The earlier verified safeguards remain: 107 unique B0=D1 target keys and
fixed 23/59/2/23 categories, typed normal/deployed/final/readout parity,
rank-readout repeat hashes, explicit 1,000 per-level contract, raw and
selected controls, no proposal write, 85-scene bootstrap with a separate
artifact, and no detector/UAV modification.

### Smoke authorization

Run a deterministic capped subset of D1 target-bearing scenes using a small
bootstrap draw count. It must produce `INVALID` **only** for
`validation_target_coverage_incomplete`; `bootstrap.json` may be `null` for
that intentionally incomplete run. All B0/D1/D2 numeric lineage comparisons,
category reconstruction, rank/control contracts, typed hashes, and no-write
bits must pass and be archived. A separate result-side smoke audit is required
before authorizing the complete 85-scene / 59-main-target D2 scan. This PASS
does not authorize a proposal-cap intervention or training.

---

## D2-smoke result audit — 2026-09-18

**VERDICT: HOLD for complete-scan authorization.** The smoke's generic
readout/lineage wiring passes, but it does not exercise D2's primary 59-target
rank-recovery branch and is therefore insufficient to authorize the full scan.

### What passed

- `exit_code` is `0`; the manifest and summary agree on `INVALID`; the sole
  invalid reason is `validation_target_coverage_incomplete`. `bootstrap.json`
  is literal `null`, which is correct for this intentionally incomplete scan.
- It processes one D1 target-bearing scene and one target. Its archived B0,
  D1, and D2 final replay IoUs are all exactly `.4926962256`; all four D1 and
  recomputed D2 stage IoUs agree, and both categories are
  `nms_suppressed`.
- All typed local controls are true: normal prediction parity, deployed-RPN
  parity, staged-final/live equality, repeated feature/logit/anchor/decoded
  rank readout, and no proposal write. The target CSV contains the complete
  three-way replay and four-stage provenance fields.
- The live rank contract is the registered `nms_pre=1000`, `max_per_img=1000`,
  `min_bbox_size=0`, default RPN threshold `0`, and NMS `.7`. The selected
  control is correctly exercised: its minimum positive rank is 523, below the
  1,000 per-level selection limit.

### Why this cannot authorize full D2

The first deterministic target scene (`002002`) contains no
`score_nms_pre_rejected` main target. Consequently
`main_targets_processed=0`, all recovery-by-cap fields are empty, the primary
median/p90 are null, and the cap=2000 recovery path is never exercised. A
generic capped smoke is allowed to omit a population CI, but it must execute
the primary branch whose full-run decision it is intended to validate.

Run one additional deliberately incomplete smoke whose selected target-bearing
scene contains at least one of the 59 main targets (and preferably retain this
already-passing selected-control smoke as complementary evidence). It must
again be `INVALID` only for coverage, while recording nonempty main recovery
at all four caps and passing all B0/D1/D2 replay/category, typed-hash, rank,
and no-write contracts. Only then request re-audit for 85-scene full-scan
authorization. This HOLD does not authorize intervention or training.

---

## Explicit-main-smoke selector review — 2026-09-18

**VERDICT: PASS — authorize one additional capped D2 smoke with
`--scene-id 002009`.**

The new selector is a bounded smoke hook. Without `--scene-id`, the existing
behavior is unchanged: D1 target scenes remain in dataset order and are either
limited by `--max-target-scenes` or all processed for the full scan. With the
flag, the program first verifies that the supplied scene belongs to the
validated D1 target-scene map and then processes exactly that one scene. It
therefore cannot silently introduce a non-lineage scene or produce a full-run
decision (`complete` remains false and coverage remains the sole expected
`INVALID`). Compilation succeeds.

Independent inspection of D1-fullval confirms `002009` is a valid target
scene and contains two B0/D1 targets: gt 24 is the raw-geometry control and
gt 25 is `score_nms_pre_rejected`, i.e. one D2 main target. This selection
will exercise the previously untouched rank-margin, all-cap recovery, and
main-summary branches while also retaining a negative control in the same
scene.

Run with a small bootstrap-draw count. The new smoke must retain the prior
three-way B0/D1/D2 replay/category, typed parity/repeat, no-write, and control
checks; it must be `INVALID` solely for incomplete coverage, with a nonempty
main recovery-by-cap record. It does not authorize the full scan until a
result-side audit passes.

---

## D2 primary-branch smoke result audit — 2026-09-18

**VERDICT: PASS — authorize the complete D2 scan over all 85 target-bearing
scenes / 107 B0 targets / 59 main targets, with the registered 10,000-draw
85-scene bootstrap (seed 20260921).**

Independent inspection of `outputs/MHBEV-DIAG-20260918/D2-smoke-main` confirms
that this complementary smoke executes the primary D2 branch that the first
selected-control smoke did not:

- `exit_code` is `0`. The manifest and summary agree on `INVALID`, with the
  sole invalid reason `validation_target_coverage_incomplete`; `bootstrap.json`
  is correctly literal `null` for the one-scene smoke.
- Explicit scene `002009` processes two lineage targets: raw-geometry control
  gt 24 and main score/`nms_pre` target gt 25. Therefore
  `main_targets_processed=1`, the cap profile is nonempty, and all four caps
  report recovery `1.0`.
- The main target's minimum positive rank is 1,099 (margin `+99` to the live
  1,000 boundary), with valid positives on levels 0 and 1. It is therefore
  correctly excluded by deployment yet retained at each registered cap
  1,250/1,500/2,000/4,000. Its raw/pre/unlimited/final IoUs are
  `.75930554/.45218939/.34874737/.32323629`; B0, D1, and D2 final replay all
  equal `.32323629`; its D1 and reconstructed categories both equal
  `score_nms_pre_rejected`.
- The raw control has no IoU≥.5 raw candidate and correctly records empty
  rank/profile fields. Its B0/D1/D2 final replay and stage-category replay
  also agree. This establishes both main and negative-control branch wiring.
- All typed normal/deployed/final parity, readout-repeat, B0/D1/D2 replay and
  category, live-RPN contract, and no-proposal-write invariants are true.

This authorizes only the registered read-only D2 full scan. Its result must
undergo a new independent result audit before any conclusion. No smoke result
authorizes a proposal-cap intervention, AP claim, or training.
