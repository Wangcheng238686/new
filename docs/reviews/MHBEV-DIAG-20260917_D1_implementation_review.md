# MHBEV-DIAG-20260917-D1 — satellite RPN stage-attribution implementation review

**Review scope (independent, read-only; 2026-09-17):** Appendix G of
`EXPERIMENT_GOVERNANCE.md`; the candidate implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_rpn_stage_attribution.py`;
the B0 diagnostic helpers/artifacts; the actual installed mmdetection
`RPNHead`; and the loaded baseline configuration lineage.  No code, data, or
experiment output was changed by this review.

**VERDICT: FAIL — do not run a D1 smoke or complete scan.**

The target filter, raw-anchor decoder, B0 replay check, fixed bootstrap design,
and read-only intent are materially on the right path.  Two deterministic
stage-semantics defects make the current program unable to establish the
Appendix-G four-stage attribution; the first one prevents it from running at
all.

## Blocking defects

1. **The declared deployed pre-NMS readout cannot execute.**
   `staged_readout` calls
   `head.predict_by_feat(..., with_nms=False)` (script line 84).  The deployed
   installed `mmdet.models.dense_heads.rpn_head.RPNHead._bbox_post_process`
   contains `assert with_nms, '`with_nms` must be True in RPNHead'`.  Thus the
   first processed scene raises an assertion before target telemetry or a
   summary can be produced.

   The repair must implement an exact, read-only pre-NMS mirror of the *live*
   RPN path: per-level sigmoid/softmax activation, resolved live `score_thr`,
   `filter_scores_and_topk(..., nms_pre, ...)`, common bbox decode, and the
   live `min_bbox_size` filter, stopping immediately before `batched_nms` and
   max-per-image truncation.  Do not silently substitute raw anchors or a
   generic BaseDenseHead call.  Since the Appendix's `with_nms=False` spelling
   is infeasible for this concrete `RPNHead`, record the upstream assertion
   and the exact source-mirrored equivalence (including source/config hash) in
   the implementation/telemetry before smoke; it is a feasibility correction,
   not a data-driven change.

2. **`max_per_img=-1` is not unlimited for this RPN implementation.**
   The candidate creates `unlimited_cfg["max_per_img"] = -1` (line 85).
   Deployed `RPNHead._bbox_post_process` always executes
   `results = results[:cfg.max_per_img]`; Python/InstanceData `[:-1]` removes
   the final NMS survivor.  Consequently this cell is neither NMS-unlimited
   nor nested faithfully with the final cell, even if the pre-NMS call above
   were fixed.

   Set `max_per_img` to a proved non-binding finite upper bound, e.g. the sum
   of the live per-level post-`nms_pre` capacities (5 × 1000 = 5000 for the
   current live contract; compute it from the actual dense layout, not a magic
   constant).  Persist the bound and assert the number of NMS survivors is at
   most that bound, so the slice cannot truncate.  Label that archived cell as
   `nms_unlimited` only after this assertion.

## Further required audit hardening before smoke

3. **Live-final semantic equality is checked only by maximum IoU.**
   D1 correctly compares `final_max_iou` to both B0's recorded target IoU and
   a `read_rpn` replay.  Equal maximum IoU, however, does not prove that its
   `predict_by_feat` final boxes/scores are the same live RPN output.  For each
   scene, compare typed hashes of the staged-final boxes/scores (and the fixed
   RPN labels, or assert their contract) with `read_rpn`'s deployed output,
   fail closed, and archive the comparison.  Keep the per-target B0 IoU check
   as the separate B0-lineage invariant.

4. **The requested cross-stage feature/anchor/raw-box invariant is not
   auditable.**  The code shares one `x`, logits, deltas and `dense_raw` call
   structurally, and repeats raw/pre output hashes.  It does not compute or
   archive typed hashes for input feature tensors, generated anchors, decoded
   raw boxes, or the resolved per-stage configs.  Appendix G.2 explicitly
   requires those to be invariant while only the registered selection/NMS/
   max-per-image switches change.  Add shape+dtype+bytes hashes (the inherited
   `array_hash` hashes bytes only) and a per-scene record of raw/pre/unlimited/
   final config fingerprints.  The pre cell must explicitly state that NMS and
   truncation are absent; unlimited and final must differ only in the proven
   non-binding versus live `max_per_img` value.

5. **Live configuration is merely recorded, not fail-closed against the
   baseline contract.**  A scene-to-scene equality check is useful but cannot
   detect loading a consistently wrong config.  Resolve and assert the live
   D1 baseline contract before processing: current source specifies
   `nms_pre=1000`, `max_per_img=1000`, NMS type `nms` with IoU `.7`, and
   `min_bbox_size=0`; absent RPN `score_thr` resolves to the framework default
   `0`.  Persist the resolved values.  This also gives the pre-NMS mirror an
   unambiguous contract.

6. **Hash telemetry is currently only boolean and is not typed.**  `stage_hash`
   uses helper `array_hash` after `instance_arrays` converts boxes/scores to
   CPU float arrays; it does not encode shape/dtype, and only summary booleans
   are retained.  Use a local typed digest and persist pre/post values (or an
   immutable per-scene composite digest) for normal prediction, deployed RPN,
   raw, pre-NMS, unlimited, and final cells.  This is required to make a later
   result-side audit independently verify the requested hash control.

## What is already sound

- The B0 filter is correct and fail-closed on the expected checkpoint,
  259-scene / 5,937-instance B0 complete run, and 107 unique targets.
- `dense_raw` reads `extract_feat → rpn_head.forward`, uses live grid priors
  and `DeltaXYWHBBoxCoder`, decodes using the sample `img_shape`, avoids
  proposal writes, and checks every level's prior/delta/score/decoded
  cardinality.
- The intended four categories and their max-IoU monotonicity check match
  Appendix G, provided the cells are repaired to be true nested subsets.
- B0 per-target deployed replay is fail-closed at `.5` and checked against the
  declared final cell; 259-scene zero-filled image-paired bootstrap, 10,000
  draws, seed `20260920`, and the observed-major-vs-rest percentile CI are
  correctly expressed.  No UAV data or training path is invoked.

## Re-review / smoke condition

After all six items are repaired, run a compile/import check and request
re-review before smoke.  The first smoke should use a deliberately incomplete
target-scene cap; it must be `INVALID` **only** for validation-target coverage
while B0 lineage/replay, exact live-final equality, nested stages, typed
feature-anchor-raw hashes, resolved-config contract, repeat controls, and
no-proposal-write all pass.  Only then may the fixed 107-target, 259-scene
scan be authorized.  Even a future D1 stage conclusion authorizes only a new
frozen satellite-stage intervention proposal, never training.

---

## Re-review after framework-semantics correction — 2026-09-17

**VERDICT: HOLD — the two initial execution blockers are repaired, but do not
run D1 smoke until the remaining Appendix-G hash/equivalence controls are
implemented and re-reviewed.**

This re-review supersedes the two specific blocking findings above, which were
true of the prior file only.

### Repairs verified in the current source

- Appendix G now accurately reflects this installed framework: its concrete
  `RPNHead` has no `score_thr` path and does not support public
  `with_nms=False` output.  `deployed_pre_nms` now directly mirrors the
  deployed RPN's per-level activation, descending `nms_pre` selection,
  selected-prior decode, and the one-foreground-class squeeze convention.
  It no longer invokes the assertion-failing public no-NMS call.
- The NMS-unlimited arm now uses `len(pre_boxes)+1`, which is strictly larger
  than the entire selected pre-NMS set.  Since RPN NMS only removes rows, this
  is a provably non-binding finite cap; it avoids the former `[:-1]` behavior
  caused by `max_per_img=-1`.
- `py_compile` and the project environment's CLI `--help` both complete.  The
  observed CUDA availability warning occurs during import-time torch probing;
  it does not affect parsing or this source-level review.

### Remaining blocking requirements

1. **Prove staged-final equals the live deployed RPN, not merely equal max
   IoU.**  The current loop only asserts
   `max_iou(first["final"], GT) == max_iou(read_rpn(...), GT)` for each B0
   target.  Different box/score populations can share that maximum.  Compare
   a typed composite hash of staged-final boxes/scores (and labels, or an
   explicit RPN-label invariant) against `read_rpn`'s live `head.predict`
   output per scene, fail closed, and persist the values/comparison.  Keep the
   B0 per-target replay check separately.

2. **Implement and archive the typed cross-stage invariant specified in
   G.2.**  Current `stage_hash` hashes only raw array bytes after conversion to
   CPU float, omits shape/dtype, and the output archives only booleans.  No
   feature, anchor, decoded-raw-box, or stage-config fingerprint is emitted.
   Add a shape+dtype+bytes digest; emit per-scene hashes for extracted feature
   inputs, each level's anchors, dense decoded raw boxes, raw/pre/unlimited/
   final boxes/scores, and resolved config fingerprints.  Assert that the
   feature/anchor/raw fingerprints are shared across all four stage cells and
   that the only config deltas are the registered pre-NMS/NMS/max-per-image
   controls.

3. **Make the live baseline configuration fail-closed.**  Checking equality
   from scene to scene cannot detect a consistently wrong loaded test config.
   Before the loop, resolve and assert the known baseline values
   (`nms_pre=1000`, `max_per_img=1000`, NMS `type=nms`, IoU `.7`,
   `min_bbox_size=0`, and no RPN `score_thr` key); record that resolved
   contract plus the non-binding unlimited bound in the artifact.  This is
   needed to substantiate the amended pre-NMS mirror.

4. **Make non-truncation and no-write evidence auditable.**  The bound is
   sound by construction but not persisted, and `no_proposal_write=True` is a
   summary literal rather than a recorded per-scene result.  Persist the
   pre-count, bound, NMS-unlimited count, `count <= bound` bit, and the
   no-proposal-write bit for every scene.  Include them in the invalid-reason
   checks.

The raw decoder/cardinality, B0 target filter, zero-filled 259-scene bootstrap
and category logic remain otherwise compatible with Appendix G.  After the
four additions, a deliberately incomplete smoke may be authorized only if it
is `INVALID` solely for incomplete target coverage and every new equivalence,
typed-hash, config, nesting, B0 replay, repeat, and no-write invariant passes.

---

## Second re-review — telemetry/equivalence completion — 2026-09-17

**VERDICT: PASS — authorize the deliberately incomplete D1 smoke only.**

The four HOLD conditions are now implemented as fail-closed checks rather than
summary-only claims.  `py_compile` succeeds in the project environment.  This
authorization is for a capped target-scene smoke; it does not authorize the
complete scan, any intervention, or training.

### Verified repair details

- **Typed stage records and exact live-final equality:** `stage_hash` now
  includes contiguous-array dtype, shape, and bytes.  Per scene `parity`
  archives normal prediction before/after digests, deployed RPN before/after
  hashes, and the staged-final hash; `staged_final_equal_live` compares the
  latter directly against the deployed `read_rpn` boxes/scores.  RPN final
  labels are additionally asserted to be all zero.  Failure of this equality
  is included in `prediction_or_deployed_rpn_parity_failed`.
- **Live config is concrete and fail-closed:** `assert_live_contract` resolves
  the actual loaded head and rejects anything other than `nms_pre=1000`,
  `max_per_img=1000`, `min_bbox_size=0`, absent/default RPN `score_thr=0`,
  NMS type `nms`, and IoU threshold `.7`.  The resolved contract is both
  archived and required to agree across scenes.
- **Shared readout provenance is archived:** every scene records feature and
  per-level anchor digests, dense raw/pre/unlimited/final stage digests, and
  each stage's explicit config map.  All stage cells are created from the same
  `extract_feat → rpn_head.forward` tensors in one `staged_readout` call;
  raw/pre/final are independently repeated and compared.  The raw decoder
  still checks per-level anchor/delta/score/decoded cardinality.
- **Unlimited/no-write control is real:** the per-scene archive carries the
  dynamically derived `len(pre)+1` bound, survivor count, non-truncation bit,
  and no-proposal-write bit.  The method raises immediately on a proposal
  write or `unlimited_count >= bound`, and either false bit enters
  `unlimited_or_no_write_contract_failed` / `INVALID`.
- **Stage semantics:** the corrected pre-NMS replay faithfully matches this
  concrete RPN's per-level activation/ranking/`nms_pre` code path and applies
  the live zero-size filter before subsequent NMS.  With the asserted live
  `min_bbox_size=0`, that filter cannot remove an IoU≥.5 positive (such a box
  necessarily has positive width and height), so it cannot alter the target
  stage attribution.  The NMS-unlimited and final cells use the live public
  RPN postprocess, with the former's bound proven non-binding.

### Smoke contract

Run all normal readouts on a deliberately capped number of B0 target scenes
and use a small bootstrap draw count suitable for wiring.  The resulting
decision must be `INVALID` **solely** for
`validation_target_coverage_incomplete`.  It must retain all of: 107-target
B0 lineage construction, exact staged-final/live equality, B0 target replay,
normal/deployed/raw/pre/final repeat controls, resolved live config,
feature-anchor-stage hash telemetry, dense cardinality, nested-stage checks,
unlimited non-truncation, and no-proposal-write checks.  Request a
result-side audit of that smoke before the complete 107-target scan.

---

## Smoke-bootstrap scope correction review — 2026-09-17

**VERDICT: PASS — retain the capped-smoke authorization.**

The minimal correction is appropriate.  A capped scan intentionally contains
only a subset of the 85 target-bearing scenes while its 259-scene bootstrap
table zero-fills all unprocessed scenes.  A resample consisting entirely of
those zero rows has no denominator, so the Appendix-G *full-scan* CI is
undefined for a smoke and must not prevent summary/telemetry emission.

The current code computes `bootstrap(...)` only when both `total > 0` and
`complete` are true.  It always constructs the complete 259-scene zero-filled
table, but does not use it to make a partial-data CI claim.  Independently,
the capped run still executes every target-scene readout and all local
fail-closed controls before setting
`validation_target_coverage_incomplete`; the decision branch cannot access a
missing bootstrap because `invalid` is already non-empty.  For a complete
107-target run, the fixed 259-scene, 10,000-draw, seed-20260920 bootstrap is
unchanged.

Thus a capped smoke may correctly archive `bootstrap: null` and be `INVALID`
only for coverage, provided all other recorded invariants pass.  This is not
a relaxation of Appendix G's full-run significance gate and does not expand
authorization beyond the previously approved smoke.

---

## D1-smoke result audit — 2026-09-17

**VERDICT: PASS — authorize the complete D1 scan over all 85 target-bearing
validation scenes / 107 B0 targets, with the registered 259-scene, 10,000-draw
bootstrap (seed 20260920).**

Independent, read-only inspection of
`outputs/MHBEV-DIAG-20260915/D1-smoke` confirms the smoke met its sole
authorization criterion:

- `exit_code` is `0`; run log ends in a written `summary.json` with no
  exception.  Manifest and summary agree on `decision=INVALID`.
- It processed exactly one target-bearing scene (`002002`) and one B0 target,
  while retaining the declared 85-scene/107-target universe.  The sole,
  explicitly recorded invalid reason is
  `validation_target_coverage_incomplete`.  `bootstrap.json` is literal
  `null`, correctly reflecting the intentionally undefined full-universe CI
  for this capped smoke.
- Every local invariant is true: normal prediction parity, deployed RPN
  before/after parity, exact staged-final/live RPN equality, raw/pre/final
  repeat equality, dense cardinality, B0 lineage/replay, and no proposal
  write.  The archived normal, deployed, and staged-final hashes agree where
  required.
- The resolved live contract is exactly the registered contract
  (`nms_pre=1000`, `max_per_img=1000`, `min_bbox_size=0`, resolved RPN
  `score_thr=0`, NMS `.7`).  Five dense levels have the expected equal
  prior/delta/score/decoded counts.  Feature/anchor/stage digests and stage
  configuration maps are present.
- The unlimited cell uses bound `4997` for 4,996 selected pre-NMS boxes and
  retains 2,347 NMS survivors; its archived non-truncation and no-write bits
  are true.  The staged final has 1,000 boxes and its boxes/scores hashes are
  exactly equal to the live `read_rpn` output.  The single target's B0 replay
  is exactly `.4926962256`; dense/raw and pre-NMS maxima are `.9221851826` and
  `.5077049732`, while unlimited/final are `.4926962256`, yielding the
  expected provisional `nms_suppressed` category without making a population
  conclusion.

The complete scan remains read-only/probe-only.  Before any interpretation,
it requires a fresh independent **result-side audit** of all 107 target rows,
the full bootstrap, all telemetry, and the Appendix-G maximum-stage gate.
Nothing in this authorization permits intervention or training.
