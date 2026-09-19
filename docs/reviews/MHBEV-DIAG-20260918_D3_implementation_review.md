# MHBEV-DIAG-20260918-D3 — frozen satellite objectness-readout implementation review

**Review scope (independent, read-only; 2026-09-18):** Appendix I of
`EXPERIMENT_GOVERNANCE.md`; candidate implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_sat_rpn_objectness_signal.py`;
and its declared B0/D1 lineage. No code, data, model, or experimental output
was changed by this review.

**VERDICT: FAIL — do not run a D3 capped smoke.**

The intended readout is promisingly narrow and several core pieces are sound:
fixed 128-scene train versus val isolation, a frozen standard-scaler/logistic
readout fit only on train, same-level score-matched negative selection, a
global label-permuted train sham, raw/deployed/staged-final parity calls, and
per-target D1 stage-IoU reconstruction. However, the current bootstrap unit,
pair identity, mapping validation, and required negative-control/archive
evidence do not yet meet Appendix I.

## Blocking defects

1. **The bootstrap has the wrong scene universe.** Appendix I requires an
   85-D1-scene paired bootstrap, with non-main scenes contributing zero pairs.
   The program builds `ids` only from `byscene`, i.e. scenes holding one of the
   59 main targets, and `stats`/`boot` use only that reduced set. This changes
   resampling weights and CI. Construct the 85 target-bearing D1 scene list
   from the full D1 target CSV, process main rows where present, then create a
   fixed 85-row `per_scene` table with zero rows for non-main scenes before
   bootstrap. Persist the universe/count and fail-close it against D1.

2. **Pair SHA does not bind the pair to scene/GT.** `pairhash(rec)` receives a
   record containing level, indices, boxes, and scores but not `scene_id` or
   `gt_index`; Appendix I expressly requires both in immutable pair identity.
   Pass scene and GT into the pair constructor and SHA record. Archive the
   pair identity used by correct and sham (or explicit same-pair condition
   hashes) in both train and validation rows.

3. **Raw-score-only tie baseline is absent.** It is preregistered both as a
   companion metric and a negative-control requirement. The script filters on
   score gap but neither writes raw-score win/tie bits nor summarizes the
   baseline rate/CI. For every valid pair record `pos_score > neg_score`,
   `<`, and equality (or a fixed tie rule), report the aggregate rate, and
   guard against interpreting a descriptor win that is simply inherited from
   the raw-score ordering. The label-permutation sham is present, but it does
   not replace this independent baseline.

4. **Anchor-to-FPN-cell mapping is under-asserted and not auditable.**
   `idx // a → (cell // w, cell % w)` is plausible for this RPN's
   location-major anchor order, but code only checks `len(prior) == a*h*w` and
   final flattened lengths. It does not assert RPN cls/delta channel anchor
   multiplicity, FPN feature spatial size equals the RPN logits H/W, FPN
   channel dimension is the registered 256, or archive cell coordinates and
   per-level layout. Add checks such as
   `cls_channels == anchors_per_cell * cls_out_channels`,
   `delta_channels == anchors_per_cell * encode_size`, and
   `feature.shape[-2:] == (h,w)`, plus a one-time/order-contract assertion
   against the live grid-prior construction. Persist `level`, anchor index,
   anchor-cell index, `(row,col)`, feature H/W and anchors/cell in rows or a
   dedicated layout CSV. Without this, a changed FPN/anchor layout can give
   every descriptor a consistent but wrong cell.

5. **Required archive outputs are incomplete.** The script does not write a
   `run_manifest.json`, and it emits no layout/cell-mapping artifact. Appendix
   I inherits the experiment archive requirement; a later result audit cannot
   fully validate lineage/config/schema from summary alone. Add manifest,
   bootstrap (already present), fixed train/val pair schemas, and a descriptor
   layout/mapping artifact. On capped smoke the bootstrap may be `null`, but
   the manifest and mapping archive remain mandatory.

## Additional required hardening

- Retain B0 rows as a map and compare live final replay both to B0
  `max_rpn_box_iou` and the D1 deployed replay field; current code compares
  only B0, while D1 final stage is compared through a separate tuple. Archive
  all three values.
- Make `repeat_feature_descriptor_hash`,
  `d1_b0_replay_and_rank_separation_fail_closed`, and `no_proposal_write`
  computed per-scene invariants in telemetry rather than summary literals.
  Current exceptions help, but archival booleans/hashes should substantiate a
  successful run.
- Persist the sham-label permutation seed and a SHA of the permuted train
  labels, and verify it is not identical to the correct label order. This
  makes the sham reproducible rather than merely plausible.

## What is correct so far

- The main D1 target count is checked at 59 and all main keys are constrained
  to the B0 RPN-miss set; train/val scene IDs are asserted disjoint.
- The positive is the highest raw-score decoded IoU≥.5 candidate, while the
  negative is same-level, all-GT IoU<.1 and closest on raw score. Both train
  and val apply the `.02` score-gap criterion.
- `rank_readout` itself uses no UAV path, stays under inference mode, repeats
  its feature/anchor/box/score/rank hashes, and does not write proposals.
- The fixed correct classifier and seed-20260923 label-permutation sham use
  the same train descriptors; the fixed 10,000/20260924 bootstrap concept and
  the decision inequalities agree with Appendix I once the 85-scene universe
  is repaired.

## Re-review and smoke condition

After these corrections, request implementation re-review. A capped smoke
must use all 128 fixed training scenes and a deterministic, nonempty subset of
the 85 D1 scenes containing at least one main target; it must be `INVALID`
only for validation coverage. It must still execute nonempty correct/sham/raw
score baseline pair paths, all descriptor-cell assertions, B0/D1 replay and
category checks, typed before/after/repeat parity, and no-write controls.
Only a result-side audit may authorize the complete 59-target / 85-scene scan.
Neither smoke nor a positive D3 result authorizes full detector training.

---

## Re-review of the descriptor/lineage revision — 2026-09-18

**VERDICT: HOLD — do not run a D3 capped smoke until the three remaining
negative-control/mapping checks below are made fail-closed.**

Most of the prior blocking implementation gaps are repaired in the current
source:

- The complete-run bootstrap now has exactly 85 D1 target-bearing scene rows;
  scenes without a main target are explicitly zero-filled. Capped runs skip
  the undefined full-population CI and remain coverage-`INVALID` only.
- Pair SHA records now bind `scene_id` and `gt_index`; raw-score win bits and
  a summary rate are emitted; B0 and D1 replay values are both compared; and
  per-scene repeat/no-write fields, descriptor layout CSV, bootstrap JSON, and
  run manifest are written.
- The mapping code now asserts RPN class/delta channel multiplicity, FPN
  spatial shape, 256 feature channels, anchor/feature cardinality, and records
  per-level layout. Train/val isolation and typed before/after/repeat paths
  remain correct. The script compiles and its CLI parses.

### Remaining blockers

1. **The raw-score negative control is reported but not evaluated.** Appendix
   I says the raw-score-only tie baseline must not systematically favor the
   correct direction. Current `raw_score_win_rate` merely reports
   `pos_score > neg_score`; it cannot make a run invalid or prevent a positive
   D3 decision when raw score itself wins most matched pairs. Pre-register a
   concrete criterion before smoke (for example, an 85-scene paired bootstrap
   CI for raw-score win minus `.5` whose upper bound must not be positive),
   compute/archive it from the same valid pair table, and add failure to
   `invalid_reasons`. Do not choose a criterion after observing D3 results.

2. **The label-permutation sham is not reproducibly verified.** The seed is
   used but neither the permuted-label SHA nor a non-identity assertion is
   saved. Persist seed, original/permuted label typed hashes, class counts, and
   `permuted != original`; make accidental identity or balance drift invalid.
   This is required to establish the sham as a genuine negative control.

3. **Cell ordering is still assumed, not runtime-asserted.** The installed
   AnchorGenerator source does document location-major order (the first `A`
   anchors are cell `(0,0)`, then `(0,1)`), so `idx // A` is currently the
   right mapping. But the probe's contract must fail closed if that framework
   behavior changes. Add a direct equivalence assertion between generated
   `grid_priors` and `sparse_priors(arange(N), (H,W), level, ...)` (or an
   equivalent sampled full-order check), record its typed digest/result, and
   persist positive/negative cell indices/row/column alongside anchor indices
   in pair CSV. Present generic layout alone does not prove every descriptor
   was taken from its intended anchor cell.

After these bounded fixes, request re-review. The smoke must use all 128 fixed
train scenes and explicit main-target scene `002009` (or another registered
main scene), execute correct/sham/raw-score baseline paths, be `INVALID` only
for coverage, and leave all new negative-control/order assertions passing.

---

## Third re-review — raw-score/sham/order-control revision — 2026-09-18

**VERDICT: HOLD — do not run D3 smoke yet.**

The raw-score and sham controls are now correctly preregistered and implemented:

- Appendix I fixes the 85-scene raw-score win-minus-.5 bootstrap upper-bound
  gate at `.15`; a complete run appends `raw_score_match_control_failed` to
  `INVALID` when exceeded.
- The sham records seed, original/permuted typed label SHA-256 values and
  class counts, and rejects an identity permutation before fitting.

However, the explicit anchor-order repair is not actually implemented as
claimed. Appendix I requires `sparse_priors` and `grid_priors` to be verified
**elementwise** at runtime. `dense` instead tests only indices `0`, `N//2`,
and `N-1`. A changed ordering in an unprobed region could pass while mapping
descriptors to the wrong FPN cell. Replace this with all-index `arange(N)`
equivalence (or an equally exhaustive proof) and archive the assertion/digest.

The pair CSV also stores only anchor indices, not derived positive/negative
cell index and `(row,col)`. Add these fields so each descriptor mapping is
directly auditable. Finally, handle complete runs with zero valid pairs as an
explicit invalid/coverage-control result rather than dereferencing `boot=None`
in the raw-score gate.

After these bounded fixes, request re-review. The cell-order contract remains
a hard smoke gate.

---

## Fourth re-review — exhaustive cell-order and zero-valid revision — 2026-09-18

**VERDICT: PASS — authorize one capped D3 smoke with all 128 fixed train
scenes and explicit validation scene `002009`.** This authorization is only
for a frozen diagnostic smoke; it does not authorize the 85-scene scan,
intervention, or detector training.

The three previously outstanding implementation details are now genuinely
fail-closed in the current source, rather than merely described in Appendix I:

- For every FPN level, `dense()` builds `arange(len(grid_priors))`, calls
  `sparse_priors` for **every** index, requires elementwise `torch.equal`, and
  separately requires equal typed digests. A mismatch raises before a pair or
  descriptor is admitted. Both `prior` and `sparse_prior` digests are stored
  in each readout's per-level hash record.
- Every pair record now carries positive and negative anchor index, cell index,
  cell row, and cell column. Since `cell = anchor_index // anchors_per_cell`
  is backed by the exhaustive grid/sparse equivalence, this makes the exact
  descriptor-cell lookup auditable in both train and validation CSVs.
- The complete-with-zero-valid case now records
  `no_valid_score_matched_pairs` as `INVALID`, leaves `boot` as `None`, and
  guards the raw-score CI dereference with `boot is not None`. Capped runs
  remain `INVALID` solely for incomplete validation coverage, as Appendix I
  specifies; their result audit must separately confirm a nonempty main-pair
  path was exercised.

The existing raw-score CI gate, nonidentity/hash-recorded sham, 85-scene
zero-filled bootstrap universe, B0/D1 four-stage replay, train/val isolation,
typed parity/repeat checks, and no-proposal-write telemetry remain present and
consistent with Appendix I. The script also passes Python compilation.

### Authorized smoke envelope

Run the fixed 128-scene train split (do **not** set `--max-train-scenes`) with
`--scene-id 002009`; `002009` is a registered D1 main-target scene (target
`gt_index=25`). A small bootstrap count is sufficient because the capped run
must not claim the full-population CI. The smoke is acceptable only if it exits
zero, is `INVALID` for exactly `validation_target_coverage_incomplete`, has a
nonempty valid score-matched main pair, and all typed/replay/order/no-write
controls pass. Its validation CSV must contain the new cell coordinates and
its descriptor telemetry must contain matching prior/sparse-prior hashes. A
result-side review remains required before authorizing the complete 59-target /
85-scene scan.

---

## Fifth re-review — deployed PAFPN descriptor width correction — 2026-09-18

**VERDICT: PASS — authorize a rerun of the exact previously authorized D3
smoke envelope.**

This is a required feasibility correction, not a relaxation of a diagnostic
gate. The failed smoke establishes that the former `f.shape[1] == 256`
assertion was false for the deployed detector. Source inspection independently
confirms the reason: `RSSAM2PAFPN` uses `mid_channels=256` internally, then
applies per-level `output_proj(mid_channels, out_channels)` to its four FPN
levels and appended P64 level. The frozen baseline configuration sets both
PAFPN `out_channels` and `rpn_head.in_channels` to 512. Thus the actual RPN
input is five 512-channel PAFPN maps; the backbone's 256-channel maps are not
the object being read by D3.

The code now requires each selected descriptor map to have
`f.shape[1] == head.in_channels`, while retaining the existing anchor/feature
H/W, channel-multiplicity, cardinality, exhaustive sparse-prior/grid-prior,
typed digest, repeat, replay, and no-write checks. Layout artifacts continue
to record the observed `feature_channels`, so the rerun must visibly report
512. Appendix I was corrected consistently: its descriptor is the current RPN
input feature and specifies the baseline's 512 dimensions, expressly excluding
the backbone's 256-dimensional map. Python compilation succeeds.

Rerun exactly the prior smoke: all fixed 128 train scenes, no
`--max-train-scenes`, and `--scene-id 002009`. It is still acceptable only
with exit zero, precisely the coverage-only `INVALID`, a nonempty valid main
pair, and every typed/replay/order/no-write invariant true. The earlier failed
run is a clean feasibility failure and supplies no scientific result. This
authorization does not extend to full validation, intervention, or training.

---

## Sixth re-review — AnchorGenerator sparse-index device contract — 2026-09-18

**VERDICT: PASS — authorize rerun of the same D3 capped smoke.**

The failed rerun is a framework device-contract failure, not a failed D3
control. The installed `AnchorGenerator.sparse_priors()` computes
`base_anchor_id = prior_idxs % A` and immediately indexes the persistent CPU
`self.base_anchors[level_idx]` with it. A CUDA index tensor is therefore
invalid, exactly as the recorded traceback reports. Its implementation then
moves the stacked sparse coordinate tensor and selected base anchors to the
explicit `device` argument before addition.

The revised all-index probe is consequently correct: it creates
`arange(N)` on CPU for the required CPU lookup, calls `sparse_priors(...,
device=dev)`, and compares its GPU-materialized output bitwise via
`torch.equal` and independently via the typed digest to GPU `grid_priors`.
No index range is sampled or omitted, and no descriptor, score, decoder,
replay, bootstrap, or decision logic changes. The script compiles.

Rerun exactly the same all-128-train / `--scene-id 002009` smoke. Retain the
previous acceptance conditions: exit zero; only
`validation_target_coverage_incomplete` as `INVALID`; nonempty valid main
pair; and all typed parity, B0/D1 replay, exhaustive prior-order, and no-write
controls true. This remains smoke-only authority, with no full scan,
intervention, or training authorization.

---

## Seventh re-review — train-pair score-gap feasibility correction — 2026-09-18

**VERDICT: PASS — authorize rerun of the same D3 capped smoke.**

The 128-scene train readout reached the intended path and failed with only
three pairs because the validation score-matching inclusion threshold was
incorrectly also applied to training. The source change is narrow and matches
the corrected Appendix I: training retains the same-level, nearest-raw-score
negative selection and archives each `score_gap`, but no longer rejects a
training pair for `gap > .02`. The 59-target held-out validation path remains
unchanged: it alone applies `gap <= .02` before a pair becomes valid, computes
the raw-score tie control, and is subject to the fixed 85-scene bootstrap CI
gate.

This preserves the anti-leakage and causal-readout boundary. The train and val
scene sets remain disjoint; fitting receives frozen descriptors and labels
only, not raw scores, UAV fields, validation rows, or validation-fitted
normalization. A broader training score distribution can make the fixed linear
readout estimable, but cannot itself pass the claim: the claim is tested only
on held-out raw-score-matched pairs, with the existing raw-score and sham
controls. Train pair count, labels, identities, and gaps remain archived for
result-side inspection.

Rerun the exact all-128-train / `--scene-id 002009` smoke. It remains valid
only under the previous exit/coverage-only-`INVALID`/nonempty-main-pair and
all-invariants conditions, and does not authorize full validation,
intervention, or training.

---

## Result audit — fourth D3 capped smoke (`D3-smoke-main`) — 2026-09-19

**VERDICT: PASS — authorize the complete frozen D3 scan of all 59 targets in
the 49 affected D1 scenes (with the fixed 85-scene bootstrap universe).** This
does not authorize any intervention or training; a separate result audit is
required before any follow-on proposal.

The smoke artifact set is internally complete for its capped scope and exited
zero. `summary.json` and the manifest report exactly 128 fixed train scenes,
158 archived train pairs, one processed target scene, one processed target,
and one valid pair. The sole result state is `INVALID` with the sole reason
`validation_target_coverage_incomplete`; `bootstrap.json` is correctly `null`
for this incomplete run. There is no hidden complete-run CI or positive-signal
claim.

The explicit main-branch validation target is present: `val_pairs.csv` records
`002009/25` as valid, preserves B0/D1/D3 replay IoU at
`0.32323628664016724`, and has score gap
`1.7969869077205658e-06`, well inside the preregistered `<=.02` validation
threshold. Its pair identity, positive/negative anchor and cell indices,
row/column coordinates, boxes, scores, classifier outputs, and correct/sham/
raw win bits are archived.

All stated smoke controls pass across the full 129-scene telemetry population
(128 train plus 1 validation): normal typed prediction parity, deployed RPN
typed parity, repeated feature/descriptor hashes, B0/D1 replay/category, and
no proposal writes are all true. Independently recomputing the archived
per-level records finds 645 layouts (129 x 5), all with 512 feature channels
and six anchors per cell, with `prior == sparse_prior` digest on every level;
all repeat and before/after hashes agree. The sham is a nonidentity permutation
with distinct typed label hashes and preserved 158/158 class counts.

The train-gap feasibility correction did not contaminate the validation
condition: train pairs have gap range approximately `3.73e-09` to `.999824`,
with 155/158 above `.02`, whereas the only validation pair was admitted only
through the unchanged `gap <= .02` branch. Neither score nor validation rows
are classifier inputs. The single-pair smoke rates (all 1.0, including sham
and raw) have no evidentiary meaning and are not used as a D3 result.

The authorized complete run must leave `--scene-id`, `--max-val-scenes`, and
`--max-train-scenes` unset; retain all 128 training scenes, process all 49
affected scenes/59 targets, and use the Appendix-I 10,000-draw, seed-20260924
85-scene bootstrap. Its result must be independently audited against the
coverage, raw-score-control, sham, CI, and decision gates before any new
experiment is considered.

---

## Full-run execution finding — `D3-fullval` — 2026-09-19

**VERDICT: FAIL — no D3 result exists; do not authorize any follow-on work.**

The authorized full run exited with code 1 and wrote only `run.log` and
`exit_code`, not summary, CSV, bootstrap, layout, or manifest artifacts. It
therefore cannot be audited as a diagnostic result. After the expensive
128-train / full-validation readouts, complete-run aggregation executes
`boot=boot(...) if complete and valid else None`. Python consequently treats
`boot` as a local variable and raises `UnboundLocalError` while resolving the
right-hand-side callable when `complete and valid` is true. Capped smoke did
not expose it because its incomplete short-circuit never evaluates that branch.

Apply only the mechanical local-name repair (for example, preserve the
function `boot` and call its result `bootstrap_result` consistently in summary
and decision code), then request re-review of the complete-run branch before
rerunning. Do not reinterpret the failed run as D3 evidence and do not
authorize intervention or training.

---

## Re-review — full-only bootstrap name-binding repair — 2026-09-19

**VERDICT: PASS — authorize one clean rerun of the same complete D3 frozen
scan.**

The patch is mechanically limited to renaming the aggregation result from
`boot` to `bootstrap_result`; the callable remains `boot(...)`. This removes
the Python local-name binding that caused the full-only `UnboundLocalError`.
All result consumers now consistently refer to `bootstrap_result`:

- the raw-score-match CI invalidation gate;
- the correct-minus-sham lower-CI positive decision branch;
- `summary.json`'s `bootstrap` field; and
- `bootstrap.json` serialization.

The existing guards remain correct: incomplete runs leave the result `None`
and are already `INVALID`; complete zero-valid runs are `INVALID` before the
conditional decision's inner bootstrap access; and complete valid runs call
the unchanged 85-scene, 10,000-draw, seed-20260924 `boot()` function. No data,
target lineage, split, pair selection, train/val gap rule, feature forward,
threshold, bootstrap mathematics, or preregistered decision criterion changed.
The script passes `py_compile`.

Rerun the complete configuration only (all 128 train scenes; all 49 affected
validation scenes / 59 targets; no scene or max-scene caps). The prior failed
attempt supplies no D3 evidence. A successful rerun still requires a separate
result audit before any intervention or training can be considered.
