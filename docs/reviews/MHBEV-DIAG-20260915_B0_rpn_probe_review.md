# MHBEV-DIAG-20260915 — B0 frozen RPN diagnosis implementation review

**Review scope (independent, read-only):** Appendix B in
`EXPERIMENT_GOVERNANCE.md`; implementation
`portable_sam2_explicit_coarse/scripts/ue/probe_small_rpn_bottleneck.py`; and
the two-scene smoke artifacts under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B0-smoke/`.

**VERDICT: HOLD — do not launch the 259-scene scan from this revision.**

The model/RPN extraction and the two required deterministic checks are sound on
the smoke scenes.  However, two pre-registered output/data-integrity contracts
are not yet encoded as hard checks.  A full run from this revision could
produce an apparently valid `PROPOSAL_OBJECTNESS`/`ROI_OR_MASK` result while
misreporting geometry coverage or scanning a changed/incomplete split.  This
is an implementation hold, not a scientific conclusion about the baseline.

## Verified pass items

1. **Deployment-equivalent RPN path.** `read_rpn()` creates a fresh inference
   sample, calls `model.extract_feat(inputs, data_samples)`, and then calls
   `model.rpn_head.predict(x, data_samples, rescale=False)`.  This is the same
   branch used in `RSPrompterAnchor.predict()` when no supplied proposals are
   present.  It neither calls `loss_and_predict` nor ROI, and asserts that the
   readout did not set `data_sample.proposals`.
2. **GT and IoU coordinate contract.** GT is taken from the UE loader at its
   1024x1024 raster resolution; boxes are xyxy; `box_iou_xyxy()` is likewise
   xyxy-to-xyxy.  Small is correctly `mask.sum() < 32**2`.  This avoids the
   old offline probe's xywh helper and follows B.1.
3. **Final-failure definition.** Final masks are obtained from normal
   `model.predict(..., rescale=False)`, matched one-to-one with Hungarian
   mask-IoU >= 0.5.  The diagnostic failure subset contains only `missed` and
   `partial_or_merge`; `matched_boundary_error` is excluded as Appendix B
   requires.
4. **No detector contamination / readout stability.** Each smoke scene uses
   fresh samples for normal prediction, RPN readout twice, and a final normal
   prediction.  In the recorded two scenes, prediction field hashes and both
   RPN hashes are equal; each readout returns exactly 1,000 finite proposals in
   the deployed 1024 coordinate range.  The smoke's `INVALID` decision is
   expected because it deliberately scans two rather than all 259 scenes; it
   is not a parity failure.
5. **Bootstrap unit.** The implementation resamples image-level
   numerator/denominator counts (not individual GT), omits draws lacking one
   of the two subsets, and records valid-draw count.  That is a defensible,
   executable reading of the pre-registered image-paired bootstrap.

## Blocking findings

1. **`failed_small_two_view_coverage` is not a ≥2-view coverage rate.**
   Per-GT fields correctly save raw view counts, but `ratio()` averages those
   counts.  Thus the smoke summary reports e.g. `correct: 2.0`, whereas the
   requested statistic is a fraction of failed small GTs satisfying
   `views_correct >= 2` (and analogously for both shams), bounded in [0, 1].
   The summary key explicitly says `two_view_coverage`, so this is not merely
   an ambiguous label.  The per-row raw counts should remain for audit, while
   the reported aggregate must be a boolean coverage fraction and its
   denominator must be recorded.
2. **The full official-data contract is not asserted.** The code regards a
   scan as data-valid whenever `n == len(dataset)`.  It does not assert the
   pre-registered `259` scenes / `5,937` GT instances, nor make either expected
   count part of the explicit invalidation condition.  A loader/filtering or
   dataset drift that still yields 259 scenes can therefore be classified as a
   mechanism decision.  Before full scan, hard-assert both expected counts
   after collection (and record their pass state in `invariants`); otherwise
   set `INVALID`.

## Required narrow correction and re-review evidence

No model, proposal, threshold, matching, or gate change is warranted.  Make
only the two contract corrections above, then provide a fresh 2-scene smoke
that shows: (a) parity/readout hashes still true; (b) each reported coverage is
a [0,1] boolean fraction with numerator/denominator; and (c) the smoke is
explicitly `INVALID` solely because it is intentionally incomplete.  The
full-run path must additionally assert 259/5,937 before allowing any of the
three Appendix-B decisions.

After that narrow re-review, the full 259-scene frozen, read-only scan may be
launched.  The eventual conclusion must remain limited to the Appendix-B
proposal-vs-ROI/mask routing rule; coverage is only necessary-query evidence,
not UAV semantic-effect evidence.

---

## R2 re-review (B0-smoke-r2)

**VERDICT: PASS — the full 259-scene, read-only B0 scan is authorized.**

The two blocking corrections above are present and are narrow contract fixes:

1. `two_view_coverage()` now evaluates `views_<condition> >= 2`, and reports
   an auditable object with `fraction`, `covered`, and `total`.  In
   `B0-smoke-r2`, each condition reports `1.0 / 1 / 1`, rather than the prior
   average-view-count value.  Raw per-GT view counts remain in `instances.csv`.
2. For `--split val`, `data_complete` now requires **all** of: processed scenes
   equals 259, the constructed official dataset contains 259 scenes, and
   collected GT rows equal 5,937.  It is a prerequisite of every non-INVALID
   mechanism decision and is written to the summary/manifest.  The R2 smoke
   truthfully reports `complete: false`, 2 scenes, and 27 GT, hence remains
   `INVALID` only because it deliberately is not the full scan.
3. The original smoke evidence remains intact: bitwise normal-predict parity,
   repeated pre-ROI RPN boxes/scores hash equality, fresh no-proposal samples,
   1,000 finite in-range RPN proposals, xyxy GT/RPN IoU, and the prescribed
   final-mask Hungarian failure categories.

This PASS authorizes only the frozen single-checkpoint, complete 259-scene
diagnostic specified in Appendix B.  Before interpreting its routing result,
verify its own summary says `data.complete: true`, both hash invariants true,
and its bootstrap uses the pre-registered 10,000 draws and seed 20260915.

---

## Full-val result audit (B0-fullval)

**VERDICT: PASS — the B0 result is valid, and the pre-registered conclusion is
`AMBIGUOUS`.  No training arm is authorized by this diagnostic.**

**Artifact scope.** Reviewed `summary.json`, `instances.csv`, `bootstrap.json`,
and `run_manifest.json` under
`portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/B0-fullval/`.

### Reproduced integrity and invariants

- Data contract passes: 259 processed scenes, 5,937 GT rows, and 5,938 CSV
  lines including its header.  The CSV itself contains 259 unique scene ids.
  `summary.data.complete` is true and the manifest agrees.
- All 259 before/readout/after normal-predict comparisons report bitwise parity;
  all 259 repeated RPN readouts report matching boxes/scores hashes.  Every
  scene has the deployed RPN maximum of 1,000 proposals.  No parity/readout
  invalidation is present.
- Independent recomputation from `instances.csv` exactly reproduces: 332 small
  GT; failure counts 177 missed + 50 partial/merge = 227; 39 matched-good; 66
  matched-boundary-error.  The latter is correctly excluded from the primary
  failure subset.
- Recomputed small RPN recalls are 0.76506024 (@0.3), 0.67168675 (@0.5),
  failure-subset 0.52863436 (@0.5), and matched-good 1.0 (@0.5).  The reported
  >=2-view fractions also recompute exactly: correct 202/227 = 0.88986784,
  XY+80m 152/227 = 0.66960352, scene-permuted 172/227 = 0.75770925.
- Re-running the image-level bootstrap from the archived per-scene counts with
  seed 20260915 and 10,000 draws reproduces all 10,000 valid replicates, mean
  failure-minus-good difference -0.47173328, and percentile CI
  [-0.54018122, -0.40552995].  `bootstrap.json` is byte-for-value consistent
  with the summary's bootstrap object.

### Gate execution and permitted next step

The decisive failure-subset RPN recall is **0.52863436**.  It is not `< 0.50`,
so the proposal/objectness rule does not fire despite the strongly negative
failure-minus-good bootstrap interval.  It is also not `>= 0.75`, so the
ROI/mask rule does not fire.  Appendix B therefore requires `AMBIGUOUS`, which
is exactly the archived decision.

Allowed: archive this valid frozen diagnosis and formulate a *new*, separately
pre-registered diagnostic/hypothesis that resolves the intermediate RPN
regime.  Forbidden: treating the negative CI as permission to relabel this
result `PROPOSAL_OBJECTNESS`; treating 0.5286 as evidence that RPN is not
involved; starting A2/A3, hot-start, or full training from B0.  Coverage remains
only necessary-query evidence and does not establish UAV semantic gain.
