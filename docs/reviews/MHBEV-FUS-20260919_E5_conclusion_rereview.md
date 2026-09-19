# MHBEV-FUS-20260919-E5 conclusion — independent re-review (second auditor)

**Reviewed object:** `docs/MHBEV-FUS-20260919_E5_conclusion.md` (decision `E5_PASS`)
**Re-review date:** 2026-09-19 · **Auditor:** separate session (second independent
reviewer, after the authoring agent and the implementation/result reviewers)
**Method:** read-only verification — artifact inspection, preregistration cross-check,
independent recomputation of the preregistered bootstrap from the archived per-scene CSV.

## Verdict: **CONFIRMED — the E5_PASS conclusion is supported.** No discrepancies found.

## Verification performed (with evidence)

1. **Preregistration match.** Appendix K (`EXPERIMENT_GOVERNANCE.md`, commit `143c9e4`)
   fixes: gates CI-lower(correct−SAT) ≥ +0.02, CI-lower(correct−mis) ≥ +0.02,
   CI-lower(PCA512 correct−mis) ≥ +0.01; paired percentile bootstrap, 10,000 draws,
   seed 20260926; full 259-val, no val skips; fit = first 80 boundary-label-valid
   scenes of the fixed 128 subset with the degenerate-skip rule; five arms fixed.
   The archived `summary.json.decision_rule` reproduces this gate verbatim.

2. **Independent bootstrap recomputation.** From
   `per_scene_boundary_r.csv` (259 rows, no NaN), re-derived with the preregistered
   seed 20260926:
   - correct−SAT: mean +0.05946, CI [0.05250, 0.06625] → pass (≥ +0.02)
   - correct−mis: mean +0.06661, CI [0.05990, 0.07339] → pass (≥ +0.02)
   - PCA512 correct−mis: mean +0.06524, CI [0.05951, 0.07105] → pass (≥ +0.01)
   Means match the archived values to 5 decimals; CIs match within percentile noise.
   All three gates pass under independent computation.

3. **Lineage and execution integrity.** `run_manifest.json`: git_commit = `143c9e4`
   (the preregistration commit); `git log 143c9e4..HEAD -- <E5 script>` is empty, so the
   executed script is exactly the reviewed code. git_dirty at run time lists 6 files
   (docs and the unrelated D3 probe script; the E5 script is not among them) — recorded
   honestly in the manifest. Model/fitted-artifact SHA256 fields present;
   `fitted_artifacts.npz` and per-scene CSV archived.

4. **Isolation and coverage.** Train-fit IDs (80) ∩ val IDs (259) = ∅ (set intersection
   empty). Val coverage 259/259 official scenes, none skipped. The single train skip is
   exactly as declared: `002646`, source index 63, reason `degenerate_p3_union_mask` —
   matching both the conclusion text and the focused implementation re-review.

5. **Review-chain coherence.** Implementation review: HOLD (selector defect) → focused
   re-review PASS (selector-only correction, skip rule preregistered before launch).
   Result review: PASS for `E5_PASS`. My earlier E4 (exploratory, torch-SGD readout,
   60-val-scene subset) was independently reviewed HOLD→superseded by E5; consistent.
   Note for future readers: E5's SAT baseline r = 0.5887 differs from E4's 0.5192
   because E5 uses deterministic ridge + preregistered cell sampling — both numbers are
   internally valid under their own preregistrations; E5 supersedes.

6. **Readout determinism.** `Ridge(alpha=1.0, solver="lsqr", tol=1e-10, max_iter=20000)`
   with sklearn PCA/scaler fit on train only; cell-sampling seed `20260925 +
   scene_index`; bootstrap seed 20260926 — all recorded in the manifest.

## Boundaries respected

The conclusion doc claims only frozen linear marginal boundary information and
explicitly disclaims detector AP, mask boundary improvement, training stability, and
UAV-depth value — matching Appendix K's interpretation boundary. The sole authorization
it derives (submit a separately preregistered BEV→boundary-refiner hot-start screening
proposal) matches K.2. I endorse this boundary: nothing in E5 authorizes implementation,
a non-zero gate, auxiliary losses, or training by itself.

## Residual caveats (non-blocking, for the record)

- E5 measures linear readability of the boundary field, not what the refiner's
  non-linear pathway will extract; the hot-start arm's expected gain remains bounded by
  the P5 oracle result (boundary-F1 +0.012 at full-strength oracle perturbation) and by
  P5's AP75 asymmetry — boundary F1 must be the co-primary endpoint.
- Cell-level occupancy conclusions (E1) are unchanged: the UAV margin is
  boundary-channel-specific; any mAP-first framing remains unsupported.
- `docs/reviews/MHBEV-FUS-20260919_E5_result_review.md` and the D3 `rerun1` review were
  uncommitted at re-review time; committed together with this record for audit-trail
  integrity.
