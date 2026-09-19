# MHBEV-FUS-20260919 E4 — independent result review

**Verdict: HOLD — E4 is a credible exploratory frozen-readout observation, but it is
not a passed, pre-registered training gate.  It does not authorize implementation,
hot-start, or full detector training.**  The sole authorized next action is a new,
pre-registered frozen confirmation probe for the *marginal* boundary signal.

Reviewed read-only on 2026-09-19: Appendix J in `EXPERIMENT_GOVERNANCE.md`, commit
`d40be42` implementation `scripts/ue/probe_fusion_information.py`, E4 artifacts in
`outputs/probe_fusion_boundary/`, and the provisional result note
`docs/reviews/MHBEV-FUS-20260919_probe_results.md`.  No model, dataset, checkpoint, or
probe implementation was changed by this review.

## What is verified

### Artifact identity and lineage

- The two committed E4 artifacts exist and match the published checksums:
  `summary.json` MD5 `a1a26698adfda374e9b14fdb287210e4` and
  `per_scene_val_ap.csv` MD5 `55c9adad42c52396f17d743c757ee7ad`.
- `summary.json` is internally consistent with the CSV to its five-decimal stored
  precision.  It declares `target="boundary"`, `n_train_used=80`, `n_val=60`.
- Code loads the dual checkpoint first and then the named single-stream checkpoint;
  therefore SAT matching keys are overwritten by the single-stream lineage whereas
  UAV/BEV-only keys remain from r3.  That is the intended mixed frozen-feature design,
  not detector fine-tuning.  All extraction is under `@torch.no_grad()` / `model.eval()`
  after the unrelated gradient telemetry stage; the boundary linear probe is the only
  fitted object.
- The artifact package is nevertheless incomplete for a formal archival run: it has no
  run manifest, resolved command/environment, checkpoint hashes, stdout log, saved
  fitted-probe weights, or train scene list.  The script and data contract allow the
  result to be reproduced, but the existing files alone do not prove the exact launch.

### Split and label audit

- Reconstructed the dataset selection from `UEDualStreamDataset`: official train has
  712 valid scenes; the fixed ratio/seed yields 128, whose first 80 are the fit set.
  Official val has 259 valid scenes; its first 60 are the evaluation set.  Their scene
  ID intersection is empty.
- All 60 CSV scene IDs are unique and exactly equal the reconstructed first-60 val IDs
  (`002002` through the ordered endpoint `002308`).  Thus there is no observed
  train/val scene leakage or substitution in the evaluation CSV.
- The boundary target is correctly constructed from the union raster of instance masks
  at the same 128x128 P3 lattice: `distanceTransform(1-occ) - distanceTransform(occ)`,
  then divided by eight.  It is a valid signed boundary-distance target.  The division
  changes only scale, not Pearson-r or a linear MSE optimum after feature scaling.
- This is label supervision for the linear readout, not an independent target.  It can
  establish linear decodability of the frozen features; it cannot establish detector
  AP, mask boundary F-score, or a causal training gain.

### Independent statistical recomputation

Using the committed 60-row CSV (so values below are subject only to its displayed
five-decimal rounding), I recomputed:

| comparison / statistic | independent result | reported result |
|---|---:|---:|
| SAT mean Pearson r | 0.5191975 | 0.5191975 |
| SAT+DBEV mean Pearson r | 0.6111292 | 0.6111293 |
| paired delta | +0.0919317 | +0.0919318 |
| improved / worse scenes | 58 / 2 | 58 / 2 |
| paired t(59), two-sided p | 13.2655, 2.345e-19 | 13.2654, 2.346e-19 |
| 100,000 resample scene-paired bootstrap CI for delta | [+0.0786, +0.1056] | not reported |
| one-sided Wilcoxon signed-rank p | 1.28e-11 | not reported |
| DBEV minus DBEVmis mean r | +0.45295 | implied by table |
| DBEV vs DBEVmis paired t(59), p | 16.1675, 2.40e-23 | not reported |

The E4 numerical claim is therefore real for this exact frozen-feature/probe setup:
adding the correct-geometry DBEV descriptor to SAT gives a robustly higher held-out
per-scene boundary-distance correlation than SAT alone, and the standalone rotated
DBEV readout is near zero (`0.03063`).  This is evidence of a useful *candidate*
mechanism, not evidence of detector improvement.

## Why the verdict is HOLD, not PASS

1. **Post-result registration is disqualifying for a gate.**  Git history shows probe
   code and output commit `d40be42` at 14:59, while Appendix J was added in `dcecf53`
   at 16:12.  Appendix J itself calls E4 “后补，占位注册于结果落盘时”.  Its metric,
   threshold, uncertainty procedure, sham gate, parity/hash invariants, and exact
   training authorization rule were therefore not fixed before E4 was run.  Per the
   governance rule, this may be retained as exploratory evidence but cannot authorize
   the next experimental level.
2. **The required marginal sham was not run.**  `DBEVmis` is fitted and evaluated only
   by itself.  The code creates `SAT+LBEV` and `SAT+DBEV`, but never creates, fits, or
   evaluates `SAT+DBEVmis`.  Consequently the result proves that correct DBEV is
   informative in isolation and that SAT+DBEV beats SAT; it does *not* exclude a gain
   from concatenating any 1024-dimensional frozen UAV descriptor, even geometrically
   wrong content.  This is particularly material because SAT has 512 dimensions while
   SAT+DBEV has 1536.
3. **No matched-dimension/capacity control exists.**  DBEVmis has the same standalone
   1024 dimensions, which is useful but insufficient for the marginal concatenation
   question.  A fit/evaluation of SAT+DBEVmis is the minimum negative control; a
   fixed-train PCA-512 (or otherwise pre-registered dimension-matched) DBEV companion
   would additionally separate content from linear capacity.
4. **Probe integrity telemetry is below the repository's frozen-probe standard.**
   There is no detector-output parity hash before/after extraction, feature repeat hash,
   no-write assertion, command manifest, or recorded fitted coefficients.  Static code
   inspection supports nonintervention (`scene_features` does not invoke prediction or
   mutate proposals), but no run-time artifact proves those invariants for this launch.
   There is likewise no archived source log to verify reported `load_state_dict`
   missing/unexpected-key counts.

The geometry formula itself is not a finding against E4: the apparent `512` constants
are a normalized lattice convention.  At 128 P3 cells,
`((i+.5)*512/128)/512 == (i+.5)/128`, the same full-tile normalized location as the
1024-pixel satellite contract.  The direct sampling uses W2C row-form equivalents of
the documented column-vector multiplication, FP32 projections, z/in-frame masking,
and a 90-degree ground-point rotation about the scene center.  No static coordinate
or split-contamination failure was found.

## Single authorized next step

Create a **new pre-registered frozen confirmation probe only**; do not modify the
detector or start a hot-start.  It must preserve the same frozen checkpoints and
train/val separation, and pre-register all of the following before execution:

- primary: paired held-out `SAT+DBEV_correct - SAT` boundary-distance correlation,
  with 10,000 scene-paired bootstrap replicates, fixed seed, and a stated lower-CI
  gate;
- required sham: identically trained/evaluated `SAT+DBEVmis - SAT`, with the gate that
  correct marginal gain exceeds the rotated-geometry sham by a pre-specified positive
  CI lower bound; retain the correct-versus-mis standalone comparison only as
  mechanism telemetry;
- capacity control: a train-only fitted, fixed-at-val matched-dimension DBEV transform
  (for example PCA-512), and a predeclared rule for interpreting it;
- detector parity before/after, repeat feature typed hashes, no-write/proposal hashes,
  exact checkpoint/config/data hashes, scene lists, command/environment manifest,
  fitted coefficients, per-scene CSV, and stdout log;
- companion end-task *screening* endpoints specified in advance (at least boundary
  F-score or boundary-IoU, segm AP75, segm mAP_s, bbox mAP_s, and overall segm/bbox
  mAP), explicitly marked as unavailable until a later authorized heat-start.

Only if that new frozen probe passes its pre-registered correct-vs-sham, capacity,
parity, and archival gates may the team submit a separate boundary-refiner hot-start
proposal for approval.  E4 alone authorizes no non-zero gate, auxiliary loss, BEV
injection implementation, or detector run.
