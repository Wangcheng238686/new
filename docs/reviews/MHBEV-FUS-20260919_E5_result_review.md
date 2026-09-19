# MHBEV-FUS-20260919-E5 — independent result review

**Verdict: PASS — `E5_PASS` is supported by the completed frozen-probe artifacts.**
The only authorization produced by this result is to submit a separate, pre-registered
**BEV → boundary-refiner hot-start screening proposal**.  It does not authorize a
detector/UAV/data modification, implementation, hot-start launch, full training, AP
claim, depth branch, or a claim that E5 itself improves masks.

Audited on 2026-09-19, read-only, from:

`/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/MHBEV-FUS-20260919-E5-fullval-rerun1`

The run manifest records committed source `143c9e4ccd670aa4170483798c13270dc31f1945`
(`feat(ue): preregister E5 frozen boundary confirmation`), the registered checkpoint
hashes/config hashes, data fingerprints, command, environment, seeds and fitted-artifact
hash.  The completed summary has exit-success status `complete_read_only_frozen_probe`.

## Artifact and lineage audit

- Bundle is complete: manifest, stdout log, summary/bootstrap JSON, 259-row per-scene
  CSV, 339-row telemetry, 80-row train-sample table, P3/world lattice archive, and
  fitted PCA/scaler/ridge artifact all exist.
- Recomputed SHA256 hashes exactly match the manifest for `summary.json` and
  `fitted_artifacts.npz`:
  `577fec4aa4f6485bf6040e90be93922d7a168f5429e7badd819404619843fb6a` and
  `bffa70a43958cf3fd482212e6695089ee97085c204a9dff762e0a5fabb9545f8`.
- Dual checkpoint load has no missing/unexpected keys.  The single-stream overlay has
  no unexpected key and its archived missing keys are exclusively the declared
  drone/geometry/gate-only modules.  The initial/final typed model-state SHA256 is
  identical: `8747375920a3d9c00d2fb5762b86b61ed89c8207b0b49e88b7898ed45a7facea`.
- `fitted_artifacts.npz` is structurally consistent: 120,000 train labels
  (`80 × 1,500`); correct/mis PCA component arrays `(512,1024)`; readout dimensions
  512 (SAT), 1536 (raw DBEV concatenations), and 1024 (PCA512 concatenations).

## Split, selector, geometry and invariant audit

- Exactly 80 fit IDs and 259 unique val IDs are recorded, with zero intersection.
  The selector records the sole prescribed skip: source index 63, scene `002646`,
  `degenerate_p3_union_mask`; the selected records end at source index 80 / `002746`.
- Telemetry ordering exactly matches the manifest: 80 train rows followed by all 259
  val rows.  Train-sample IDs/source indices match the selected records and every
  sample seed equals `20260925 + source_index`.
- The P3 mapping archive has the expected `(16384,4)` local lattice, `(80,2)` train
  centers, `(259,2)` val centers and matching ID arrays.  Correct and rotated geometry
  each retain valid-view count at least one at every cell; mean valid views is archived
  per scene (up to 4.1579).
- All 339 rows have equal before/after typed prediction digest, including each of
  bbox, score, label and mask.  All repeat feature mapping hashes match; every
  before/after proposal-absence flag is true; every row has the same model-state hash
  and `model_training=False`.  Thus no observed detector, proposal, state, or
  feature-repeat contamination invalidates the probe.

## Independent metric recomputation

I recomputed all values from `per_scene_boundary_r.csv` and regenerated the registered
10,000-draw paired percentile bootstrap in its exact order using seed `20260926`.
Every reported CI and paired t statistic reproduced bit-for-bit at CSV precision.

| registered comparison | mean Δr | 95% paired bootstrap CI | gate | result |
|---|---:|---:|---:|---|
| `SAT+DBEV_correct − SAT` | +0.05946 | [+0.05270, +0.06644] | lower ≥ +0.02 | pass |
| `SAT+DBEV_correct − SAT+DBEV_mis` | +0.06661 | [+0.05994, +0.07356] | lower ≥ +0.02 | pass |
| `SAT+PCA512(DBEV_correct) − SAT+PCA512(DBEV_mis)` | +0.06524 | [+0.05952, +0.07105] | lower ≥ +0.01 | pass |

Companion means are SAT `.58868`, correct raw DBEV `.64815`, mismatched raw DBEV
`.58153`, correct PCA512 `.64905`, and mismatched PCA512 `.58381` (all `n=259`).
Paired t statistics are 16.79, 19.36 and 22.09 respectively, with two-sided p-values
`3.03e-43`, `3.40e-52`, and `2.08e-61`; these are corroborating statistics only, as
pre-registered.

## Decision and boundary

All seven recorded invariants are true and each of the three registered lower-CI gates
passes.  The result is therefore legitimately **`E5_PASS`**: under the fixed frozen
linear readout, correctly projected DBEV contains held-out boundary-distance information
that exceeds SAT alone, the equally wide 90-degree geometry mismatch, and the 512-D
capacity control.

This does **not** show any detector outcome or causal mask benefit.  Per Appendix K,
the next permitted action is only a new, separately reviewed and pre-registered
boundary-refiner hot-start proposal with its own zero/nonzero-gate, training, parity,
telemetry and full end-task multi-metric gates.  No direct implementation or training
is approved by this review.
