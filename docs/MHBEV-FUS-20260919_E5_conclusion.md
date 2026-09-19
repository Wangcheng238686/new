# E5 conclusion — DBEV marginal boundary information confirmation

**Experiment:** `MHBEV-FUS-20260919-E5`  
**Status:** complete frozen read-only confirmation; no detector/UAV/data mutation or training.  
**Decision:** `E5_PASS`.

## Scope and integrity

E5 is the preregistered replacement for exploratory E4.  It fits deterministic
linear ridge readouts only on 80 boundary-label-valid scenes from the fixed
128-scene training subset, then evaluates every one of the 259 official val
scenes.  The sole skipped train entry is source index 63, scene `002646`, whose
P3 union mask is degenerate; no val scene is skipped.

All 339 readout scenes pass typed normal-prediction parity, repeated feature
hash, no-proposal-write, eval-state and model-state invariants.  The complete
manifest, train sample hashes, PCA/scaler/ridge coefficients, telemetry,
per-scene values, bootstrap and stdout log are archived with the data-disk
artifact.

## Preregistered results

| Comparison | Mean boundary-r difference | 10,000-draw 259-scene paired bootstrap 95% CI | Gate |
|---|---:|---:|---|
| `SAT+DBEV_correct − SAT` | +0.05946 | [+0.05270, +0.06644] | pass (lower >= +.02) |
| `SAT+DBEV_correct − SAT+DBEV_mis` | +0.06661 | [+0.05994, +0.07356] | pass (lower >= +.02) |
| `SAT+PCA512(correct) − SAT+PCA512(mis)` | +0.06524 | [+0.05952, +0.07105] | pass (lower >= +.01) |

The capacity-matched result and the marginal rotated-geometry sham rule out the
two decisive E4 alternatives: merely concatenating a high-dimensional UAV
descriptor, or benefiting from generic linear capacity.  This is credible
evidence that correctly projected DBEV provides marginal frozen linear
information for the P3 boundary-distance target.

## Decision boundary

E5 does **not** demonstrate detector AP, mask boundary F-score, training
stability, or a UAV-depth benefit.  It therefore authorizes only a separately
preregistered and independently reviewed **BEV-to-boundary-refiner hot-start
screening proposal**.  It does not authorize implementation, training, a
nonzero gate, auxiliary loss, or a full run by itself.

## Artifacts

- Full output: `/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/MHBEV-FUS-20260919-E5-fullval-rerun1/`
- Independent implementation audit: `docs/reviews/MHBEV-FUS-20260919_E5_implementation_review.md`
- Independent result audit: `docs/reviews/MHBEV-FUS-20260919_E5_result_review.md`
