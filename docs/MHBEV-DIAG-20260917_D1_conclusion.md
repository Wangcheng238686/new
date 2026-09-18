# D1 conclusion — frozen satellite RPN stage attribution

**Experiment:** `MHBEV-DIAG-20260917-D1`  
**Status:** complete, read-only frozen diagnostic; no training or detector/UAV mutation.  
**Registered decision:** `D1_MIXED`.

## Scope and validated contract

D1 decomposes the fixed B0 slice of 107 small-object final failures that have
no deployed post-NMS RPN proposal at IoU >= .5.  It replays the exact baseline
checkpoint over all 85 affected scenes (the official 259-scene validation
universe), and assigns every target to exactly one nested RPN loss stage:
raw dense geometry, live per-level `nms_pre`, live NMS, or final
`max_per_img` truncation.

The deployed RPN contract was fail-closed as `nms_pre=1000`,
`max_per_img=1000`, `min_bbox_size=0`, no score threshold, classwise NMS
IoU=.7.  For this mmdetection version, no-NMS RPN readout is unsupported and
`max_per_img=-1` truncates one item; D1 therefore source-mirrors the live
per-level selection and uses a proved non-binding finite cap for the
NMS-unlimited cell.

All registered checks passed: 107/107 fixed B0 target keys, 85/85 affected
scenes, bitwise normal-prediction and deployed-RPN parity, staged-final equals
live RPN, repeated raw/pre/final hashes, typed feature/anchor/stage hashes,
dense-anchor cardinality (425 level records), B0 final-IoU replay, and no
proposal write.  Every unlimited cell was non-truncated.

## Results

| Fixed B0 RPN-miss stage | Targets | Share |
|---|---:|---:|
| Raw dense geometry unavailable | 23 | 21.50% |
| Per-level `nms_pre` selection rejects aligned candidate | 59 | 55.14% |
| NMS suppresses it | 2 | 1.87% |
| Final `max_per_img=1000` truncates it | 23 | 21.50% |

The largest observed cell is per-level `nms_pre` (59/107), but it does **not**
pass the preregistered dominance gate: its share is below .60, and its
259-scene paired-bootstrap (10,000 samples, seed 20260920) major-minus-rest
95% CI is `[-0.090909, 0.292042]`.  The only valid conclusion is therefore
`D1_MIXED`, not a dominant single-stage conclusion.

## Decision boundary

D1 rules out treating NMS as the principal explanation (2 targets), but does
not authorize a direct intervention, UAV fusion redesign, hot-start, or
training.  The next admissible experiment is a new frozen D2 rank-profile
probe: characterize the 59 per-level preselection losses by FPN level, rank,
distance to the top-1000 cutoff, and a preregistered cap counterfactual, while
retaining raw-geometry and final-truncation cells as controls.  That probe can
separate a recoverable selection-boundary issue from a broad satellite
objectness/anchor deficit before any model change is proposed.

## Artifacts

- Full output: `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/D1-fullval/`
- Smoke output: `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260915/D1-smoke/`
- Implementation/smoke review: `docs/reviews/MHBEV-DIAG-20260917_D1_implementation_review.md`
- Independent result review: `docs/reviews/MHBEV-DIAG-20260917_D1_result_review.md`
