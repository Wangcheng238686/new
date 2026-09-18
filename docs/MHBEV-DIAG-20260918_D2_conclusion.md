# D2 conclusion — frozen RPN rank profile

**Experiment:** `MHBEV-DIAG-20260918-D2`  
**Status:** complete, frozen read-only diagnostic; no detector/UAV mutation or training.  
**Registered decision:** `D2_BROAD_RANK_DEFICIT`.

## Question and validated scope

D2 tests the 59 D1 targets whose correct raw RPN anchors exist but are rejected
by the deployed per-level `nms_pre=1000` selection.  For every raw IoU>=.5
candidate it reconstructs the deployment's level-local descending rank and
asks whether a larger **per-level** rank cap would retain one correct
candidate.  The preregistered primary counterfactual is cap=2000; 1250, 1500,
and 4000 are profile-only companion caps.

The full scan covered all 107 B0 targets in 85 affected validation scenes.
Every B0/D1/D2 target key, B0 RPN IoU, D1 four-stage IoU, and recomputed D1
category matched fail-closed.  Typed normal-prediction/deployed-RPN/staged-final
parity, repeated rank-readout hashes, live RPN configuration, no-proposal-write,
and both controls all passed.  Raw-geometry control: 23/23 have no correct raw
candidate.  Selected-stage control: 25/25 NMS or max-per-image targets have a
correct candidate ranked <=1000.

## Results on the fixed 59-target main set

| Per-level counterfactual cap | Correct candidate retained | Recovery |
|---:|---:|---:|
| 1250 | 10 / 59 | 16.95% |
| 1500 | 14 / 59 | 23.73% |
| 2000 (primary) | 20 / 59 | 33.90% |
| 4000 | 28 / 59 | 47.46% |

The main-set minimum-positive-rank median is **4163** and p90 is **31,278.4**;
the correct candidates are thus usually far beyond the deployed top-1000
boundary, rather than just below it.  The candidate levels are L0=55,
L1=28, L2=8 (a target can have correct candidates on multiple levels).

At the primary cap=2000, the 85-scene paired bootstrap (10,000 samples,
seed 20260921) is **[22.03%, 46.00%]**.  This fails both preregistered gates:
33.90% < 70%, and the CI lower bound 22.03% < 50%.

## Decision boundary

Increasing `nms_pre` / per-level candidate cap is **not** an admissible next
route: it is not authorized for a config-only validation, training, or UAV
fusion redesign.  It would retain too few of the fixed RPN-miss targets even
at 2x cap, while imposing compute/memory cost.

The next scientific branch must be a new frozen hypothesis, not a cap sweep:
either satellite anchor/bbox geometry (the separate 23 raw-geometry failures)
or a satellite objectness-quality diagnosis for the broadly low-ranked
candidates.  Neither conclusion establishes an UAV benefit; B3 and C1 remain
the relevant frozen negative evidence for UAV score/objectness readout.

## Artifacts

- Full output: `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260918/D2-fullval/`
- Generic smoke: `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260918/D2-smoke/`
- Main-path smoke: `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260918/D2-smoke-main/`
- Implementation/smoke audit: `docs/reviews/MHBEV-DIAG-20260918_D2_implementation_review.md`
- Independent result audit: `docs/reviews/MHBEV-DIAG-20260918_D2_result_review.md`
