# D3 conclusion — frozen satellite RPN objectness readout

**Experiment:** `MHBEV-DIAG-20260918-D3`  
**Status:** complete, frozen read-only diagnostic; no detector/UAV mutation or
training.  
**Registered decision:** `D3_NO_FROZEN_SAT_OBJECTNESS_SIGNAL`.

## Question and validated scope

D3 tests whether the frozen satellite-view PAFPN feature at an RPN anchor
contains targetness information beyond the deployed raw RPN score.  The fixed
validation set is the 59 D1 `score_nms_pre_rejected` small-target failures:
the correct raw anchor exists but loses before the deployed pre-NMS selection.
For every target, D3 compares the linear readout's probability for the
highest-raw-score positive among decoded raw candidates with IoU >= `.5`
against a same-level background anchor whose raw score differs by at most
`.02`.

The successful clean rerun covered all 128 registered train scenes (158
training pairs), all 49 affected validation scenes and all 59 targets.  It
uses the preregistered 85-scene bootstrap, 10,000 draws, seed `20260924`.
The first full attempt is retained as a software-failure artifact only: it
ended at aggregation with a Python local-name error and supplies no evidence.
The reviewed mechanical rerun repair changed neither model, checkpoint,
split, pair definition, thresholds, nor bootstrap mathematics.

All required guards passed in the rerun: typed normal-prediction parity,
deployed-RPN parity, repeated feature/descriptor hashes, fail-closed B0/D1
replay/category, no proposal write, and exact sparse-prior/grid-prior parity.
All 885 layout records (177 scenes x 5 levels) use the deployed 512-channel
PAFPN input and six anchors per cell.  The score-match control also passes:
the raw-score win-rate-minus-tie CI upper bound is below `.15`.

## Results

| Held-out, score-matched measure | Result |
|---|---:|
| Valid pairs / fixed targets | 59 / 59 (100%) |
| Frozen satellite readout wins | 25 / 59 = **42.37%** |
| Permuted-label sham wins | 40 / 59 = **67.80%** |
| Raw-score wins | 25 / 59 = 42.37% |
| Correct minus sham, 85-scene bootstrap 95% CI | **[-43.33%, -7.69%]** |
| Raw-score win rate minus tie, 95% CI | [-20.49%, 5.41%] |

The preregistered signal gate requires at least 48 valid pairs, readout
win-rate >= 65%, and a correct-minus-sham CI lower bound above zero.  Coverage
and the score-match calibration pass, but the readout has 42.37% wins and its
correct-minus-sham interval is wholly below zero.  It therefore fails the
positive gate decisively.

## Decision boundary

The credible conclusion is narrow but useful: **the frozen satellite PAFPN
descriptor, with this linear anchor-level objectness readout, does not provide
usable targetness information for this fixed class of small RPN misses.**  It
does not justify adding a satellite/UAV RPN objectness or quality-loss branch,
training a fusion module, changing UAV inputs, or increasing candidate caps.
It also does not prove that every geometric or depth-aware UAV approach must
fail; those are distinct hypotheses and require new preregistration plus a
new frozen diagnostic.

The only authorized next step is to formulate a materially different frozen
diagnostic.  No intervention or training is authorized by D3.

## Artifacts

- Successful full output: `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260918/D3-fullval-rerun1/`
- Retained failed output (not evidence): `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260918/D3-fullval/`
- Smoke output: `portable_sam2_explicit_coarse/outputs/MHBEV-DIAG-20260918/D3-smoke-main/`
- Implementation, failure, and rerun review: `docs/reviews/MHBEV-DIAG-20260918_D3_implementation_review.md`
- Independent result audit: `docs/reviews/MHBEV-DIAG-20260918_D3_rerun1_result_review.md`
