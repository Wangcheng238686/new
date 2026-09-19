# D3 fullval rerun1 result audit — 2026-09-19

**VERDICT: PASS (valid negative result).** `D3_NO_FROZEN_SAT_OBJECTNESS_SIGNAL`
is credible under Appendix I. It closes the frozen satellite-FPN linear
objectness-readout route. It authorizes neither an intervention nor detector
training.

## Artifact and lineage integrity

`D3-fullval-rerun1` exited 0 and contains summary, bootstrap, train/validation
pair CSVs, descriptor layout, manifest, and log. The registered checkpoint and
B0/D1 locations agree with the frozen lineage. Scope is complete: 128 train
scenes / 158 train pairs, 49 affected validation scenes, all 59 D1 main
targets, and an 85-scene bootstrap universe. The validation CSV has 59 rows
and 59 valid strict score-matched pairs; layout has 885 records, exactly
177 readout scenes x 5 levels.

All summary invariants are true: normal prediction typed parity, deployed-RPN
typed parity, repeated descriptor hash, fail-closed B0/D1 replay/category, and
no proposal write. Independently checking telemetry found every level's
`prior` digest equal to `sparse_prior`, every layout at 512 channels and six
anchors/cell, and B0/D1/D3 replay IoUs equal within `1e-5` for every target.
The sham permutation is nonidentity (distinct typed hashes) with preserved
158/158 labels.

## Independent recomputation

From `val_pairs.csv`, correct wins are 25/59 = 0.4237288, sham wins are
40/59 = 0.6779661, and raw-score wins are 25/59 = 0.4237288. All valid pairs
retain the validation-only gap gate; maximum gap is `3.401678986847401e-05`,
well below `.02`. The reported mean gap is `1.8598909203520937e-06`; level
counts are 40/14/5 at levels 0/1/2.

I independently rebuilt the 85 scene rows from the validation CSV (zero rows
for non-target scenes), redrew 10,000 bootstrap samples with seed 20260924,
and reproduced both archived CIs exactly:

| quantity | recomputed 95% CI | gate | result |
|---|---:|---:|---|
| correct minus sham win rate | [-0.4333333, -0.0769231] | lower > 0 for signal | fail |
| raw-score win rate minus .5 | [-0.2049180, 0.0540916] | upper <= .15 | pass |

All 10,000 bootstrap draws contain at least one pair. Thus score matching is
adequately calibrated, but the frozen descriptor readout is not merely below
the positive threshold: it is significantly worse than its label-permuted
sham on this held-out slice. The summary's no-signal decision, no invalid
reasons, and coverage 1.0 are consistent with the preregistered rule.

## Allowed next step

Close D3 as a negative frozen-readout result. The only permissible next route
is a separately preregistered **new frozen diagnostic with a materially
different hypothesis**; it must not reuse D3 as authority for a satellite RPN
objectness/quality-loss intervention, candidate-cap validation, UAV change,
or any training. Any proposed route must first pass its own implementation
review and frozen gate.
