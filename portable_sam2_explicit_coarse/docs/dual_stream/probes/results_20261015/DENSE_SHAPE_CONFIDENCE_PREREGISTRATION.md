# Dense source shape/confidence frozen diagnostic preregistration
2026-10-15; EXP-0006/0007; one existing E72 strict EMA, correct UAV, fixed 153 validation images/4496 GT-box appearances/237 global buildings. No training, optimizer, backward, RPN, bbox/AP, support fix, amplitude sweep or model edits.

## Three fixed cells
- baseline: original P sparse and original P dense.
- pred_binary_pm4: original P sparse, dense produced from original predicted 64x64 coarse binary (logit >=0) mapped to -4/+4, through unchanged original canvas transform, frozen PE, learned alpha and box support.
- gt_binary_pm4: original P sparse, dense produced from actual get_coarse_targets 64x64 GT binary mapped to -4/+4 through the same pipeline. Exact historical PG gt_dense_only replay.
Original sparse means actual decoder 2/6 tokens, including unchanged box tokens and invalid-point neutralization. Sparse is replaced at actual decoder input even though upstream source point mining runs in B/G. It is NOT a CPU padded token copy. Boxes, visual features, PE, high-resolution visual inputs, model parameters and postprocessing are fixed. Coarse source original values before replacement must match across cells. Baseline and G final per-GT IoU, sparse/dense SHA must exactly replay prior factorial output.

## Metrics/selection/uncertainty/gates
Only one complete deterministic fixed-cell run after admitted three-image smoke, no checkpoint selection or cell selection.
Two primary comparisons, all 4496 appearances mean final pasted-mask GT-box IoU:
1. pred_binary_pm4 - baseline; isolates fixed finite amplitude/binarization versus original within-shape logit distribution, NOT pure temperature calibration.
2. gt_binary_pm4 - pred_binary_pm4; compares predicted versus GT64 shape at equal +/-4 amplitude, conditional on original sparse.
For each, support gate mean >= .005 AND image-paired sum/count bootstrap 97.5% interval lower >0. Bootstrap 1000 resamples seed1780, quantiles .0125/.9875 (two-comparison Bonferroni marginal intervals, target family95%). Finite-resample exploratory intervals do not cover spatial correlation, selected checkpoint or training seed uncertainty. Historical G-baseline is secondary descriptive 95% only.
Always report all size groups (area<32^2;32^2<=area<96^2;>=96^2), P2/nonP2 and dense support0/nonzero, plus global-building equal-weight means. No posthoc selected successful subgroup gate. Building appearances are repeated, not independent buildings.

## Companion diagnostic schema/boundaries
Per GT: identity, final IoU, final paste XOR pixels, support cells, final dense-to-no-mask norm, actual sparse/dense SHA, original sparse SHA, original/source/GT64 foreground pixels, source binary IoU vs GT64, sparse token count. Binary IoU is a source check, not learned gain. Final changes are not quality improvements by themselves. Support is 32x32 box center support, NOT UAV visibility. No labels enter model except the declared G replacement.
Per image: baseline/G historical IoU and prompts exact, actual sparse exact, actual visual tensor values exact, features SHA unchanged, RGB/geometry unchanged, original coarse exact, support0 final masks exact, hooks removed, production chunk sizes and target ranges, smoke sham status. Model parameter version unchanged.
Visual guard compares all actual decoder image embedding, PE and high-res values on GPU against unchanged source features/PE expanded in the exact original one-image order; scalar arguments and kwargs keyset asserted. This avoids repeated CPU SHA of expanded large visual maps while retaining all-value checks. Features SHA once before and after per image. Actual decoder prompt per-GT SHA retained.

## Admission / budget
GPU0 only after parent admission. Smoke first three manifest images 287/288/295: 45 GT, both historical endpoints exact, actual original sparse held exact in all cells, source/input sham all three modes final mask and decoder prompts exact; support0 pixel exact; no extra observer copies. At most 6 mask forwards/image smoke; full 3 source forwards/image plus one visual extraction. Full multichunk assertions retained (smoke doesn't cover all chunk/token shapes). CPU summary from artifacts only. Expected full minutes rather than previous 37-minute repeated expanded-feature CPU hashing; conservative wall budget 20min, stop on assert/bug, no unregistered extra GPU run. Do not infer pure model inference performance from diagnostic elapsed time.
