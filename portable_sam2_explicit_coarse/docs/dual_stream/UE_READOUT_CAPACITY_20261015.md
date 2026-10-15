# Frozen readout capacity and numeric integrity audit

2026-10-15. Scope: EXP-0006 E72 EMA frozen dual-trained satellite path before UAV fusion, plus frozen DepthCVT projected BEV. This is not the independent E66 single stream. Current development subset only; no formal network training, backward, new model fit, AP, forward pass or checkpoint update in this audit.

## Inputs and exact reproduction

Read all 281 cached NPZ files from `/tmp/ue_uav_experiments/20261014_ad_seed42/final_analysis/uav_complementarity/features/`, verify every NPZ SHA against extraction run_manifest, separate train/validation statistics, and apply only the five already saved ridge coefficient/normalization sets. Train128 images,12621 RoIs,2239 visible buildings; validation153 images,4496 RoIs,237 buildings. The 64 grid rows per RoI are correlated samples, not independent buildings.

Reproduced all SG/S64/SU/SX/SN validation predictions with max absolute difference <=1.11e-16. SG has39 active variables, SU/SX/S64 have71; only SU/SX/S64 are equal active visual dimension controls. No new lambda, seed, readout, selection or validation sweep.

Audit source `/tmp/ue_readout_capacity_audit.py`; complete numeric result `/tmp/ue_readout_capacity_audit.json`; log `/tmp/ue_readout_capacity_audit.log`.

## Findings supported directly by cached numbers

### A. There is no clear feature collapse or numeric ridge suppression

- UAV projected32 feature standard deviations are finite and nonzero. Correlation covariance participation ratio is12.34 train /12.43 validation (out of32), versus satellite32 16.72/15.69. This is a correlated representation, not32 independent dimensions, but it has substantial variation.
- SU standardized training covariance minimum eigenvalue is0.082802 and maximum6.84820; all71 dimensions are active.
- Fixed lambda0.01 yields spectral ridge retention `eig/(eig+.01)` from0.89224 to0.99854, median0.97948. Training effective degrees of freedom is68.985/71. This is the actual training covariance statement; validation covariance is not a new fit or independent ridge diagnostic.
- Therefore **the claim that strong regularization numerically erased the UAV contribution is not supported**. The saved SU extra-UAV coefficient block is nonzero (L2 .007386); its output contribution RMS is.008399 on train /.006399 validation, compared with satellite block.022729/.020949. These correlated-block contributions are descriptive and cannot be treated as causal importance.
- A covariance-only unsupervised linear residualization of UAV32 against the39 shared satellite/geometry variables leaves67.43% train /69.84% validation UAV variance. This means the sampled UAV vector is not almost entirely a linear copy of shared inputs. It does **not** establish that the remaining variance predicts shape or corresponds to useful cross-view detail. No label residual is used in this calculation and no new prediction model is fitted.

### B. UAV features vary less within each RoI

The sum of per-channel within-RoI variance divided by total global feature variance is:

| Source | train | validation |
|---|---:|---:|
| satellite32 | .68308 | .69557 |
| extra satellite32 | .71980 | .74563 |
| aligned UAV32 | .22971 | .41685 |
| shifted UAV32 | .22830 | .37777 |
| noise32 sanity control | .98439 | .98433 |

Thus, in these cached coordinates, UAV variation is more dominated by between-RoI appearance than satellite variation is. Shape residual variance itself is mostly within RoI (.82189/.79583). This is a concrete **spatial-specificity mismatch clue**, not proof that BEV spatial detail has been destroyed. Feature channels differ, satellite contains the production extra sine PE, scale assignments differ (P2/P3 for sat versus fixed stride4 projection for UAV), and the ratio also depends on object size, context and zero support. Noise has high within-RoI variation and is useless, illustrating that more variation is not itself more information.

### C. A train/validation difficulty gap exists, without evidence of catastrophic scale drift

Uncorrected cached8² coarse occupancy residual MSE is .019210 train /.038097 validation. Mean signed residual GT−coarse is .017487/.049016. This is not an independent test gap because E72 has already used this validation set for selection, and train samples were seen by the original representation learning.

UAV channel validation mean shift relative to train std has RMS .28117, and std ratio min/median/max .76418/.91759/1.22833. Satellite32 corresponding mean shift RMS .17216 and std ratios .87886/.98451/1.15637. The marginal UAV drift is somewhat larger, but the finite ratios do not demonstrate catastrophic normalization failure; no causal attribution to domain shift is justified without a controlled test.

## Structural limitations confirmed in source, but not established as the main bottleneck

1. **Random channel projection**: satellite512→32 (plus independent32 control), UAV256→32. Matrices are fixed independent Gaussian projections, not task-aware. Cache contains no original channels, so actual task information discarded by compression cannot be estimated or recovered from these NPZs. Nonzero rank cannot certify semantic information retention.
2. **Spatial readout bottleneck**: RoIAlign14² then bilinear8² features; GT64² and probability64² are area-pooled8². Label contains occupancy fractions, but no exact within-cell boundary arrangement. This endpoint cannot measure fine contours or final SAM mask behavior. The native projected BEV is still128²; one cannot infer from8² output alone that the BEV itself lacks detail.
3. **Globally shared additive linear function per cell**: each of64 cells is evaluated independently with the same71 coefficients. x/y, coarse probability, box size, confidence and coverage enter additively. No learned `UAV×position`, `UAV×satellite`, `UAV×coarse`, neighbor or shape-topology interactions are possible. Many context-dependent correction rules are not representable. The model is still a valid, reproducible small-capacity accessibility test; a weak result is not evidence that a nonlinear decoder will succeed.
4. **Asymmetric positional information**: satellite RoI has the inherited extra global sine PE before random projection, UAV does not receive that explicit PE in this diagnostic. Shared localxy helps but cannot perform multiplicative feature/position gating. The diagnostic compares actual available paths, not identical feature semantics or identical PE construction. Standardization removes per-channel scale differences, not this structural difference.
5. **128-image fit**: 12621 train RoIs/2239 unique visible buildings support a71-coefficient fit; no learning curve exists, so neither data sufficiency nor insufficiency is established. Do not interpret807744 rows as independent examples or simply assert a lack of train data.

## Decision

This audit found **no obvious numeric failure, full collapse, wrong standardization, or excessive ridge shrinkage** explaining the weak effect. It found **reduced UAV within-RoI spatial variation and a narrowly additive coarse readout**, each worth keeping as a hypothesis. It cannot distinguish task-information loss before the cache from information present but inaccessible to this readout.

Current CPU audit is sufficient to stop widening the readout in this round. Combine with the other agents' geometry/support/error-strata evidence. Do not fit new predictors on the repeatedly inspected validation subset just to pass the old gate. If a future readout is justified, select exactly one protocol from training evidence and capacity-matched satellite/shift controls before one validation evaluation; treat that as exploratory reuse, not a fresh independent confirmation. No extra fit or new feature extraction is requested here.

## Other-agent overlap

The JSON contains a precomputed descriptive empty/partial/full occupancy breakdown using already saved predictions, but `bev_error_strata` owns the independent error-strata interpretation. It is not a new endpoint, subgroup gate, or evidence for selecting a new readout. No confidence intervals or subgroup searches are added in this audit.
