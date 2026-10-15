# Frozen feature complementarity diagnostic — v1 preregistration; execution awaiting root review

Route A frozen representation diagnostic; no formal network training, AP or checkpoint updates. Only six-image smoke uses original SAM inference for coarse-route equivalence; formal extraction does not run SAM. Current development subset only, E72 validation-selected EMA; this is not independent generalization evidence.

## Question and limits
Can a fixed low-capacity readout use aligned UAV/BEV appearance to predict satellite coarse-mask residuals better than satellite appearance alone, beyond geometry/support and equal-capacity spatial mismatch? Positive result establishes linear-readout accessibility in the frozen representation, not UAV final-mask benefit, formal-model gain, or a claim about unlabeled deployment. Negative result only rejects this readout/resolution/budget.

## Inputs and sampling
Manifest frozen SHA495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59, train627/val153. Select128train images by smallest SHA256('uav-complementarity-v1:'+image_id), no masks/quality selection; use all153val. All GT boxes and visible local IDs in selected images. Decode local/global table for per-building weighting. Manifest known building IDs train2523/val237, shared0; actual visible IDs will also be checked. Report image footprint overlap/nearest origin distance and repeated-building appearances; same scene, nearby imagery and encoder training exposure remain limitations.

Satellite representation: E72 satellite backbone/neck BEFORE any UAV fusion, then original extra positional encoding and mask RoI extractor. It is a frozen dual-trained satellite path, not independently trained single-stream E66.
Baseline coarse: exact original SmallMaskDecoder on those satellite-only RoI features, sigmoid64x64; GT targets via original get_coarse_targets. This differs from correct-dual P3/P4 coarse for RoIs routed there, intentionally to avoid leaking UAV into satellite baseline.
UAV representation: E72 DepthCVT BEV projected with unchanged target geometry to128x128 using production BEVToSatellite. RoiAlign on projected map at spatial_scale1/4, aligned flag matching production extractor; output14x14 then bilinear8x8. No normalization by GT support or GT instance mask. Confidence and coverage from production projection use same sampling. Wrong-location representation: cyclic roll projectedUAV32cells (128sat pixels) BOTHx/y, while keeping original target geometry maps; this is feature mismatch only, not a physical intervention.
Readout spatial unit: fixed8x8 normalized RoI grid. Downsample satelliteRoI14->8, UAVRoI14->8, coarse probability64->8 via area, GT64->8 via area. All64cells perROI are used; includes GTbackground in box, no foreground-balanced sampling.

## Fixed cells / equal capacity
Shared:32dim satellite random projection + coarse8 probability + normalized localx/y + logboxwidth/height + confidence+coverage (39dims). Extra32 slots:
- S64: second independent32dim satellite projection.
- SG: zeros (geometry-only control for extra appearance).
- SU: aligned32dim UAV projection.
- SX: same UAV projection spatially mismatched as above.
- SN: independent fixed standard Gaussian noise32 perimage seeded by imagehash, no additional encoder.
Total71variables plus intercept, same regularizer and coefficient shape. S64/SU/SX have equal active visual dimension; SG intentional inactive-slot control. Fixed Gaussian matrices seeds1811/1812/1813, scale1/sqrt(inputC), no PCA/label fit. Their source hash is persisted.

## Fit and endpoint
One ridge regression percell predicts residual y=GT8fraction-coarse8probability. Standardization learned on train rows only, std floor1e-6. Equal ROI contribution (64fixedrows/ROI); ridge objective mean square +0.01*sum(beta^2) with standardized variables; intercept unpenalized. Lambda fixed0.01, no grid/search/validation threshold choice. All feature normalization and coefficients saved. Fitted only128train images; val GT used only after fit.
Primary: ROI-mean8x8 Brier score after p=clip(coarse8+predresidual,0,1), paired loss reduction SU vs SG and SU vs SX, controlBrier minus alignedBrier is positive. These are the TWO primary contrasts. Companion: binary8x8 occupancy IoU at threshold.5 (GToccupancy>=.5), not coarse64IoU/finalmask/AP. S64 and SN descriptive capacity controls, not postselected primaries. Report baseline/trainfit/validation Brier and IoU, total rows/ROIs/images, active variable count, buildingequal Brier, and size groups (boxarea32^2/96^2). Evaluate ALL and supportpositive RoI groups regardless outcome; supportpositive means mean sampled production coverage>0, not GTmask overlap. Gate based on ALL only, no selected support subgroup.
1000imagepaired sum/count bootstrap seed1790. Two primary intervals use Bonferroni97.5% marginal percentile bounds, quantiles.0125/.9875, yielding exploratory joint95% coverage under image independence. Same scene, spatialcorrelation, E72validationselection, one trainingseed and oneprojectionseed are outside this uncertainty.
Positive gate: Brier reduction>=0.001 and Bonferroni lowerbound>0 for BOTH primary contrasts. This is fixed absolute ROImean Brier reduction, not finalmask effect. No separate64IoU requirement. Gatefail=insufficient positive evidence under fixed linear diagnostic, no automatic training escalation. True positives require reporting S64/noise descriptive controls and honest scope, not overclaiming stronger backbone or independentlytrainedsingle comparison.

## Integrity and costs
Frozen eval/inference_mode feature extraction, no optimizer or backward. Parameters version unchanged; no hooks returning replacement; preUAVneck reference cloned if needed. Train/val split disjoint IDs+visibleglobalIDs checked. Representative3train+3validation images validate directcoarse equals full-maskhead capturedcoarse under zeroUAV. No SAM full masks aftersmoke. Feature cache shape/IDs/SHA/script+protocol+checkpoint/config provenance. Estimated ~281image forward passes incl5viewUAV, approximately0.3–1million linear rows; cap35minutes extraction, stop only operational cap with incomplete status (not result-selected). No extrapolation to all627train fit or data still producing.
