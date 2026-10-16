# RoI-UAV feasibility probe independent review

2026-10-16. Reviewer did not design or implement either producer probe. Scope: existing artifacts and CPU-only computations; no fitting, model forward, training, or label edits.

## Review criteria fixed before producer results

1. Candidate headroom is not UAV-reachable improvement. Per-GT cached maximum IoUs do not preserve candidate identity and cannot establish pre-NMS one-to-one recoverability, direct NMS suppression chains, or deployable AP gains.
2. A GT-box intervention changes prompts and is an oracle; it is not a candidate reranking result. Existing correct/zero/shift differences are conditional dependence of an already trained dual model, not proof that a new RoI interface is useful.
3. Global-ID/topology uncertainty is a material confound. A visually fragmented GT could be erroneous building grouping. Report visible geometry separately from confidence in canonical instance identity. Unflagged is not clean.
4. Evidence cases must retain selection and denominators, distinguish observed appearances from unique buildings/pairs, show matched satellite and UAV RGB/depth references, and avoid claiming human-verified source correspondence from projection alone.
5. Negative correspondence controls should preserve source quality when practical. A shifted projection or wrong-view image may diagnose association sensitivity but does not measure learned feature utility. No automatic extra-modality benefit follows from source visibility alone.
6. Verdict boundaries: ‘足以继续设计’ requires explicit candidate-stage opportunity plus at least plausible corresponding source evidence, with no training claim; ‘需补关键证据’ applies when source correspondence or additional instance cues remain ambiguous; ‘不支持’ requires evidence that the proposed interface has no relevant support/headroom in the investigated cases, not merely failure to prove it.

## Pending

Producer protocols/results not yet available at initial review. Numerical independent checks and image inspection to follow.

VERDICT: PENDING; this document is not a pass gate.

## Protocol and implementation review

- Read both producer protocols before conclusions, candidate `probe.py`, UAV `freeze_cases.py` and `run_probe.py`, scope, prior governance, adjacent-pair diagnosis and label-risk assessment.
- Candidate cache does contain each GT's best eligible box coordinates. Therefore it is legitimate to deduplicate these real coordinates into a partial geometry set E, and E plus final boxes into U, then test distinct matches. This qualifies the initial caution about missing candidate identities: all-candidate identities/scores and full suppression chains remain absent, but the saved-coordinate subset can be analyzed.
- Deduplication prevents identical coordinates from manufacturing two detections. The cardinality-first reward exceeds every possible total IoU advantage; exact threshold semantics are >=.5/.75, NMS compatibility <=.5. All final pair assignments reproduce prior results, and own sparse cardinality checks independently agree.
- Pair-local oracle ranking can put two mutually non-suppressing boxes first. It does not guarantee a single global ranking can realize all pair opportunities, preserve other detections, or improve AP. Report preserves this limitation. Coordinate-compatible surviving boxes have not been assigned/generated new masks.
- UAV reads metric camera-Z depth (cm/100), full camera transforms and actual surfaces. No ground-plane substitute. Same-camera numerical roundtrip is a consistency check, not external calibration validation. All 30 depth maps are fully valid under the producer's depth range, so lack of an additional validity gate in the cross-view grid does not affect these results.
- Shifted-region control is fixed before execution and preserves underlying camera/depth/RGB. It verifies position-specific screen associations only. Shifted regions are not a sham modality inference and do not measure learned source utility.
- 4m XY cooccupancy is a proxy on screen-associated raw surfaces, not actual CVT target-cell attention/mass, nor proof that learned BEV pooling erased instance identity.

## Independent numerical reproduction

Scripts `review_recompute.py`, `review_uav_recompute.py`; exact output `review_recompute.json`.

- Reproduced all six category/lexicographic selections and 12 unique global building IDs from original pair records, fragment records, GT RLE areas and edges.
- Independently reconstructed real-coordinate E/F/U sets from raw stage and final files. Checked 44,808 pair/set feasibility and hard-NMS-compatibility values, all 12,930 saved witnesses (indices, coordinates, distinctness, GT IoUs, mutual IoU), and 1,836 sparse maximum-matching cardinalities. All reported summary counts reproduced.
- Candidate results: dual@.5 371 final box-failed pair appearances, 95 partial-U pair-local ranking-compatible; dual@.75 1116/391. Interior non-small unflagged 1378 appearances: .5 33 failed/22 compatible (8 physical pairs), .75 185/93 (37 physical pairs). No learned recovery is inferred.
- Reimplemented camera calculations independently using homogeneous transforms and inverse K, without importing producer or production camera helpers. All 240 support/shift/Jaccard/unique-satellite-pixel quantities, 120 cross-view depth residual bins, and all six 4m support-cell sets reproduced exactly. All 30 views have finite positive in-range depth.

## Visual inspection and linkage

Read all six context montages, covering satellite and all five UAV views. Solid/dotted contours and legends distinguish correct/shifted screen associations. Readability adequate; these are engineering evidence figures, not statistical estimates.

- 507: stepped roof/facade context is visible in several views. Some source contours extend along facades; satellite overlap does not prove every marked source surface belongs to the target. This is the strongest selected linkage to candidate ranking opportunity at .5.
- 931: tall-building facades and a roof step are visible, but source views vary greatly in target coverage and one provides little support. The current partial candidate set shows no pair-local ranking opportunity at either threshold. Source visibility alone does not repair absent saved geometry.
- 928: stepped buildings and high-rise facades are visible; screen support may include surfaces at different depths. Bbox already succeeds at .5, so this is not evidence for recovering a .5 candidate-ranking failure.
- 1196: height/roof configuration is visible but cyan/magenta source regions partly lie on facade strips; no candidate subset opportunity at either threshold. Canonical ownership remains unresolved.
- 287 and 288: additional side views and roof steps exist, but these are .5 success controls. They have candidate opportunities at .75, not evidence of .5 failure recovery.

Cross-joining selected case IDs/slots with candidate records: only 507 is a dual@.5 ranking-compatible failed case; at .75 only 287 and 288 are compatible. Do not present all six cases as evidence on rank-recoverable failures.

## Findings / remaining gaps

1. No blocking computation error found in these CPU probes.
2. Nonzero candidate geometry and actual source observations establish necessary conditions, not sufficient modality/representation usefulness.
3. The key missing test is **same candidate under satellite information versus correctly associated UAV source features, with position/identity negative controls**, without GT-assisted candidate selection leaking into deployment claims. No such consumption test ran here.
4. GT topology flags are neither confirmed wrong labels nor a clean-set certificate; all geometry opportunity counts remain conditional on current labels. Repetition and developer-set selection prohibit independent generalization claims.
5. An interface that only changes candidate ranking cannot recover geometry absent from the complete eligible set. This cache is partial; absence here cannot establish that limitation for the full model.

## Final verdict

VERDICT: PASS — protocol/implementation/numerical reproduction within declared CPU frozen-artifact scope.

Direction judgment: **有必要条件但缺关键消费证据 / 需补关键证据**. This supports further bounded probe design, not learned module value, architecture effectiveness, AP gain or training authorization. It does not support rejecting the UAV hypothesis. ‘足以继续设计’ is valid only in the narrow sense of retaining this interface as a candidate and designing the missing consumption probe; it is not a passed usefulness gate. ‘不支持’ is not justified because real candidate opportunities and additional observable source structure exist in at least one linked failure case.

## Root synthesis final review

Read `UE_ROI_UAV_FEASIBILITY_20261016.md` in full after both producer reports. VERDICT: PASS. Candidate table, stratified counts, physical-pair counts, six-case support-view counts and candidate linkage agree with independently checked records. Recomputed aggregate depth counts (11,525 / 7,726 within2m / 3,799 behind / 0 in front), maximum source/shift Jaccard .9485799146, maximum roundtrip 1.0629719327e-11px, pose error 0m and maximum FOV error 9.2755753e-7deg all agree with synthesis rounding. The revised UAV summary adds derived coverage fields; independent original quantities remain consistent. All 178 source hashes and all 19 QA-listed artifact hashes independently match their current files.

The synthesis preserves the decisive boundary: **本轮未训练、未fit、未执行新模型forward，正式RoI-UAV模块未实现；只有必要条件，缺关键消费证据。** It does not present partial candidate geometry as AP recovery, screen overlap as physical instance truth, position negatives as source utility, or unflagged labels as clean. The text explicitly withholds model implementation/training approval and keeps 507 as a fixed diagnostic anchor rather than a general benefit claim. No blocking issue or requested wording correction remains in this synthesis review.
