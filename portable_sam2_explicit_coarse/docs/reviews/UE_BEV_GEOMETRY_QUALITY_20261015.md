# Independent review — six-image raw BEV geometry audit

VERDICT: **PASS for the preregistered descriptive six-image CPU geometry audit.** Not a semantic alignment, full-validation or model-gain pass.

Reviewer did not implement or run the original raw-depth audit. Reviewed `PREREGISTRATION.md`, `audit.py`, production geometry/surfaces/candidates/cameras/footprints code and production cache-key builder. Recomputed saved-array displacement counts, means, all p50/p90/p99 and >1/4/8px fractions, plus candidate height/XY compatibility, from six raw_geometry NPZs and their existing immutable geometry caches. No raw depth reload, network forward, GPU or fit. All checks pass.

Independent executable: `/tmp/ue_bev_geometry_quality_independent_review.py`; numeric output `/tmp/ue_bev_geometry_quality_independent_review.json`; log `.log` alongside it.

## Protocol and identity

- Six validation IDs [1218,950,306,1216,948,505] use the stated SHA order; fixed before GT/results, no GT used.
- Source script/protocol/production geometry module SHAs match summary. All six rebuilt production payloads were compared array-for-array to the original cache in the implementation, and the reviewer independently verified all six cache file SHAs and candidate statistics.
- Cache-key composition matches `data/ue_geometry_cache.py`: geometry signature, float local origin, split floor, exact camera text, and resolved EXR path/size/mtime. No cache write occurs. The original cache key alone uses metadata rather than depth content, but this audit additionally records depth/camera file content SHAs and rebuilds from the files, so it does not rely solely on stale-cache metadata.
- Parsed production settings are the actual fixed512m/128cell grid,512² images,4-pixel depth sample,2m zbin,minimum3 top-bin samples. Hardcoded arithmetic in the audit matches these exact settings; script is an experiment snapshot rather than general config-agnostic code.

## Masks, populations and displacement decomposition

- Raw depth valid selection is finite, positive and <10000m. Camera loader guarantees finite transforms, and `backproject` therefore yields finite raw points for these finite inputs. The audit independently uses the same stride/sample centers as production.
- Points outside BEV are excluded before indexing. Valid/invalid height counts and selected-bin counts use the saved in-BEV point population; cell and point populations remain separate.
- A=rawXYZ, B=rawXY + referenceZ, C=referenceXYZ. Thus B−A isolates z replacement under the same camera, C−B isolates XY cell centering under fixed z, C−A measures total displacement. Magnitudes are not expected to add.
- All three displacement measures share **one common finite and positive-camera-depth mask**, additionally height_valid. The optional satellite-frame subset is defined by the raw A pixel only, as preregistered. It is not a requirement that all A/B/C fall inside image.
- Reviewer verified every stored displacement vector is finite in these six outputs and common equals height_valid here; selected-bin is a subset of height_valid. Selected-bin logic correctly reconstructs the highest zbin with>=3 sampled points rather than flooring an averaged float32 reference z.
- Pooled sample counts: all valid-height431153; raw-satellite-frame384632; selected-bin263776; selected-bin and raw-frame237784. These are repeated five-view point samples, not buildings or independent observations.

## Independently pooled examples

| Population | metric | p50 px | p90 px | p99 px | >4px |
|---|---|---:|---:|---:|---:|
| all valid-height | z replacement | .01459 | 4.38600 |16.67459|10.876%|
| all valid-height | XY centering |2.33926|3.29747|3.92688|.647%|
| selected bin | z replacement |.00417|.02835|.09925|0%|
| selected bin | XY centering |2.31100|3.24630|3.85440|.373%|

These support the specific decomposition: selected-bin z replacement is small; points in other observed bins can have larger z-induced displacement. They do **not** show which points correspond to target roofs or which proposals receive attention.

## Surface/candidate interpretation constraints

- Multi-bin counts describe distinct2m quantized sampled heights; bin boundaries, sloped surfaces and facade sampling can create multiple bins. They do not identify multiple semantic surfaces, buildings, roof layers or mask classes.
- The production single-surface representation selects the highest qualifying bin, not a GT roof. Lower-bin displacement should not automatically be called an alignment bug because that point is not necessarily intended to define the selected reference surface.
- A4m cell and its centering produce expected quantization. A2.3px median centering distance is an approximation budget, not proof of erroneous calibration, and cannot directly explain the final segmentation gap.
- Candidate checks count valid candidate **slots/votes**, unweighted across cells/views; XY core means |dx|<=2m AND |dy|<=2m, separate from |dz|<=2m. Production proposal scoring favors a height-compatible core but also admits contextual anchors. Compatibility ratios are not attention weights, semantic correctness or actual used information.
- Reviewer independently confirmed projected cells are a subset of observed&height_valid&candidate-available. Candidate comparisons use the saved payload's float32 coordinates, matching the audited calculation; results describe payload precision and the chosen2m threshold.

## Limits / recommendation

PASS to archive and use as descriptive geometric evidence. No hypothesis gate or CI was preregistered. No result-selected settings change is warranted. This six-image no-GT audit cannot establish external calibration truth, actual feature mixing, target-specific roof recoverability or final-mask/AP causality.
