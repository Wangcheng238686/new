# Fixed six-image CPU raw geometry audit

Authorized by root before execution, 2026-10-15. Validation IDs sorted by SHA256('bev-geometry-quality-v1:'+str(id)), first six, no selection using outputs or GT. Frozen manifest 495841b2174242c87661e6b40dba2ddcca13d64b13798116efc8c933c679ca59. No GT loaded. No model/GPU/fit/AP or settings modifications.

Production geometry config is parsed from original EXP-0006 training log. Load all five 512² camera-Z EXR views at production stride4; record source hashes. Rebuild production geometry from these same inputs, compare every array to existing cache if cache is available, record cache identity. No cache writes.

Trace point counts outside BEV, in observed cells without valid top-height, valid-height cells, candidate and projected cells. Unique observed/valid/projected cell statistics separately. Multi-bin describes depth-bin coexistence, not semantic surface identity. Selected points are points belonging to the highest qualifying height-bin, not semantic roof labels.

For each point in valid-height cell compare satellite projections: A raw xyz; B raw xy with selected cell reference z; C reference xyz. Common finite positive-depth A/B/C set only; also separately restrict A to satellite frame. Report ||B-A|| z-change, ||C-B|| xy-center quantization, ||C-A|| total; mean/p50/p90/p99/>1,4,8px. Separate all valid-height points and selected-bin points; point weighted, repeat view points not independent instances. Coordinates recorded metric local world; reference z exact production top-bin mean. Quantization is expected representation approximation, not presumed bug.

Report unique point-height-bin coexistence cells and candidate anchor dz/xy compatibility on observed&height_valid&candidate cells and projected cells; candidate proportions are unweighted proposals, not trained attention nor semantic mismatch. No hypothesis pass/fail threshold, CIs, optimization or result-selected followup. Full tensors saved only under/tmp for audit. Limit six images; stop/report operational errors without switching samples.
