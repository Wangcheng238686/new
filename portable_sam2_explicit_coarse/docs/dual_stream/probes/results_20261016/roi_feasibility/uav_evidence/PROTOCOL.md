# Frozen source-correspondence feasibility probe

2026-10-16. CPU only; no fit, model inference, annotation edits or training.

Select from existing pair records in lexicographic (image_id, slots) order. Both GTs must have visible area >=1024 and not touch image border. Select two dual@0.5 failures containing existing fragment proxy, then two other dual failures, then two single+dual successes. Exclude any already-selected global building ID across categories. Stop at available cases; do not replace cases on visual quality. Satellite GT is an unverified visible-instance reference; fragment proxy is relative to that GT, not proof of physical fragmentation.

For all selected cases and all five views: use production camera contract version2, metric cameras and camera-Z EXR(cm converted to m). Backproject original pixels using production helpers, project world surfaces to satellite camera; no ground-plane surrogate. Nearest satellite pixel falls within a GT region means screen-overlap support ONLY. Satellite depth and UAV instance labels are absent: this does not establish physical identity or satellite visibility. Show RGB plus support contours and contextual crop for both targets. Record invalid-depth fraction, pose/metadata agreement, direct camera roundtrip error, source support and world XY 4m cooccupancy (diagnostic proxy for coarse spatial mixing, not actual learned BEV weights). Pair RGB-D cannot establish hidden building identity or learnability.

Cross-view geometry check: fixed source pixel grid16, retain locally smooth3x3 depths(range<1m), project to each other view nearest pixel; report signed residual, <=2m consistency, >2m behind and <-2m in front separately. Behind may be true occlusion. This is geometry consistency, not instance correspondence validation.

Read every resulting case image. Conclusions are per case, not estimates of full-data benefits. No automatic pass/fail gate on a visually selected threshold; feasibility remains conditional on annotation semantics and future supervised verification.

Before depth execution add fixed positional negative: shift both satellite GT regions +32px in x; if either target bbox would exceed image, shift both -32px. Same camera/depth; record source support count and within-view binary support Jaccard against correct regions. Nonoverlap or count changes show position specificity only, not learned UAV information or physical truth. Show negative support with dotted contours on all context crops.
