# Implementation notes

The three groups are actual decoder-input dense interventions with original sparse P always fixed. The source mask decoder is still executed normally in B/G; a shape-injector output hook substitutes only the coarse tensor before original canvas/PE/gate/support processing. A decoder prehook then restores P sparse (including box tokens, invalid-point behavior and actual sequence length). Tail decoder, thresholds and mask paste remain unchanged.

This makes GT_binary_pm4 exactly the earlier factorial PG cell, even though its source point mining upstream uses G; input sparse is P by actual-value assertion/SHA. Baseline PP and historical PG endpoint/prompt SHA equality are per GT, stronger than matching aggregate means.

The visual guard compares every value of decoder embedding, positional embedding and both high-res maps on GPU against the one-image fixed source expanded in GT order. Source features are hashed before and after the entire image. This retains value-level input isolation while removing repeated CPU copies/hashing of large per-GT expanded visual maps. Output elapsed is still diagnostic runtime, not a performance benchmark.

The predicted +/-4 comparison preserves the 64x64 sign shape exactly; resampling through the original canvas can nevertheless change signs/shapes downstream because interpolating hard +/-4 differs from interpolating variable logits. Therefore that contrast is a binarization/amplitude/within-shape distribution intervention, not a pure confidence temperature intervention, nor a promise that the final rasterized shape is fixed.

GT versus predicted +/-4 changes only 64x64 shape values at equal finite amplitude and fixed original points. Dense encoding is nonlinear and point/dense consistency can differ. A positive result supports conditional headroom from this shape-source intervention, not learned coarse quality or UAV attainability.

For the 32x32 dense-box support0 subset all three dense prompts must equal the no-mask base and final masks must be pixel-exact equal to PP. This is a negative control. It is unrelated to UAV geometry coverage. No support rule is fixed in this experiment.
