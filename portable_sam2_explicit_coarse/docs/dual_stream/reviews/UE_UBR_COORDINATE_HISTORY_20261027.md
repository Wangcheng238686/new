# UBR wiring history audit (read only source review, 2026-10-27)

## Finding
For the current roi_local checkpoint contract, native mask logits and captured post-decoder 32-channel upscaled_embedding must be carried through the same bbox-defined output lattice before joint spatial UBR consumption. Do not infer an image-coordinate output lattice merely from repeated image-coordinate input embeddings. This is a conservative inherited contract, not a claim that every learned feature is a physically calibrated image descriptor.

## Exact history
- `6dc6aa49f1ae45db49c20b5f19c8665d1dcd9a51`: defaults changed to roi_local after full_image supervision collapsed on sparse WHU (background shortcut). Changing to full_image is a model/supervision experiment, not a coordinate bug fix.
- `49670c5f`: existing target/paste semantics preserved in split target mixin. Current `rsprompter/cspm/mask_head_targets.py:31-35` explicitly says each decoder output is interpreted as ROI-local, assigned GT is proposal-cropped, prediction uses bbox paste (git blame traces these lines here).
- `eef70ca21255e127ca89b6cf8e0bc98bd0679320`: UDPR port to roi_local; main 28x28 target unchanged, native selected-point loss uses per-proposal native-resolution GT crops; deployed .4 threshold means boundary_logit -0.4055, not zero. Current `rsprompter/detection/anchor_roi_head.py:267-281` inherits this.
- `ddb95247`: comments explicitly document sigmoid -> bbox probability paste; current `rsprompter/cspm/mask_head.py:677-689` preserves parent FCN path.
- Structural move `95026ae` does not change numerical semantics.

## Explicit contrary history / retired mistake
`docs/dual_stream/UE_UDPR_POINT_PROTOCOL_20261023.md:3` marks old S1 retired for treating native output grid as full image. `UE_CONSUMER_BLINDSPOT_PROBES_20261023.md:7-27` explains that complete image/high-res repetition was insufficient evidence of global output coordinates. S1-R maps native cell centers through proposal bbox and uses the same mapping for candidate lookup and diagnostic GT.
These two documents and `udpr_point_roi_probe.py` currently have no tracked git log entries (untracked working-tree historical artifacts); do not invent a commit for them.
Original preserved evidence: `/data/wangcheng/checkpoint/ue_udpr_point_20261023/retirement.json` confirms wrong UAV/diagnostic mapping and source retention. Corrected `/data/wangcheng/checkpoint/ue_udpr_point_roi_20261023/cpu_gate.json` records translation/scale/target-crop verification, zero-projection identity, no-support identity, and source/query gradients.

## Actual current tensor path
- `rsprompter/cspm/mask_head.py:376-400`: image embeddings, PE, high-res features repeat per proposal; they are not RoI-cropped here.
- `rsprompter/decoder/sam2_decoder.py:111-126`: capture final output_upscaling activation (act2), not raw global conv_s0 feature.
- `../sam2/sam2/modeling/sam/mask_decoder.py:217-234`: instance-conditioned transformer output is upscaled, global high-res skip features are added, and mask logits are pointwise `hyper_in @ upscaled_embedding`. The captured 32ch output and logits therefore share an identical native lattice. Captured features are not identical to the earlier repeated global input.
- `rsprompter/decoder/sam2_decoder.py:220-225`: native dimensions equality asserted.
- `rsprompter/cspm/mask_head_targets.py:24-49,307-325`: training uses proposal crop target and resizes mask predictions to target resolution.
- `_predict_by_feat_single` in `mask_head.py:677-689`: roi_local delegates to FCNMaskHead sigmoid + bbox probability paste; full_image branch separately resizes image canvas and ignores bbox.
- installed `mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:336,450-470`: sigmoid before grid_sample; image pixel centers +.5 map via bbox to normalized grid; align_corners=False and zero padding.

## Required adapter contract
1. Declare frame explicitly; current post-decoder 32ch default should be native ROI, never automatically global just because input was global.
2. Warp logits by sigmoid -> exact historical probability paste -> finite inverse-logit for UBR. Feature warp uses same normalized bbox grid, applied directly to channels, without sigmoid.
3. Finite background inverse-logit is intentional approximation; preserve original probabilities/binary output in zero-residual parity oracle and publish clipping epsilon. Sigmoid(interpolate(logits)) is not interpolate(sigmoid(logits)).
4. At zero residual, native raw tensors unchanged, final mask scores and binary masks agree with original FCN deployment including .4 threshold.
5. Native final-mask GT uses continuous/aligned crop; native UDPR auxiliary GT used integer floor/ceil crop plus nearest resize. They are not numerically identical samplers. UBR satellite-grid GT should be explicitly declared rather than silently inheriting wrong crop.
6. Use floating bbox and transformed image coordinates; do not round bbox or use ori_shape without handling scale factor. Test translation, fractional boxes, nonuniform scaling, border clipping, empty boxes and RoI batch identity/order.
7. A genuine pre-decoder conv_s0 feature is image-frame and should be resized/cropped in image coordinates; do not generalize post-decoder ROI treatment to all 32ch tensors.

## Limits
Historical/global input spatial bias and output ROI supervision coexist. This is a learned legacy reinterpretation, not proof decoder has ideal geometric equivariance. Keeping raw post-decoder feature and its own logits together honors the deployed semantics; changing one to global while pasting the other yields a new mixed-frame model requiring independent justification. Synthetic parity gates verify the declared contract, not downstream AP.
