# Independent UBR P1 wiring contract review

2026-10-27. Read-only reviewer: `/root/review_ubr_wiring_contract`. No production source edits, training launches, or commits. Reviewed current HEAD 884f5d3 and historical commits below. This is a contract review, not AP evidence.

## Verdict

**Conditional go for a standalone explicit coordinate adapter; block integrated training until target/loss and prediction-context contracts are fixed.** Existing component acceptance does not establish end-to-end identity with the detector. In particular, a global 512-square UBR output must never be passed through the inherited roi-local final-mask loss or bbox-paste predictor.

## Source and history evidence

- `rsprompter/cspm/mask_head.py` forward: original image embeddings, prompt PE, and the first two full-image high-resolution feature tensors are repeated per image's RoIs. They are not RoIAlign-cropped. Prompt boxes and coarse masks condition the SAM2 decode.
- `rsprompter/decoder/sam2_decoder.py`: high-res features pass through conv_s0/conv_s1; the hook on output_upscaling's final activation captures the 32-channel upscaled embedding. Vendor `sam2/sam2/modeling/sam/mask_decoder.py:220-236` forms mask logits by mask-token hypernetwork matrix multiplication with that exact spatial embedding. Thus captured feature and mask logits share the native decoder lattice, but this does not make that lattice a physically cropped satellite image.
- Current legacy `roi_local` reinterprets that output lattice as proposal-local for supervision and prediction. `_MaskHeadTargetsMixin.get_targets` crops assigned GT with proposals using MMDetection mask_target. `_predict_by_feat_single` delegates sigmoid -> bbox probability paste -> threshold to FCNMaskHead.
- **6dc6aa4**: explicit historical full_image-to-roi_local default switch. Full-image sparse foreground BCE collapsed toward background (reported logit mean around -18). This is evidence against silently reviving full-image main loss, not proof that every full-image loss will fail.
- **eef70ca**: UDPR port intentionally kept historical 28-square roi-local main mask loss; selected-point tail loss separately used native-grid RoI GT. It also fixed deployed probability threshold .4's logit anchor. Removing UDPR does not remove the need to separate coordinate contracts.
- `anchor_roi_head._mask_forward` supplies RoI-extracted mask features for coarse/prompt construction while SAM2 image/high-res tensors remain full-image. `prompt_pos_priors` are the prompt and mask-target priors; jitter is currently explicitly forbidden. `extra_pe` is injected at the RoI-head loss/predict entry, not an arbitrary number of times.

## Explicit coordinate choice for P1

The least disruptive mapping is: **retain legacy native output semantics and warp the decoder feature lattice together with its associated mask probability using the same proposal-to-image sampler**. This is a declared latent-grid mapping; do not claim the feature has been physically cropped from a satellite feature map. Using the associated native feature is internally consistent with the hypernetwork's native-lattice mask construction.

An alternative, independently valid new model is to leave the captured feature in its original full-image interpretation and interpolate it to image resolution while pasting only the roi-local mask. That is a different feature-to-mask contract and should not be introduced accidentally. It is not established as the superior route here.

Keep image embeddings, prompt PE, dense shape-prior injection and raw high-res tensors unchanged for the P1 baseline comparison. Do not crop/reposition them while adding this adapter.

### Exact paste contract

Use the installed MMDetection `_do_paste_mask` semantics: image pixel centers x+.5/y+.5, continuous xyxy, align_corners=False, zero padding. Sigmoid BEFORE resampling probabilities. Do not integer-crop or clamp fractional proposals to image boundaries before constructing this mapping. Multi-channel decoder features can use the same sampling grid. Inference geometry and all context must be on the same augmented image lattice; output rescale happens once at the end.

The installed FCNMaskHead calculates canvas from ori_shape and scale_factor when rescale=False, and rescales boxes when rescale=True. Avoid in-place box mutation leaking to subsequent context/feature sampling. Padding, non-square sizes and anisotropic scale factors require explicit tests or explicit rejection for first 512-square scope.

## Blocking loss question: choose explicitly

A global refined output cannot go into old `loss_and_target` with final_mask_coordinate_mode=roi_local. That function resizes predictions to the crop target resolution (historically 28-square), so it would compare the entire image resized to 28-square with an instance RoI crop. Shapes may match while supervision is geometrically wrong.

A paste -> refine -> inverse crop round trip also is not identity, even with zero residual: interpolation is not invertible and finite logit clamping changes gradients.

Two honest strategies:

1. **Minimal preservation strategy, recommended for wiring first:** retain unchanged raw roi-local main mask loss; add UBR band focal+Dice on assigned full-image GT masks, delete UDPR margin. UBR receives its direct .1-weight band loss, while decoder receives the preserved main loss and UBR loss. This requires documenting that the final whole-image mask loss does NOT train UBR; it is a deliberate correction of the earlier proposed contract, not silent implementation. Loss group count remains unchanged.
2. **Final-global-loss strategy:** define a new explicitly normalized full-image/support-balanced main loss on refined output and assigned full-image GT. Re-establish matched satellite baseline and empty/large/small-instance balance. This is a new supervision intervention and cannot be called an unchanged main-mask loss. Historical 6dc6aa4 makes naive full-image BCE particularly unsafe.

Do not claim either strategy has been selected merely because the adapter passes tests. Root must lock the contract before integrated training. Residual-only inverse mapping could retain native zero identity but would create separate local-trained/global-deployed behavior; do not add this extra mechanism merely to avoid the explicit choice.

## Prediction context hazards

1. Training positive proposals are not the inference prediction population. Only current pre-UBR predictions can form neighbors.
2. No-grad is not eval. LoRA dropout and any buffers remain training-mode unless handled; mode and state/RNG changes must be explicit. Predicting from already-extracted train-mode features is not numerically identical to a full eval-mode forward; disclose rather than overclaim.
3. Avoid double extra-PE: passing already PE-injected x to roi_head.predict invokes _inject_extra_pe again.
4. Extra decode calls overwrite `_captured_upscaled_embedding`, `_last_tail_outputs`, coarse/forward telemetry caches. Keep required training outputs local and clear capture before every call. Never read whichever tensor happened to be captured last.
5. Stable predicted candidate IDs exclude predicted self, but sampled positive training RoIs often have no identical predicted candidate ID. Their lineage rule requires an explicit deterministic geometric association without GT identity, or a declared training-context difference. Do not solve it by GT instance matching.
6. Original protected-prefix/hand-scored supplemental candidates are not equivalent to standard combined NMS. P1 context helper must preserve the actual selected P1 population; P2 is the declared place to change candidate selection.

## Zero-initialization parity gates

Baseline must be the **same checkpoint with UDPR disabled**, not the previously refined UDPR result. Removing a trained UDPR changes predictions by design; zero UBR cannot reproduce it.

Required gates:

- Same real image embeddings, prompt PE, RoIs and high-res tensors: decoder raw logits and captured embedding identical to tail-disabled reference; coarse prompt output and detection boxes/scores unchanged.
- Compare image-aligned probabilities against installed FCNMaskHead paste for fractional/clipped/tiny boxes, multiple image IDs, empty RoIs, non-square image and scaled-output cases. Check threshold .4, including exact and near-threshold values; CPU/GPU and AMP-to-FP32 boundaries.
- Define epsilon, finite background logit, max probability error and saturation counts. Exact 0/1 inverse logits cannot be finite and exactly preserve continuous probabilities. Compare final binary mask exactly, probability within locked tolerance. If numerical threshold ties differ, solve explicitly rather than dropping test pixels.
- No second sigmoid or bbox paste after UBR. UBR output already lives on image grid.
- Geometry landmark/checkerboard and mirrored/moved RoIs demonstrate correspondence; invalid geometry must be zero with mask consistent.
- Zero residual outside/inside band exact on UBR input; final image mask parity with old paste. Loss parity applies only to unchanged native loss; must not invent whole-loss parity for changed targets.
- Nonzero UBR residual must change appropriate global pixels, including desired growth beyond proposal boundary; verify finite-logit clamp has not made growth impossible. Full/empty masks preserve their intentionally empty band.
- Gradients: decoder through appropriate declared loss, UBR out head on first step and hidden layers after nonzero head update; geometry consumer reachable; empty-rank graph touches UBR params and performs matching collectives.

## Executed numerical counterexamples

Using repository cvt2 Python / PyTorch (no source mutation):

1. Linear interpolation of logits [-10, 2] then sigmoid gives .0179862; sigmoid then interpolate gives .4404212. At threshold .4 the resulting mask differs. Logit paste is not a valid substitute.
2. Bilinear 4->3->4 resampling of [0,1,0,1] gives [.166667,.375,.625,.833333], maximum error .625. An inverse crop/resize does not recover the original native signal.
3. Clamping probabilities [0,1,.4] to [1e-6,1-1e-6], logit and sigmoid gives max absolute error ~1.073e-6 in FP32; finite-logit conversion cannot provide literal probability identity at 0/1. This simple example preserved .4 binary threshold, but all tie cases still need explicit tests.

## Implementation hold points

- Lock native-vs-global main-loss target contract.
- Lock training positive-RoI to predicted-self exclusion/association, without GT neighbor identity.
- Lock feature mapping interpretation and probability epsilon.
- Lock P1 scope for canvas/rescale/padding and reject unsupported augmentation explicitly.
- Pass real-model raw-output and probability-paste parity before four-GPU integrated resource gate.

No evidence presently requires changing frozen training sources or interrupting existing experiments.

## 主代理裁定与独立复审结论

采用上文最小保留策略：原native主mask loss不变，全图UBR band loss权重.1。active UBR从真实预测pool中匹配，使用稳定预测ID，不将GT正RoI猜测关联到预测ID。复审认可这一简化。固定bbox IoU>=.5、每图最多64、最高IoU分配、同GT可多预测；GT不加入预测pool、不参与邻居。仅正预测UBR监督不等于候选拒伪训练。完整资源门必须包含context decode与可微active redecode/反传。

代码复审找到finite_probability_logits允许过小epsilon时FP32上界舍入为1的具体问题；主代理已将参数下界限制为float32 eps，补反例测试。默认1e-6不受影响。零输出部署恒等保留原概率而不是容忍threshold tie误差。CPU测试与四卡三精度证据见主审计文档；不据此认定完整生产head已接好。
