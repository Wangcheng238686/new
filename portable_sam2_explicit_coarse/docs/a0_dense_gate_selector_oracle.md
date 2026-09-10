# A0 instance-selective dense-gate Oracle

## Question

The corrected ten-class A0 audit finds that forcing the global learned-canvas
dense gate from its checkpoint value to `alpha=1` changes mAP by only
`+0.002403`, with 500-image-bootstrap 95% CI `[-0.001495,+0.006008]`.
This does **not** establish stable global-gate value.  It leaves a narrower
question: could an instance-wise gate benefit by opening only where the
already-delivered dense prompt helps?

## Frozen post-hoc protocol

`inference/probes/dense_gate_selector_oracle.py` consumes the existing
`learned_gate_current` and `learned_gate_1.0` artifacts from exactly the same
checkpoint and image IDs.  It hard-fails unless every detection has identical
image id, class, bbox, detector score and mask score in the two arms.  Thus it
can replace only `segmentation`; proposal, ranking and all model inputs are
outside this audit.

Historical canvas records predate `dataset.num_classes` in their manifest.
For that legacy case, `--contract-manifest` supplies only the dataset contract
after hard-checking equal checkpoint, processed image IDs and image count.  It
upgrades newly emitted output manifests and never edits the historical inputs.

For each detection that has a same-class GT box match with IoU >= 0.5, it
computes mask IoU for current and forced-alpha masks.  The Oracle selects the
forced mask iff that individual IoU strictly improves; unmatched detections and
ties remain current.  GT is therefore used only after inference as a selector
label, never as a canvas, feature, prompt or decoder input.

The control is not a global same-count sample: for every image it selects the
same number of matched detections as the Oracle, uniformly at random.  It is
therefore resistant to a spurious gain caused simply by changing more masks in
proposal-dense images.  The tool emits 32 deterministic controls plus the
per-detection audit table.

## Decision gate

Only proceed to a train-only readout/gate if all are true:

1. Oracle mAP is credibly above both `current` and global `alpha=1`, using a
   correct ten-class image-paired bootstrap on its exported records.
2. Oracle also exceeds the per-image-count-matched random control; at least a
   fixed-seed control receives its own paired CI and the Oracle is not within
   the random point-estimate range.
3. A gate trained on NWPU train-520 from pre-existing inference-time features
   predicts the Oracle decision on val-130.  The validation labels must never
   be fitted or used to choose features/hyperparameters.

Failure at any gate stops this route.  The Oracle is an upper bound, not an
ablation row and not evidence that a learned module improves mAP.

## Completed frozen result (2026-09-09)

The original canvas artifacts and the geometry audit manifest identify the
same A0 model checkpoint and exact 130 NWPU validation IDs.  The latter was
used only to restore the missing ten-class contract.  The selector hard checks
all 918 detection non-mask fields; 728 have a same-class box match and the
Oracle changes 248 masks.  Its 32 per-image-count-matched random controls span
0.668426--0.671151 mAP (mean 0.669949).

| comparison | mAP delta | 500× image-paired ten-class 95% CI |
|---|---:|---:|
| current → GT selector Oracle | +0.006307 | [+0.003364, +0.009406] |
| global `alpha=1` → Oracle | +0.003904 | [+0.002262, +0.005646] |
| fixed per-image-count random (seed 44) → Oracle | +0.004020 | [+0.001664, +0.006485] |

Thus all three frozen *upper-bound* gates pass.  This does not yet establish
learnability: the mandatory next test is a train-520-only readout of the
Oracle label, evaluated once on val-130 without using validation labels for
feature or threshold selection.

## Next diagnostic: pre-PromptEncoder readability gate

The candidate must respect the intended prompt-mining location.  It will be a
small per-ROI readout **before** `PromptEncoder`, using only tensors already
available in the production path:

- pooled 14×14 detector RoI feature `x` delivered to `ShapePriorInjector`;
- raw 64×64 coarse-logit confidence/entropy/boundary statistics and the
  resulting full-image canvas statistics;
- normalised proposal geometry, detector confidence and class identity.

It may not use GT, forced-alpha masks, decoder masks/tokens, or an additional
SAM2 forward at inference.  A zero-initialised eventual gate has to preserve
the A0 global gate at initialisation, then output an ROI-specific alpha in
`[0,1]` for the existing dense residual, not render a new canvas:

\[
E_i=E_{base}+\alpha_i\,S_{box,i}(E_{dense,i}-E_{base}).
\]

This is deliberately **not** another native-visual canvas generator.  The
already completed A0 native-SAM high-res readability audit showed aligned
features beat their spatially permuted control but did not beat raw coarse for
the old coarse-error task; SAM image embeddings failed both comparisons.  That
evidence does not rule out their use as an auxiliary gate feature, but it rules
out claiming that they are a validated replacement information source here.

Before that module exists, a frozen train-520/val-130 logistic readout must
predict the post-hoc Oracle label.  We will compare raw-statistics/geometry
only, full aligned pre-PE features, and a row-permuted aligned-feature control.
Its validation-selected masks must exceed both the raw-only and permutation
controls with paired ten-class bootstrap.  Otherwise this direction stops:
the Oracle would be a GT-only routing oracle rather than evidence for a
learnable prompt-mining mechanism.
