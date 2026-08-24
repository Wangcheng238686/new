# WHU-1024 ablation matrix

| ID | Image emb. | Neck | Prompt route | Final mask | FiLM | Box token | Dense prompt | P2 refiner |
|---|---:|---|---|---|---:|---:|---:|---:|
| B0 | 32×32 legacy | aggregator | legacy 5-token MLP | ROI-local | - | - | - | - |
| B1 | 32×32 legacy | PAFPN | legacy 5-token MLP | ROI-local | - | - | - | - |
| M0 | 32×32 legacy | aggregator | legacy 5-token MLP | full-image | - | - | - | - |
| M1 | 32×32 legacy | PAFPN | legacy 5-token MLP | full-image | - | - | - | - |
| C1 | 32×32 legacy | aggregator | coarse 2P2N | full-image | no | no | no | no |
| C2 | 32×32 legacy | PAFPN | coarse 2P2N | full-image | no | no | no | no |
| C2-L | 32×32 legacy | PAFPN | coarse 2P2N | full-image + ROI-focused loss | no | no | no | no |
| C2-R | 32×32 legacy | PAFPN | coarse 2P2N + ROI-SAM2 | ROI-local | no | no | no | no |
| C3 | 32×32 legacy | PAFPN | coarse 2P2N | full-image | no | yes | no | no |
| C4 | 32×32 legacy | PAFPN | coarse 2P2N | full-image | no | yes | yes | no |
| C4-densefix | 32×32 legacy | PAFPN | coarse 2P2N | full-image | no | yes | yes (fixed 0.5) | no |
| C4-densefix-unfreeze | 32×32 legacy | PAFPN | coarse 2P2N | full-image | no | yes | yes (fixed 0.5, ds unfrozen) | no |
| R0 | 64×64 official | aggregator | legacy 5-token MLP | ROI-local | - | - | - | - |
| R1-C3 | 64×64 official | PAFPN | coarse 2P2N | full-image | no | yes | no | no |
| R1-C4 | 64×64 official | PAFPN | coarse 2P2N | full-image | no | yes | yes | no |
| R1-C4-RD | 64×64 official | PAFPN | coarse 2P2N | full-image | no | yes | raw logits (detached) | no |
| R1-C4-G | 64×64 official | PAFPN | coarse 2P2N | full-image | no | yes | hard-EDT Gaussian (detached) | no |
| C5-v2 | 64×64 official | PAFPN | coarse 2P2N | full-image | no | yes | yes | yes |
| C3-roi_local | 32×32 legacy | PAFPN | coarse 2P2N | **ROI-local** | no | yes | no | no |
| R1-C4-RD-roi_local | 64×64 official | PAFPN | coarse 2P2N | **ROI-local** | no | yes | raw logits (detached) | no |
| Paper PromptMiner RD+P2 | 64×64 official | PAFPN | coarse 2P2N | **ROI-local** | yes | yes | raw logits (detached) | yes |

The C1–C5 / R1-C* rows above were trained under the historical `full-image`
final-mask contract, which collapses under sparse WHU targets (see the
supervision-collapse diagnosis below). Since that diagnosis, the coarse route
defaults to **ROI-local**; the two `-roi_local` rows re-evaluate the same
architecture under healthy supervision. `full-image` is now only M0/M1's
committed control (explicit), not a coarse-route default.

B0/B1 use the historical ROI-local final-mask target and bbox paste. M0/M1
reuse the same MLP, positional embedding, trainable zero no-mask embedding and
all optimization hyperparameters, but switch only the final-mask contract to
full-image target plus one resize at prediction. Therefore B0→M0 and B1→M1
isolate the coordinate contract, while M0→C1 and M1→C2 compare MLP against
coarse under the same full-image contract.

C2-L keeps C2's model inputs, 2P2N points, full-image decoder coordinates and
single-resize inference unchanged. It changes only final-mask supervision:
the positive proposal box is expanded by 1.20x on the decoder grid, foreground
and background BCE are balanced inside that support, ROI Dice has weight 1.0,
and an outside-ROI BCE term with weight 0.05 suppresses full-canvas leakage.
The proposal support is training-only; validation and checkpoint inference do
not crop or paste the predicted mask.

C2-R is a separate ROI-SAM2 alternative. For each positive proposal it uses
aligned RoIAlign to crop the 32×32 image embedding and the 128×128/64×64
high-resolution SAM2 features, then normalizes each crop back to the feature's
native spatial size. The coarse 2P2N coordinates are mapped to a canonical
0–1024 ROI-local PromptEncoder canvas. MaskDecoder therefore predicts an
ROI-local mask, training uses the same proposal crop as the ROI target, and
validation/checkpoint inference paste the mask through the detected box.

```bash
bash scripts/ablations/m0_aggregator_mlp_full_image.sh
bash scripts/ablations/m1_pafpn_mlp_full_image.sh
```

All coarse experiments currently default to `SHAPE_CONTEXT_FUSION=roi_only`.
This is a true no-FiLM route: global context projection, box-conditioned
cross-attention, gamma/beta heads and the context gate are not constructed.
FiLM remains available only for a later controlled ablation:

```bash
SHAPE_CONTEXT_FUSION=gated_spatial_film \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
```

The default run tag includes the resolved fusion name, so this command writes
to a `*_gated_spatial_film` experiment rather than mixing with `*_roi_only`.

B0 is the new-mainline bridge control. The byte-exact historical baseline remains
under `../../legacy_baseline` at repository level and is launched with
`../../../scripts/reproduce_legacy_segm.sh`.

B0/B1/M0/M1 reproduce the historical MLP bridge's image positional embedding: the
64×64 parameter is zero-initialized and trainable, then interpolated to the
selected MaskDecoder image-embedding size (32×32 for B0/B1/M0/M1). It is deliberately
not replaced by a sinusoidal or PromptEncoder random-Fourier PE.
The B0 wrapper defaults to `RUN_TAG=b0_aggregator_mlp_aligned`, keeping its
logs/checkpoints separate from invalid runs produced before the ROI-target
repair. `ABLATION_ID` remains `b0_aggregator_mlp`.
C1–C4 and the coarse-strategy wrappers append
`_semanticfix_<shape_context_fusion>` to their default run tags so corrected
contracts and no-FiLM/FiLM checkpoints cannot overwrite each other.

Every wrapper first validates the resolved MMEngine config. Architecture values
are fixed by the wrapper and cannot be inherited from stale environment values.

R0 isolates the SAM2 spatial-resolution change on top of B0. The dedicated
R1 attribution matrix uses R1-C3, R1-C4 and C5-v2 at the same official 64×64
resolution. R1-C3→R1-C4 changes only the dense prompt switch, while
R1-C4→C5-v2 changes only the P2 boundary refiner switch. They
set `SAM_IMAGE_EMBED_STRIDE=16`, select the 64×64 stride-16 FPN feature for the
MaskDecoder, use 256×256/128×128 high-resolution features, and configure a
64×64 PromptEncoder grid (dense-mask input: 256×256). The detector still
receives the same four-level FPN. Existing B0/B1, M0/M1 and C1–C4 remain stride-32/32×32 for
historical reproducibility.

```bash
bash scripts/ablations/r0_b0_aggregator_mlp_emb64.sh
bash scripts/ablations/r1_c3_pafpn_coarse_points_box_emb64.sh
bash scripts/ablations/r1_c4_pafpn_coarse_points_box_dense_emb64.sh
bash scripts/ablations/r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh
bash scripts/ablations/r1_c4_g_pafpn_coarse_points_box_gaussian_emb64.sh
bash scripts/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh
```

Run the three attribution experiments sequentially on one two-GPU machine so
they do not compete for memory. Their default protocol is the common 20% train,
100% validation, seed 44, EMA-off screening setup. `RUN_IN_BACKGROUND=0` makes
each command block until completion, so pasting the block remains truly serial:

```bash
RUN_IN_BACKGROUND=0 bash scripts/ablations/r1_c3_pafpn_coarse_points_box_emb64.sh
RUN_IN_BACKGROUND=0 bash scripts/ablations/r1_c4_pafpn_coarse_points_box_dense_emb64.sh
RUN_IN_BACKGROUND=0 bash scripts/ablations/r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64.sh
RUN_IN_BACKGROUND=0 bash scripts/ablations/r1_c4_g_pafpn_coarse_points_box_gaussian_emb64.sh
RUN_IN_BACKGROUND=0 bash scripts/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh
```

## R1-C4-G: Gaussian dense representation

R1-C4-G is a representation ablation, not a P2 or full SAMRefiner experiment.
It retains R1-C4's detector, PAFPN, 64x64 image embedding, coarse head, original
2P2N miner, proposal box token, frozen SAM2 PromptEncoder/no-mask embedding,
learnable global-sigmoid dense gate (`alpha_init=0.25`), losses and optimizer.
It changes only the detached dense observation relative to R1-C4-RD.

The required causal comparisons are:

- R1-C4 vs R1-C4-RD: effect of removing final-mask gradient through raw dense;
- R1-C4-RD vs R1-C4-G: raw vs Gaussian representation under the same detach;
- R1-C3 vs R1-C4-G: net benefit over no dense prompt.

For each ROI, R1-C4-G thresholds `sigmoid(coarse.detach())` at 0.5, applies an
exact Euclidean distance transform to the 64x64 hard foreground, selects the
single global maximum as the centre, and uses the full foreground area. Centre
and area are analytically mapped through the proposal to the native 256x256
full-image PromptEncoder mask canvas. The positive-only prompt is

```text
G(x,y) = 15 * exp(-((x-xc)^2 + (y-yc)^2) / (4 * A_canvas))
```

It is not temperature-scaled, signed or clamped. The full-image Gaussian is
isotropic before R1-C4's existing post-encoding proposal-box support is applied.
An empty coarse foreground is marked invalid and its encoded mask is replaced
per ROI by the exact pretrained no-mask base, rather than treating an encoded
all-zero canvas as no-mask. Exact EDT requires `scipy.ndimage` in the configured
Python environment. No Gaussian-specific loss or warm-up is added.

Both controls use the common screening protocol: independent initialization
from the same SAM2 Base+ checkpoint, seed 44, 20% train, 100% validation, 80
maximum epochs, validation every epoch and EMA off. Checkpoint inference reads
`dense_prompt_cfg` from `cfg.model`, so transform, detach, threshold, omega and
gamma are reconstructed without relying on the wrapper environment.

R1-C4-G is promoted to full-data training only when all pre-registered checks
hold: best smoothed segm/mAP is at least 0.005 above R1-C4-RD; the benefit is
present for at least three validations and the last-five-validation mean is not
lower; it also beats R1-C3; bbox/mAP drops by no more than 0.01; and Gaussian
valid ratio, alpha and applied-delta ratio show that the route is active without
NaN or abnormal saturation. These screening criteria are not a multi-seed or
full-data proof.

## C4 dense-gate repair (densefix)

The C4 screening run showed that the dense prompt contributed no measurable
segm/mAP gain over C3 (both tracking within epoch noise). The learnable
`global_sigmoid` dense gate moved only slightly from its 0.25 initialization
(α: 0.250→0.255 over the first eight epochs), while the dense embedding's
`applied_delta_ratio` remained around 0.13. This means the applied perturbation
was roughly **13%**, not 0.13%, of the base embedding norm. The observation
motivates a gate-strength ablation but does not by itself prove that the learned
gate is faulty; a low coefficient may also be the model's response to redundant
dense information.

C4-densefix is the single-variable gate-strength control. It retains C4's
prompt route, coordinate contract and lack of P2 refiner, while replacing the
trainable `global_sigmoid` coefficient with a fixed **0.5** coefficient. The
model fingerprint changes because the gate mode is part of `cfg.model`, even
though it remains in the C4 experiment family. The available fixed-only run was
terminated early, so it cannot by itself close the gate attribution.

The variant lives in a dedicated config so the committed C4 config is unchanged:

- `configs/whu1024_baseplus_explicit_coarse_densefix.py` — same as the coarse
  config except `shape_prior_cfg.shape_scale_mode="fixed"` and
  `prompt_scale_init=0.5`.
- The wrapper selects it via `CONFIG_OVERRIDE` (a new optional `_run_ablation.sh`
  hook that, when unset, leaves every existing experiment on its committed
  config path). `ABLATION_ID` stays the C4 architecture identity; `RUN_TAG`
  carries a `_densefix` suffix so logs/checkpoints never collide with C4.

```bash
# Defaults to GPU 1,2 (C2-r's former slot, shared with C3).
bash scripts/ablations/c4_pafpn_coarse_points_box_dense_densefix.sh

# Retune the fixed gate strength:
SHAPE_DENSE_ALPHA_INIT=0.75 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense_densefix.sh
```

### C4 densefix + unfreeze mask_downscaling

Canvas-quality investigation revealed a second bottleneck: the PromptEncoder
`mask_downscaling` convolution stack (3 layers, 4,684 params) is frozen with
the rest of the PE when `freeze_all=True`.  These pretrained conv weights were
optimised for SAM's original strong, full-image mask prompts and cannot adapt to
the 98%-sparse coarse-mask canvas that the shape injector produces, even when
the fixed gate lifts the injection strength.

`c4_pafpn_coarse_points_box_dense_densefix_unfreeze` builds on the densefix
gate repair (fixed α=0.5) and additionally leaves `mask_downscaling` trainable
via the checkpointed `prompt_encoder_cfg.train_mask_downscaling=True` field.
The remaining PE components (point embeddings, no-mask embedding, positional
encoding) stay frozen. A new
environment hook `PROMPT_ENCODER_LR_MULT` (default 0.0, here overridden to 1.0)
replaces the previously hard-coded zero multiplier so the 4,684 trainable
parameters receive a non-zero learning rate.

```bash
# Reuses the same densefix config.  GPU 1,2 (shared with C3).
bash scripts/ablations/c4_pafpn_coarse_points_box_dense_densefix_unfreeze.sh

# The wrapper resolves this launcher input into cfg.model:
PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=1 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense_densefix_unfreeze.sh
```

## C3/C4/densefix-unfreeze/C5-v2 stride-32 vs stride-16 diagnosis (2026-07-30)

Four coarse-route experiments were trained with the standard 20% train / 100%
validation screening protocol, `SEGM_SCORE_MODE=detector`, `EMA off`, and all hit
`EARLY_STOPPING_PATIENCE=10` (early-stop start at epoch 20) around epoch 32.
The table reports the smoothed segm/mAP the early-stopper tracked, the best raw
segm/mAP, and the best bbox/mAP.

| Experiment | Stride / grid | Dense gate | mask_ds | Refiner | best smooth segm/mAP | best raw segm/mAP | best bbox/mAP |
|---|---|---|---|---|---:|---:|---:|
| C3 | 32 / 32×32 | — | frozen | no | 0.4544 | 0.4773 | 0.7120 |
| C4 | 32 / 32×32 | global_sigmoid (α≈0.25) | frozen | no | 0.4549 | 0.4852 | 0.7150 |
| C4-densefix-unfreeze | 32 / 32×32 | fixed (α=0.5) | **unfrozen** | no | 0.4526 | 0.4663 | 0.7080 |
| C5-v2 | 16 / 64×64 | global_sigmoid (α≈0.25) | frozen | yes | **0.6321** | **0.6480** | 0.7116 |

Source logs (each is the canonical full run; `pid` identifies the launcher):

```text
logs/ablations/c3_pafpn_coarse_points_box_semanticfix_roi_only_tr0.2_va1.0_20260729_235545_pid374570.log
logs/ablations/c4_pafpn_coarse_points_box_dense_semanticfix_roi_only_tr0.2_va1.0_20260729_235711_pid375196.log
logs/ablations/c4_pafpn_coarse_points_box_dense_densefix_unfreeze_tr0.2_va1.0_20260730_031855_pid429545.log
logs/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64_tr0.2_va1.0_20260729_235752_pid375615.log
```

A killed-early C4-densefix run (fixed gate, frozen mask_downscaling) is also on
record for the gate-only check; it was superseded by densefix-unfreeze:

```text
logs/ablations/c4_pafpn_coarse_points_box_dense_densefix_tr0.2_va1.0_20260730_020955_pid409970.log
```

### Dense monitor across the three stride-32 runs

The `applied_delta_ratio` (dense embedding perturbation relative to the frozen
`no_mask_embed` baseline) confirms each repair actually lifted the injected
signal, yet mAP did not move:

| Run | α (final) | applied_delta_ratio (final) | best smooth segm/mAP |
|---|---:|---:|---:|
| C4 (global_sigmoid, mask_ds frozen) | 0.277 | 0.146 | 0.4549 |
| C4-densefix-unfreeze (fixed 0.5, mask_ds unfrozen) | 0.500 | 0.307 | 0.4526 |

### What is and is not proven

These runs establish that, **on stride-32/32×32 with
`restrict_dense_prompt_to_box=True`**, the combined intervention of fixing the
dense coefficient at 0.5 and training `mask_downscaling` yields no measurable
segm/mAP gain over the sparse-only baseline. The injected ratio rose from 0.146
to 0.307 (a ~2× increase) and the result still tracks C3 within epoch noise.
Because the completed run changes both variables, their individual effects are
not identified; the fixed-only run ended too early.

Two variables remain **untested**, so "dense is useless" must not be concluded
globally:

1. **`restrict_dense_prompt_to_box`**: this was never switched off. With the
   hard 0/1 box support, the dense delta touches only the ~3×3 embedding pixels
   that a typical ~100×100 px WHU building box maps to on a 32×32 grid (~0.9%
   coverage, matching the logged `canvas_nonzero_ratio≈0.5–2%`). The dense
   information therefore lands on a handful of pixels that the 2P2N points and
   the box token already localise.
2. **Image-embedding resolution**: C5-v2 runs at stride-16/64×64, where the same
   box covers ~6×6 pixels and dense shape structure has room to diverge from
   point localisation. C5-v2 also enables the P2 boundary refiner. The local
   R1-C4 control is now running and reached segm/mAP 0.6149 at epoch 6 without
   the refiner, versus C5-v2's final best 0.6480. This strongly supports a large
   stride-16 contribution, but the comparison remains preliminary until R1-C4
   finishes under the same early-stop protocol.

### Hypothesis on the stride-32 mechanism

Inference (not proven): at stride-32, a ~100×100 px WHU building box quantises
to ~3×3 embedding pixels. Inside such a small region, the 2P2N points plus the
box token already occupy nearly every spatial position, so the continuous shape
distribution the dense canvas carries has little room to differ from the
localisation the sparse prompts already provide. At stride-16 the same box maps
to ~6×6 pixels, leaving spatial structure the dense prompt could exploit. This
is consistent with the segm/mAP gap being present from epoch 1 (0.517 vs 0.20)
before the refiner has learned much. The ongoing R1-C4 result is consistent
with this mechanism, but final attribution must use its completed curve rather
than the current epoch-6 snapshot.

### What the P2 refiner actually changed

The per-epoch refiner statistics show raw ≈ refined throughout (dice differs
in the 4th decimal, boundary-F1 in the 3rd):

| Epoch | raw_dice | refined_dice | raw_bF1 | refined_bF1 |
|---|---|---|---|---|
| 6 | 0.8009 | 0.8011 | 0.8233 | 0.8301 |
| 32 | 0.8333 | 0.8319 | 0.9158 | 0.9160 |

So on the coarse-mask quality metrics the refiner's direct correction is
marginal; this is consistent with the 64×64 route carrying most of the gain,
but the completed R1-C4 control is still required. The refiner's loss share
did climb steadily (`p2_boundary` from 2.9% to 6.7%), but this has not been
shown to convert into segm/mAP that a stride-16-without-refiner run could not
also reach.

### Next experiments needed to close the attribution

1. **Finish R1-C4** (`SAM_IMAGE_EMBED_STRIDE=16`, dense, no refiner) under the
   same early-stop protocol. Its epoch-6 segm/mAP is 0.6149; the completed best
   and smoothed curves will quantify the stride-16 contribution and bound the
   refiner's incremental contribution against C5-v2.
2. **Run R1-C3** (`SAM_IMAGE_EMBED_STRIDE=16`, points+box, no dense, no
   refiner). R1-C3→R1-C4 is the strict dense-prompt attribution at official
   resolution; R1-C4→C5-v2 is the strict P2-refiner attribution.
3. Optionally, a stride-32 run with `restrict_dense_prompt_to_box=False` to test
   whether the box restriction — not the resolution — is the dense bottleneck.
   This requires exposing the flag (currently config-only).

## full_image mask-supervision collapse hypothesis (2026-07-31)

B1 (MLP, stride-32, roi_local) reaches best segm/mAP 0.6888 over 74 epochs while
every coarse-route experiment (full_image) plateaus at ~0.45 (stride-32) or
~0.65 (stride-16) by epoch 25-32. The gap is not explained by route, resolution,
dense or the refiner alone. The training logs expose a supervision-level cause:

| metric (ep32) | B1 (roi_local) | C3 (full_image) |
|---|---:|---:|
| loss_mask | 0.12 | **0.0012** |
| mask logit_mean | +1.0 | **-18** |
| mask prob_mean | 0.52 | **0.0048** |
| mask prob >= 0.5 | 0.52 | **0.0048** |
| target_fill | 0.52 | 0.0048 |

Under `final_mask_coordinate_mode=full_image`, each decoder mask is supervised
against a full-image GT in which a WHU building covers ~0.5% of pixels. Standard
BCE then collapses to an all-background shortcut: the model drives almost every
logit to ~-18, predicts near-zero probability everywhere, and `loss_mask` falls
to ~0.001 — not a good fit but a degenerate one. segm/mAP is then carried almost
entirely by detector box localisation, not mask quality, so the coarse route
plateaus regardless of dense/refiner additions.

B1 avoids this because `roi_local` crops the GT inside each proposal, giving a
balanced target (~0.5 fill) that standard BCE can actually learn (loss_mask
~0.12). The coarse route currently forces `full_image` for every non-ROI-SAM
experiment (`_run_ablation.sh` coarse branch + the config).

### Why C2-L (roi_balanced_dice) did not fix it

C2-L keeps the `full_image` coordinate space and only reweights the loss: a
rectangular 1.2x-box support gives balanced BCE+Dice inside and a 0.05-weight
background BCE outside. This fails because (a) the rectangular support does not
match the real mask shape so "inside" is still mostly background, and (b) the
weakened outside supervision lets the full-grid prediction drift, lowering IoU.
C2-L scored *worse* than the standard full_image run, which is consistent with
"rebalancing a loss inside the wrong coordinate space is harmful".

### roi_local results: collapse confirmed, coarse route surpasses MLP

Switching the coarse route to roi_local was confirmed in one epoch: mask
`logit_mean` returned from ~-18 to ~+0.2, `loss_mask` rose from ~0.001 to ~0.63,
and `target_fill` from ~0.005 to ~0.5. The full_image BCE collapse hypothesis is
confirmed — the coarse route's low plateau was a supervision artefact.

Following the diagnosis, the default `final_mask_coordinate_mode` for every
non-ROI-SAM coarse route was switched from `full_image` to `roi_local`
(`_run_ablation.sh` coarse branch + the committed config). `smoke_all.sh` step
[3/5] now asserts each wrapper's resolved coordinate contract, so a silent
regression fails CI rather than training under the wrong supervision. M0/M1 keep
their committed `full_image` (explicit) as the only remaining full_image controls.

Final best segm/mAP over the 20% train / 100% val screening protocol (all
roi_local unless noted, all trained to convergence without early-stop kill):

| Experiment | Stride | Route | best segm/mAP | vs B1 |
|---|---|---|---:|---:|
| C3 (full_image, collapse) | 32 | coarse points_box | 0.4773 | −0.211 |
| **C3-roi_local** | 32 | coarse points_box | **0.6107** | −0.078 |
| R1-C4-RD (full_image, collapse) | 16 | coarse + raw_detach dense | 0.6450 | −0.044 |
| **R1-C4-RD-roi_local** | 16 | coarse + raw_detach dense | **0.6957** | **+0.007** |
| B1 (MLP baseline) | 32 | MLP | 0.6888 | — |

Conclusions from the converged runs:

1. **roi_local repair lifts the coarse route to and beyond the MLP baseline.**
   R1-C4-RD-roi_local (coarse + dense + stride-16) reaches 0.6957, surpassing B1
   (0.6888). At stride-32, C3-roi_local still trails B1 by ~0.08, confirming
   stride-16 is genuinely needed for the coarse route on these small WHU targets.
2. **stride-16 is a real, separable gain under healthy supervision.** Same route,
   same roi_local, stride-32→16: C3-roi_local 0.6107 → R1-C4-RD-roi_local 0.6957
   (+0.085), no longer confounded with the full_image collapse.
3. **The dense prompt + coarse route has measurable validation value.** Earlier
   full_image runs could not show it because the mask was never learned. The
   held-out test result below shows that the aggregate gain is modest and comes
   mainly from large objects, so this is not a uniform replacement for B1.

### B1 vs R1-C4-RD-roi_local validation and test (2026-07-31)

Both runs use 20% train / 100% validation, seed 44, detector scores, no EMA and
ROI-local bbox paste. Each row evaluates the **segm-best** checkpoint, not the
separately saved bbox-best checkpoint. Validation contains 627 images / 15,926
instances; test contains 2,220 images / 70,063 instances. AP50 values in the
training logs are printed to three decimals; test values come from the saved
standalone-inference `metrics.json`.

| Experiment / split | checkpoint | bbox/mAP | bbox/AP50 | bbox/AP75 | segm/mAP | segm/AP50 | segm/AP75 |
|---|---|---:|---:|---:|---:|---:|---:|
| B1 validation | `best_model_epoch70.pth` | 0.7454 | 0.891 | 0.8395 | 0.6888 | 0.891 | 0.828 |
| B1 test | same checkpoint | 0.7235 | 0.8863 | 0.8220 | 0.6796 | 0.8872 | 0.8212 |
| R1-C4-RD-roi_local validation | `best_model_epoch80.pth` | 0.7450 | 0.891 | 0.8400 | 0.6957 | 0.892 | 0.829 |
| R1-C4-RD-roi_local test | same checkpoint | 0.7236 | 0.8867 | 0.8225 | 0.6839 | 0.8880 | 0.8137 |

The test-set delta (R1 minus B1) is `+0.0001` bbox/mAP and `+0.0043`
segm/mAP. The segmentation gain is scale-dependent rather than uniform:

| Test segm metric | B1 | R1-C4-RD-roi_local | delta |
|---|---:|---:|---:|
| mAP_s | 0.4038 | 0.3810 | −0.0228 |
| mAP_m | 0.7338 | 0.7362 | +0.0024 |
| mAP_l | 0.7226 | 0.7553 | +0.0326 |

Thus the validation advantage shrinks from `+0.0069` to `+0.0043` on test.
R1 improves large-object masks but loses small-object AP and `segm/AP75`; bbox
performance is effectively unchanged. The comparable standalone artifacts are:

```text
outputs/b1_best_epoch70_test/{metrics.json,run_manifest.json,predictions.json}
outputs/r1_c4_rd_roi_local_best_epoch80_test/{metrics.json,run_manifest.json,predictions.json}
```

Source logs (roi_local, converged):

```text
logs/ablations/c3_pafpn_coarse_points_box_roi_local_tr0.2_va1.0_20260731_063609_pid847889.log
logs/ablations/r1_c4_rd_pafpn_coarse_points_box_raw_detach_roi_local_emb64_tr0.2_va1.0_20260731_070728_pid855471.log
```

```bash
# coarse route now defaults to roi_local; wrappers without an explicit export
# also run roi_local. Verify with the smoke harness (asserts the contract):
FULL_MODEL_SMOKE=0 bash scripts/ablations/smoke_all.sh
```

Run the C2 final-mask signal ablation with the common 20% train / 100% validation
screening protocol:

```bash
bash scripts/ablations/c2l_pafpn_coarse_points_roi_loss.sh
bash scripts/ablations/c2r_pafpn_coarse_points_roi_sam.sh
```

## Machine environment configuration

Every active shell entry loads `configs/environment.sh` through
`scripts/load_environment.sh`. This is the single machine-specific file to edit
after copying the project to another host. It contains:

- the Python interpreter;
- the vendored SAM2 source and Base+ pretrained checkpoint;
- the WHU dataset root;
- checkpoint, log and temporary roots;
- default visible GPUs, DDP process count, per-rank batch, accumulation and AMP.

Values exported by the outer shell still take precedence for one-off runs. To
keep the repository copy unchanged, select another complete config file:

```bash
PORTABLE_SAM2_ENV_FILE=/absolute/path/to/environment.sh \
  bash scripts/ablations/b0_aggregator_mlp.sh
```

The resolved config path and checkpoint/log/tmp roots are included in every
real run and dry-run hyperparameter snapshot.

## Dataset subsets

The ablation wrappers currently default to 20% of the official training split and the
complete official validation split:

```bash
TRAIN_SUBSET_RATIO=0.2
VAL_SUBSET_RATIO=1.0
VAL_BATCH_SIZE=1
VAL_EVERY_N_EPOCHS=1
COMPUTE_VAL_LOSS=0
```

Override both explicitly when a different screening ratio is needed:

```bash
TRAIN_SUBSET_RATIO=0.1 VAL_SUBSET_RATIO=0.2 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
```

Use `TRAIN_SUBSET_RATIO=1.0 VAL_SUBSET_RATIO=1.0` for a full-data run.

Subsets are deterministic image-level samples. Train uses `SUBSET_SEED`
(default 44); validation uses `SUBSET_SEED + 10000`. All ablations therefore see
the same sample IDs when the ratio and seed are unchanged.

Validation also follows the historical WHU1024 baseline execution contract:
only global rank 0 iterates the full validation loader, the other DDP ranks
wait on a CPU/Gloo epoch-control broadcast, and bbox/segm COCO evaluation uses
detector scores. The long rank-0 validation wait therefore does not occupy an
NCCL collective. Every epoch still predicts the complete validation split and
runs bbox/segm COCO evaluation. For screening, `COMPUTE_VAL_LOSS=0` skips only
the additional `model.loss(...)` forward; metric logging, best checkpoints and
early stopping are unchanged. Set `COMPUTE_VAL_LOSS=1` when validation loss is
explicitly required. Empty-GT images remain in COCO evaluation so false positives
are counted. B0/B1 retain the historical ROI-local target and detected-box paste.
M0/M1 and C1–C5 retain the SAM2 decoder's native full-image grid, train against resized
full-image GT and resize once at prediction. Standalone checkpoint inference
reads the same `final_mask_coordinate_mode`; component/preflight smoke checks
both post-processing contracts.
`VAL_BATCH_SIZE` can be overridden, but `1` is the comparison default.
Predicted dense masks are batch-encoded to COCO RLE immediately per image and
released instead of being retained for an epoch-end float32 scan. Ground-truth
RLE records are cached after the first deterministic validation pass.
The headline/best-bbox metric defaults to the historical `bbox/mAP`; detailed
validation still reports `bbox/mAP_75`. Set `SAVE_BBOX_BEST_METRIC` explicitly
only when intentionally changing that selection criterion.

Segmentation ranking is governed by one checkpoint-persisted model contract,
`segm_score_mode`. Every current B0/B1, M0/M1, C1-C5 and R0/R1 wrapper defaults
to `SEGM_SCORE_MODE=detector`, so training validation and standalone inference
both rank bbox and segm detections by detector `scores`. The optional
`mask_quality` mode is accepted only when the model configuration also enables
a quality head; a missing `mask_scores` field is a hard error rather than a
silent fallback. Candidate thresholding, NMS and `max_per_img` remain detector
score operations in either mode.

## Historical baseline hyperparameter contract

All B0-M1/C1-C5 wrappers inherit the same WHU1024 baseline optimization defaults so
that the matrix changes architecture rather than training conditions:

```bash
BATCH_SIZE=1
GRAD_ACCUM_STEPS=4
NPROC_PER_NODE=2
CUDA_VISIBLE_DEVICES=1,2
LEARNING_RATE=5e-4
SAT_BACKBONE_LR_MULT=1.0
SAT_OTHER_LR_MULT=1.0
MASK_DECODER_LR_MULT=1.0
NO_MASK_LR_MULT=1.0
WARMUP_ITERS=100
WEIGHT_DECAY=0.05
DET_LOSS_WEIGHT_STAGE1=1.0
DET_LOSS_WEIGHT_STAGE2=1.0
DET_LOSS_WEIGHT_STAGE3=1.0
EARLY_STOPPING_PATIENCE=10
EARLY_STOPPING_START_EPOCH=20
EARLY_STOPPING_SMOOTH_WINDOW=5
EARLY_STOPPING_MIN_DELTA=5e-4
EMA_ENABLED=0
EMA_EVAL=0
EMA_SAVE_BEST=0
TORCH_DDP_TIMEOUT_SECONDS=1800
TORCH_DDP_CONTROL_TIMEOUT_SECONDS=86400
SHAPE_CONTEXT_FUSION=roi_only
SEGM_SCORE_MODE=detector
```

With the default two ranks on physical GPUs 1 and 2, the effective global batch
size remains `2 x 1 x 4 = 8`. Learning rate, warmup and epoch count therefore
remain aligned with the previous `4 x 1 x 2 = 8` public setting. B0/B1 also
reproduce the legacy MLP route's zero-initialized, trainable `no_mask_embed`;
detector loss remains at weight 1.0 for every epoch, rather than silently
dropping to 0.75/0.50 after epochs 5/10. Final-mask logs expose the ROI target
fill ratio, mask-logit statistics and thresholded foreground ratio so a
full-image-target/all-empty-mask regression is visible in the first epoch.
coarse routes explicitly load and freeze the SAM2 no-mask embedding as part of
their PromptEncoder-based architecture. Coarse model construction fails if the
checkpoint path/key/value is invalid; full-model preflight also checks that the
MaskHead value was really loaded and remains frozen, so it cannot silently fall
back to zeros. Screening runs default to no EMA construction, no EMA validation,
and no EMA best-checkpoint selection. A deliberate full-data EMA run must opt in
with `EMA_ENABLED=1`; `EMA_EVAL` and `EMA_SAVE_BEST` then inherit that value.

The shared 1800-second NCCL timeout covers training collectives. Rank-0-only
validation and COCO evaluation use a separate CPU/Gloo control group with a
default 86400-second timeout, preventing idle workers from timing out inside an
NCCL broadcast/barrier. Override either layer from an outer wrapper when needed:

```bash
TORCH_DDP_TIMEOUT_SECONDS=3600 TORCH_DDP_CONTROL_TIMEOUT_SECONDS=172800 \
  bash scripts/ablations/b0_aggregator_mlp.sh --master-port 29601
```

The legacy `NCCL_TIMEOUT` environment name remains a fallback when
`TORCH_DDP_TIMEOUT_SECONDS` is unset. The resolved value and source are written
to the log snapshot together with the Gloo control backend and timeout.

## B0 screening reproduction diagnosis (2026-07-29)

The EMA-off 20%-train/100%-validation B0 run was compared against the frozen
full-train historical WHU1024 baseline. Epoch numbers are not directly
comparable: B0 has 74 optimizer steps per epoch, while the full-data baseline
has approximately 368. The meaningful alignment is therefore five B0 epochs
per one historical epoch.

| Optimizer-step alignment | B0 bbox/segm mAP | Historical bbox/segm mAP |
|---|---:|---:|
| B0 epoch 5 / historical epoch 1 (~370 steps) | 0.5291 / 0.2675 | 0.4801 / 0.2858 |
| B0 epoch 10 / historical epoch 2 (~740 steps) | 0.6121 / 0.4584 | 0.5632 / 0.4509 |
| B0 epoch 15 / historical epoch 3 (~1110 steps) | 0.6387 / 0.4979 | 0.6549 / 0.4623 |
| B0 epoch 20 / historical epoch 4 (~1480 steps) | 0.6792 / 0.5444 | 0.6508 / 0.5054 |

The aligned loss and metric trends confirm that the migrated B0 model/loss/
validation wiring is behaving as a usable screening control: bbox and segm
increase on the same scale, and train/validation loss decrease without an
all-empty-mask regression. This is evidence for successful *screening-baseline*
reproduction, not a claim of byte-exact or final-metric reproduction.

Three intentional or inherited protocol differences prevent interpreting this
20% run as the complete historical reproduction:

1. It repeatedly trains on a deterministic 20% subset; 80 B0 epochs contain
   5920 optimizer updates, versus roughly 29440 in 80 historical full-data
   epochs.
2. Screening EMA is disabled. The historical run switches validation to EMA at
   epoch 5, so only its live-weight epochs 1-4 are cleanly comparable here.
3. The historical scheduler computes cosine `T_max` in mini-batches but calls
   `scheduler.step()` only after each accumulated optimizer update. With
   `GRAD_ACCUM_STEPS=2`, its cosine decay is approximately two times slower than
   intended. The migrated trainer deliberately fixes `T_max` to optimizer-step
   units. At the ~1480-step comparison point, B0 is at about `4.34e-4`, while
   the historical run remains near `4.99e-4`.

Source logs used for this diagnosis:

```text
portable_sam2_explicit_coarse/logs/ablations/
  b0_aggregator_mlp_aligned_emaoff_gloo_v1_tr0.2_va1.0_20260729_095000_pid1719009.log
/home/wangcheng2021/project/portable_sam2_fusion_new/whu1024_reproduce_bundle/
  portable_sam2_fusion_new/logs/ablation_hyperparam_whu1024_baseplus/
  whu1024_bs1a2_fixddp_baseplus_ddp4_20260725_200053.log
```

## Smoke and dry-run

```bash
# Config, command, independent subset and representative full-model checks.
bash scripts/ablations/smoke_all.sh

# Fast checks without loading four complete SAM2 models.
FULL_MODEL_SMOKE=0 bash scripts/ablations/smoke_all.sh

# Inspect one resolved experiment without starting training.
DRY_RUN=1 CHECK_DATA=1 PREFLIGHT_MODEL=1 \
  TRAIN_SUBSET_RATIO=0.01 VAL_SUBSET_RATIO=0.02 \
  bash scripts/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh
```

All `DRY_RUN=1` smoke/preflight invocations write only to the invoking terminal
and never create files under `logs/ablations/`, even when `LOG_DIR` or
`LOG_FILE` is set. Real training runs continue to save their complete terminal
log and resolved hyperparameter snapshot.

## Background execution

Training launchers default to `RUN_IN_BACKGROUND=1`. After config validation
and the resolved-parameter snapshot, the runner starts torchrun through
`nohup setsid`, disconnects stdin, prints the launcher PID, and returns control
to the shell. It does not create a PID file. Closing the terminal therefore
does not stop the experiment.

```bash
# Default: detached background training.
bash scripts/ablations/b0_aggregator_mlp.sh

# Follow the exact log path printed by the launcher.
tail -f logs/ablations/<run>.log

# Foreground mode for interactive debugging.
RUN_IN_BACKGROUND=0 bash scripts/ablations/b0_aggregator_mlp.sh
```

`DRY_RUN=1` never starts a background process. The printed `background_pid`
identifies the detached torchrun launcher; inspect it with `ps -fp <pid>`.
Sending a normal termination signal to that launcher lets torchrun shut down
its workers.

## Serial queue after B0

`monitor_b0_then_serial.sh` can remain detached while B0 is running, then execute
the remaining main ablation matrix one at a time. The default order is B1, M0,
M1, C1-C5, R0, R1. Each child wrapper is forced into foreground mode relative
to the monitor, so the next experiment starts only after the current torchrun
exits. A failed, killed, or missing task is recorded and skipped; it does not
stop the rest of the queue. The monitor does not create a PID file and uses a
non-blocking lock to prevent duplicate queues.

```bash
B0_LOG=/absolute/path/to/current_b0.log \
nohup setsid bash scripts/ablations/monitor_b0_then_serial.sh \
  >/dev/null 2>&1 &
```

Monitor events are written to
`logs/ablations/serial_after_b0_<timestamp>.log`; every experiment continues to
write its full resolved snapshot and training output to its own normal ablation
log. `POLL_SECONDS` controls the B0 polling interval. `ABLATION_QUEUE` can
replace the default whitespace-separated wrapper list. `--print-plan` validates
and prints the resolved queue without waiting or launching training. Set one
`RUN_SUFFIX` on B0 and the monitor to isolate a complete restarted matrix in new
log/checkpoint directories without changing the per-experiment IDs.

When `MASTER_PORT` is unset, each launcher derives a port from its PID to avoid
collisions between concurrently started background experiments. A fixed port
can be passed from the outermost wrapper with `--master-port <port>` or
`--master-port=<port>`. The precedence is CLI option, `MASTER_PORT` environment
variable, then automatic derivation. The resolved port and source are included
in the log snapshot.

```bash
# Parallel runs: assign a different rendezvous port to each outer launcher.
bash scripts/ablations/b0_aggregator_mlp.sh --master-port 29601
bash scripts/ablations/c2_pafpn_coarse_points.sh --master-port=29602
bash scripts/ablations/c2l_pafpn_coarse_points_roi_loss.sh --master-port=29603
bash scripts/ablations/c2r_pafpn_coarse_points_roi_sam.sh --master-port=29604
```

Useful common overrides include `MAX_EPOCHS`, `BATCH_SIZE`,
`GRAD_ACCUM_STEPS`, `LEARNING_RATE`, `NPROC_PER_NODE`, `CUDA_VISIBLE_DEVICES`,
`MAX_TRAIN_BATCHES`, `MAX_VAL_BATCHES`, `INIT_FROM`, `RESUME_FROM`, and
`CHECKPOINT_DIR`. Override a shared optimization value only when intentionally
starting a separate hyperparameter experiment; otherwise comparisons no longer
belong to the B0/B1, M0/M1 and C1-C5 architecture-only matrix.

`SAM_IMAGE_EMBED_STRIDE` is fixed by the two resolution wrappers: `16` means
official 64×64 at 1024 input; all legacy wrappers default to `32`. Do not
override it inside a named run, because the contract checker treats it as part
of the experiment identity.

`FINAL_MASK_COORDINATE_MODE` is likewise fixed by the named wrappers: B0/B1/R0
and C2-R use `roi_local`; M0/M1 and the other coarse routes use `full_image`.
Launch the corresponding wrapper instead of overriding this variable on another
experiment ID.

`FINAL_MASK_LOSS_MODE` is `standard` for the original matrix and fixed to
`roi_balanced_dice` by C2-L. C2-L defaults to
`FINAL_MASK_ROI_EXPAND_RATIO=1.20`, `FINAL_MASK_ROI_BCE_WEIGHT=1.0`,
`FINAL_MASK_ROI_DICE_WEIGHT=1.0`, and `FINAL_MASK_OUTSIDE_BCE_WEIGHT=0.05`.
These values are parsed into `cfg.model`, checked before launch, printed in the
hyperparameter snapshot and recovered from checkpoints during inference.
`ROI_SAM_ENABLED=1` is fixed only by C2-R, and
`ROI_SAM_SAMPLING_RATIO=2` controls the aligned RoIAlign samples per bin. Both
are validated before launch; the resolved `roi_sam_cfg` is part of the model
fingerprint and checkpoint reconstruction contract.

Additional CLI arguments are appended to the exact trainer command:

```bash
bash scripts/ablations/c2_pafpn_coarse_points.sh \
  --prompt-debug-stats 1 --prompt-debug-stats-interval 20
```

## Coarse strategy controls

All C1-C5 wrappers default to adaptive-validity 2P2N from epoch 1 and a fixed
coarse BCE+Dice weight of `0.10`:

```bash
SHAPE_POINT_ADAPTIVE_VALIDITY=1
POINT_WARMUP_ENABLED=0
SHAPE_LOSS_SCHEDULE_MODE=fixed
SHAPE_PRIOR_LOSS_WEIGHT=0.10
```

`adaptive_validity=1` invalidates point slots that fail confidence, topology,
or spacing requirements. `adaptive_validity=0` is a real fixed-validity
control: it always emits two probability maxima and two minima, with distance
constraints for the second point. Coarse runs enable prompt diagnostics every
50 iterations by default, including the all-invalid ROI ratio.

The focused C2 strategy matrix lives in `coarse_strategy/`:

| ID | Point validity | Coarse loss |
|---|---|---|
| S0 | adaptive | fixed 0.10 |
| S1 | adaptive | fixed 0.20 |
| S2 | adaptive | epoch 1-5: 0.20; epoch 6+: 0.10 |

```bash
bash scripts/ablations/coarse_strategy/s0_c2_fixed_w010.sh
bash scripts/ablations/coarse_strategy/s1_c2_fixed_w020.sh
bash scripts/ablations/coarse_strategy/s2_c2_two_stage_w020_w010.sh
```

## Terminal logs and hyperparameter snapshots

Every non-dry-run ablation wrapper saves stdout and stderr to a project-local log:

```text
logs/ablations/<run_tag>_tr<train_ratio>_va<val_ratio>_<timestamp>_pid<pid>.log
```

In default background mode the terminal displays the launch summary, PID and
log path, then returns; subsequent training output goes only to the log. With
`RUN_IN_BACKGROUND=0`, output is mirrored to both terminal and log.

The log begins with a `resolved_hyperparameters_begin` /
`resolved_hyperparameters_end` block. It records the resolved architecture
route, dataset ratios, optimizer and EMA settings, effective global batch size,
coarse point/loss strategy, prompt-diagnostic settings, initialization/resume
paths, validation-loss switch, data/checkpoint locations, git commit, and the
exact torchrun command.
This is the resolved execution snapshot, so it should be used for experiment
review rather than relying only on wrapper defaults.

Override the directory or the exact file when needed:

```bash
LOG_DIR=/path/to/logs bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
LOG_FILE=/path/to/exact.log bash scripts/ablations/b0_aggregator_mlp.sh
```

`DRY_RUN=1` ignores both logging overrides and creates no log file. Project-local
runtime logs are ignored by Git.
