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
| R0 | 64×64 official | aggregator | legacy 5-token MLP | ROI-local | - | - | - | - |
| R1-C4 | 64×64 official | PAFPN | coarse 2P2N | full-image | no | yes | yes | no |
| C5-v2 | 64×64 official | PAFPN | coarse 2P2N | full-image | no | yes | yes | yes |

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

R0 and R1-C4 isolate the SAM2 spatial-resolution change on top of B0 and C4.
C5-v2 then changes only the P2 boundary refiner switch. They
set `SAM_IMAGE_EMBED_STRIDE=16`, select the 64×64 stride-16 FPN feature for the
MaskDecoder, use 256×256/128×128 high-resolution features, and configure a
64×64 PromptEncoder grid (dense-mask input: 256×256). The detector still
receives the same four-level FPN. Existing B0/B1, M0/M1 and C1–C4 remain stride-32/32×32 for
historical reproducibility.

```bash
bash scripts/ablations/r0_b0_aggregator_mlp_emb64.sh
bash scripts/ablations/r1_c4_pafpn_coarse_points_box_dense_emb64.sh
bash scripts/ablations/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh
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
