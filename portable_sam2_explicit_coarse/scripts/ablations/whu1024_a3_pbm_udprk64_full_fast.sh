#!/usr/bin/env bash
# WHU-1024 full-data training of the A3 recipe (PBM/D5-B base + UDPR-v1 K64,
# P2-BRR disabled).  This is the WHU port of the NWPU matrix300 `a3` arm that
# the ICASSP paper describes as PromptMiner-SAM2 (CSPM P+B+M prompts + UDPR):
# the current WHU headline row came from the P2-BRR form, which the paper
# never describes, so this run replaces it with the architecture the paper
# actually claims.  Decisions locked with the user on 2026-09-15 (Q1-Q10):
#   - full A3 recipe port, from scratch, no NWPU checkpoint involvement;
#   - 150 epochs, val every 2 epochs, multi-alias selection
#     (segm-best / bbox-best / composite / last);
#   - A-series augmentation (hflip+vflip 0.5, multi-scale 896-1152 @ 0.5);
#   - 2 GPUs x 48 GB, EMA tracking-only + AMP, effective batch stays 8 and
#     optimizer steps per epoch stay 368 (identical LR schedule to fast150).
#
# Model contract (architecture_id = r1_c4_pafpn_coarse_points_box_dense_
# emb64_udprk64, same ID family as the NWPU a3 arms):
#   points_box_dense @ stride16 pafpn, dense source NOT detached (a-series),
#   prompt-encoder mask-downscaling unfrozen (D5-B, lr_mult 0.1),
#   final-mask loss roi_balanced_dice in full-image coordinates,
#   segm scored by detector confidence, UDPR residual_v1 K64 hidden128
#   (zero extra mode keys -> historic v1 checkpoint contract).
#
# BATCH_SIZE=2 x GRAD_ACCUM_STEPS=2 on 2 ranks: the fp32 bs2 profile peaked
# at 28.3 GB on 40 GB GPUs (fast150 note); with AMP + FINAL_MASK_TARGET_
# SIZE=256 it fits the 48 GB budget with margin while halving micro-steps.
# Fall back to BATCH_SIZE=1 GRAD_ACCUM_STEPS=4 (same effective batch and
# step count) if the smoke run OOMs.
#
# Usage:
#   bash scripts/ablations/whu1024_a3_pbm_udprk64_full_fast.sh          # train
#   DRY_RUN=1 bash scripts/ablations/whu1024_a3_pbm_udprk64_full_fast.sh
#   RESUME_OWN=/path/to/last_checkpoint.pth bash ...                    # crash recovery
set -Eeuo pipefail

# --- identity / architecture contract ---
export ABLATION_ID="r1_c4_pafpn_coarse_points_box_dense_emb64_udprk64"
export RUN_TAG="${RUN_TAG:-whu1024_a3_pbm_udprk64_full_fast}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export SAM_IMAGE_EMBED_STRIDE="16"

# Same-run crash recovery: explicit opt-in only, never inherited silently.
if [ -n "${RESUME_OWN:-}" ]; then
  export RESUME_FROM="${RESUME_OWN}"
fi

# --- A3 model surface (mirrors vhr10_p2v2_dev.sh a3 arm + a-series locks) ---
export P2_BOUNDARY_REFINER_ENABLED="0"
export DECODER_TAIL_REFINER_ENABLED="1"
export DECODER_TAIL_NUM_POINTS="${A3_NUM_POINTS:-64}"
export DECODER_TAIL_HIDDEN_DIM="128"
export DECODER_TAIL_POINT_LOSS_WEIGHT="1.0"
export DECODER_TAIL_DELTA_LOGIT_MAX="2.0"
# DECODER_TAIL_MODE stays unset: residual_v1 without materialized mode keys.

export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH="0"
export SHAPE_DENSE_ALPHA_INIT="0.5"
export SHAPE_DENSE_TEMPERATURE="1.0"
export SHAPE_DENSE_OUTSIDE_FILL="0.0"
export SHAPE_CONTEXT_FUSION="roi_only"
export SHAPE_POINT_ADAPTIVE_VALIDITY="1"
export SHAPE_LOSS_SCHEDULE_MODE="fixed"
export SHAPE_LOSS_STAGE1_END="5"
export SHAPE_LOSS_WEIGHT_STAGE1="0.20"
export SHAPE_LOSS_WEIGHT_STAGE2="0.10"
export SHAPE_PRIOR_LOSS_WEIGHT="0.10"
export COARSE_MASK_OUTPUT_SIZE="64"

export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING="1"
export PROMPT_ENCODER_LR_MULT="0.1"

export SEGM_SCORE_MODE="detector"
export FINAL_MASK_COORDINATE_MODE="full_image"
export FINAL_MASK_LOSS_MODE="roi_balanced_dice"
export FINAL_MASK_ROI_EXPAND_RATIO="1.20"
export FINAL_MASK_ROI_BCE_WEIGHT="1.0"
export FINAL_MASK_ROI_DICE_WEIGHT="1.0"
export FINAL_MASK_OUTSIDE_BCE_WEIGHT="0.05"

export ROI_SAM_ENABLED="0"
export POINT_WARMUP_ENABLED="0"
export CANVAS_RENDERER_ENABLED="0"
export DENSE_CAPACITY_HEAD_ENABLED="0"

# --- launch form: 2x48GB, effective batch 8, 368 optimizer steps/epoch ---
# AMP=0 (fp32): the first launch (2026-09-15 23:20) diverged to all-NaN at
# ~optimizer step 200-220 -- right after the 100-step warmup reached full
# lr 5e-4 -- under AMP=1 fp16 (first non-finite loss 00:35:11 epoch=1; 3498
# skipped steps by E6; val DT=0; dir archived *_nan_amp1_bak_20260916).
# fp16+lr5e-4 is proven on NWPU A-series (sparse) and on the P2BRR/roi_local
# WHU form, but not on A3 x dense WHU; fp32 is the conservative superset.
# bs1 x accum4 mirrors the 2026-08-24 paper run's per-GPU micro form and is
# memory-safe in fp32; optimizer steps/epoch stay 368 (schedule unchanged).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export AMP="${AMP:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
# NOTE: FINAL_MASK_TARGET_SIZE is deliberately NOT exported here.  Its last
# consumer was removed in c82cdd1 (2026-08-25); the env is read by nothing,
# and fast150's comment claiming it "frees memory" is stale.  The fi target
# grid is defined by mask_preds.shape[-2:] in sam2_mask_head_targets.py.

# --- schedule / data (fast150 parity: seed 44, LR 5e-4 for effective 8) ---
export MAX_EPOCHS="${MAX_EPOCHS:-150}"
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
export SUBSET_SEED="${SUBSET_SEED:-44}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_ITERS="${WARMUP_ITERS:-100}"

# --- validation cadence + multi-alias selection ---
export VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-2}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
# patience counts VALIDATIONS: 10 x val-every-2 = the 20-epoch patience the
# fast150 wrapper tuned for the 150-epoch cosine (annealing tail from ~E84).
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
export EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-90}"
export EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-5e-4}"
export EARLY_STOPPING_SMOOTH_WINDOW="${EARLY_STOPPING_SMOOTH_WINDOW:-5}"
export SAVE_BBOX_BEST_METRIC="${SAVE_BBOX_BEST_METRIC-bbox/mAP}"
export SAVE_COMPOSITE_BEST="${SAVE_COMPOSITE_BEST:-1}"
export SAVE_COMPOSITE_WEIGHTS="${SAVE_COMPOSITE_WEIGHTS:-0.5*bbox/mAP+0.5*segm/mAP}"
export SAVE_LAST_MODEL="${SAVE_LAST_MODEL:-1}"
export TEST_MAX_PER_IMG="${TEST_MAX_PER_IMG:-100}"

# --- EMA: tracking-only shadow; post-training raw-vs-EMA val decides test ---
export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_EVAL="${EMA_EVAL:-0}"
export EMA_SAVE_BEST="${EMA_SAVE_BEST:-0}"

# --- A-series augmentation (hflip 0.5 + vflip 0.5 + multi-scale @0.5) ---
export TRAIN_FLIP_PROB="${TRAIN_FLIP_PROB:-0.5}"
export TRAIN_VFLIP_PROB="${TRAIN_VFLIP_PROB:-0.5}"
export TRAIN_MULTI_SCALE_RESIZE_PROB="${TRAIN_MULTI_SCALE_RESIZE_PROB:-0.5}"
export TRAIN_MULTI_SCALE_MODE="${TRAIN_MULTI_SCALE_MODE:-value}"
export TRAIN_MULTI_SCALE_IMG_SCALE="${TRAIN_MULTI_SCALE_IMG_SCALE:-896:896,960:960,1024:1024,1088:1088,1152:1152}"

exec bash "$(dirname "$0")/../_run_ablation.sh" "$@"
