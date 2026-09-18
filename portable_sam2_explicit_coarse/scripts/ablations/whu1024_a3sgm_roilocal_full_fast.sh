#!/usr/bin/env bash
# WHU-1024 full-data run: the NWPU-paper a3sgm decoder-tail (coupled margin
# tail + decoupled aux clip) mounted on the OLD fast150 supervision base
# (roi_local + standard final-mask loss, PE frozen, dense detached) -- i.e.
# P2BRR's slot in the fast150 recipe is taken by the margin tail, everything
# else stays fast150.  Purpose: test whether the paper's tail module can
# reproduce the fast150 trajectory/effect when the supervision foundation is
# the one that works on WHU.
#
# WHU-specific anchor: log(0.4/0.6)=-0.4055 -- this repo's WHU
# test_cfg deploys mask_thr_binary=0.4 (same as NWPU; the runtime anchor
# belongs to its 0.4 threshold); the roi_local predict path now cross-checks
# this at first inference.
#
# Model contract: r1_c4_rd_..._emb64_udprk64 family (points_box_dense @
# stride16 pafpn, dense detached, P2BRR off, tail margin K64).  The tail's
# point loss consumes per-RoI GT crops on the decoder's native grid while the
# main final-mask loss keeps the historical 28x28 roi_local targets.
#
# Launch form mirrors the proven WHU fp32 form: 2 GPUs x bs1 x accum4 =
# effective 8, 368 optimizer steps/epoch, AMP off (this host's fp16 is slow
# and diverged on the A3 form), EMA tracking-only, val every epoch (fast150's
# cadence, for trajectory comparability).
set -Eeuo pipefail

# --- identity / architecture ---
export ABLATION_ID="r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64_udprk64"
export RUN_TAG="${RUN_TAG:-whu1024_a3sgm_roilocal_full_fast}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export SAM_IMAGE_EMBED_STRIDE="16"

if [ -n "${RESUME_OWN:-}" ]; then
  export RESUME_FROM="${RESUME_OWN}"
fi

# --- fast150 supervision base (P2BRR slot emptied) ---
export P2_BOUNDARY_REFINER_ENABLED="0"
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH="1"
export FINAL_MASK_COORDINATE_MODE="roi_local"
# FINAL_MASK_LOSS_MODE stays default "standard" (fast150 semantics).
export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING="0"
export PROMPT_ENCODER_LR_MULT="0.0"
# SHAPE_DENSE_ALPHA_INIT stays default 0.25 (fast150).

# --- a3sgm decoder tail (margin, coupled) ---
export DECODER_TAIL_REFINER_ENABLED="1"
export DECODER_TAIL_NUM_POINTS="64"
export DECODER_TAIL_HIDDEN_DIM="128"
export DECODER_TAIL_POINT_LOSS_WEIGHT="1.0"
export DECODER_TAIL_DELTA_LOGIT_MAX="2.0"
export DECODER_TAIL_MODE="residual_v1_margin"
export DECODER_TAIL_CROSSING_MARGIN="0.10"
export DECODER_TAIL_BOUNDARY_LOGIT="-0.4054651081081644"
export DECOUPLED_AUX_GRAD_CLIP="1"

# --- inert-module locks ---
export ROI_SAM_ENABLED="0"
export POINT_WARMUP_ENABLED="0"
export CANVAS_RENDERER_ENABLED="0"
export DENSE_CAPACITY_HEAD_ENABLED="0"
export SHAPE_PRIOR_LOSS_WEIGHT="0.10"

# --- launch form: fp32, effective batch 8, 368 steps/epoch ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export AMP="${AMP:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"

# --- schedule / data (fast150 parity) ---
export MAX_EPOCHS="${MAX_EPOCHS:-150}"
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
export SUBSET_SEED="${SUBSET_SEED:-44}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_ITERS="${WARMUP_ITERS:-100}"

# --- validation every epoch (fast150 cadence) + multi-alias selection ---
export VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
export EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-90}"
export EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-5e-4}"
export EARLY_STOPPING_SMOOTH_WINDOW="${EARLY_STOPPING_SMOOTH_WINDOW:-5}"
export SAVE_BBOX_BEST_METRIC="${SAVE_BBOX_BEST_METRIC-bbox/mAP}"
export SAVE_COMPOSITE_BEST="${SAVE_COMPOSITE_BEST:-1}"
export SAVE_COMPOSITE_WEIGHTS="${SAVE_COMPOSITE_WEIGHTS:-0.5*bbox/mAP+0.5*segm/mAP}"
export SAVE_LAST_MODEL="${SAVE_LAST_MODEL:-1}"
export TEST_MAX_PER_IMG="${TEST_MAX_PER_IMG:-100}"

# --- EMA tracking-only ---
export EMA_ENABLED="${EMA_ENABLED:-1}"
export EMA_EVAL="${EMA_EVAL:-0}"
export EMA_SAVE_BEST="${EMA_SAVE_BEST:-0}"

# --- fast150 augmentation ---
export TRAIN_VFLIP_PROB="${TRAIN_VFLIP_PROB:-0.5}"
export TRAIN_MULTI_SCALE_RESIZE_PROB="${TRAIN_MULTI_SCALE_RESIZE_PROB:-0.5}"
export TRAIN_MULTI_SCALE_MODE="${TRAIN_MULTI_SCALE_MODE:-value}"
export TRAIN_MULTI_SCALE_IMG_SCALE="${TRAIN_MULTI_SCALE_IMG_SCALE:-896:896,960:960,1024:1024,1088:1088,1152:1152}"

exec bash "$(dirname "$0")/../_run_ablation.sh" "$@"
