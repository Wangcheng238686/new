#!/usr/bin/env bash
# Accelerated 150-epoch variant of paper_promptminer_rd_p2_whu_full.sh.
#
# Same architecture and data protocol as the paper run; every knob below is a
# wall-clock optimization agreed after the AMP subset A/B check (18 epochs,
# mean |dSegM/mAP| = 0.021, within run-to-run noise):
#   - AMP=1 (fp16 autocast; ampcheck log 20260824_114159)
#   - FINAL_MASK_TARGET_SIZE=256 (per-ROI supervision at the decoder's native
#     grid; frees the memory that unlocks BATCH_SIZE=2)
#   - BATCH_SIZE=2 x GRAD_ACCUM_STEPS=2 (effective global batch stays 8)
#   - VAL_BATCH_SIZE=4 (eval-mode batching does not change metrics)
#   - CUDNN_BENCHMARK=1 (autotuned convs; drops bit-level reproducibility)
# Best-model selection and every downstream report run at the COCO default
# maxDet=100 (TEST_MAX_PER_IMG=100): all comparison baselines and ablations
# in this project are scored at maxDets=100, so the training-side contract
# is pinned to the same value to keep one unified metric contract.
# Early stopping is widened for the 150-epoch cosine (annealing tail starts
# around epoch 84; patience=10 would risk firing during the mid-plateau).
#
# Public runners (_run_ablation.sh and friends) keep all-acceleration-off
# defaults; only this wrapper opts in.
set -Eeuo pipefail

export ABLATION_ID="c5v2_pafpn_coarse_p2_boundary_refiner_emb64"
export RUN_TAG="${RUN_TAG:-paper_promptminer_rd_p2_whu_full_fast}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE="roi_local"

# --- accelerated protocol defaults (outer shell may still override) ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export AMP="${AMP:-1}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
export MAX_EPOCHS="${MAX_EPOCHS:-150}"
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
# VAL_BATCH_SIZE stays conservative: the validation memory spike is dominated
# by per-image decoder batching (measured 40.4 GB peak at val batch=1 next to
# fp32 training's 28.3 GB), and TEST_MAX_PER_IMG=100 caps the model side too;
# batch=2 keeps most of the throughput gain without betting on margins.
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
export FINAL_MASK_TARGET_SIZE="${FINAL_MASK_TARGET_SIZE:-256}"
export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-1}"
export TEST_MAX_PER_IMG="${TEST_MAX_PER_IMG:-100}"
export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
export EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-90}"
export EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-5e-4}"
export EARLY_STOPPING_SMOOTH_WINDOW="${EARLY_STOPPING_SMOOTH_WINDOW:-5}"

# --- identical to the paper run ---
export SUBSET_SEED="${SUBSET_SEED:-44}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export WARMUP_ITERS="${WARMUP_ITERS:-100}"
export EMA_ENABLED="${EMA_ENABLED:-0}"
export EMA_EVAL="${EMA_EVAL:-0}"
export EMA_SAVE_BEST="${EMA_SAVE_BEST:-0}"
export PROMPT_ENCODER_LR_MULT="${PROMPT_ENCODER_LR_MULT:-0.0}"
export SHAPE_PRIOR_LOSS_WEIGHT="${SHAPE_PRIOR_LOSS_WEIGHT:-0.10}"
export P2_BOUNDARY_REFINER_PROJECTED_CHANNELS="${P2_BOUNDARY_REFINER_PROJECTED_CHANNELS:-64}"
export P2_BOUNDARY_REFINER_MID_CHANNELS="${P2_BOUNDARY_REFINER_MID_CHANNELS:-64}"
export P2_BOUNDARY_REFINER_LOSS_WEIGHT="${P2_BOUNDARY_REFINER_LOSS_WEIGHT:-0.05}"
export POINT_WARMUP_ENABLED="${POINT_WARMUP_ENABLED:-0}"

exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
