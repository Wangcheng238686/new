#!/usr/bin/env bash
# Paper-reproduction entry for the single-stream PromptMiner-SAM2 formulation
# in main5.tex.  This is intentionally separate from R1-C4-RD: R1-C4-RD is
# the historical no-P2 baseline, while this wrapper enables P2-BRR and keeps
# the paper's detached raw-coarse dense-prompt contract.
set -Eeuo pipefail

# The resolved architecture ID remains C5-v2 because it is the registered
# PAFPN + stride-16 + points/box/dense + P2-BRR architecture.  RUN_TAG keeps
# the paper protocol and its raw-detached dense input distinguishable from
# historical C5-v2 runs through the checkpoint fingerprint and directory.
export ABLATION_ID="c5v2_pafpn_coarse_p2_boundary_refiner_emb64"
export RUN_TAG="${RUN_TAG:-paper_promptminer_rd_p2_whu_full}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE="roi_local"

# main5.tex single-stream protocol.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export MAX_EPOCHS="${MAX_EPOCHS:-100}"
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
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
