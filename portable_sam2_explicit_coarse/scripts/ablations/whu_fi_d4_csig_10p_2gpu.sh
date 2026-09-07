#!/usr/bin/env bash
# D4: dense-canvas representation change under the full_image spatial
# contract (design doc: docs/d3_d4_p2_escalation_canvas_design.md v3 §3).
#
# Single configuration factor: SHAPE_DENSE_TRANSFORM raw_logits ->
# confidence_signed (bounded, confidence-weighted canvas; zero-crossing at
# the 0.5 contour).  detach stays 0 — no gaussian-style forced detach, no
# gradient-path confound.  Tests whether a bounded, confidence-weighted
# canvas is better consumed by the frozen PromptEncoder; it does NOT create
# new information.  Judgement: best mAP vs control 0.6144 / D2fi Mask
# 0.6133; ties count as neutral, not promotion.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_GPU_LIST="${DEV_GPU_LIST:-0,1}"
IFS=',' read -r -a DEV_GPUS <<< "${DEV_GPU_LIST}"
if (( ${#DEV_GPUS[@]} != 2 )); then
  echo "DEV_GPU_LIST must contain exactly two GPU ids, got ${DEV_GPU_LIST}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${DEV_GPU_LIST}"
export NPROC_PER_NODE=2
export BATCH_SIZE=1
export GRAD_ACCUM_STEPS=4
export AMP=1
export MAX_EPOCHS=15
export TRAIN_SUBSET_RATIO=0.1
export VAL_SUBSET_RATIO=0.1
export SUBSET_SEED=44
export VAL_EVERY_N_EPOCHS=3
export VAL_BATCH_SIZE=2
export EMA_ENABLED=0
export EMA_EVAL=0
export EMA_SAVE_BEST=0
export RUN_IN_BACKGROUND=0
export SHAPE_DENSE_DETACH=0
export SHAPE_DENSE_ALPHA_INIT=0.10
export SHAPE_DENSE_TEMPERATURE=1.0

# The single intended difference vs whu_d1fi_pb_mask_*: must be set BEFORE
# the row wrapper sources whu_p2_matrix_common.sh (which now defers to a
# preset value instead of forcing raw_logits).
export SHAPE_DENSE_TRANSFORM=confidence_signed

source "${SCRIPT_DIR}/whu_fullimage_overlay.sh"

export RUN_TAG="whu_d4fi_mask_csig_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"
