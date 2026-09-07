#!/usr/bin/env bash
# D3: P2 combination-recipe escalation under the full_image spatial contract
# (design doc: docs/d3_d4_p2_escalation_canvas_design.md v3 §2).
#
# Escalation knobs (the ONLY intended differences vs whu_d2fi_full_*):
#   envelope  ±0.05 -> ±0.30  (beta 0.10->0.30, delta_logit_max 0.50->1.00)
#   boundary loss weight 0.05 -> 0.20
# This is a COMBINATION recipe (~24x near-init auxiliary gradient on the P2
# head): it cannot attribute gains to amplitude vs supervision.  Mechanism
# criterion: refinement gain boundary_f1 >= +0.01 AND dice >= 0 (sustained);
# fallback line: best < 0.605 -> revert conservative envelope.
# Controls already complete: D2fi Mask 0.6133 / conservative Full 0.6069.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_GPU_LIST="${DEV_GPU_LIST:-2,3}"
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

source "${SCRIPT_DIR}/whu_fullimage_overlay.sh"

export P2_BOUNDARY_REFINER_BETA=0.30
export P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX=1.00
export P2_BOUNDARY_REFINER_LOSS_WEIGHT=0.20
export RUN_TAG="whu_d3fi_full_p2esc030_d100_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_full.sh" "$@"
