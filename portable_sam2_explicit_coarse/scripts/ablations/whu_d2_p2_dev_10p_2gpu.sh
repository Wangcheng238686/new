#!/usr/bin/env bash
# D2: after D1 freezes the dense setup, test whether existing P2BR is positive.
# Runs the frozen Mask control and a conservative P2BR candidate serially.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_GPU_LIST="${DEV_GPU_LIST:-1,2}"
IFS=',' read -r -a DEV_GPUS <<< "${DEV_GPU_LIST}"
if (( ${#DEV_GPUS[@]} != 2 )); then
  echo "DEV_GPU_LIST must contain exactly two GPU ids, got ${DEV_GPU_LIST}" >&2
  exit 2
fi

# Two GPUs × batch 1 × accumulation 4 = effective global batch 8.
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

# D1's candidate dense setup, to be frozen before D2 is actually launched.
export SHAPE_DENSE_DETACH=0
export SHAPE_DENSE_ALPHA_INIT=0.10
export SHAPE_DENSE_TEMPERATURE=1.0

export RUN_TAG="whu_d2_mask_alpha010_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"

# Existing P2BR only: reduce the old ±0.4-logit envelope to ±0.05 at init
# (beta × delta_logit_max) while retaining the final-mask gradient path.
export P2_BOUNDARY_REFINER_BETA=0.10
export P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX=0.50
export P2_BOUNDARY_REFINER_LOSS_WEIGHT=0.05
export RUN_TAG="whu_d2_full_alpha010_p2b010_d050_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_full.sh" "$@"
