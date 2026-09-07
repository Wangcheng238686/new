#!/usr/bin/env bash
# D2 under the full_image spatial contract: frozen Mask control vs the
# conservative P2BR candidate.
# Protocol mirrors whu_d2_p2_dev_10p_2gpu.sh exactly (2 GPUs x batch 1 x
# accum 4, 15 epochs, WHU 10%/10%, dense alpha=0.10 / T=1.0 / detach=0,
# P2BR beta=0.10 x delta_logit_max=0.50, loss_weight=0.05); ONLY the
# final-mask coordinate/loss contract differs.  The whu_d2fi_* run tags
# keep results disjoint from the roi_local D-series.
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

export RUN_TAG="whu_d2fi_mask_alpha010_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"

# Same conservative P2BR envelope as the roi_local D2 (residual cap is now
# visible in telemetry via P2BR/saturation_threshold = 0.99 * 0.10 * 0.50).
export P2_BOUNDARY_REFINER_BETA=0.10
export P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX=0.50
export P2_BOUNDARY_REFINER_LOSS_WEIGHT=0.05
export RUN_TAG="whu_d2fi_full_alpha010_p2b010_d050_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_full.sh" "$@"
