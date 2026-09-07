#!/usr/bin/env bash
# D1 under the full_image spatial contract: fixed Point+Box control vs
# Point+Box+Mask dense candidate.
# Protocol mirrors whu_d1_dense_dev_10p_2gpu.sh exactly (2 GPUs x batch 1 x
# accum 4, 15 epochs, WHU 10%/10%, frozen dense knobs alpha=0.10 / T=1.0 /
# detach=0); ONLY the final-mask coordinate/loss contract differs.  The
# whu_d1fi_* run tags keep results disjoint from the roi_local D-series.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_GPU_LIST="${DEV_GPU_LIST:-1,2}"
IFS=',' read -r -a DEV_GPUS <<< "${DEV_GPU_LIST}"
if (( ${#DEV_GPUS[@]} != 2 )); then
  echo "DEV_GPU_LIST must contain exactly two GPU ids, got ${DEV_GPU_LIST}" >&2
  exit 2
fi

# Two GPUs x batch 1 x accumulation 4 = effective global batch 8, identical
# to the roi_local D-series dev protocol.
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

export RUN_TAG="whu_d1fi_pb_control_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box.sh" "$@"

export RUN_TAG="whu_d1fi_pb_mask_alpha010_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"
