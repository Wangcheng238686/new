#!/usr/bin/env bash
# D5-A: dense gate-init control under the full_image spatial contract
# (per docs/d5_review_handoff.md §3).  Single intended differences vs the
# historical Mask arm (whu_d1fi_pb_mask / whu_d2fi_mask):
#   SHAPE_DENSE_ALPHA_INIT 0.10 -> 0.50  (learnable global_sigmoid gate
#   INITIAL value — NOT a fixed gate; densefix config is deliberately not
#   used because it forces shape_scale_mode="fixed").
# P2 stays disabled; PE stays fully frozen (train_mask_downscaling=0,
# PROMPT_ENCODER_LR_MULT=0).  D5-B adds the PE adaptation on top of this.
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
export SHAPE_DENSE_ALPHA_INIT=0.50
export SHAPE_DENSE_TEMPERATURE=1.0
export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=0
export PROMPT_ENCODER_LR_MULT=0.0

# Hard-locked protocol values (review 2026-09-06): common.sh now defers
# SHAPE_DENSE_TRANSFORM to a preset, so leftover D4-csig shell state would
# otherwise leak in SILENTLY (contract check stays OK either way).  The same
# holds for smoke leftovers (WARMUP_ITERS/MAX_*_BATCHES/INIT_FROM/...), so a
# formal D5 launch must not inherit any of them from the environment.
export SHAPE_DENSE_TRANSFORM=raw_logits
export WARMUP_ITERS=100
unset MAX_TRAIN_BATCHES MAX_VAL_BATCHES INIT_FROM RESUME_FROM CONFIG_OVERRIDE

source "${SCRIPT_DIR}/whu_fullimage_overlay.sh"

export RUN_TAG="whu_d5a_mask_gate050_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"
