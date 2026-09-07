#!/usr/bin/env bash
# D6 candidate arm: Point+Box+Mask with SMALL gate init + PE adaptation.
# The untested combination after D5 (per D6 review):
#   alpha init 0.10 (learnable global_sigmoid gate) — dense enters weakly,
#   avoiding the early-training amplification seen at init 0.5 (D5-A/B);
#   mask_downscaling unfrozen (4,684 params) with PROMPT_ENCODER_LR_MULT=0.1.
# Differences vs D5-B (whu_fi_d5b_*): SHAPE_DENSE_ALPHA_INIT 0.50 -> 0.10.
# Differences vs historical Mask (alpha 0.10, PE frozen): the two PE knobs.
# No P2, no loss change, no new modules.  Acceptance per the three-gate
# protocol in the review; nothing here guarantees a Mask gain.
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
export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=1
export PROMPT_ENCODER_LR_MULT=0.1

# Hard-locked protocol values (same policy as D5).
export SHAPE_DENSE_TRANSFORM=raw_logits
export WARMUP_ITERS=100
unset MAX_TRAIN_BATCHES MAX_VAL_BATCHES INIT_FROM RESUME_FROM CONFIG_OVERRIDE

source "${SCRIPT_DIR}/whu_fullimage_overlay.sh"

export RUN_TAG="whu_d6fi_mask_gate010_pe010_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"
