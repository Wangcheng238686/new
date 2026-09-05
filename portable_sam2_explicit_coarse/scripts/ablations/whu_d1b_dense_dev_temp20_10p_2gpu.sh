#!/usr/bin/env bash
# D1 iteration 2 (2026-09-05): dense temperature recalibration.
#
# D1 round 1 verdict (WHU 10% dev, seed 44, 15ep): dense alpha=0.10/T=1.0 was
# neutral — pb 0.5814 vs mask 0.5824 (+0.0010, inside the ±0.006 duplicate-arm
# noise floor), while applied_delta_ratio was healthy (0.05-0.09) and the raw
# canvas was clamp-saturated (min=-8/max=8). Per the implementation guide this
# iteration turns the ONE allowed calibration knob: raw-logit temperature
# 1.0 -> 2.0 (canvas = clamp(logits/T, ±8): halved saturation, graded field).
# SHAPE_DENSE_ALPHA_INIT stays frozen at 0.10 so the D2 lineage (also 0.10)
# remains valid unless this iteration wins.
#
# The Point+Box control is temperature-invariant (no dense canvas) and is
# REUSED from round 1 (whu_d1_pb_control_10p_2gpu, best 0.5814) — same code
# state, seed and subset; only the mask arm reruns.
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
export SHAPE_DENSE_DETACH=0
export SHAPE_DENSE_ALPHA_INIT=0.10
export SHAPE_DENSE_TEMPERATURE=2.0

export RUN_TAG="whu_d1b_pb_mask_alpha010_temp20_10p_2gpu"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"
