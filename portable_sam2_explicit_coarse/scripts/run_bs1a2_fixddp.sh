#!/usr/bin/env bash
# =============================================================================
# Reproduce the WHU1024 BEST-segm run: whu1024_bs1a2_fixddp
#   - segm/mAP = 0.7543 (peak @ epoch 17), bbox/mAP = 0.767
#   - IIMR OFF, multi-scale aug OFF
#   - 4-GPU DDP, bs=1/gpu, accum=2 -> effective bs=8, lr=5e-4
#
# Differs from run_baseline_noms.sh ONLY in per-GPU batch size and grad-accum
# steps (bs=1/accum=2 vs bs=2/accum=1; same effective batch of 8). The original
# run early-stopped at epoch 27; the segm peak was hit at epoch 17.
#
# This is a thin wrapper that delegates to run_hparam_whu1024_baseplus_4gpu.sh.
# Override any variable below or any variable understood by the base script.
# =============================================================================
set -euo pipefail

# --- experiment identity ---
export RUN_TAG="${RUN_TAG:-bs1a2_fixddp}"

# --- the switches that define this run ---
export BEST_BS="${BEST_BS:-1}"          # bs=1 per GPU
export BEST_ACCUM="${BEST_ACCUM:-2}"    # grad-accum=2 -> effective bs = 1*2*4 = 8
export MULTI_SCALE_RESIZE_PROB="${MULTI_SCALE_RESIZE_PROB:-0}"
export IIMR_ENABLED="${IIMR_ENABLED:-0}"

# --- LR / schedule ---
export BEST_LR="${BEST_LR:-5e-4}"
export BEST_MULT="${BEST_MULT:-1.0}"
export MAX_EPOCHS="${MAX_EPOCHS:-80}"

# --- GPU layout ---
export GPU_LIST="${GPU_LIST:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_hparam_whu1024_baseplus_4gpu.sh" "$@"
