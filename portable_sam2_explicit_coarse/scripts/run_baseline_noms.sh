#!/usr/bin/env bash
# =============================================================================
# Reproduce the WHU1024 BEST-bbox run: whu1024_baseline_noms
#   - bbox/mAP = 0.788 (peak @ epoch 78/80), segm/mAP = 0.748
#   - IIMR OFF, multi-scale aug OFF (noms = no multi-scale), fused FPN
#   - 4-GPU DDP, bs=2/gpu, accum=1 -> effective bs=8, lr=5e-4
#
# This is a thin wrapper that exports the exact hyper-parameters recorded in
# logs/.../whu1024_baseline_noms_4gpu_baseplus_ddp_20260614_161842.params and
# delegates to run_hparam_whu1024_baseplus_4gpu.sh (do not edit that script).
# Override any variable below or any variable understood by the base script.
# =============================================================================
set -euo pipefail

# --- experiment identity ---
export RUN_TAG="${RUN_TAG:-baseline_noms_4gpu}"

# --- the two switches that define this run (vs the script defaults) ---
export MULTI_SCALE_RESIZE_PROB="${MULTI_SCALE_RESIZE_PROB:-0}"   # noms = NO multi-scale
export IIMR_ENABLED="${IIMR_ENABLED:-0}"                          # baseline: IIMR disabled

# --- batch / LR (defaults already match, restated for clarity) ---
export BEST_BS="${BEST_BS:-2}"
export BEST_ACCUM="${BEST_ACCUM:-1}"
export BEST_LR="${BEST_LR:-5e-4}"
export BEST_MULT="${BEST_MULT:-1.0}"
export MAX_EPOCHS="${MAX_EPOCHS:-80}"

# --- GPU layout ---
export GPU_LIST="${GPU_LIST:-0,1,2,3}"
export NUM_GPUS="${NUM_GPUS:-4}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_hparam_whu1024_baseplus_4gpu.sh"
