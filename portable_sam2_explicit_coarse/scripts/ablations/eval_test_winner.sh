#!/usr/bin/env bash
# Test-split report for the winning weights of the fast-150 run.
# Usage: bash scripts/ablations/eval_test_winner.sh [gpu_id]
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/scripts/load_environment.sh"

RUN_DIR="/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/paper_promptminer_rd_p2_whu_full_fast_tr1.0_va1.0"
CKPT="${RUN_DIR}/best_model_epoch150_ema_merged.pth"
OUT="${RUN_DIR}/inference_test_ema_merged"
GPU="${1:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
exec "${PYTHON}" "${ROOT}/inference/infer_from_checkpoint.py" \
  --checkpoint "${CKPT}" \
  --weights model \
  --split test \
  --device cuda:0 \
  --output-dir "${OUT}"
