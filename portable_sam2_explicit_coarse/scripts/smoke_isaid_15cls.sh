#!/usr/bin/env bash
# Smoke test for the iSAID 15-class route: tiny subsets, a few train batches,
# then one full validation pass with 15-class COCO evaluation.
# Runs single-GPU; pass the GPU id as $1 (default 2, the idle one).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

SMOKE_GPU="${1:-2}"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU}"

LOG_DIR="${PORTABLE_SAM2_LOG_ROOT}/smoke"
mkdir -p "${LOG_DIR}" "${PORTABLE_SAM2_CHECKPOINT_ROOT}/smoke_isaid_15cls"

LOG_FILE="${LOG_DIR}/isaid_15cls_smoke_$(date +%Y%m%d_%H%M%S).log"

"${PYTHON}" -u train/train_rsprompter_fusion.py \
  --config configs/isaid_baseplus_explicit_coarse.py \
  --data-root "${ISAID_DATA_ROOT:-/data1/wangcheng/dataset/iSAID}" \
  --use-isaid-coco \
  --image-size 1024 1024 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --epochs 1 \
  --max-train-batches 30 \
  --lr 5e-4 \
  --warmup-iters 10 \
  --weight-decay 0.05 \
  --train-subset-ratio 0.005 \
  --val-subset-ratio 0.01 \
  --val-batch-size 1 \
  --val-every-n-epochs 1 \
  --compute-val-loss 0 \
  --early-stopping-patience 999 \
  --early-stopping-start-epoch 999 \
  --seed 44 \
  --amp 0 \
  --ema-enabled 0 \
  --checkpoint-dir "${PORTABLE_SAM2_CHECKPOINT_ROOT}/smoke_isaid_15cls" \
  --prompt-debug-stats 0 \
  2>&1 | tee "${LOG_FILE}"

echo "smoke log: ${LOG_FILE}"
