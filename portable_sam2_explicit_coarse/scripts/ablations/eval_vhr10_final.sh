#!/usr/bin/env bash
# Final evaluation for the VHR-10 fast-400 run: raw and EMA weights of the
# best checkpoint on the RSPrompter 520/130 val split (the report split).
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
source "${ROOT}/scripts/load_environment.sh"

RUN_DIR="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/vhr10_fast400_tr1.0_va1.0"
CKPT="${RUN_DIR}/best_model.pth"
GPU="${1:-0}"

export CUDA_VISIBLE_DEVICES="${GPU}"

for W in model ema; do
  echo "===== VHR-10 final eval: weights=${W} ====="
  "${PYTHON}" "${ROOT}/inference/infer_from_checkpoint.py" \
    --checkpoint "${CKPT}" \
    --weights "${W}" \
    --split validation \
    --device cuda:0 \
    --output-dir "${RUN_DIR}/inference_validation_${W}" 2>&1 | tail -6
done
echo "===== done ====="
