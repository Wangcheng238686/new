#!/usr/bin/env bash
# Evaluate a UE satellite-only checkpoint on the official UE split.
# CUDA_VISIBLE_DEVICES=2 maps physical GPU 2 to cuda:0 inside Python.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Keep these before load_environment.sh: its local config honors pre-exported
# values and otherwise contains paths from the older /data mount.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHON="${PYTHON:-/data2/wangcheng/envs/cvt2/bin/python}"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-/data1/wangcheng/dataset/ue_coco_with_uav}"
export CHECKPOINT="${CHECKPOINT:-/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth}"
export UE_SPLIT="${UE_SPLIT:-val}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export RUN_TAG="${RUN_TAG:-ue_single_c5v2_e86_${UE_SPLIT}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/${RUN_TAG}}"

# shellcheck source=../load_environment.sh
source "$ROOT/scripts/load_environment.sh"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python is not executable: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint is missing: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -d "$UE_DATA_ROOT/$UE_SPLIT/satellite" ]]; then
  echo "UE split is missing: $UE_DATA_ROOT/$UE_SPLIT/satellite" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
exec "$PYTHON" -u inference/infer_from_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-format ue --ue-split "$UE_SPLIT" \
  --data-root "$UE_DATA_ROOT" \
  --image-size 1024 1024 --batch-size 1 --num-workers "$NUM_WORKERS" \
  --device cuda:0 --sam2-ckpt "$SAM2_CKPT" \
  --output-dir "$OUTPUT_DIR" "$@"
