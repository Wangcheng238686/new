#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

export SAM2_MODEL_SIZE=base_plus
export SAM2_REPO="${SAM2_REPO:-$(cd "${PROJECT_ROOT}/../sam2" && pwd)}"
export SAM2_CKPT="${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}"
export WHU1024_DATA_ROOT="${WHU1024_DATA_ROOT:-/data/wangcheng/dataset/WHU}"
export MAX_EPOCHS="${MAX_EPOCHS:-80}"
export EXPLICIT_PROMPT_MODE="${EXPLICIT_PROMPT_MODE:-points_box_dense}"
export DENSEBR_ENABLED="${DENSEBR_ENABLED:-0}"

PYTHON="${PYTHON:-/data/wangcheng/envs/cvt2/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/${EXPLICIT_PROMPT_MODE}_densebr${DENSEBR_ENABLED}}"

exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT:-29500}" \
  train/train_rsprompter_fusion.py \
  --config configs/whu1024_baseplus_explicit_coarse.py \
  --data-root "${WHU1024_DATA_ROOT}" \
  --use-whu-coco \
  --image-size 1024 1024 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${MAX_EPOCHS}" \
  --lr "${LEARNING_RATE}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --seed 44 \
  --amp 0 \
  --ema-enabled 1 \
  --ema-decay 0.999 \
  --ema-eval 1 \
  --ema-save-best 1 \
  --val-every-n-epochs 1 \
  --early-stopping-patience 10 \
  --early-stopping-start-epoch 20 \
  --mask-decoder-lr-mult 1.0 \
  --no-mask-lr-mult 0.0 \
  --prompt-encoder-lr-mult 0.0 \
  --shape-prior-lr-mult 1.0 \
  --densebr-lr-mult 1.0

