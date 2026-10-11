#!/usr/bin/env bash
# SURFBEV-B0 only.  This script deliberately has no UAV option or path.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-$ROOT/data/ue_raw_hero_uav_pairset_20261011}"
export B0_SEED="${B0_SEED:-44}"
export RUN_TAG="${RUN_TAG:-surfacebev_b0_raw512_s${B0_SEED}}"
export WORK_DIR="${WORK_DIR:-$ROOT/work_dirs/surfacebev/${RUN_TAG}}"
export EXPLICIT_PROMPT_MODE=points_box_dense
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM=raw_logits
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE=roi_local
exec /home/wangcheng/miniconda3/bin/conda run --no-capture-output -n cvt2 python train/train_rsprompter_fusion.py \
  --config configs/ue_raw_hero_b0.py --run-tag "$RUN_TAG" --data-root "$UE_DATA_ROOT" \
  --use-ue-raw-surface --image-size 512 512 --batch-size 1 --grad-accum-steps 8 \
  --epochs 100 --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed "$B0_SEED" \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --ema-enabled 0 --ema-eval 0 --ema-save-best 0 --checkpoint-dir "$WORK_DIR" --gpu-id 0 "$@"
