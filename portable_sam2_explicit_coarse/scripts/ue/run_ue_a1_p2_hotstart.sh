#!/usr/bin/env bash
# Governed A1 hot-start screen.  Run one independent seed per invocation.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-/data1/wangcheng/dataset/ue_coco_with_uav}"
export A1_SEED="${A1_SEED:-44}"
export RUN_TAG="${RUN_TAG:-a1_p2_z0_hotstart_s${A1_SEED}_tr128_va259}"
export WORK_DIR="${WORK_DIR:-$ROOT/work_dirs/ue/${RUN_TAG}}"
export INIT_FROM="${INIT_FROM:-/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_work_dirs/ue/ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth}"

export EXPLICIT_PROMPT_MODE=points_box_dense
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM=raw_logits
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE=roi_local

exec /data2/wangcheng/envs/cvt2/bin/python train/train_rsprompter_fusion.py \
  --config configs/ue_a1_p2_z0_hotstart.py --run-tag "$RUN_TAG" \
  --data-root "$UE_DATA_ROOT" --use-ue-dual --image-size 1024 1024 \
  --batch-size 1 --grad-accum-steps 8 --epochs 30 --lr 5e-4 --drone-lr-mult 1.0 \
  --weight-decay 0.05 --warmup-iters 100 --train-subset-ratio 0.1797752808988764 \
  --val-subset-ratio 1.0 --seed "$A1_SEED" --val-batch-size 1 --val-every-n-epochs 1 \
  --compute-val-loss 0 --amp 1 --ema-enabled 0 --ema-eval 0 --ema-save-best 0 \
  --checkpoint-dir "$WORK_DIR" --gpu-id 0 --train-a1-only \
  --init-from "$INIT_FROM" --allow-cross-arch-init "$@"
