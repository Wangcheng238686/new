#!/usr/bin/env bash
# UE dual stream + BEV injection into the SAM2 image embedding canvas.
# Identical to run_ue_dual_stream.sh except UE_DUAL_INJECT_EMBEDDING=1.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-/data1/wangcheng/dataset/ue_coco_with_uav}"
export UE_DUAL_INJECT_EMBEDDING=1
export RUN_TAG="${RUN_TAG:-ue_dual_c5v2_p2p3_emb_tr128_va259_s44}"
export WORK_DIR="${WORK_DIR:-$ROOT/work_dirs/ue/${RUN_TAG}}"

# Same 5fff72f WHU single-stream protocol as run_ue_single_stream.sh.
export EXPLICIT_PROMPT_MODE=points_box_dense
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM=raw_logits
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE=roi_local

exec /home/wangcheng/miniconda3/bin/conda run --no-capture-output -n cvt2 python train/train_rsprompter_fusion.py \
  --config configs/ue_dual_stream_p2p3_sparse.py \
  --run-tag "$RUN_TAG" \
  --data-root "$UE_DATA_ROOT" --use-ue-dual --image-size 1024 1024 \
  --batch-size 1 --grad-accum-steps 8 --epochs 100 \
  --lr 5e-4 --drone-lr-mult 1.0 --weight-decay 0.05 --warmup-iters 100 \
  --train-subset-ratio 0.1797752808988764 --val-subset-ratio 1.0 --seed 44 \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 \
  --amp 1 --ema-enabled 0 --ema-eval 0 --ema-save-best 0 \
  --checkpoint-dir "$WORK_DIR" --gpu-id 0 "$@"
