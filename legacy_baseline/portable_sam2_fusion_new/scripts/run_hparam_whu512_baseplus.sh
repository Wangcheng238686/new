#!/usr/bin/env bash
set -euo pipefail

cd /home/wangcheng2021/project/portable_sam2_fusion_new/portable_sam2_fusion_new

PYTHON=/data/wangcheng/envs/cvt2/bin/python
TRAIN=train/train_rsprompter_fusion.py
CONFIG=configs/rsprompter_anchor_whu512_base_plus_singlecls.py
DATA=/data/wangcheng/dataset/WHU-512
CKPT_BASE=/data/wangcheng/checkpoint/ablation_hyperparam_whu512_baseplus
LOG_DIR=logs/ablation_hyperparam_whu512_baseplus

SAM2_CKPT=${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}
SAM2_BASEPLUS_URL=${SAM2_BASEPLUS_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt}
SAM2_BASEPLUS_BYTES=${SAM2_BASEPLUS_BYTES:-323493298}

mkdir -p "$LOG_DIR" "$(dirname "$SAM2_CKPT")"

need_download=0
if [ ! -f "$SAM2_CKPT" ]; then
  need_download=1
else
  actual_bytes=$(stat -c%s "$SAM2_CKPT" || echo 0)
  if [ "$actual_bytes" -ne "$SAM2_BASEPLUS_BYTES" ]; then
    echo "[WARN] checkpoint size mismatch: ${actual_bytes} != ${SAM2_BASEPLUS_BYTES}, re-downloading"
    rm -f "$SAM2_CKPT"
    need_download=1
  fi
fi

if [ "$need_download" -eq 1 ]; then
  echo "[INFO] Downloading SAM2 base_plus checkpoint..."
  wget -c "$SAM2_BASEPLUS_URL" -O "$SAM2_CKPT"
fi

export SAM2_CKPT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
TS=$(date +%Y%m%d_%H%M%S)

BEST_BS=${BEST_BS:-2}
BEST_ACCUM=${BEST_ACCUM:-2}
BEST_LR=${BEST_LR:-5e-4}
BEST_MULT=${BEST_MULT:-1.0}
IIMR_NUM_UNCERTAINTY_POINTS=${IIMR_NUM_UNCERTAINTY_POINTS:-4}
IIMR_INTERMEDIATE_LOSS_WEIGHT=${IIMR_INTERMEDIATE_LOSS_WEIGHT:-0.0}
IIMR_USE_LEARNABLE_RESIDUAL=${IIMR_USE_LEARNABLE_RESIDUAL:-0}
IIMR_MEMORY_RESIDUAL_WEIGHT=${IIMR_MEMORY_RESIDUAL_WEIGHT:-0.02}
IIMR_RESIDUAL_WARMUP_EPOCHS=${IIMR_RESIDUAL_WARMUP_EPOCHS:-8}
IIMR_DYNAMIC_FUSION_MODE=${IIMR_DYNAMIC_FUSION_MODE:-gated}
RUN_TAG=${RUN_TAG:-iter2_dp${IIMR_NUM_UNCERTAINTY_POINTS}}

COMMON_ARGS=(
  --config "$CONFIG"
  --data-root "$DATA"
  --dataset-format whu_coco
  --whu-train-ann-file "annotations/train.json"
  --whu-val-ann-file "annotations/val.json"
  --whu-train-img-subdir "train/image"
  --whu-val-img-subdir "val/image"
  --whu-single-class 1
  --whu-enable-category-mapping 0
  --eval-category-name building
  --image-size 512 512
  --epochs 80
  --seed 44
  --val-ratio 1.0
  --val-batch-size 1
  --val-every-n-epochs 1
  --early-stopping-patience 10
  --early-stopping-smooth-window 5
  --early-stopping-min-epochs 20
  --early-stopping-min-delta 5e-4
  --early-stopping-metric segm_map
  --sat-other-lr-mult 1.0
  --warmup-iters 50
  --weight-decay 0.05
  --deterministic 1
  --amp 0
  --ema-enabled 1
  --ema-decay 0.999
  --ema-update-every 1
  --ema-eval 1
  --ema-save-best 1
  --ema-eval-start-epoch 5
  --iimr-enabled 1
  --iimr-num-iterations 2
  --iimr-use-dynamic-prompting 1
  --iimr-num-uncertainty-points "$IIMR_NUM_UNCERTAINTY_POINTS"
  --iimr-dynamic-balance-ratio 0.25
  --iimr-dynamic-start-iteration 1
  --iimr-dynamic-point-mix 1.0
  --iimr-use-roi-pos-encoding 1
  --iimr-use-geometry-aware-kv 0
  --iimr-detach-mask-path 0
  --iimr-use-topology-memory 0
  --iimr-intermediate-loss-weight "$IIMR_INTERMEDIATE_LOSS_WEIGHT"
  --iimr-use-learnable-residual "$IIMR_USE_LEARNABLE_RESIDUAL"
  --iimr-memory-residual-weight "$IIMR_MEMORY_RESIDUAL_WEIGHT"
  --iimr-enable-after-epochs 10
  --iimr-residual-warmup-epochs "$IIMR_RESIDUAL_WARMUP_EPOCHS"
  --iimr-residual-start-weight 0.0
  --iimr-dynamic-fusion-mode "$IIMR_DYNAMIC_FUSION_MODE"
  --iimr-use-boundary-uncertainty "${IIMR_USE_BOUNDARY_UNCERTAINTY:-0}"
  --iimr-adaptive-balance "${IIMR_ADAPTIVE_BALANCE:-0}"
)

GPU_ID=${GPU_ID:-2}
LOG_FILE="$LOG_DIR/whu512_${RUN_TAG}_baseplus_gpu${GPU_ID}_${TS}.log"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u -m torch.distributed.run --standalone --nproc_per_node 1 "$TRAIN" "${COMMON_ARGS[@]}" \
  --checkpoint-dir "$CKPT_BASE/whu512_${RUN_TAG}" \
  --batch-size "$BEST_BS" --grad-accum-steps "$BEST_ACCUM" \
  --lr "$BEST_LR" --sat-backbone-lr-mult "$BEST_MULT" \
  > "$LOG_FILE" 2>&1 &

PID=$!
echo "[LAUNCHED] TS=$TS"
echo "[PID] WHU512_BASEPLUS=$PID log=$LOG_FILE"
echo "[INFO] SAM2_CKPT=$SAM2_CKPT"
echo "[INFO] RUN_TAG=$RUN_TAG iimr_num_uncertainty_points=$IIMR_NUM_UNCERTAINTY_POINTS"
echo "[INFO] iimr_intermediate_loss_weight=$IIMR_INTERMEDIATE_LOSS_WEIGHT"
echo "[INFO] iimr_use_learnable_residual=$IIMR_USE_LEARNABLE_RESIDUAL"
echo "[INFO] image_size=512x512 batch_size=$BEST_BS grad_accum_steps=$BEST_ACCUM"
echo "[INFO] iimr_memory_residual_weight=$IIMR_MEMORY_RESIDUAL_WEIGHT residual_warmup_epochs=$IIMR_RESIDUAL_WARMUP_EPOCHS"
