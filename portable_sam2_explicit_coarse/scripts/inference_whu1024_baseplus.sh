#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}"
INFERENCE_BIN="${PROJECT_ROOT}/inference/inference_rsprompter_fusion.py"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID:-0}"

if [ -z "${SAM2_CKPT:-}" ]; then
  export SAM2_CKPT="/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt"
fi

SAT_TEST_DATA_ROOT="${SAT_TEST_DATA_ROOT:-/data/wangcheng/dataset/WHU}"
WHU_TEST_ANN_FILE="${WHU_TEST_ANN_FILE:-2.4 annotation/annotation/test.json}"
WHU_TEST_IMG_SUBDIR="${WHU_TEST_IMG_SUBDIR:-2.2 test/test}"

IMAGE_SIZE_H="${IMAGE_SIZE_H:-1024}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-1024}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/wangcheng/checkpoint/ablation_hyperparam_whu1024_baseplus/whu1024_baseline_noms_4gpu/best_model.pth}"

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/rsprompter_anchor_whu1024_base_plus_singlecls.py}"

NMS_IOU_THR="${NMS_IOU_THR:-0.45}"
RCNN_SCORE_THR="${RCNN_SCORE_THR:-0.0}"
MASK_THR_BINARY="${MASK_THR_BINARY:-0.5}"
MAX_PER_IMG="${MAX_PER_IMG:-300}"
NUM_WORKERS="${NUM_WORKERS:-2}"

CKPT_WEIGHT_SOURCE="${CKPT_WEIGHT_SOURCE:-ema}"
EMA_FALLBACK_TO_RAW="${EMA_FALLBACK_TO_RAW:-1}"
TRAIN_ARGS_PRIORITY="${TRAIN_ARGS_PRIORITY:-checkpoint}"

SCORE_THR="${SCORE_THR:-0.3}"
MAX_NUM="${MAX_NUM:-100}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-whu1024_baseline_nms$(echo "$NMS_IOU_THR" | tr -d .)}"
WORK_DIR="${WORK_DIR:-${PROJECT_ROOT}/inference_results/${RUN_NAME}-${TIMESTAMP}}"
SHOW_DIR="${SHOW_DIR:-${WORK_DIR}/visualizations}"

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[ERROR] 配置文件不存在: $CONFIG_PATH"
  exit 1
fi

if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[WARN] 权重文件不存在: $CHECKPOINT_PATH"
  echo "[WARN] 请通过 CHECKPOINT_PATH 指定正确模型文件"
fi

mkdir -p "$WORK_DIR"

SKIP_VIS="${SKIP_VIS:-0}"
if [ "$SKIP_VIS" = "1" ]; then
  SHOW_DIR_ARG=""
else
  mkdir -p "$SHOW_DIR"
  SHOW_DIR_ARG="--show-dir $SHOW_DIR"
fi

echo "========================================"
echo "WHU 1024 Base+ 推理"
echo "========================================"
echo "Config: $CONFIG_PATH"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "SAM2_CKPT: $SAM2_CKPT"
echo "Data root: $SAT_TEST_DATA_ROOT"
echo "Image size: ${IMAGE_SIZE_H}x${IMAGE_SIZE_W}"
echo "NMS IoU: $NMS_IOU_THR"
echo "RCNN score thr: $RCNN_SCORE_THR"
echo "Mask thr: $MASK_THR_BINARY"
echo "Max per img: $MAX_PER_IMG"
echo "Num workers: $NUM_WORKERS"
echo "Ckpt weight source: $CKPT_WEIGHT_SOURCE (ema fallback to raw: $EMA_FALLBACK_TO_RAW)"
echo "Train args priority: $TRAIN_ARGS_PRIORITY"
echo "Work dir: $WORK_DIR"
echo "Show dir: $SHOW_DIR"
echo "========================================"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/inference}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_${TIMESTAMP}.log"

VIS_ARGS=()
if [ "$SKIP_VIS" != "1" ]; then
  VIS_ARGS=(--show-dir "$SHOW_DIR" --show-score-thr "$SCORE_THR" --max-num "$MAX_NUM")
fi

echo "Log: $LOG_FILE"
echo "========================================"

"$PYTHON_BIN" -u "$INFERENCE_BIN" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  --data-root "$SAT_TEST_DATA_ROOT" \
  --dataset-format whu_coco \
  --whu-val-ann-file "$WHU_TEST_ANN_FILE" \
  --whu-val-img-subdir "$WHU_TEST_IMG_SUBDIR" \
  --whu-single-class 1 \
  --whu-enable-category-mapping 0 \
  --image-size "$IMAGE_SIZE_H" "$IMAGE_SIZE_W" \
  --num-workers "$NUM_WORKERS" \
  --work-dir "$WORK_DIR" \
  "${VIS_ARGS[@]}" \
  --gpu-id "$GPU_ID" \
  --ckpt-weight-source "$CKPT_WEIGHT_SOURCE" \
  --ema-fallback-to-raw "$EMA_FALLBACK_TO_RAW" \
  --train-args-priority "$TRAIN_ARGS_PRIORITY" \
  --cfg-options \
    model.test_cfg.rcnn.nms.iou_threshold="$NMS_IOU_THR" \
    model.test_cfg.rcnn.score_thr="$RCNN_SCORE_THR" \
    model.test_cfg.rcnn.mask_thr_binary="$MASK_THR_BINARY" \
    model.test_cfg.rcnn.max_per_img="$MAX_PER_IMG" \
  2>&1 | tee "$LOG_FILE"
