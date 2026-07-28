#!/bin/bash
# IIMR 推理脚本（默认对齐 WHU/base_plus 训练口径，支持按需覆盖）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -z "${SAM2_CKPT:-}" ]; then
  export SAM2_CKPT="/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt"
fi

SAT_TEST_DATA_ROOT="${SAT_TEST_DATA_ROOT:-/data/wangcheng/dataset/WHU-512}"
WHU_TEST_ANN_FILE="${WHU_TEST_ANN_FILE:-annotations/test.json}"
WHU_TEST_IMG_SUBDIR="${WHU_TEST_IMG_SUBDIR:-test/image}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
GPU_ID="${GPU_ID:-0}"

IIMR_ENABLED="${IIMR_ENABLED:--1}"
IIMR_NUM_ITERATIONS="${IIMR_NUM_ITERATIONS:--1}"
IIMR_USE_TOPOLOGY_MEMORY="${IIMR_USE_TOPOLOGY_MEMORY:--1}"
IIMR_DETACH_MASK_PATH="${IIMR_DETACH_MASK_PATH:--1}"
IIMR_MEMORY_RESIDUAL_WEIGHT="${IIMR_MEMORY_RESIDUAL_WEIGHT:--1}"   # [哨兵值 -1] -1=继承config默认(0.3); 显式设值覆盖(如0.10/0.15/0.30/0.50)

# Checkpoint weight source controls:
# - CKPT_WEIGHT_SOURCE: raw | ema | auto
# - EMA_FALLBACK_TO_RAW: 1 | 0
# - USE_CKPT_TRAIN_ARGS: 1 | 0
# Backward-compatible alias:
# - USE_EMA_STATE: 1 -> ema, 0 -> raw, -1 -> ignore
USE_EMA_STATE="${USE_EMA_STATE:--1}"
if [ -z "${CKPT_WEIGHT_SOURCE+x}" ]; then
  if [ "${USE_EMA_STATE}" -eq 1 ]; then
    CKPT_WEIGHT_SOURCE="ema"
  elif [ "${USE_EMA_STATE}" -eq 0 ]; then
    CKPT_WEIGHT_SOURCE="raw"
  else
    CKPT_WEIGHT_SOURCE="ema"
  fi
fi
EMA_FALLBACK_TO_RAW="${EMA_FALLBACK_TO_RAW:-1}"
USE_CKPT_TRAIN_ARGS="${USE_CKPT_TRAIN_ARGS:-1}"

if [ "${USE_CKPT_TRAIN_ARGS}" -eq 1 ]; then
  TRAIN_ARGS_PRIORITY="checkpoint"
else
  TRAIN_ARGS_PRIORITY="env"
fi

if [ "${CKPT_WEIGHT_SOURCE}" != "raw" ] && [ "${CKPT_WEIGHT_SOURCE}" != "ema" ] && [ "${CKPT_WEIGHT_SOURCE}" != "auto" ]; then
  echo "错误：CKPT_WEIGHT_SOURCE 只能是 raw/ema/auto，当前值=${CKPT_WEIGHT_SOURCE}"
  exit 1
fi

if [ -z "${CHECKPOINT_PATH:-}" ]; then
  CHECKPOINT_PATH="$(ls -dt /data/wangcheng/checkpoint/ablation_hyperparam_whu512_baseplus/whu512_*/best_model.pth 2>/dev/null | head -n 1)"
fi

CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/rsprompter_anchor_whu512_base_plus_singlecls.py}"
INFERENCE_BIN="${PROJECT_ROOT}/inference/inference_rsprompter_fusion.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="${PROJECT_ROOT}/inference_results/iimr_aligned-${TIMESTAMP}"
SHOW_DIR="${SHOW_DIR:-${WORK_DIR}/visualizations}"

if [ -z "$CHECKPOINT_PATH" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误：权重文件不存在：$CHECKPOINT_PATH"
    echo "请设置 CHECKPOINT_PATH，例如:"
    echo "export CHECKPOINT_PATH=/data/wangcheng/checkpoint/xxx/best_model.pth"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误：配置文件不存在：$CONFIG_PATH"
    exit 1
fi

mkdir -p "$WORK_DIR"
mkdir -p "$SHOW_DIR"

SCORE_THR="${SCORE_THR:-0.3}"
MAX_NUM="${MAX_NUM:-100}"
IMAGE_SIZE_H="${IMAGE_SIZE_H:-512}"
IMAGE_SIZE_W="${IMAGE_SIZE_W:-512}"
NMS_IOU_THR="${NMS_IOU_THR:-0.45}"
RCNN_SCORE_THR="${RCNN_SCORE_THR:-0.0}"
MASK_THR_BINARY="${MASK_THR_BINARY:-0.5}"
MAX_PER_IMG="${MAX_PER_IMG:-300}"

# 结构性口径覆盖：默认依赖 checkpoint train_args 恢复，只有 USE_CKPT_TRAIN_ARGS=0 时才会写入 cfg-options
FPN_TYPE="${FPN_TYPE:-}"
TOPDOWN_DOWNSTREAM_HEAD="${TOPDOWN_DOWNSTREAM_HEAD:-}"
DENSE_PROMPT_MODE="${DENSE_PROMPT_MODE:-}"

CFG_OPTIONS=()
CFG_OPTIONS+=("model.test_cfg.rcnn.nms.iou_threshold=${NMS_IOU_THR}")
CFG_OPTIONS+=("model.test_cfg.rcnn.score_thr=${RCNN_SCORE_THR}")
CFG_OPTIONS+=("model.test_cfg.rcnn.mask_thr_binary=${MASK_THR_BINARY}")
CFG_OPTIONS+=("model.test_cfg.rcnn.max_per_img=${MAX_PER_IMG}")
if [ "${USE_CKPT_TRAIN_ARGS}" -ne 1 ]; then
  [ "${IIMR_ENABLED}" -ge 0 ] && CFG_OPTIONS+=("model.roi_head.iimr.enabled=${IIMR_ENABLED}")
  [ "${IIMR_NUM_ITERATIONS}" -gt 0 ] && CFG_OPTIONS+=("model.roi_head.iimr.num_iterations=${IIMR_NUM_ITERATIONS}")
  [ "${IIMR_USE_TOPOLOGY_MEMORY}" -ge 0 ] && CFG_OPTIONS+=("model.roi_head.iimr.use_topology_memory=${IIMR_USE_TOPOLOGY_MEMORY}")
  [ "${IIMR_DETACH_MASK_PATH}" -ge 0 ] && CFG_OPTIONS+=("model.roi_head.iimr.detach_mask_path=${IIMR_DETACH_MASK_PATH}")
  [ "${IIMR_MEMORY_RESIDUAL_WEIGHT}" != "-1" ] && CFG_OPTIONS+=("model.roi_head.iimr.memory_residual_weight=${IIMR_MEMORY_RESIDUAL_WEIGHT}")
  [ -n "${FPN_TYPE}" ] && CFG_OPTIONS+=("model.neck.fpn_type=${FPN_TYPE}")
  [ -n "${TOPDOWN_DOWNSTREAM_HEAD}" ] && CFG_OPTIONS+=("model.roi_head.topdown_downstream_head=${TOPDOWN_DOWNSTREAM_HEAD}")
  [ -n "${DENSE_PROMPT_MODE}" ] && CFG_OPTIONS+=("model.roi_head.mask_head.dense_prompt_mode=${DENSE_PROMPT_MODE}")
fi

echo "========================================"
echo "IIMR 推理（默认对齐 WHU/base_plus 训练配置）"
echo "========================================"
echo "配置：$CONFIG_PATH"
echo "权重：$CHECKPOINT_PATH"
echo "SAM2 权重：$SAM2_CKPT"
echo "测试数据：$SAT_TEST_DATA_ROOT"
echo "WHU ann/img：${WHU_TEST_ANN_FILE} / ${WHU_TEST_IMG_SUBDIR}"
echo "图像尺寸：${IMAGE_SIZE_H}x${IMAGE_SIZE_W}"
echo "NMS IoU：${NMS_IOU_THR}"
echo "RCNN score_thr：${RCNN_SCORE_THR}"
echo "Mask 二值阈值：${MASK_THR_BINARY}"
echo "max_per_img：${MAX_PER_IMG}"
echo "可视化目录：$SHOW_DIR"
echo "指标目录：$WORK_DIR"
echo "========================================"
echo "IIMR 覆盖项（-1 表示继承训练配置）："
echo "  IIMR_ENABLED=${IIMR_ENABLED}"
echo "  IIMR_NUM_ITERATIONS=${IIMR_NUM_ITERATIONS}"
echo "  IIMR_USE_TOPOLOGY_MEMORY=${IIMR_USE_TOPOLOGY_MEMORY}"
echo "  IIMR_DETACH_MASK_PATH=${IIMR_DETACH_MASK_PATH}"
echo "  IIMR_MEMORY_RESIDUAL_WEIGHT=${IIMR_MEMORY_RESIDUAL_WEIGHT}"
echo "  CKPT_WEIGHT_SOURCE=${CKPT_WEIGHT_SOURCE}"
echo "  EMA_FALLBACK_TO_RAW=${EMA_FALLBACK_TO_RAW}"
echo "  USE_CKPT_TRAIN_ARGS=${USE_CKPT_TRAIN_ARGS}"
echo "  TRAIN_ARGS_PRIORITY=${TRAIN_ARGS_PRIORITY}"
echo "  FPN_TYPE=${FPN_TYPE:-<from-checkpoint-or-config>}"
echo "  TOPDOWN_DOWNSTREAM_HEAD=${TOPDOWN_DOWNSTREAM_HEAD:-<from-checkpoint-or-config>}"
echo "  DENSE_PROMPT_MODE=${DENSE_PROMPT_MODE:-<from-checkpoint-or-config>}"
if [ "${USE_CKPT_TRAIN_ARGS}" -eq 1 ]; then
  echo "  NOTE: 已锁定 checkpoint train_args 为最高优先级，结构性环境变量覆盖将被忽略"
fi
echo "========================================"

CMD=("$PYTHON_BIN" "$INFERENCE_BIN"
  "$CONFIG_PATH"
  "$CHECKPOINT_PATH"
  --data-root "$SAT_TEST_DATA_ROOT"
  --dataset-format whu_coco
  --whu-val-ann-file "$WHU_TEST_ANN_FILE"
  --whu-val-img-subdir "$WHU_TEST_IMG_SUBDIR"
  --whu-single-class 1
  --whu-enable-category-mapping 0
  --image-size "$IMAGE_SIZE_H" "$IMAGE_SIZE_W"
  --show-dir "$SHOW_DIR"
  --work-dir "$WORK_DIR"
  --show-score-thr "$SCORE_THR"
  --max-num "$MAX_NUM"
  --gpu-id "$GPU_ID"
  --ckpt-weight-source "$CKPT_WEIGHT_SOURCE"
  --ema-fallback-to-raw "$EMA_FALLBACK_TO_RAW"
  --train-args-priority "$TRAIN_ARGS_PRIORITY")

if [ ${#CFG_OPTIONS[@]} -gt 0 ]; then
  CMD+=(--cfg-options)
  CMD+=("${CFG_OPTIONS[@]}")
fi

{
  echo "[CMD] 推理命令快照:"
  printf '  %q ' "${CMD[@]}"
  echo
  echo "--------------------------------------------------"
}

"${CMD[@]}"

echo ""
echo "========================================"
echo "IIMR 推理完成"
echo "========================================"
echo "可视化结果：$SHOW_DIR"
echo "评估指标：$WORK_DIR"
echo "========================================"
