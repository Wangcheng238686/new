#!/bin/bash
# RSPrompter with SAM2-Hiera-Large v12 - 统一学习率版本 - 推理脚本

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -z "${SAM2_CKPT}" ]; then
  export SAM2_CKPT="/data/wangcheng/pretrained-models/sam2/sam2_hiera_large.pt"
fi

SAT_TEST_DATA_ROOT="${SAT_TEST_DATA_ROOT:-/data/wangcheng/dataset/unversity-big-after-without-negative/university-s-test}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU_ID="${GPU_ID:-0}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data/wangcheng/checkpoint/rsprompter-sam2-large-v12-uniform-lr/best_model.pth}"
CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_anchor_satS_v11_sam2_large_full.py"
INFERENCE_BIN="${PROJECT_ROOT}/inference/inference_rsprompter_fusion.py"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
WORK_DIR="${PROJECT_ROOT}/inference_results/rsprompter-sam2-large-v12-uniform-lr-${TIMESTAMP}"
SHOW_DIR="${SHOW_DIR:-${WORK_DIR}/visualizations}"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误：权重文件不存在：$CHECKPOINT_PATH"
    echo "请设置 CHECKPOINT_PATH 环境变量"
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

echo "========================================"
echo "RSPrompter(SAM2) Hiera-Large v12 推理"
echo "统一学习率版本"
echo "========================================"
echo "配置：$CONFIG_PATH"
echo "权重：$CHECKPOINT_PATH"
echo "SAM2 权重：$SAM2_CKPT"
echo "测试数据：$SAT_TEST_DATA_ROOT"
echo "图像尺寸：${IMAGE_SIZE_H}x${IMAGE_SIZE_W}"
echo "可视化目录：$SHOW_DIR"
echo "指标目录：$WORK_DIR"
echo "========================================"
echo "模型特性："
echo "  ✅ 统一学习率 (backbone_lr_mult=1.0)"
echo "  ✅ use_high_res_features=True"
echo "  ✅ loRA r=16, alpha=32"
echo "========================================"

"$PYTHON_BIN" "$INFERENCE_BIN" \
  "$CONFIG_PATH" \
  "$CHECKPOINT_PATH" \
  --data-root "$SAT_TEST_DATA_ROOT" \
  --image-size "$IMAGE_SIZE_H" "$IMAGE_SIZE_W" \
  --show-dir "$SHOW_DIR" \
  --work-dir "$WORK_DIR" \
  --show-score-thr "$SCORE_THR" \
  --max-num "$MAX_NUM" \
  --gpu-id "$GPU_ID"

echo ""
echo "========================================"
echo "RSPrompter(SAM2 Large v12) 推理完成"
echo "========================================"
echo "可视化结果：$SHOW_DIR"
echo "评估指标：$WORK_DIR"
echo "========================================"
