#!/bin/bash
# RSPrompter with SAM2-Hiera-Large v12 - 统一学习率版本
# 改进：使用统一学习率（不分层），保持 batchsize 和数据增强不变

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# SAM2 Large checkpoint
if [ -z "${SAM2_CKPT}" ]; then
  export SAM2_CKPT="/data/wangcheng/pretrained-models/sam2/sam2_hiera_large.pt"
fi

SAT_DATA_ROOT="${SAT_DATA_ROOT:-/data/wangcheng/dataset/unversity-big-after-without-negative/university-s-train}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

BATCH_SIZE="${BATCH_SIZE:-8}"                  
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"        
EPOCHS="${EPOCHS:-50}"                           
LEARNING_RATE="${LEARNING_RATE:-1e-4}"           

# ============================================
# 关键改进：统一学习率（不分层训练）
# ============================================
# 将所有模块的学习率倍率设为 1.0，实现统一学习率
SAM2_BACKBONE_LR_MULT="${SAM2_BACKBONE_LR_MULT:-1.0}" 
SAM2_OTHER_LR_MULT="${SAM2_OTHER_LR_MULT:-1.0}"       

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/wangcheng/checkpoint/rsprompter-sam2-large-v12-uniform-lr}"

WARMUP_ITERS="${WARMUP_ITERS:-50}"

USE_FSDP="${USE_FSDP:-0}"
FSDP_SHARDING="${FSDP_SHARDING:-FULL_SHARD}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-1000000}"

VAL_RATIO="${VAL_RATIO:-0.2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"

FSDP_ARGS=()
if [ "$USE_FSDP" = "1" ]; then
  FSDP_ARGS+=(--use-fsdp --fsdp-sharding "$FSDP_SHARDING" --fsdp-min-num-params "$FSDP_MIN_NUM_PARAMS")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/rsprompter_sam2_large_v12_uniform_lr_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "${LOG_DIR}/rsprompter_sam2_large_v12_uniform_lr_latest.log"

# 使用 v11-full 的配置（包含 high_res_features 等改进）
CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_anchor_satS_v11_sam2_large_full.py"
TRAIN_BIN="${PROJECT_ROOT}/train/train_rsprompter_fusion.py"

echo "========================================"
echo "RSPrompter(SAM2) Hiera-Large v12"
echo "统一学习率版本（对比实验）"
echo "========================================"
echo "训练参数:"
echo "  - 模型：SAM2 Hiera-Large"
echo "  - 权重：$SAM2_CKPT"
echo "  - 图像尺寸：512x512"
echo "  - 总轮数：$EPOCHS"
echo "  - 基础学习率：$LEARNING_RATE"
echo "  - 批大小：$BATCH_SIZE (梯度累积：$GRAD_ACCUM_STEPS) [effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS))]"
echo "  - LR 倍率：backbone=${SAM2_BACKBONE_LR_MULT} other=${SAM2_OTHER_LR_MULT}"
echo "  - 验证：ratio=${VAL_RATIO} interval=${VAL_EVERY_N_EPOCHS} patience=${EARLY_STOPPING_PATIENCE}"
echo "  - 权重目录：$CHECKPOINT_DIR"
echo "========================================"
echo "  batch_size=${BATCH_SIZE:-8}"
echo "  grad_accum=${GRAD_ACCUM_STEPS:-4}"
echo "  epochs=${EPOCHS:-50}"
echo "  lr=${LEARNING_RATE:-1e-4}"
echo "  WARMUP_ITERS=${WARMUP_ITERS:-50}"
echo "  数据增强策略：仅水平翻转 (flip_prob=0.5)"
echo "========================================"
echo "关键改进："
echo "  🆕 统一学习率 (backbone_lr_mult=1.0, 不再分层)"
echo "  use_high_res_features=True"
echo "  LoRA r=16, alpha=32, dropout=0.05"
echo "  with_sincos=True"
echo "  prompt_shape=(100,5)"
echo "========================================"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  "$TRAIN_BIN" \
  --config "$CONFIG_PATH" \
  --data-root "$SAT_DATA_ROOT" \
  --image-size 512 512 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LEARNING_RATE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --val-ratio "$VAL_RATIO" \
  --val-batch-size "$VAL_BATCH_SIZE" \
  --val-every-n-epochs "$VAL_EVERY_N_EPOCHS" \
  --early-stopping-patience "$EARLY_STOPPING_PATIENCE" \
  --sat-backbone-lr-mult "$SAM2_BACKBONE_LR_MULT" \
  --sat-other-lr-mult "$SAM2_OTHER_LR_MULT" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --warmup-iters "$WARMUP_ITERS" \
  --weight-decay 0.05 \
  "${FSDP_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "RSPrompter(SAM2 Large v12 统一学习率) 训练完成"
echo "========================================"
