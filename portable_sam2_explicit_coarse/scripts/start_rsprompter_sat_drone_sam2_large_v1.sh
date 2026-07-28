#!/bin/bash
# RSPrompter with SAM2-Hiera-Large - 卫星-无人机融合版本
# 整合: SAM2 Large + LoRA + 高度引导融合 + 深度感知 BEV

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ -z "${SAM2_CKPT}" ]; then
  export SAM2_CKPT="/data/wangcheng/pretrained-models/sam2/sam2_hiera_large.pt"
fi

SAT_DATA_ROOT="${SAT_DATA_ROOT:-/data/wangcheng/dataset/unversity-big-after-without-negative/university-s-train}"
DRONE_DATA_ROOT="${DRONE_DATA_ROOT:-/data/wangcheng/dataset/unversity-big-after-without-negative/university-d-train-selected-5}"
DRONE_IMAGE_SIZE="${DRONE_IMAGE_SIZE:-512}"
NUM_VIEWS="${NUM_VIEWS:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
EPOCHS="${EPOCHS:-50}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"

SAM2_BACKBONE_LR_MULT="${SAM2_BACKBONE_LR_MULT:-1}"
DRONE_LR_MULT="${DRONE_LR_MULT:-1.0}"
SCENE_ALIGN_LR_MULT="${SCENE_ALIGN_LR_MULT:-2.0}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/wangcheng/checkpoint/rsprompter-sat-drone-sam2-large-v1-single-gpu}"

WARMUP_ITERS="${WARMUP_ITERS:-100}"

USE_FSDP="${USE_FSDP:-0}"
FSDP_SHARDING="${FSDP_SHARDING:-FULL_SHARD}"
FSDP_MIN_NUM_PARAMS="${FSDP_MIN_NUM_PARAMS:-1000000}"

VAL_RATIO="${VAL_RATIO:-0.2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-${BATCH_SIZE}}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-5}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"

RESUME_FROM="${RESUME_FROM:-}"

FSDP_ARGS=()
if [ "$USE_FSDP" = "1" ]; then
  FSDP_ARGS+=(--use-fsdp --fsdp-sharding "$FSDP_SHARDING" --fsdp-min-num-params "$FSDP_MIN_NUM_PARAMS")
fi

RESUME_ARGS=()
if [ -n "$RESUME_FROM" ]; then
  RESUME_ARGS+=(--resume-from "$RESUME_FROM")
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/rsprompter_sat_drone_sam2_large_v1_${TIMESTAMP}.log"
mkdir -p "$LOG_DIR"
ln -sf "$LOG_FILE" "${LOG_DIR}/rsprompter_sat_drone_sam2_large_v1_latest.log"

CONFIG_PATH="${PROJECT_ROOT}/configs/rsprompter_sat_drone_sam2_large_v1.py"
TRAIN_BIN="${PROJECT_ROOT}/train/train_rsprompter_fusion.py"

echo "========================================"
echo "RSPrompter(SAM2) Hiera-Large"
echo "卫星-无人机融合版本"
echo "========================================"
echo "训练参数:"
echo "  - 模型：SAM2 Hiera-Large (卫星+无人机)"
echo "  - 权重：$SAM2_CKPT"
echo "  - 图像尺寸：512x512"
echo "  - 无人机尺寸：${DRONE_IMAGE_SIZE}x${DRONE_IMAGE_SIZE}"
echo "  - 无人机视角数：$NUM_VIEWS"
echo "  - 总轮数：$EPOCHS"
echo "  - 基础学习率：$LEARNING_RATE"
echo "  - 批大小：$BATCH_SIZE (梯度累积：$GRAD_ACCUM_STEPS) [effective: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))]"
echo "  - LR 倍率：backbone=${SAM2_BACKBONE_LR_MULT} drone=${DRONE_LR_MULT} scene_align=${SCENE_ALIGN_LR_MULT}"
echo "  - 验证：ratio=${VAL_RATIO} interval=${VAL_EVERY_N_EPOCHS} patience=${EARLY_STOPPING_PATIENCE}"
echo "  - 权重目录：$CHECKPOINT_DIR"
echo "========================================"
echo "关键特性："
echo "  ✅ SAM2 Hiera-Large backbone"
echo "  ✅ LoRA r=16, alpha=32, dropout=0.05"
echo "  ✅ 高度引导的空间融合"
echo "  ✅ 深度感知 BEV 生成"
echo "  ✅ 512 通道维度"
echo "  ✅ prompt_shape=(100,5)"
echo "  ✅ use_high_res_features=True"
echo "========================================"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC_PER_NODE" \
  "$TRAIN_BIN" \
  --config "$CONFIG_PATH" \
  --data-root "$SAT_DATA_ROOT" \
  --use-drone \
  --drone-data-root "$DRONE_DATA_ROOT" \
  --drone-image-size "$DRONE_IMAGE_SIZE" "$DRONE_IMAGE_SIZE" \
  --num-views "$NUM_VIEWS" \
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
  --drone-lr-mult "$DRONE_LR_MULT" \
  --scene-align-lr-mult "$SCENE_ALIGN_LR_MULT" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --warmup-iters "$WARMUP_ITERS" \
  --weight-decay 0.01 \
  "${FSDP_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "RSPrompter(SAM2 Large 卫星-无人机融合) 训练完成"
echo "========================================"
