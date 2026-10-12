#!/usr/bin/env bash
# PromptMiner-SAM2 (a3sgm) — WHU-1024 全量训练瘦入口。
#
# 模型形态全部在静态配置 configs/a3sgm_whu1024.py（与论文 WHU 检查点
# best_model_epoch118 的训练期架构契约指纹逐字节一致）。本脚本只负责
# 机器环境、fast150 数据/调度策略与发射形态（fp32、有效 batch 8、150ep）。
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${MAINLINE}/scripts/load_environment.sh"

export RUN_TAG="${RUN_TAG:-whu1024_a3sgm_roilocal_full_fast}"

# --- fast150 数据/调度策略（与历史 run 完全一致）---
export MAX_EPOCHS=150
export VAL_EVERY_N_EPOCHS=1
export WARMUP_ITERS=100
export CUDNN_BENCHMARK=1
export EARLY_STOPPING_PATIENCE=20
export EARLY_STOPPING_START_EPOCH=90
export EARLY_STOPPING_MIN_DELTA=5e-4
export EARLY_STOPPING_SMOOTH_WINDOW=5
export SAVE_BBOX_BEST_METRIC=bbox/mAP
export SAVE_COMPOSITE_BEST=1
export SAVE_COMPOSITE_WEIGHTS="0.5*bbox/mAP+0.5*segm/mAP"
export SAVE_LAST_MODEL=1
export SHAPE_PRIOR_LOSS_WEIGHT=0.10
# A 系增强五元组（hflip+vflip+多尺度 896-1152）
export TRAIN_FLIP_PROB=0.5
export TRAIN_VFLIP_PROB=0.5
export TRAIN_MULTI_SCALE_RESIZE_PROB=0.5
export TRAIN_MULTI_SCALE_MODE=value
export TRAIN_MULTI_SCALE_IMG_SCALE=896:896,960:960,1024:1024,1088:1088,1152:1152

RESUME_ARGS=()
if [ -n "${RESUME_OWN:-}" ]; then
  RESUME_ARGS=(--resume-from "${RESUME_OWN}")
fi

CKPT_DIR="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/${RUN_TAG}_tr1.0_va1.0"

cd "${MAINLINE}"
exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT:-32633}" \
  train/train_rsprompter_fusion.py \
  --config configs/a3sgm_whu1024.py \
  --data-root "${WHU1024_DATA_ROOT}" \
  --use-whu-coco \
  --image-size 1024 1024 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${MAX_EPOCHS}" \
  --lr 5e-4 \
  --checkpoint-dir "${CKPT_DIR}" \
  --train-subset-ratio 1.0 --val-subset-ratio 1.0 \
  --val-batch-size 4 \
  --seed 44 \
  --amp "${AMP}" \
  --ema-enabled 1 --ema-decay 0.999 --ema-eval 0 --ema-save-best 0 \
  --ema-update-every 1 --ema-eval-start-epoch 5 \
  --val-every-n-epochs "${VAL_EVERY_N_EPOCHS}" \
  --compute-val-loss 0 \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --early-stopping-smooth-window "${EARLY_STOPPING_SMOOTH_WINDOW}" \
  --sat-backbone-lr-mult 1.0 --sat-other-lr-mult 1.0 \
  --mask-decoder-lr-mult 1.0 --no-mask-lr-mult 1.0 \
  --shape-prior-lr-mult 1.0 \
  --prompt-encoder-lr-mult 0.0 \
  --warmup-iters "${WARMUP_ITERS}" \
  --weight-decay 0.05 \
  --det-loss-stage1-end 5 --det-loss-stage2-end 10 \
  --det-loss-weight-stage1 1.0 --det-loss-weight-stage2 1.0 --det-loss-weight-stage3 1.0 \
  --prompt-debug-stats 1 --prompt-debug-stats-interval 50 \
  "${RESUME_ARGS[@]}"
