#!/usr/bin/env bash
# PromptMiner-SAM2 (a3sgm) — WHU-1024 UBR fast60 短调度家族瘦入口（EXP-0048）。
#
# 里程碑触发的预算决策（2026-11-15 用户拍定，见
# docs/evidence/whu_nwpu_ubr_rerun_protocol_20261114.md"决策修订"节）：
# EXP-0046（fast150）于 E22 val segm 0.7001 ≥ 旧 UDPR 论文 run ep98 val best
# 0.6948 达标后按用户指示停止；本 run 为**从头重训的干净 60ep 家族**（不是
# resume 截断），余弦按 60ep 全程退火。与 fast150（论文协议）相对：
#   - MAX_EPOCHS 150 -> 60（唯一协议差，构成新调度家族，不与 150 家族同表直比）
#   - VAL_SHARD_ACROSS_RANKS=1（工程等价优化：分片全量精确重组，继承规则落档）
#   - 其余逐参不变（fp32、4×1×2 有效 batch 8、seed44、A 系增强、PE lr_mult 0、
#     早停参数保留但 E90 起点在 60ep 下自然失效——预算即停止条件）
# config 复用 configs/ablations/whu1024_ubr_fast150_20261114.py（模型定义不含
# epoch 数，epoch 全由 CLI 传入；模型差分与 EXP-0046 完全相同：UDPR 关停 +
# --native-ubr 安装 S 形零证据尾）。
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${MAINLINE}/scripts/load_environment.sh"

export RUN_TAG="${RUN_TAG:-whu1024_a3sgm_ubr_fast60_20261115}"

# --- fast60 数据/调度策略（除 MAX_EPOCHS/分片验证外与 fast150 历史合同一致）---
export MAX_EPOCHS=60
export VAL_EVERY_N_EPOCHS=1
export VAL_SHARD_ACROSS_RANKS=1
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
  --master_port="${MASTER_PORT:-32635}" \
  train/train_rsprompter_fusion.py \
  --config configs/ablations/whu1024_ubr_fast150_20261114.py \
  --data-root "${WHU1024_DATA_ROOT}" \
  --dataset whu \
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
  --ema-enabled 1 --ema-decay 0.999 --ema-eval 0 \
  --ema-update-every 1 --ema-eval-start-epoch 5 \
  --val-every-n-epochs "${VAL_EVERY_N_EPOCHS}" \
  --compute-val-loss 0 \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --early-stopping-smooth-window "${EARLY_STOPPING_SMOOTH_WINDOW}" \
  --backbone-lr-mult 1.0 --other-lr-mult 1.0 \
  --mask-decoder-lr-mult 1.0 --no-mask-lr-mult 1.0 \
  --shape-prior-lr-mult 1.0 \
  --prompt-encoder-lr-mult 0.0 \
  --warmup-iters "${WARMUP_ITERS}" \
  --weight-decay 0.05 \
  --prompt-debug-stats 1 --prompt-debug-stats-interval 50 \
  --native-ubr \
  "${RESUME_ARGS[@]}"
