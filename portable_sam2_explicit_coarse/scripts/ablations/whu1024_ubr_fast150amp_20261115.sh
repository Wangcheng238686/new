#!/usr/bin/env bash
# PromptMiner-SAM2 (a3sgm) — WHU-1024 UBR fast150-AMP 重跑瘦入口（EXP-0049）。
#
# = fast150 协议（150ep 完整论文调度、fp32→AMP 之外逐参一致）。
# 预算决策反转（2026-11-15，test 证据）：E22 截断点 test 0.6897/0.7285 未退火
# 即达旧行 94.8%/96.5%，但 val 里程碑系选择偏差、收敛上限未知——撤销 fast60
# 砍预算（EXP-0048 未发射），恢复完整 150ep 调度；唯一标注差分=精度（AMP），
# 实测需求 ~12.5G/卡、预期 1.5-2x/步。E22-fp32（EXP-0046）保持独立数据点。
#
# 逐参克隆 scripts/ablations/whu1024_a3sgm_roilocal_full_fast.sh（fast150
# 冻结协议：fp32、有效 batch 8、150ep、A 系增强、seed44），唯一差异：
#   1) config 指向 configs/ablations/whu1024_ubr_fast150_20261114.py
#      （a3sgm_whu1024.py + decoder_tail_refiner 关停）；
#   2) --native-ubr 安装 S 形零证据 native UBR 尾（6625 参数，seed2080）。
# 预登记：docs/evidence/whu_nwpu_ubr_rerun_protocol_20261114.md。
set -Eeuo pipefail
# AMP 默认开启（本家族唯一协议差分）
export AMP="${AMP:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${MAINLINE}/scripts/load_environment.sh"

export RUN_TAG="${RUN_TAG:-whu1024_a3sgm_ubr_fast150amp_20261115}"

# --- fast150 数据/调度策略（与历史 run 完全一致）---
export MAX_EPOCHS=150
export VAL_EVERY_N_EPOCHS=1
export VAL_SHARD_ACROSS_RANKS=1  # 分卡验证继承规则（工程等价，见预登记文档）
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
  --master_port="${MASTER_PORT:-32636}" \
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
  --mask-roi-chunk-size 8 \
  "${RESUME_ARGS[@]}"
