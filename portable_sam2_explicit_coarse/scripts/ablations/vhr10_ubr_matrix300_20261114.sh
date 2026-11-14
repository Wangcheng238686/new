#!/usr/bin/env bash
# PromptMiner-SAM2 (a3sgm) — NWPU VHR-10 UBR matrix300 协议重跑瘦入口（EXP-0047）。
#
# 逐参克隆 scripts/ablations/vhr10_a3sgm_matrix300.sh（matrix300 冻结协议：
# AMP、4×1×2 有效 batch 8、300ep、分片验证、PE downscaling 可训、
# decoupled-aux-clip、seed44），唯一差异：
#   1) config 默认指向 configs/ablations/vhr10_ubr_matrix300_20261114.py
#      （a3sgm_vhr10.py + decoder_tail_refiner 关停；full_image +
#      roi_balanced_dice 坐标/监督契约不变）；
#   2) --native-ubr 安装 S 形零证据 native UBR 尾（6625 参数，seed2080）。
#      --decoupled-aux-clip 保留：aux 前缀表已等价加入 native_refiner.。
#      --decoder-tail-lr-mult 1.0 保留（UDPR 已不在，无对应参数组；native
#      refiner 落默认组 lr×1.0，语义等价）。
# 预登记：docs/evidence/whu_nwpu_ubr_rerun_protocol_20261114.md。
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAINLINE="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${MAINLINE}/scripts/load_environment.sh"

export RUN_TAG="${RUN_TAG:-vhr10_a3sgm_ubr_matrix300_20261114}"

# --- matrix300 数据/调度策略（与历史 run 完全一致）---
export MAX_EPOCHS=300
export VAL_EVERY_N_EPOCHS=5
export WARMUP_ITERS=100
export CUDNN_BENCHMARK=1
export EARLY_STOPPING_PATIENCE=0
export EARLY_STOPPING_START_EPOCH=9999
export EARLY_STOPPING_MIN_DELTA=5e-4
export EARLY_STOPPING_SMOOTH_WINDOW=5
export SAVE_BBOX_BEST_METRIC=bbox/mAP
export SAVE_COMPOSITE_BEST=1
export SAVE_COMPOSITE_WEIGHTS="0.5*bbox/mAP+0.5*segm/mAP"
export SAVE_LAST_CHECKPOINT=1
export SAVE_LAST_MODEL=1
export SHAPE_PRIOR_LOSS_WEIGHT=0.10
export VAL_SHARD_ACROSS_RANKS=1
# A 系增强五元组（hflip+vflip+多尺度 896-1152）
export TRAIN_FLIP_PROB=0.5
export TRAIN_VFLIP_PROB=0.5
export TRAIN_MULTI_SCALE_RESIZE_PROB=0.5
export TRAIN_MULTI_SCALE_MODE=value
export TRAIN_MULTI_SCALE_IMG_SCALE=896:896,960:960,1024:1024,1088:1088,1152:1152

# config：UBR delta（保留 ABLATION_CONFIG 覆盖机制）
CONFIG="${ABLATION_CONFIG:-configs/ablations/vhr10_ubr_matrix300_20261114.py}"

RESUME_ARGS=()
if [ -n "${RESUME_OWN:-}" ]; then
  RESUME_ARGS=(--resume-from "${RESUME_OWN}")
fi

CKPT_DIR="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/${RUN_TAG}_tr1.0_va1.0"

cd "${MAINLINE}"
exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT:-32809}" \
  train/train_rsprompter_fusion.py \
  --config "${CONFIG}" \
  --data-root "${VHR10_DATA_ROOT}" \
  --dataset vhr10 \
  --image-size 1024 1024 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${MAX_EPOCHS}" \
  --lr 5e-4 \
  --warmup-iters "${WARMUP_ITERS}" \
  --weight-decay 0.05 \
  --prompt-encoder-lr-mult 0.1 \
  --decoder-tail-lr-mult 1.0 \
  --decoupled-aux-clip \
  --native-ubr \
  --checkpoint-dir "${CKPT_DIR}" \
  --train-subset-ratio 1.0 --val-subset-ratio 1.0 \
  --val-batch-size 2 \
  --val-every-n-epochs "${VAL_EVERY_N_EPOCHS}" \
  --compute-val-loss 0 \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}" \
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}" \
  --early-stopping-smooth-window "${EARLY_STOPPING_SMOOTH_WINDOW}" \
  --amp "${AMP}" \
  --ema-enabled 1 --ema-decay 0.999 --ema-eval 0 \
  --seed 44 \
  --prompt-debug-stats 1 --prompt-debug-stats-interval 50 \
  "${RESUME_ARGS[@]}"
