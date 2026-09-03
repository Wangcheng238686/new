#!/usr/bin/env bash
# VHR-10 fast-400 entry: the WHU fast-150 protocol transplanted to the NWPU
# VHR-10 10-class instance-segmentation benchmark (RSPrompter 520/130 split).
#
# Protocol (all validated on the WHU fast-150 run unless noted):
#   - 4 GPUs x BATCH_SIZE=1 x GRAD_ACCUM_STEPS=2 (effective batch 8)
#   - AMP fp16, CUDNN_BENCHMARK, expandable_segments, mask target 256
#   - Best-model selection on val segm/mAP at maxDet=100, best-only
#     retention + in-place last_checkpoint (crash resume); bbox-best disabled.
#   - EMA shadow tracking (EMA_EVAL=0): raw weights drive val/selection,
#     post-run compare raw vs EMA via infer_from_checkpoint --weights.
#   - Augmentation: hflip+vflip 0.5, multi-scale jitter 1024+-12.5% (p=0.5,
#     value mode, fixed 1024 canvas so frozen SAM2 always sees native input).
#   - Early stopping DISABLED: run the full 400-epoch cosine (~26k optimizer
#     steps); best-only retention keeps the peak regardless.
#   - --prompt-encoder-lr-mult 0.0 MUST be passed explicitly: the argparse
#     default is 0.1, the WHU protocol freezes the prompt encoder.
#
# Dataset notes (multi-class vs WHU):
#   10 classes, RSPrompter 80/20 split (520/130), val doubles as the report
#   split exactly like the RSPrompter baseline (test_dataloader == val).
#   Class imbalance (bridge 94 .. airplane ~600) is left unweighted, matching
#   the baseline; per-class AP is a post-run analysis from predictions.json.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# --- run-specific exports BEFORE the env loader (outer value wins) ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# SAM2 全链路权重切到 large（backbone/decoder/prompt encoder 同源）
export SAM2_CKPT="${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_large.pt}"

# 架构契约导出段与 fast400 同步（C5-v2 完整协议）
export NECK_TYPE=pafpn
export PROMPT_ROUTE=coarse
export EXPLICIT_PROMPT_MODE=points_box_dense
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM=raw_logits
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE=roi_local
export P2_BOUNDARY_REFINER_PROJECTED_CHANNELS="${P2_BOUNDARY_REFINER_PROJECTED_CHANNELS:-64}"
export P2_BOUNDARY_REFINER_MID_CHANNELS="${P2_BOUNDARY_REFINER_MID_CHANNELS:-64}"
export P2_BOUNDARY_REFINER_LOSS_WEIGHT="${P2_BOUNDARY_REFINER_LOSS_WEIGHT:-0.05}"
export SHAPE_PRIOR_LOSS_WEIGHT="${SHAPE_PRIOR_LOSS_WEIGHT:-0.10}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export AMP=1
export BATCH_SIZE=1
export GRAD_ACCUM_STEPS=2
export MAX_EPOCHS=600
export FINAL_MASK_TARGET_SIZE=256
export CUDNN_BENCHMARK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TEST_MAX_PER_IMG=100
export EMA_ENABLED=1 EMA_EVAL=0 EMA_SAVE_BEST=0
export SAVE_BBOX_BEST_METRIC=""
export SAVE_LAST_CHECKPOINT=1
export TRAIN_FLIP_PROB=0.5 TRAIN_VFLIP_PROB=0.5
export TRAIN_MULTI_SCALE_RESIZE_PROB=0.5
export TRAIN_MULTI_SCALE_MODE=value
export TRAIN_MULTI_SCALE_IMG_SCALE="896:896,960:960,1024:1024,1088:1088,1152:1152"

source "${PROJECT_ROOT}/scripts/load_environment.sh"

RUN_TAG="${RUN_TAG:-vhr10_large600}"
SUBSET_TAG="tr1.0_va1.0"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/${RUN_TAG}_${SUBSET_TAG}}"
LOG_DIR="${PORTABLE_SAM2_LOG_ROOT}/ablations"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_${SUBSET_TAG}_${TS}_pid$$.log"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$( [[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo 1 || echo 0 )"
echo "============================================================"
echo "vhr10_large600 resolved:"
echo "  git_commit=${GIT_COMMIT} git_dirty=${GIT_DIRTY}"
echo "  split=${VHR10_TRAIN_ANN_FILE} / ${VHR10_VAL_ANN_FILE}"
echo "  gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE}"
echo "  batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS} (effective $((BATCH_SIZE*GRAD_ACCUM_STEPS*NPROC_PER_NODE)))"
echo "  epochs=${MAX_EPOCHS} lr=5e-4 warmup=100 seed=44 amp=1"
echo "  ema=${EMA_ENABLED}/${EMA_EVAL} maxdet=${TEST_MAX_PER_IMG} early_stop=off"
echo "  aug: vflip=${TRAIN_VFLIP_PROB} ms=${TRAIN_MULTI_SCALE_RESIZE_PROB}@${TRAIN_MULTI_SCALE_MODE}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  log_file=${LOG_FILE}"
echo "============================================================"

# Full-state crash resume: RESUME_FROM=<ckpt.pth> bash scripts/ablations/vhr10_large600.sh
# restores weights+optimizer+cosine schedule+AMP scaler+EMA shadow+epoch counter
# and continues the original schedule (at most the in-flight epoch is lost).
RESUME_FROM="${RESUME_FROM:-}"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")
  echo "  resume_from=${RESUME_FROM}"
fi

exec "${PYTHON}" -m torch.distributed.run \
  "--nproc_per_node=${NPROC_PER_NODE}" \
  "--master_port=${MASTER_PORT:-$((20000 + $$ % 20000))}" \
  train/train_rsprompter_fusion.py \
  --config configs/vhr10_large_explicit_coarse.py \
  --data-root "${VHR10_DATA_ROOT}" \
  --use-vhr10-coco \
  --image-size 1024 1024 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${MAX_EPOCHS}" \
  --lr 5e-4 \
  --warmup-iters 100 \
  --weight-decay 0.05 \
  --prompt-encoder-lr-mult 0.0 \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --train-subset-ratio 1.0 \
  --val-subset-ratio 1.0 \
  --val-batch-size 2 \
  --val-every-n-epochs 1 \
  --compute-val-loss 0 \
  --early-stopping-patience 9999 \
  --early-stopping-start-epoch 9999 \
  --amp 1 \
  --ema-enabled 1 \
  --ema-decay 0.999 \
  --ema-eval 0 \
  --ema-save-best 0 \
  --seed 44 \
  --prompt-debug-stats 1 \
  --prompt-debug-stats-interval 50 \
  "${EXTRA_ARGS[@]}"
