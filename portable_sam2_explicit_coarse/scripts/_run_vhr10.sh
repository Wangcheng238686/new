#!/usr/bin/env bash
# Shared VHR-10 runner: one thin entry per experiment in scripts/ablations/
# (vhr10_fast400.sh / vhr10_jitter400.sh / vhr10_large400.sh / vhr10_large600.sh
# / vhr10_c5v2_ft200.sh) exec this file with the variant name as $1.
#
# The architecture exports block lives HERE and ONLY here. The 2026-08-30
# incident trained two degraded C4 variants because a per-wrapper copy of this
# block silently dropped exports (P2 boundary refiner loss share 0.00%, SAM
# embedding at 32x32, dense input not detached). Never re-inline it elsewhere.
#
# Protocol (validated on the WHU fast-150 run unless noted):
#   - 4 GPUs x BATCH_SIZE=1 x GRAD_ACCUM_STEPS=2 (effective batch 8)
#   - AMP fp16, expandable_segments, mask target 256
#   - Best-model selection on val segm/mAP at maxDet=100, best-only retention
#     + in-place last_checkpoint (crash resume); bbox-best disabled.
#   - EMA shadow tracking (EMA_EVAL=0): raw weights drive val/selection,
#     post-run compare raw vs EMA via infer_from_checkpoint --weights.
#   - Augmentation: hflip+vflip 0.5, multi-scale jitter 1024+-12.5% (p=0.5,
#     value mode, fixed 1024 canvas so frozen SAM2 always sees native input).
#   - Early stopping DISABLED: run the full cosine schedule; best-only
#     retention keeps the peak regardless.
#   - --prompt-encoder-lr-mult 0.0 MUST be passed explicitly: the argparse
#     default is 0.1; the protocol freezes the prompt encoder.
#
# Dataset: NWPU VHR-10 10-class instance segmentation, RSPrompter-release
# 80/20 split (520 train / 130 val), val doubles as the report split exactly
# like the RSPrompter baseline (test_dataloader == val). Class imbalance is
# left unweighted, matching the baseline; per-class AP is a post-run analysis
# from predictions.json.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANT="${1:-}"
case "${VARIANT}" in
  fast400)
    RUN_TAG_DEFAULT="vhr10_c5v2_400"
    CONFIG_PATH="configs/vhr10_baseplus_explicit_coarse.py"
    DEFAULT_EPOCHS=400
    DEFAULT_LR="5e-4"
    INIT_ARGS=()
    ;;
  jitter400)
    RUN_TAG_DEFAULT="vhr10_jitter400"
    CONFIG_PATH="configs/vhr10_jitter_explicit_coarse.py"
    DEFAULT_EPOCHS=400
    DEFAULT_LR="5e-4"
    INIT_ARGS=()
    ;;
  large400)
    RUN_TAG_DEFAULT="vhr10_large400"
    CONFIG_PATH="configs/vhr10_large_explicit_coarse.py"
    DEFAULT_EPOCHS=400
    DEFAULT_LR="5e-4"
    INIT_ARGS=()
    ;;
  large600)
    RUN_TAG_DEFAULT="vhr10_large600"
    CONFIG_PATH="configs/vhr10_large_explicit_coarse.py"
    DEFAULT_EPOCHS=600
    DEFAULT_LR="5e-4"
    INIT_ARGS=()
    ;;
  ft200)
    RUN_TAG_DEFAULT="vhr10_c5v2_ft200"
    CONFIG_PATH="configs/vhr10_baseplus_explicit_coarse.py"
    DEFAULT_EPOCHS=200
    DEFAULT_LR="1e-4"
    # Fine-tune continuation of the c5v2-400 winner at the lower LR tier.
    INIT_ARGS=(--init-from
      "${INIT_FROM:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_c5v2_400_tr1.0_va1.0/last_checkpoint.pth}")
    ;;
  "")
    echo "Usage: _run_vhr10.sh <fast400|jitter400|large400|large600|ft200>" >&2
    exit 2
    ;;
  *)
    echo "Unknown VHR-10 variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

# --- architecture contract exports (single source of truth; see header) ---
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

# --- variant-specific overrides on top of the shared protocol ---
case "${VARIANT}" in
  large400|large600)
    # SAM2 full-chain weights switched to large (backbone/decoder/prompt
    # encoder from the same hiera-large release).
    export SAM2_CKPT="${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_large.pt}"
    ;;
esac

# --- shared runtime protocol ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export AMP=1
export BATCH_SIZE=1
export GRAD_ACCUM_STEPS=2
export MAX_EPOCHS="${MAX_EPOCHS:-${DEFAULT_EPOCHS}}"
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

RUN_TAG="${RUN_TAG:-${RUN_TAG_DEFAULT}}"
SUBSET_TAG="tr1.0_va1.0"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/${RUN_TAG}_${SUBSET_TAG}}"
LOG_DIR="${PORTABLE_SAM2_LOG_ROOT}/ablations"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${RUN_TAG}_${SUBSET_TAG}_${TS}_pid$$.log"
mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# Full-state crash resume (large600 lineage): restores weights + optimizer +
# cosine schedule + AMP scaler + EMA shadow + epoch counter and continues the
# original schedule (at most the in-flight epoch is lost).
#   RESUME_FROM=<ckpt.pth> bash scripts/ablations/vhr10_large600.sh
RESUME_FROM="${RESUME_FROM:-}"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
  EXTRA_ARGS+=(--resume-from "${RESUME_FROM}")
fi

# Dry run: verify env/protocol resolution and print the launch plan, but do
# not start torchrun (same semantics as DRY_RUN=1 in _run_ablation.sh).
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "dry_run=1 — launch skipped; would run: ${CONFIG_PATH} epochs=${MAX_EPOCHS} lr=${DEFAULT_LR}"
  exit 0
fi

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$( [[ -n "$(git status --porcelain 2>/dev/null)" ]] && echo 1 || echo 0 )"
echo "============================================================"
echo "vhr10_${VARIANT} resolved:"
echo "  git_commit=${GIT_COMMIT} git_dirty=${GIT_DIRTY}"
echo "  split=${VHR10_TRAIN_ANN_FILE} / ${VHR10_VAL_ANN_FILE}"
echo "  config=${CONFIG_PATH}"
echo "  gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE}"
echo "  batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS} (effective $((BATCH_SIZE*GRAD_ACCUM_STEPS*NPROC_PER_NODE)))"
echo "  epochs=${MAX_EPOCHS} lr=${DEFAULT_LR} warmup=100 seed=44 amp=1"
echo "  ema=${EMA_ENABLED}/${EMA_EVAL} maxdet=${TEST_MAX_PER_IMG} early_stop=off"
echo "  aug: vflip=${TRAIN_VFLIP_PROB} ms=${TRAIN_MULTI_SCALE_RESIZE_PROB}@${TRAIN_MULTI_SCALE_MODE}"
echo "  checkpoint_dir=${CHECKPOINT_DIR}"
echo "  log_file=${LOG_FILE}"
if [[ -n "${RESUME_FROM}" ]]; then
  echo "  resume_from=${RESUME_FROM}"
fi
echo "============================================================"

exec "${PYTHON}" -m torch.distributed.run \
  "--nproc_per_node=${NPROC_PER_NODE}" \
  "--master_port=${MASTER_PORT:-$((20000 + $$ % 20000))}" \
  train/train_rsprompter_fusion.py \
  --config "${CONFIG_PATH}" \
  --data-root "${VHR10_DATA_ROOT}" \
  --use-vhr10-coco \
  --image-size 1024 1024 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --epochs "${MAX_EPOCHS}" \
  --lr "${DEFAULT_LR}" \
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
  "${INIT_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
