#!/usr/bin/env bash
# R1-C4-RD on ue_coco (single-stream, no drone): the same architecture as the
# WHU R1-C4-RD ablation (PAFPN, stride-16/64x64, coarse points_box_dense,
# detached raw-logits dense prompt, no P2 refiner), trained on the leakage-free
# ue_coco split (annotations/{train,val,test}.clean.json + unified images/).
#
# This wrapper is self-contained: it does NOT call scripts/ablations/_run_ablation.sh
# (which is WHU-specific) and instead builds the validate + torchrun command
# directly. WHU experiments are untouched.
#
# Differences from the WHU wrapper:
#   - data-root        : /data1/wangcheng/dataset/ue_coco
#   - ann_file         : annotations/{train,val}.clean.json (grid-cell split)
#   - img subdir       : images (unified dir, decoupled from split)
#   - image-size       : 1024 1024 (ue_coco is native 512; resize up to match
#                        the WHU R1-C4-RD embedding geometry: 1024/16 = 64x64)
#   - checkpoint/log   : under an ue_coco sub-root, kept separate from WHU
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=../load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

# ---- Architecture switches (identical to WHU R1-C4-RD) ----
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH=1
export SAM2_MODEL_SIZE=base_plus
# ROI-local final-mask coordinates: the WHU investigation showed full_image
# supervision collapses on sparse buildings; ue_coco mirrors the healthy
# roi_local contract used by the WHU R1-C4-RD-roi_local baseline.
export FINAL_MASK_COORDINATE_MODE="roi_local"
# coarse route model config (architecture-only; data comes from CLI below).
export PROMPT_GENERATOR_MODE="explicit_mask"
export PROMPT_SPARSE_MODE="shape_point"
export PROMPT_ENCODER_ENABLED=1
export SHAPE_PRIOR_ENABLED=1
export SHAPE_CONTEXT_FUSION="roi_only"
export SHAPE_POINT_ADAPTIVE_VALIDITY=1
export SHAPE_LOSS_SCHEDULE_MODE="fixed"
export SHAPE_PRIOR_LOSS_WEIGHT=0.10
export SEGM_SCORE_MODE=detector
# ue_coco is its own dataset; subset ratios default to full (1.0) since the
# clean split is already small (886/189/183 images).
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"

# ---- ue_coco dataset paths (override the WHU defaults baked into the trainer) ----
UECOCO_DATA_ROOT="${UECOCO_DATA_ROOT:-/data1/wangcheng/dataset/ue_coco}"
export WHU_TRAIN_ANN_FILE="annotations/train.clean.json"
export WHU_VAL_ANN_FILE="annotations/val.clean.json"
export WHU_TRAIN_IMG_SUBDIR="images"
export WHU_VAL_IMG_SUBDIR="images"

# ---- Run identity ----
ABLATION_ID="uecoco_r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64"
RUN_TAG="${RUN_TAG:-${ABLATION_ID}}"
SUBSET_TAG="tr${TRAIN_SUBSET_RATIO}_va${VAL_SUBSET_RATIO}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_ROOT="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ue_coco"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${CHECKPOINT_ROOT}/${RUN_TAG}_${SUBSET_TAG}}"
LOG_DIR="${PORTABLE_SAM2_LOG_ROOT}/ablations_uecoco"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${RUN_TAG}_${SUBSET_TAG}_${RUN_TIMESTAMP}_pid$$.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "terminal_log=${LOG_FILE}"

# ---- DDP / launch params (4-card, tuned for 4090 single-card memory) ----
# 4 ranks x batch 1 x grad-accum 2 = effective batch 8 (matches the WHU
# screening contract of 2 x 1 x 4 = 8) while halving per-rank activation
# memory pressure vs grad-accum 4. batch_size=1 is already the per-card
# minimum; effective batch is preserved via accumulation.
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
# Parse --master-port from CLI first (consumes it so it is not also appended as
# a trainer arg), then fall back to env, then auto-derive from PID.
TRAINER_EXTRA_ARGS=()
while (($# > 0)); do
  case "$1" in
    --master-port) MASTER_PORT="$2"; shift 2;;
    --master-port=*) MASTER_PORT="${1#*=}"; shift;;
    *) TRAINER_EXTRA_ARGS+=("$1"); shift;;
  esac
done
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT="$((20000 + ($$ % 20000)))"
fi
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES}}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-1}"
# grad-accum 2 with 4 ranks keeps effective batch = 8 (4 x 1 x 2).
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
SUBSET_SEED="${SUBSET_SEED:-44}"

# ---- 1) Validate the architecture contract (same model config as WHU R1-C4-RD) ----
# expected-architecture-id must match what the (reused) WHU config resolves to;
# ABLATION_ID carries the uecoco_ prefix only for log/checkpoint separation.
VALIDATE_ARGS=(
  --config configs/whu1024_baseplus_explicit_coarse.py
  --ablation-id "${ABLATION_ID}"
  --expected-architecture-id "r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64"
  --expected-neck "${NECK_TYPE}"
  --prompt-route "${PROMPT_ROUTE}"
  --explicit-prompt-mode "${EXPLICIT_PROMPT_MODE}"
  --p2-boundary-refiner-enabled "${P2_BOUNDARY_REFINER_ENABLED}"
  --expected-final-mask-mode "roi_local"
  --expected-final-mask-loss-mode "standard"
  --expected-roi-sam-enabled "0"
  --expected-image-embed-stride "${SAM_IMAGE_EMBED_STRIDE}"
  --expected-segm-score-mode "${SEGM_SCORE_MODE}"
  --train-subset-ratio "${TRAIN_SUBSET_RATIO}"
  --val-subset-ratio "${VAL_SUBSET_RATIO}"
  --subset-seed "${SUBSET_SEED}"
  --data-root "${UECOCO_DATA_ROOT}"
)
"${PYTHON}" "scripts/ablations/validate_ablation_contract.py" "${VALIDATE_ARGS[@]}"

# ---- 2) Build the torchrun command ----
CMD=(
  "${PYTHON}" -m torch.distributed.run
  "--nproc_per_node=${NPROC_PER_NODE}"
  "--master_port=${MASTER_PORT}"
  train/train_rsprompter_fusion.py
  --config configs/whu1024_baseplus_explicit_coarse.py
  --data-root "${UECOCO_DATA_ROOT}"
  --use-whu-coco
  --image-size 1024 1024
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --epochs "${MAX_EPOCHS}"
  --lr "${LEARNING_RATE}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --train-subset-ratio "${TRAIN_SUBSET_RATIO}"
  --val-subset-ratio "${VAL_SUBSET_RATIO}"
  --val-batch-size "${VAL_BATCH_SIZE}"
  --seed "${SUBSET_SEED}"
  --amp "${AMP}"
  --ema-enabled 0
  --ema-decay 0.999
  --ema-eval 0
  --ema-save-best 0
  --val-every-n-epochs 1
  --compute-val-loss 0
  --early-stopping-patience 10
  --early-stopping-start-epoch 20
  --early-stopping-min-delta 5e-4
  --early-stopping-smooth-window 5
  --sat-backbone-lr-mult 1.0
  --sat-other-lr-mult 1.0
  --mask-decoder-lr-mult 1.0
  --no-mask-lr-mult 1.0
  --prompt-encoder-lr-mult 0.0
  --shape-prior-lr-mult 1.0
  --p2-boundary-refiner-lr-mult 1.0
  --warmup-iters 100
  --weight-decay 0.05
  --det-loss-stage1-end 5
  --det-loss-stage2-end 10
  --det-loss-weight-stage1 1.0
  --det-loss-weight-stage2 1.0
  --det-loss-weight-stage3 1.0
  --ema-update-every 1
  --ema-eval-start-epoch 5
  --prompt-debug-stats 1
  --prompt-debug-stats-interval 50
)

# Allow extra trainer args from the CLI (master-port already consumed above).
CMD+=("${TRAINER_EXTRA_ARGS[@]}")

echo "============================================================"
echo "ue_coco R1-C4-RD (leakage-free split)"
echo "ablation_id=${ABLATION_ID}"
echo "data_root=${UECOCO_DATA_ROOT}"
echo "train_ann=annotations/train.clean.json  val_ann=annotations/val.clean.json  img=images/"
echo "image_size=1024x1024 (resized from native 512)"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "log_file=${LOG_FILE}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

case "${RUN_IN_BACKGROUND:-1}" in
  0)
    echo "launch_mode=foreground"
    exec "${CMD[@]}"
    ;;
  1)
    echo "launch_mode=background"
    nohup setsid "${CMD[@]}" </dev/null >>"${LOG_FILE}" 2>&1 &
    LAUNCH_PID=$!
    echo "background_pid=${LAUNCH_PID}"
    echo "follow_log=tail -f ${LOG_FILE}"
    echo "status_command=ps -fp ${LAUNCH_PID}"
    ;;
esac
