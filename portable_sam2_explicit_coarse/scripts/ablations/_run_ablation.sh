#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

: "${ABLATION_ID:?wrapper must set ABLATION_ID}"
: "${NECK_TYPE:?wrapper must set NECK_TYPE}"
: "${PROMPT_ROUTE:?wrapper must set PROMPT_ROUTE}"
: "${DENSEBR_ENABLED:?wrapper must set DENSEBR_ENABLED}"

case "${NECK_TYPE}" in
  aggregator|pafpn) ;;
  *) echo "NECK_TYPE must be aggregator or pafpn, got ${NECK_TYPE}" >&2; exit 2 ;;
esac

case "${PROMPT_ROUTE}" in
  mlp)
    CONFIG_PATH="configs/whu1024_baseplus_clean.py"
    EXPECTED_EXPLICIT_MODE="none"
    export PROMPT_GENERATOR_MODE="rsprompter_mlp"
    export PROMPT_SPARSE_MODE="point"
    export PROMPT_ENCODER_ENABLED=0
    export SHAPE_PRIOR_ENABLED=0
    export EXPLICIT_PROMPT_MODE="none"
    export DENSEBR_ENABLED=0
    ;;
  coarse)
    CONFIG_PATH="configs/whu1024_baseplus_explicit_coarse.py"
    : "${EXPLICIT_PROMPT_MODE:?coarse wrapper must set EXPLICIT_PROMPT_MODE}"
    case "${EXPLICIT_PROMPT_MODE}" in
      points|points_box|points_box_dense) ;;
      *) echo "invalid EXPLICIT_PROMPT_MODE=${EXPLICIT_PROMPT_MODE}" >&2; exit 2 ;;
    esac
    EXPECTED_EXPLICIT_MODE="${EXPLICIT_PROMPT_MODE}"
    export PROMPT_GENERATOR_MODE="explicit_mask"
    export PROMPT_SPARSE_MODE="shape_point"
    export PROMPT_ENCODER_ENABLED=1
    export SHAPE_PRIOR_ENABLED=1
    ;;
  *) echo "PROMPT_ROUTE must be mlp or coarse, got ${PROMPT_ROUTE}" >&2; exit 2 ;;
esac

export NECK_TYPE
export SAM2_MODEL_SIZE=base_plus
export SAM2_REPO="${SAM2_REPO:-$(cd "${PROJECT_ROOT}/../sam2" && pwd)}"
export SAM2_CKPT="${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}"
export WHU1024_DATA_ROOT="${WHU1024_DATA_ROOT:-/data/wangcheng/dataset/WHU}"
export MAX_EPOCHS="${MAX_EPOCHS:-80}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/portable_sam2_explicit_coarse_mpl}"

PYTHON="${PYTHON:-/data/wangcheng/envs/cvt2/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
SUBSET_SEED="${SUBSET_SEED:-44}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

RUN_TAG="${RUN_TAG:-${ABLATION_ID}}"
SUBSET_TAG="tr${TRAIN_SUBSET_RATIO}_va${VAL_SUBSET_RATIO}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/${RUN_TAG}_${SUBSET_TAG}}"

VALIDATE_ARGS=(
  --config "${CONFIG_PATH}"
  --ablation-id "${ABLATION_ID}"
  --expected-neck "${NECK_TYPE}"
  --prompt-route "${PROMPT_ROUTE}"
  --explicit-prompt-mode "${EXPECTED_EXPLICIT_MODE}"
  --densebr-enabled "${DENSEBR_ENABLED}"
  --train-subset-ratio "${TRAIN_SUBSET_RATIO}"
  --val-subset-ratio "${VAL_SUBSET_RATIO}"
  --subset-seed "${SUBSET_SEED}"
  --data-root "${WHU1024_DATA_ROOT}"
)
if [ "${CHECK_DATA:-0}" = "1" ]; then
  VALIDATE_ARGS+=(--check-data)
fi
if [ "${PREFLIGHT_MODEL:-0}" = "1" ]; then
  VALIDATE_ARGS+=(--build-model)
fi
"${PYTHON}" "${SCRIPT_DIR}/validate_ablation_contract.py" "${VALIDATE_ARGS[@]}"

CMD=(
  "${PYTHON}" -m torch.distributed.run
  "--nproc_per_node=${NPROC_PER_NODE}"
  "--master_port=${MASTER_PORT:-29500}"
  train/train_rsprompter_fusion.py
  --config "${CONFIG_PATH}"
  --data-root "${WHU1024_DATA_ROOT}"
  --use-whu-coco
  --image-size 1024 1024
  --batch-size "${BATCH_SIZE}"
  --grad-accum-steps "${GRAD_ACCUM_STEPS}"
  --epochs "${MAX_EPOCHS}"
  --lr "${LEARNING_RATE}"
  --checkpoint-dir "${CHECKPOINT_DIR}"
  --train-subset-ratio "${TRAIN_SUBSET_RATIO}"
  --val-subset-ratio "${VAL_SUBSET_RATIO}"
  --seed "${SUBSET_SEED}"
  --amp "${AMP:-0}"
  --ema-enabled "${EMA_ENABLED:-1}"
  --ema-decay "${EMA_DECAY:-0.999}"
  --ema-eval "${EMA_EVAL:-1}"
  --ema-save-best "${EMA_SAVE_BEST:-1}"
  --val-every-n-epochs "${VAL_EVERY_N_EPOCHS:-1}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-10}"
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH:-20}"
  --mask-decoder-lr-mult "${MASK_DECODER_LR_MULT:-1.0}"
  --no-mask-lr-mult 0.0
  --prompt-encoder-lr-mult 0.0
  --shape-prior-lr-mult "${SHAPE_PRIOR_LR_MULT:-1.0}"
  --densebr-lr-mult "${DENSEBR_LR_MULT:-1.0}"
)

if [ "${MAX_TRAIN_BATCHES:-0}" -gt 0 ]; then
  CMD+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi
if [ "${MAX_VAL_BATCHES:-0}" -gt 0 ]; then
  CMD+=(--max-val-batches "${MAX_VAL_BATCHES}")
fi
if [ -n "${INIT_FROM:-}" ]; then
  CMD+=(--init-from "${INIT_FROM}")
fi
if [ -n "${INIT_EXCLUDE_PREFIXES:-}" ]; then
  CMD+=(--init-exclude-prefixes "${INIT_EXCLUDE_PREFIXES}")
fi
if [ -n "${RESUME_FROM:-}" ]; then
  CMD+=(--resume-from "${RESUME_FROM}")
fi

echo "============================================================"
echo "ablation=${ABLATION_ID} neck=${NECK_TYPE} prompt=${PROMPT_ROUTE}"
echo "explicit_mode=${EXPECTED_EXPLICIT_MODE} densebr=${DENSEBR_ENABLED}"
echo "subset=train:${TRAIN_SUBSET_RATIO} val:${VAL_SUBSET_RATIO} seed:${SUBSET_SEED}"
echo "config=${CONFIG_PATH}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
printf 'command='
printf ' %q' "${CMD[@]}"
printf '\n'
echo "============================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
exec "${CMD[@]}"
