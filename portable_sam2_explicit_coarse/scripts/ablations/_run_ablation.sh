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
MASTER_PORT="${MASTER_PORT:-29500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-0.1}"
VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
SUBSET_SEED="${SUBSET_SEED:-44}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
AMP="${AMP:-0}"
EMA_ENABLED="${EMA_ENABLED:-1}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_EVAL="${EMA_EVAL:-1}"
EMA_SAVE_BEST="${EMA_SAVE_BEST:-1}"
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-20}"
MASK_DECODER_LR_MULT="${MASK_DECODER_LR_MULT:-1.0}"
SHAPE_PRIOR_LR_MULT="${SHAPE_PRIOR_LR_MULT:-1.0}"
DENSEBR_LR_MULT="${DENSEBR_LR_MULT:-1.0}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
INIT_FROM="${INIT_FROM:-}"
INIT_EXCLUDE_PREFIXES="${INIT_EXCLUDE_PREFIXES:-}"
RESUME_FROM="${RESUME_FROM:-}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

RUN_TAG="${RUN_TAG:-${ABLATION_ID}}"
SUBSET_TAG="tr${TRAIN_SUBSET_RATIO}_va${VAL_SUBSET_RATIO}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/${RUN_TAG}_${SUBSET_TAG}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_SAFE_TAG="${RUN_TAG//\//_}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/ablations}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${LOG_SAFE_TAG}_${SUBSET_TAG}_${RUN_TIMESTAMP}_pid$$.log}"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "terminal_log=${LOG_FILE}"

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
  "--master_port=${MASTER_PORT}"
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
  --val-batch-size "${VAL_BATCH_SIZE}"
  --seed "${SUBSET_SEED}"
  --amp "${AMP}"
  --ema-enabled "${EMA_ENABLED}"
  --ema-decay "${EMA_DECAY}"
  --ema-eval "${EMA_EVAL}"
  --ema-save-best "${EMA_SAVE_BEST}"
  --val-every-n-epochs "${VAL_EVERY_N_EPOCHS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}"
  --mask-decoder-lr-mult "${MASK_DECODER_LR_MULT}"
  --no-mask-lr-mult 0.0
  --prompt-encoder-lr-mult 0.0
  --shape-prior-lr-mult "${SHAPE_PRIOR_LR_MULT}"
  --densebr-lr-mult "${DENSEBR_LR_MULT}"
)

if [ "${MAX_TRAIN_BATCHES}" -gt 0 ]; then
  CMD+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi
if [ "${MAX_VAL_BATCHES}" -gt 0 ]; then
  CMD+=(--max-val-batches "${MAX_VAL_BATCHES}")
fi
if [ -n "${INIT_FROM}" ]; then
  CMD+=(--init-from "${INIT_FROM}")
fi
if [ -n "${INIT_EXCLUDE_PREFIXES}" ]; then
  CMD+=(--init-exclude-prefixes "${INIT_EXCLUDE_PREFIXES}")
fi
if [ -n "${RESUME_FROM}" ]; then
  CMD+=(--resume-from "${RESUME_FROM}")
fi

GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
EFFECTIVE_GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))

echo "============================================================"
echo "resolved_hyperparameters_begin"
echo "timestamp=${RUN_TIMESTAMP}"
echo "git_commit=${GIT_COMMIT}"
echo "ablation_id=${ABLATION_ID}"
echo "run_tag=${RUN_TAG}"
echo "neck_type=${NECK_TYPE}"
echo "prompt_route=${PROMPT_ROUTE}"
echo "prompt_generator_mode=${PROMPT_GENERATOR_MODE}"
echo "prompt_sparse_mode=${PROMPT_SPARSE_MODE}"
echo "prompt_encoder_enabled=${PROMPT_ENCODER_ENABLED}"
echo "shape_prior_enabled=${SHAPE_PRIOR_ENABLED}"
echo "explicit_prompt_mode=${EXPECTED_EXPLICIT_MODE}"
echo "densebr_enabled=${DENSEBR_ENABLED}"
echo "train_subset_ratio=${TRAIN_SUBSET_RATIO}"
echo "val_subset_ratio=${VAL_SUBSET_RATIO}"
echo "val_batch_size=${VAL_BATCH_SIZE}"
echo "subset_seed=${SUBSET_SEED}"
echo "python=${PYTHON}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "master_port=${MASTER_PORT}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "batch_size_per_rank=${BATCH_SIZE}"
echo "grad_accum_steps=${GRAD_ACCUM_STEPS}"
echo "effective_global_batch_size=${EFFECTIVE_GLOBAL_BATCH_SIZE}"
echo "epochs=${MAX_EPOCHS}"
echo "learning_rate=${LEARNING_RATE}"
echo "amp=${AMP}"
echo "ema_enabled=${EMA_ENABLED}"
echo "ema_decay=${EMA_DECAY}"
echo "ema_eval=${EMA_EVAL}"
echo "ema_save_best=${EMA_SAVE_BEST}"
echo "val_every_n_epochs=${VAL_EVERY_N_EPOCHS}"
echo "early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
echo "early_stopping_start_epoch=${EARLY_STOPPING_START_EPOCH}"
echo "mask_decoder_lr_mult=${MASK_DECODER_LR_MULT}"
echo "no_mask_lr_mult=0.0"
echo "prompt_encoder_lr_mult=0.0"
echo "shape_prior_lr_mult=${SHAPE_PRIOR_LR_MULT}"
echo "densebr_lr_mult=${DENSEBR_LR_MULT}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "init_from=${INIT_FROM}"
echo "init_exclude_prefixes=${INIT_EXCLUDE_PREFIXES}"
echo "resume_from=${RESUME_FROM}"
echo "check_data=${CHECK_DATA:-0}"
echo "preflight_model=${PREFLIGHT_MODEL:-0}"
echo "dry_run=${DRY_RUN:-0}"
echo "sam2_repo=${SAM2_REPO}"
echo "sam2_checkpoint=${SAM2_CKPT}"
echo "data_root=${WHU1024_DATA_ROOT}"
echo "config=${CONFIG_PATH}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "log_file=${LOG_FILE}"
printf 'command='
printf ' %q' "${CMD[@]}"
printf '\n'
echo "resolved_hyperparameters_end"
echo "============================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
exec "${CMD[@]}"
