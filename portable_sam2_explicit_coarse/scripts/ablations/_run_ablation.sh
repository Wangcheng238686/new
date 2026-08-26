#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=../load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

: "${ABLATION_ID:?wrapper must set ABLATION_ID}"
: "${NECK_TYPE:?wrapper must set NECK_TYPE}"
: "${PROMPT_ROUTE:?wrapper must set PROMPT_ROUTE}"
: "${P2_BOUNDARY_REFINER_ENABLED:?wrapper must set P2_BOUNDARY_REFINER_ENABLED}"

# Launcher-only options are accepted by every outer wrapper because each one
# forwards its arguments here. Keep all other options for the trainer.
CLI_MASTER_PORT=""
TRAINER_EXTRA_ARGS=()
while (($# > 0)); do
  case "$1" in
    --master-port)
      if (($# < 2)); then
        echo "--master-port requires a value" >&2
        exit 2
      fi
      CLI_MASTER_PORT="$2"
      shift 2
      ;;
    --master-port=*)
      CLI_MASTER_PORT="${1#*=}"
      shift
      ;;
    --)
      shift
      TRAINER_EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      TRAINER_EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

case "${NECK_TYPE}" in
  aggregator|pafpn) ;;
  *) echo "NECK_TYPE must be aggregator or pafpn, got ${NECK_TYPE}" >&2; exit 2 ;;
esac

ROI_SAM_ENABLED="${ROI_SAM_ENABLED:-0}"
case "${ROI_SAM_ENABLED}" in
  0|1) ;;
  *) echo "ROI_SAM_ENABLED must be 0 or 1, got ${ROI_SAM_ENABLED}" >&2; exit 2 ;;
esac
ROI_SAM_SAMPLING_RATIO="${ROI_SAM_SAMPLING_RATIO:-2}"
case "${ROI_SAM_SAMPLING_RATIO}" in
  ''|*[!0-9]*) echo "ROI_SAM_SAMPLING_RATIO must be >= 0" >&2; exit 2 ;;
esac

case "${PROMPT_ROUTE}" in
  mlp)
    if [[ "${ROI_SAM_ENABLED}" != "0" ]]; then
      echo "ROI-SAM is only supported by the explicit coarse route" >&2
      exit 2
    fi
    CONFIG_PATH="configs/whu1024_baseplus_clean.py"
    EXPECTED_EXPLICIT_MODE="none"
    export PROMPT_GENERATOR_MODE="rsprompter_mlp"
    export PROMPT_SPARSE_MODE="point"
    export PROMPT_ENCODER_ENABLED=0
    export SHAPE_PRIOR_ENABLED=0
    export EXPLICIT_PROMPT_MODE="none"
    export P2_BOUNDARY_REFINER_ENABLED=0
    FINAL_MASK_COORDINATE_MODE="${FINAL_MASK_COORDINATE_MODE:-roi_local}"
    case "${FINAL_MASK_COORDINATE_MODE}" in
      roi_local|full_image) ;;
      *) echo "invalid FINAL_MASK_COORDINATE_MODE=${FINAL_MASK_COORDINATE_MODE}" >&2; exit 2 ;;
    esac
    DEFAULT_PROMPT_DEBUG_STATS=0
    ;;
  coarse)
    CONFIG_PATH="configs/whu1024_baseplus_explicit_coarse.py"
    # A wrapper may swap in a config variant (e.g. densefix) that differs only
    # in frozen model-architecture knobs. Unset by default, so every existing
    # coarse experiment keeps its committed config path unchanged.
    if [[ -n "${CONFIG_OVERRIDE:-}" ]]; then
      CONFIG_PATH="${CONFIG_OVERRIDE}"
    fi
    : "${EXPLICIT_PROMPT_MODE:?coarse wrapper must set EXPLICIT_PROMPT_MODE}"
    case "${EXPLICIT_PROMPT_MODE}" in
      points|box|mask|points_box|points_box_dense) ;;
      *) echo "invalid EXPLICIT_PROMPT_MODE=${EXPLICIT_PROMPT_MODE}" >&2; exit 2 ;;
    esac
    EXPECTED_EXPLICIT_MODE="${EXPLICIT_PROMPT_MODE}"
    export PROMPT_GENERATOR_MODE="explicit_mask"
    export PROMPT_SPARSE_MODE="shape_point"
    export PROMPT_ENCODER_ENABLED=1
    export SHAPE_PRIOR_ENABLED=1
    if [[ "${ROI_SAM_ENABLED}" == "1" ]]; then
      if [[ "${EXPLICIT_PROMPT_MODE}" != "points" ]]; then
        echo "ROI-SAM is a strict points-only C2 variant" >&2
        exit 2
      fi
      FINAL_MASK_COORDINATE_MODE="roi_local"
    else
      FINAL_MASK_COORDINATE_MODE="${FINAL_MASK_COORDINATE_MODE:-roi_local}"
    fi
    DEFAULT_PROMPT_DEBUG_STATS=1
    ;;
  *) echo "PROMPT_ROUTE must be mlp or coarse, got ${PROMPT_ROUTE}" >&2; exit 2 ;;
esac

export NECK_TYPE
export FINAL_MASK_COORDINATE_MODE
export ROI_SAM_ENABLED ROI_SAM_SAMPLING_RATIO
SHAPE_DENSE_TRANSFORM="${SHAPE_DENSE_TRANSFORM:-raw_logits}"
case "${SHAPE_DENSE_TRANSFORM}" in
  raw_logits|confidence_signed|gaussian_edt) ;;
  *) echo "invalid SHAPE_DENSE_TRANSFORM=${SHAPE_DENSE_TRANSFORM}" >&2; exit 2 ;;
esac
SHAPE_DENSE_DETACH="${SHAPE_DENSE_DETACH:-0}"
case "${SHAPE_DENSE_DETACH}" in
  0|1) ;;
  *) echo "SHAPE_DENSE_DETACH must be 0 or 1" >&2; exit 2 ;;
esac
SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD="${SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD:-0.5}"
SHAPE_GAUSSIAN_OMEGA="${SHAPE_GAUSSIAN_OMEGA:-15.0}"
SHAPE_GAUSSIAN_GAMMA="${SHAPE_GAUSSIAN_GAMMA:-4.0}"
if [[ "${SHAPE_DENSE_TRANSFORM}" == "gaussian_edt" && "${SHAPE_DENSE_DETACH}" != "1" ]]; then
  echo "gaussian_edt requires SHAPE_DENSE_DETACH=1" >&2
  exit 2
fi
export SHAPE_DENSE_TRANSFORM SHAPE_DENSE_DETACH
export SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD SHAPE_GAUSSIAN_OMEGA SHAPE_GAUSSIAN_GAMMA
SAM_IMAGE_EMBED_STRIDE="${SAM_IMAGE_EMBED_STRIDE:-32}"
case "${SAM_IMAGE_EMBED_STRIDE}" in
  16|32) ;;
  *)
    echo "SAM_IMAGE_EMBED_STRIDE must be 16 or 32, got ${SAM_IMAGE_EMBED_STRIDE}" >&2
    exit 2
    ;;
esac
export SAM_IMAGE_EMBED_STRIDE
export SAM2_MODEL_SIZE=base_plus
export MAX_EPOCHS="${MAX_EPOCHS:-80}"
if [ -n "${CLI_MASTER_PORT}" ]; then
  MASTER_PORT="${CLI_MASTER_PORT}"
  MASTER_PORT_SOURCE="cli"
elif [ -n "${MASTER_PORT:-}" ]; then
  MASTER_PORT_SOURCE="environment"
else
  # Avoid collisions when several detached experiments are launched from the
  # same host. The active PID range makes concurrent wrappers resolve to
  # different ports; users can still pin MASTER_PORT explicitly.
  MASTER_PORT="$((20000 + ($$ % 20000)))"
  MASTER_PORT_SOURCE="auto_pid"
fi
case "${MASTER_PORT}" in
  ''|*[!0-9]*)
    echo "MASTER_PORT must be an integer in [1, 65535], got ${MASTER_PORT}" >&2
    exit 2
    ;;
esac
if ((MASTER_PORT < 1 || MASTER_PORT > 65535)); then
  echo "MASTER_PORT must be in [1, 65535], got ${MASTER_PORT}" >&2
  exit 2
fi
if [ -n "${TORCH_DDP_TIMEOUT_SECONDS:-}" ]; then
  DDP_TIMEOUT_SOURCE="TORCH_DDP_TIMEOUT_SECONDS"
elif [ -n "${NCCL_TIMEOUT:-}" ]; then
  TORCH_DDP_TIMEOUT_SECONDS="${NCCL_TIMEOUT}"
  DDP_TIMEOUT_SOURCE="NCCL_TIMEOUT"
else
  TORCH_DDP_TIMEOUT_SECONDS="1800"
  DDP_TIMEOUT_SOURCE="baseline_default"
fi
case "${TORCH_DDP_TIMEOUT_SECONDS}" in
  ''|*[!0-9]*)
    echo "TORCH_DDP_TIMEOUT_SECONDS must be a positive integer, got ${TORCH_DDP_TIMEOUT_SECONDS}" >&2
    exit 2
    ;;
esac
if ((TORCH_DDP_TIMEOUT_SECONDS < 1)); then
  echo "TORCH_DDP_TIMEOUT_SECONDS must be >= 1, got ${TORCH_DDP_TIMEOUT_SECONDS}" >&2
  exit 2
fi
export TORCH_DDP_TIMEOUT_SECONDS
TORCH_DDP_CONTROL_TIMEOUT_SECONDS="${TORCH_DDP_CONTROL_TIMEOUT_SECONDS:-86400}"
case "${TORCH_DDP_CONTROL_TIMEOUT_SECONDS}" in
  ''|*[!0-9]*)
    echo "TORCH_DDP_CONTROL_TIMEOUT_SECONDS must be a positive integer, got ${TORCH_DDP_CONTROL_TIMEOUT_SECONDS}" >&2
    exit 2
    ;;
esac
if ((TORCH_DDP_CONTROL_TIMEOUT_SECONDS < 1)); then
  echo "TORCH_DDP_CONTROL_TIMEOUT_SECONDS must be >= 1, got ${TORCH_DDP_CONTROL_TIMEOUT_SECONDS}" >&2
  exit 2
fi
export TORCH_DDP_CONTROL_TIMEOUT_SECONDS
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
SUBSET_SEED="${SUBSET_SEED:-44}"
EMA_ENABLED="${EMA_ENABLED:-0}"
EMA_DECAY="${EMA_DECAY:-0.999}"
EMA_EVAL="${EMA_EVAL:-${EMA_ENABLED}}"
EMA_SAVE_BEST="${EMA_SAVE_BEST:-${EMA_ENABLED}}"
for ema_flag_name in EMA_ENABLED EMA_EVAL EMA_SAVE_BEST; do
  ema_flag_value="${!ema_flag_name}"
  if [[ "${ema_flag_value}" != "0" && "${ema_flag_value}" != "1" ]]; then
    echo "${ema_flag_name} must be 0 or 1, got ${ema_flag_value}" >&2
    exit 2
  fi
done
if [[ "${EMA_ENABLED}" == "0" && ("${EMA_EVAL}" != "0" || "${EMA_SAVE_BEST}" != "0") ]]; then
  echo "EMA_EVAL and EMA_SAVE_BEST must be 0 when EMA_ENABLED=0" >&2
  exit 2
fi
VAL_EVERY_N_EPOCHS="${VAL_EVERY_N_EPOCHS:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
COMPUTE_VAL_LOSS="${COMPUTE_VAL_LOSS:-0}"
if [[ "${COMPUTE_VAL_LOSS}" != "0" && "${COMPUTE_VAL_LOSS}" != "1" ]]; then
  echo "COMPUTE_VAL_LOSS must be 0 or 1, got ${COMPUTE_VAL_LOSS}" >&2
  exit 2
fi
# `${VAR-...}` without the colon: an explicitly exported empty string means
# "disable the bbox-best checkpoint" (best-only retention wrappers); only a
# completely unset variable falls back to the default metric.
SAVE_BBOX_BEST_METRIC="${SAVE_BBOX_BEST_METRIC-bbox/mAP}"
SEGM_SCORE_MODE="${SEGM_SCORE_MODE:-detector}"
case "${SEGM_SCORE_MODE}" in
  detector|mask_quality) ;;
  *)
    echo "SEGM_SCORE_MODE must be detector or mask_quality, got ${SEGM_SCORE_MODE}" >&2
    exit 2
    ;;
esac
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_START_EPOCH="${EARLY_STOPPING_START_EPOCH:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-5e-4}"
EARLY_STOPPING_SMOOTH_WINDOW="${EARLY_STOPPING_SMOOTH_WINDOW:-5}"
SAT_BACKBONE_LR_MULT="${SAT_BACKBONE_LR_MULT:-1.0}"
SAT_OTHER_LR_MULT="${SAT_OTHER_LR_MULT:-1.0}"
MASK_DECODER_LR_MULT="${MASK_DECODER_LR_MULT:-1.0}"
NO_MASK_LR_MULT="${NO_MASK_LR_MULT:-1.0}"
PROMPT_ENCODER_LR_MULT="${PROMPT_ENCODER_LR_MULT:-0.0}"
PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING="${PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING:-${UNFREEZE_MASK_DOWNSCALING:-0}}"
case "${PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING}" in
  0|1) ;;
  *) echo "PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING must be 0 or 1, got ${PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING}" >&2; exit 2 ;;
esac
SHAPE_PRIOR_LR_MULT="${SHAPE_PRIOR_LR_MULT:-1.0}"
SHAPE_CONTEXT_FUSION="${SHAPE_CONTEXT_FUSION:-roi_only}"
case "${SHAPE_CONTEXT_FUSION}" in
  roi_only|gated_spatial_film|legacy_multiplicative) ;;
  *)
    echo "SHAPE_CONTEXT_FUSION must be roi_only, gated_spatial_film or legacy_multiplicative, got ${SHAPE_CONTEXT_FUSION}" >&2
    exit 2
    ;;
esac
P2_BOUNDARY_REFINER_LR_MULT="${P2_BOUNDARY_REFINER_LR_MULT:-1.0}"
P2_BOUNDARY_REFINER_PROJECTED_CHANNELS="${P2_BOUNDARY_REFINER_PROJECTED_CHANNELS:-64}"
P2_BOUNDARY_REFINER_MID_CHANNELS="${P2_BOUNDARY_REFINER_MID_CHANNELS:-64}"
P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX="${P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX:-2.0}"
P2_BOUNDARY_REFINER_LOSS_WEIGHT="${P2_BOUNDARY_REFINER_LOSS_WEIGHT:-0.05}"
ALLOW_CROSS_ARCH_INIT="${ALLOW_CROSS_ARCH_INIT:-0}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
DET_LOSS_STAGE1_END="${DET_LOSS_STAGE1_END:-5}"
DET_LOSS_STAGE2_END="${DET_LOSS_STAGE2_END:-10}"
DET_LOSS_WEIGHT_STAGE1="${DET_LOSS_WEIGHT_STAGE1:-1.0}"
DET_LOSS_WEIGHT_STAGE2="${DET_LOSS_WEIGHT_STAGE2:-1.0}"
DET_LOSS_WEIGHT_STAGE3="${DET_LOSS_WEIGHT_STAGE3:-1.0}"
EMA_UPDATE_EVERY="${EMA_UPDATE_EVERY:-1}"
EMA_EVAL_START_EPOCH="${EMA_EVAL_START_EPOCH:-5}"
SHAPE_POINT_ADAPTIVE_VALIDITY="${SHAPE_POINT_ADAPTIVE_VALIDITY:-1}"
SHAPE_LOSS_SCHEDULE_MODE="${SHAPE_LOSS_SCHEDULE_MODE:-fixed}"
SHAPE_PRIOR_LOSS_WEIGHT="${SHAPE_PRIOR_LOSS_WEIGHT:-0.10}"
FINAL_MASK_LOSS_MODE="${FINAL_MASK_LOSS_MODE:-standard}"
case "${FINAL_MASK_LOSS_MODE}" in
  standard|roi_balanced_dice) ;;
  *) echo "FINAL_MASK_LOSS_MODE must be standard or roi_balanced_dice, got ${FINAL_MASK_LOSS_MODE}" >&2; exit 2 ;;
esac
FINAL_MASK_ROI_EXPAND_RATIO="${FINAL_MASK_ROI_EXPAND_RATIO:-1.20}"
FINAL_MASK_ROI_BCE_WEIGHT="${FINAL_MASK_ROI_BCE_WEIGHT:-1.0}"
FINAL_MASK_ROI_DICE_WEIGHT="${FINAL_MASK_ROI_DICE_WEIGHT:-1.0}"
FINAL_MASK_OUTSIDE_BCE_WEIGHT="${FINAL_MASK_OUTSIDE_BCE_WEIGHT:-0.05}"
SHAPE_LOSS_STAGE1_END="${SHAPE_LOSS_STAGE1_END:-5}"
SHAPE_LOSS_WEIGHT_STAGE1="${SHAPE_LOSS_WEIGHT_STAGE1:-0.20}"
SHAPE_LOSS_WEIGHT_STAGE2="${SHAPE_LOSS_WEIGHT_STAGE2:-0.10}"
POINT_WARMUP_ENABLED="${POINT_WARMUP_ENABLED:-0}"
POINT_WARMUP_NO_POINT_EPOCHS="${POINT_WARMUP_NO_POINT_EPOCHS:-0}"
POINT_WARMUP_ONE_PAIR_EPOCHS="${POINT_WARMUP_ONE_PAIR_EPOCHS:-0}"
POINT_WARMUP_FULL_START_EPOCH="${POINT_WARMUP_FULL_START_EPOCH:-$((POINT_WARMUP_NO_POINT_EPOCHS + POINT_WARMUP_ONE_PAIR_EPOCHS + 1))}"
PROMPT_DEBUG_STATS="${PROMPT_DEBUG_STATS:-${DEFAULT_PROMPT_DEBUG_STATS}}"
PROMPT_DEBUG_STATS_INTERVAL="${PROMPT_DEBUG_STATS_INTERVAL:-50}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
INIT_FROM="${INIT_FROM:-}"
INIT_EXCLUDE_PREFIXES="${INIT_EXCLUDE_PREFIXES:-}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"
export CUDA_VISIBLE_DEVICES
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export SHAPE_POINT_ADAPTIVE_VALIDITY
export SHAPE_CONTEXT_FUSION
export SHAPE_LOSS_SCHEDULE_MODE
export SHAPE_PRIOR_LOSS_WEIGHT
export FINAL_MASK_LOSS_MODE FINAL_MASK_ROI_EXPAND_RATIO
export FINAL_MASK_ROI_BCE_WEIGHT FINAL_MASK_ROI_DICE_WEIGHT
export FINAL_MASK_OUTSIDE_BCE_WEIGHT
export SHAPE_LOSS_STAGE1_END
export SHAPE_LOSS_WEIGHT_STAGE1
export SHAPE_LOSS_WEIGHT_STAGE2
export POINT_WARMUP_ENABLED
export POINT_WARMUP_NO_POINT_EPOCHS
export POINT_WARMUP_ONE_PAIR_EPOCHS
export POINT_WARMUP_FULL_START_EPOCH
export P2_BOUNDARY_REFINER_ENABLED P2_BOUNDARY_REFINER_PROJECTED_CHANNELS
export P2_BOUNDARY_REFINER_MID_CHANNELS P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX
export P2_BOUNDARY_REFINER_LOSS_WEIGHT
export SAVE_BBOX_BEST_METRIC
export SEGM_SCORE_MODE
export PROMPT_ENCODER_LR_MULT
export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING

RUN_TAG="${RUN_TAG:-${ABLATION_ID}}"
EXPECTED_ARCHITECTURE_ID="${EXPECTED_ARCHITECTURE_ID:-${ABLATION_ID}}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
if [[ -n "${RUN_SUFFIX}" ]]; then
  RUN_TAG="${RUN_TAG}_${RUN_SUFFIX}"
fi
SUBSET_TAG="tr${TRAIN_SUBSET_RATIO}_va${VAL_SUBSET_RATIO}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/${RUN_TAG}_${SUBSET_TAG}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_SAFE_TAG="${RUN_TAG//\//_}"
LOG_DIR="${LOG_DIR:-${PORTABLE_SAM2_LOG_ROOT}/ablations}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  # Smoke/preflight output stays on the invoking terminal.  Do not create one
  # parameter-only file per wrapper under the real experiment log directory.
  LOG_FILE=""
  echo "terminal_log=disabled(dry_run)"
else
  LOG_FILE="${LOG_FILE:-${LOG_DIR}/${LOG_SAFE_TAG}_${SUBSET_TAG}_${RUN_TIMESTAMP}_pid$$.log}"
  mkdir -p "$(dirname "${LOG_FILE}")"
  exec > >(tee -a "${LOG_FILE}") 2>&1
  echo "terminal_log=${LOG_FILE}"
fi

VALIDATE_ARGS=(
  --config "${CONFIG_PATH}"
  --ablation-id "${ABLATION_ID}"
  --expected-architecture-id "${EXPECTED_ARCHITECTURE_ID}"
  --expected-neck "${NECK_TYPE}"
  --prompt-route "${PROMPT_ROUTE}"
  --explicit-prompt-mode "${EXPECTED_EXPLICIT_MODE}"
  --p2-boundary-refiner-enabled "${P2_BOUNDARY_REFINER_ENABLED}"
  --expected-final-mask-mode "${FINAL_MASK_COORDINATE_MODE}"
  --expected-final-mask-loss-mode "${FINAL_MASK_LOSS_MODE}"
  --expected-roi-sam-enabled "${ROI_SAM_ENABLED}"
  --expected-image-embed-stride "${SAM_IMAGE_EMBED_STRIDE}"
  --expected-segm-score-mode "${SEGM_SCORE_MODE}"
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
  --compute-val-loss "${COMPUTE_VAL_LOSS}"
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
  --early-stopping-start-epoch "${EARLY_STOPPING_START_EPOCH}"
  --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
  --early-stopping-smooth-window "${EARLY_STOPPING_SMOOTH_WINDOW}"
  --sat-backbone-lr-mult "${SAT_BACKBONE_LR_MULT}"
  --sat-other-lr-mult "${SAT_OTHER_LR_MULT}"
  --mask-decoder-lr-mult "${MASK_DECODER_LR_MULT}"
  --no-mask-lr-mult "${NO_MASK_LR_MULT}"
  --prompt-encoder-lr-mult "${PROMPT_ENCODER_LR_MULT}"
  --shape-prior-lr-mult "${SHAPE_PRIOR_LR_MULT}"
  --p2-boundary-refiner-lr-mult "${P2_BOUNDARY_REFINER_LR_MULT}"
  --warmup-iters "${WARMUP_ITERS}"
  --weight-decay "${WEIGHT_DECAY}"
  --det-loss-stage1-end "${DET_LOSS_STAGE1_END}"
  --det-loss-stage2-end "${DET_LOSS_STAGE2_END}"
  --det-loss-weight-stage1 "${DET_LOSS_WEIGHT_STAGE1}"
  --det-loss-weight-stage2 "${DET_LOSS_WEIGHT_STAGE2}"
  --det-loss-weight-stage3 "${DET_LOSS_WEIGHT_STAGE3}"
  --ema-update-every "${EMA_UPDATE_EVERY}"
  --ema-eval-start-epoch "${EMA_EVAL_START_EPOCH}"
  --prompt-debug-stats "${PROMPT_DEBUG_STATS}"
  --prompt-debug-stats-interval "${PROMPT_DEBUG_STATS_INTERVAL}"
)

if [ "${MAX_TRAIN_BATCHES}" -gt 0 ]; then
  CMD+=(--max-train-batches "${MAX_TRAIN_BATCHES}")
fi
if [ "${MAX_VAL_BATCHES}" -gt 0 ]; then
  CMD+=(--max-val-batches "${MAX_VAL_BATCHES}")
fi
if [ -n "${INIT_FROM}" ]; then
  CMD+=(--init-from "${INIT_FROM}")
  if [ "${ALLOW_CROSS_ARCH_INIT}" = "1" ]; then
    CMD+=(--allow-cross-arch-init)
  fi
fi
if [ -n "${INIT_EXCLUDE_PREFIXES}" ]; then
  CMD+=(--init-exclude-prefixes "${INIT_EXCLUDE_PREFIXES}")
fi
if [ -n "${RESUME_FROM}" ]; then
  CMD+=(--resume-from "${RESUME_FROM}")
fi
CMD+=("${TRAINER_EXTRA_ARGS[@]}")

GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
# HEAD alone cannot identify the executed code when the working tree is dirty
# (the resumed paper run executed 5fff72f + uncommitted edits). Record the
# dirty flag, the modified/untracked code files, and a diff fingerprint so a
# log line uniquely pins the exact tree state.
GIT_STATUS_SHORT="$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null || true)"
GIT_DIRTY="$( [[ -n "${GIT_STATUS_SHORT}" ]] && echo 1 || echo 0 )"
# The grep must not fail under `set -o pipefail` when the tree is clean
# (empty status -> no matches -> grep exit 1 -> silent errexit killed the
# runner right after contract validation on the first clean-tree launch).
GIT_DIRTY_FILES="$(echo "${GIT_STATUS_SHORT}" | awk '{print $2}' | grep -E '\.(py|sh|md)$' | paste -sd, - | cut -c1-400 || true)"
GIT_DIFF_SHA="$(git -C "${PROJECT_ROOT}" diff HEAD 2>/dev/null | sha256sum | cut -c1-16)"
[[ -z "${GIT_DIFF_SHA}" ]] && GIT_DIFF_SHA=none
EFFECTIVE_GLOBAL_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NPROC_PER_NODE))

echo "============================================================"
echo "resolved_hyperparameters_begin"
echo "timestamp=${RUN_TIMESTAMP}"
echo "environment_config=${PORTABLE_SAM2_ENV_FILE}"
echo "git_commit=${GIT_COMMIT}"
echo "git_dirty=${GIT_DIRTY}"
echo "git_dirty_code_files=${GIT_DIRTY_FILES:-none}"
echo "git_diff_sha16=${GIT_DIFF_SHA}"
echo "ablation_id=${ABLATION_ID}"
echo "expected_architecture_id=${EXPECTED_ARCHITECTURE_ID}"
echo "run_tag=${RUN_TAG}"
echo "run_suffix=${RUN_SUFFIX}"
echo "neck_type=${NECK_TYPE}"
echo "prompt_route=${PROMPT_ROUTE}"
echo "prompt_generator_mode=${PROMPT_GENERATOR_MODE}"
echo "prompt_sparse_mode=${PROMPT_SPARSE_MODE}"
echo "prompt_encoder_enabled=${PROMPT_ENCODER_ENABLED}"
echo "shape_prior_enabled=${SHAPE_PRIOR_ENABLED}"
echo "explicit_prompt_mode=${EXPECTED_EXPLICIT_MODE}"
echo "shape_dense_transform=${SHAPE_DENSE_TRANSFORM}"
echo "shape_dense_detach=${SHAPE_DENSE_DETACH}"
echo "shape_gaussian_foreground_threshold=${SHAPE_GAUSSIAN_FOREGROUND_THRESHOLD}"
echo "shape_gaussian_omega=${SHAPE_GAUSSIAN_OMEGA}"
echo "shape_gaussian_gamma=${SHAPE_GAUSSIAN_GAMMA}"
echo "p2_boundary_refiner_enabled=${P2_BOUNDARY_REFINER_ENABLED}"
echo "final_mask_coordinate_mode=${FINAL_MASK_COORDINATE_MODE}"
echo "final_mask_loss_mode=${FINAL_MASK_LOSS_MODE}"
echo "final_mask_roi_expand_ratio=${FINAL_MASK_ROI_EXPAND_RATIO}"
echo "final_mask_roi_bce_weight=${FINAL_MASK_ROI_BCE_WEIGHT}"
echo "final_mask_roi_dice_weight=${FINAL_MASK_ROI_DICE_WEIGHT}"
echo "final_mask_outside_bce_weight=${FINAL_MASK_OUTSIDE_BCE_WEIGHT}"
echo "roi_sam_enabled=${ROI_SAM_ENABLED}"
echo "roi_sam_sampling_ratio=${ROI_SAM_SAMPLING_RATIO}"
echo "sam_image_embedding_stride=${SAM_IMAGE_EMBED_STRIDE}"
echo "sam_image_embedding_size=$((1024 / SAM_IMAGE_EMBED_STRIDE))"
echo "train_subset_ratio=${TRAIN_SUBSET_RATIO}"
echo "val_subset_ratio=${VAL_SUBSET_RATIO}"
echo "val_batch_size=${VAL_BATCH_SIZE}"
echo "subset_seed=${SUBSET_SEED}"
echo "python=${PYTHON}"
echo "nproc_per_node=${NPROC_PER_NODE}"
echo "master_port=${MASTER_PORT}"
echo "master_port_source=${MASTER_PORT_SOURCE}"
echo "ddp_timeout_seconds=${TORCH_DDP_TIMEOUT_SECONDS}"
echo "ddp_timeout_source=${DDP_TIMEOUT_SOURCE}"
echo "ddp_control_backend=gloo"
echo "ddp_control_timeout_seconds=${TORCH_DDP_CONTROL_TIMEOUT_SECONDS}"
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
echo "compute_val_loss=${COMPUTE_VAL_LOSS}"
echo "save_bbox_best_metric=${SAVE_BBOX_BEST_METRIC}"
echo "segm_score_mode=${SEGM_SCORE_MODE}"
echo "early_stopping_patience=${EARLY_STOPPING_PATIENCE}"
echo "early_stopping_start_epoch=${EARLY_STOPPING_START_EPOCH}"
echo "early_stopping_min_delta=${EARLY_STOPPING_MIN_DELTA}"
echo "early_stopping_smooth_window=${EARLY_STOPPING_SMOOTH_WINDOW}"
echo "sat_backbone_lr_mult=${SAT_BACKBONE_LR_MULT}"
echo "sat_other_lr_mult=${SAT_OTHER_LR_MULT}"
echo "mask_decoder_lr_mult=${MASK_DECODER_LR_MULT}"
echo "no_mask_lr_mult=${NO_MASK_LR_MULT}"
echo "prompt_encoder_lr_mult=${PROMPT_ENCODER_LR_MULT}"
echo "prompt_encoder_train_mask_downscaling=${PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING}"
echo "shape_prior_lr_mult=${SHAPE_PRIOR_LR_MULT}"
echo "shape_context_fusion=${SHAPE_CONTEXT_FUSION}"
echo "p2_boundary_refiner_lr_mult=${P2_BOUNDARY_REFINER_LR_MULT}"
echo "p2_boundary_refiner_projected_channels=${P2_BOUNDARY_REFINER_PROJECTED_CHANNELS}"
echo "p2_boundary_refiner_mid_channels=${P2_BOUNDARY_REFINER_MID_CHANNELS}"
echo "p2_boundary_refiner_delta_logit_max=${P2_BOUNDARY_REFINER_DELTA_LOGIT_MAX}"
echo "p2_boundary_refiner_loss_weight=${P2_BOUNDARY_REFINER_LOSS_WEIGHT}"
echo "allow_cross_arch_init=${ALLOW_CROSS_ARCH_INIT}"
echo "warmup_iters=${WARMUP_ITERS}"
echo "weight_decay=${WEIGHT_DECAY}"
echo "det_loss_stage1_end=${DET_LOSS_STAGE1_END}"
echo "det_loss_stage2_end=${DET_LOSS_STAGE2_END}"
echo "det_loss_weight_stage1=${DET_LOSS_WEIGHT_STAGE1}"
echo "det_loss_weight_stage2=${DET_LOSS_WEIGHT_STAGE2}"
echo "det_loss_weight_stage3=${DET_LOSS_WEIGHT_STAGE3}"
echo "ema_update_every=${EMA_UPDATE_EVERY}"
echo "ema_eval_start_epoch=${EMA_EVAL_START_EPOCH}"
echo "shape_point_adaptive_validity=${SHAPE_POINT_ADAPTIVE_VALIDITY}"
echo "shape_loss_schedule_mode=${SHAPE_LOSS_SCHEDULE_MODE}"
echo "shape_prior_loss_weight=${SHAPE_PRIOR_LOSS_WEIGHT}"
echo "shape_loss_stage1_end=${SHAPE_LOSS_STAGE1_END}"
echo "shape_loss_weight_stage1=${SHAPE_LOSS_WEIGHT_STAGE1}"
echo "shape_loss_weight_stage2=${SHAPE_LOSS_WEIGHT_STAGE2}"
echo "point_warmup_enabled=${POINT_WARMUP_ENABLED}"
echo "point_warmup_no_point_epochs=${POINT_WARMUP_NO_POINT_EPOCHS}"
echo "point_warmup_one_pair_epochs=${POINT_WARMUP_ONE_PAIR_EPOCHS}"
echo "point_warmup_full_start_epoch=${POINT_WARMUP_FULL_START_EPOCH}"
echo "prompt_debug_stats=${PROMPT_DEBUG_STATS}"
echo "prompt_debug_stats_interval=${PROMPT_DEBUG_STATS_INTERVAL}"
echo "max_train_batches=${MAX_TRAIN_BATCHES}"
echo "max_val_batches=${MAX_VAL_BATCHES}"
echo "init_from=${INIT_FROM}"
echo "init_exclude_prefixes=${INIT_EXCLUDE_PREFIXES}"
echo "resume_from=${RESUME_FROM}"
echo "check_data=${CHECK_DATA:-0}"
echo "preflight_model=${PREFLIGHT_MODEL:-0}"
echo "dry_run=${DRY_RUN:-0}"
echo "run_in_background=${RUN_IN_BACKGROUND}"
echo "sam2_repo=${SAM2_REPO}"
echo "sam2_checkpoint=${SAM2_CKPT}"
echo "data_root=${WHU1024_DATA_ROOT}"
echo "checkpoint_root=${PORTABLE_SAM2_CHECKPOINT_ROOT}"
echo "log_root=${PORTABLE_SAM2_LOG_ROOT}"
echo "tmp_root=${PORTABLE_SAM2_TMP_ROOT}"
echo "config=${CONFIG_PATH}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "log_file=${LOG_FILE:-disabled(dry_run)}"
printf 'command='
printf ' %q' "${CMD[@]}"
printf '\n'
echo "resolved_hyperparameters_end"
echo "============================================================"

if [ "${DRY_RUN:-0}" = "1" ]; then
  exit 0
fi

case "${RUN_IN_BACKGROUND}" in
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
    exit 0
    ;;
  *)
    echo "RUN_IN_BACKGROUND must be 0 or 1, got ${RUN_IN_BACKGROUND}" >&2
    exit 2
    ;;
esac
