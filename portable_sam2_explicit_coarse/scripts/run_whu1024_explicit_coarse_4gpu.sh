#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

export SAM2_MODEL_SIZE=base_plus
export SAM2_REPO="${SAM2_REPO:-$(cd "${PROJECT_ROOT}/../sam2" && pwd)}"
export SAM2_CKPT="${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}"
export WHU1024_DATA_ROOT="${WHU1024_DATA_ROOT:-/data/wangcheng/dataset/WHU}"
export MAX_EPOCHS="${MAX_EPOCHS:-80}"
export EXPLICIT_PROMPT_MODE="${EXPLICIT_PROMPT_MODE:-points_box_dense}"
export DENSEBR_ENABLED="${DENSEBR_ENABLED:-0}"
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/${EXPLICIT_PROMPT_MODE}_densebr${DENSEBR_ENABLED}_semanticfix_${SHAPE_CONTEXT_FUSION:-roi_only}}"
export RUN_TAG="${RUN_TAG:-explicit_${EXPLICIT_PROMPT_MODE}_densebr${DENSEBR_ENABLED}_semanticfix_${SHAPE_CONTEXT_FUSION:-roi_only}}"

case "${EXPLICIT_PROMPT_MODE}:${DENSEBR_ENABLED}" in
  points:0)
    WRAPPER="scripts/ablations/c2_pafpn_coarse_points.sh"
    ;;
  points_box:0)
    WRAPPER="scripts/ablations/c3_pafpn_coarse_points_box.sh"
    ;;
  points_box_dense:0)
    WRAPPER="scripts/ablations/c4_pafpn_coarse_points_box_dense.sh"
    ;;
  points_box_dense:1)
    WRAPPER="scripts/ablations/c5_pafpn_coarse_densebr.sh"
    ;;
  *)
    echo "Unsupported EXPLICIT_PROMPT_MODE/DENSEBR_ENABLED combination: ${EXPLICIT_PROMPT_MODE}/${DENSEBR_ENABLED}" >&2
    exit 2
    ;;
esac

exec bash "${WRAPPER}" "$@"
