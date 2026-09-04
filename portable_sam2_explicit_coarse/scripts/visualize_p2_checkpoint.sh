#!/usr/bin/env bash
# Render P2BoundaryRefiner inference panels from a training checkpoint.
# Usage: bash scripts/visualize_p2_checkpoint.sh CHECKPOINT [extra visualize_p2_refiner.py args]
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

# Architecture env must mirror scripts/ablations/paper_promptminer_rd_p2_whu_full.sh
# so the recorded config reproduces the checkpoint's architecture contract.
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE="roi_local"
export P2_BOUNDARY_REFINER_PROJECTED_CHANNELS="${P2_BOUNDARY_REFINER_PROJECTED_CHANNELS:-64}"
export P2_BOUNDARY_REFINER_MID_CHANNELS="${P2_BOUNDARY_REFINER_MID_CHANNELS:-64}"
export P2_BOUNDARY_REFINER_LOSS_WEIGHT="${P2_BOUNDARY_REFINER_LOSS_WEIGHT:-0.05}"

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/visualize_p2_checkpoint.sh CHECKPOINT [extra args]" >&2
  exit 2
fi

CHECKPOINT="$1"
shift

exec "${PYTHON}" inference/probes/visualize_p2_refiner.py \
  --checkpoint "${CHECKPOINT}" \
  --config configs/whu1024_baseplus_explicit_coarse.py \
  "$@"
