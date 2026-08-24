#!/usr/bin/env bash
# Single-content PromptEncoder x P2 refiner ablation initialized from the
# converged R1-C4-RD ROI-local segm-best checkpoint.
set -Eeuo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: bash scripts/ablations/prompt_content_p2_from_r1_best.sh POINT|BOX|MASK 0|1 [launcher args]" >&2
  exit 2
fi

CONTENT="${1,,}"
P2_ENABLED="$2"
shift 2
case "${CONTENT}" in
  point) EXPLICIT_MODE="points" ;;
  box) EXPLICIT_MODE="box" ;;
  mask) EXPLICIT_MODE="mask" ;;
  *) echo "prompt content must be point, box or mask, got ${CONTENT}" >&2; exit 2 ;;
esac
case "${P2_ENABLED}" in
  0|1) ;;
  *) echo "P2 switch must be 0 or 1, got ${P2_ENABLED}" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

R1_BEST_DEFAULT="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/r1_c4_rd_pafpn_coarse_points_box_raw_detach_roi_local_emb64_tr0.2_va1.0/best_model_epoch80.pth"
export INIT_FROM="${INIT_FROM:-${R1_BEST_DEFAULT}}"
if [[ ! -f "${INIT_FROM}" ]]; then
  echo "R1-C4-RD initialization checkpoint not found: ${INIT_FROM}" >&2
  exit 2
fi

export ABLATION_ID="pafpn_coarse_${EXPLICIT_MODE}_emb64_p2br${P2_ENABLED}"
export EXPECTED_ARCHITECTURE_ID="${ABLATION_ID}"
export RUN_TAG="${RUN_TAG:-prompt_content_${CONTENT}_p2_${P2_ENABLED}_from_r1_best}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="${EXPLICIT_MODE}"
export P2_BOUNDARY_REFINER_ENABLED="${P2_ENABLED}"
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE="roi_local"
export ALLOW_CROSS_ARCH_INIT=1

exec bash "${SCRIPT_DIR}/_run_ablation.sh" "$@"
