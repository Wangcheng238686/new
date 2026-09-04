#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

export SAM2_MODEL_SIZE=base_plus
export EXPLICIT_PROMPT_MODE="${EXPLICIT_PROMPT_MODE:-points_box_dense}"
export SAM_IMAGE_EMBED_STRIDE="${SAM_IMAGE_EMBED_STRIDE:-16}"

# P2-BRR is meaningful only for the dense full row.  Preserve convenient
# point/point+box invocations while rejecting any explicit unsupported mix in
# the routing table below.
if [[ "${EXPLICIT_PROMPT_MODE}" == "points_box_dense" ]]; then
  export P2_BOUNDARY_REFINER_ENABLED="${P2_BOUNDARY_REFINER_ENABLED:-1}"
else
  export P2_BOUNDARY_REFINER_ENABLED="${P2_BOUNDARY_REFINER_ENABLED:-0}"
fi

case "${EXPLICIT_PROMPT_MODE}:${P2_BOUNDARY_REFINER_ENABLED}:${SAM_IMAGE_EMBED_STRIDE:-32}" in
  points:0:16)
    WRAPPER="scripts/ablations/whu_p2_matrix_point.sh"
    ;;
  points_box:0:16)
    WRAPPER="scripts/ablations/whu_p2_matrix_point_box.sh"
    ;;
  points_box_dense:0:16)
    WRAPPER="scripts/ablations/whu_p2_matrix_point_box_mask.sh"
    ;;
  points_box_dense:1:16)
    WRAPPER="scripts/ablations/whu_p2_matrix_full.sh"
    ;;
  *)
    echo "Unsupported prompt/refiner/stride combination: ${EXPLICIT_PROMPT_MODE}/${P2_BOUNDARY_REFINER_ENABLED}/${SAM_IMAGE_EMBED_STRIDE:-32}" >&2
    exit 2
    ;;
esac

exec bash "${WRAPPER}" "$@"
