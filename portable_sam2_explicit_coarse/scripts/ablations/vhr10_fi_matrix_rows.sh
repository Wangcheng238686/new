#!/usr/bin/env bash
# NWPU fi ablation-matrix row entries (prep only; launch awaits the plan
# review in docs/nwpu_fi_matrix_plan.md and user authorization).
#
# Usage: bash vhr10_fi_matrix_rows.sh <point|point_box|point_box_mask|full> [args...]
# Each row presets its prompt combination + P2 switch and the fi contract,
# then runs the shared VHR-10 protocol (fast400 lineage: full data, 4 GPUs,
# EMA tracking, seed 44).  Legacy VHR-10 runs are untouched (separate
# vhr10_fi_matrix_* run tags).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROW="${1:?usage: vhr10_fi_matrix_rows.sh <point|point_box|point_box_mask|full>}"
shift || true

case "${ROW}" in
  point)
    export EXPLICIT_PROMPT_MODE="points"
    export P2_BOUNDARY_REFINER_ENABLED=0
    export RUN_TAG="vhr10_fi_matrix_point"
    ;;
  point_box)
    export EXPLICIT_PROMPT_MODE="points_box"
    export P2_BOUNDARY_REFINER_ENABLED=0
    export RUN_TAG="vhr10_fi_matrix_point_box"
    ;;
  point_box_mask)
    export EXPLICIT_PROMPT_MODE="points_box_dense"
    export P2_BOUNDARY_REFINER_ENABLED=0
    export RUN_TAG="vhr10_fi_matrix_point_box_mask"
    ;;
  full)
    export EXPLICIT_PROMPT_MODE="points_box_dense"
    export P2_BOUNDARY_REFINER_ENABLED=1
    export RUN_TAG="vhr10_fi_matrix_full"
    ;;
  *)
    echo "Unknown row: ${ROW}" >&2
    exit 2
    ;;
esac

source "${SCRIPT_DIR}/vhr10_fi_overlay.sh"
exec bash "${SCRIPT_DIR}/../_run_vhr10.sh" fast400 "$@"
