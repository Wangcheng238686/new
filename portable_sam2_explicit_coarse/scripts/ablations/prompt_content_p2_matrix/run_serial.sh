#!/usr/bin/env bash
# Serial 3x2 prompt-content/P2 matrix, warm-started from R1-C4-RD epoch80.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
MATRIX_LOG="logs/ablations/prompt_content_p2_matrix_serial_${RUN_TIMESTAMP}.log"
mkdir -p "$(dirname "${MATRIX_LOG}")"
exec > >(tee -a "${MATRIX_LOG}") 2>&1

echo "matrix_started=$(date '+%Y-%m-%d %H:%M:%S')"
echo "matrix_log=${MATRIX_LOG}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<environment-default>}"

wrappers=(
  e1_point_p2_off.sh
  e2_point_p2_on.sh
  e3_box_p2_off.sh
  e4_box_p2_on.sh
  e5_mask_p2_off.sh
  e6_mask_p2_on.sh
)

for wrapper in "${wrappers[@]}"; do
  echo "experiment_started=${wrapper} time=$(date '+%Y-%m-%d %H:%M:%S')"
  bash "${SCRIPT_DIR}/${wrapper}" "$@"
  echo "experiment_completed=${wrapper} time=$(date '+%Y-%m-%d %H:%M:%S')"
done

echo "matrix_completed=$(date '+%Y-%m-%d %H:%M:%S')"
