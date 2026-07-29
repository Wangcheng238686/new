#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_ROOT}/portable_sam2_explicit_coarse/scripts/load_environment.sh"
bash "${PROJECT_ROOT}/scripts/verify_legacy_baseline.sh"

export SAM2_SOURCE_PATH="${SAM2_SOURCE_PATH:-${PROJECT_ROOT}/sam2}"
export CKPT_BASE="${CKPT_BASE:-${PORTABLE_SAM2_CHECKPOINT_ROOT}/legacy_reproduction}"
cd "${PROJECT_ROOT}/legacy_baseline/portable_sam2_fusion_new"
exec bash scripts/run_baseline_noms.sh
