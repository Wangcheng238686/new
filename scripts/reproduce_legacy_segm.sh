#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "${PROJECT_ROOT}/scripts/verify_legacy_baseline.sh"

export SAM2_SOURCE_PATH="${SAM2_SOURCE_PATH:-${PROJECT_ROOT}/sam2}"
export CKPT_BASE="${CKPT_BASE:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/legacy_reproduction}"
cd "${PROJECT_ROOT}/legacy_baseline/portable_sam2_fusion_new"
exec bash scripts/run_bs1a2_fixddp.sh
