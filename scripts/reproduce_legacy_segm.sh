#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
bash "${PROJECT_ROOT}/scripts/verify_legacy_baseline.sh"

cd "${PROJECT_ROOT}/legacy_baseline/portable_sam2_fusion_new"
exec bash scripts/run_bs1a2_fixddp.sh

