#!/usr/bin/env bash
set -euo pipefail
telemetry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
if (($# == 0)); then set -- start; fi
exec /data/wangcheng/envs/cvt2/bin/python "$telemetry_dir/../analysis/ue_native_telemetry_owner_20261028.py" "$@"
