#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
exec "${UE_EXPERIMENT_PYTHON:-/data/wangcheng/envs/cvt2/bin/python}" "$experiment_dir/ue_staged_uav_control.py" "$@"
