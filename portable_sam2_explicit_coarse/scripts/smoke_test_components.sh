#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
PYTHON="${PYTHON:-/data/wangcheng/envs/cvt2/bin/python}"
exec "${PYTHON}" scripts/smoke_test_components.py

