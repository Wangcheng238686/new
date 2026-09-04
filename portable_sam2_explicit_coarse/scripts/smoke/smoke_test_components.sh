#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"
exec "${PYTHON}" tools/smoke_test_components.py
