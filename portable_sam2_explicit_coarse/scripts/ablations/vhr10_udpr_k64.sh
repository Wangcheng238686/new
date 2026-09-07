#!/usr/bin/env bash
# Frozen-A0 heat-start entry for the independent UDPR-K64 experiment.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" udpr64 "$@"
