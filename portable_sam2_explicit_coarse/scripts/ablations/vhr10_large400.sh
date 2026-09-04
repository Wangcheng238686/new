#!/usr/bin/env bash
# VHR-10 large-400: SAM2 hiera-large full-chain weights, otherwise the C5-v2
# fast400 protocol. Shared protocol lives in scripts/_run_vhr10.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../_run_vhr10.sh" large400 "$@"
