#!/usr/bin/env bash
# VHR-10 jitter-400: plan-D box+point dual jitter arm (jitter config variant).
# NOTE: never launched as of the 2026-09 refactor; kept as a preregistered
# entry. Shared protocol lives in scripts/_run_vhr10.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../_run_vhr10.sh" jitter400 "$@"
