#!/usr/bin/env bash
# VHR-10 fast-400: the WHU fast-150 protocol transplanted to NWPU VHR-10.
# Base+/stride-16 C5-v2; RUN_TAG vhr10_c5v2_400. Shared protocol and the
# architecture exports block live in scripts/_run_vhr10.sh (single source).
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../_run_vhr10.sh" fast400 "$@"
