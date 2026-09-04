#!/usr/bin/env bash
# VHR-10 ft-200: fine-tune continuation of vhr10_c5v2_400 at lr 1e-4 for 200
# epochs; best val segm/mAP 0.6617 (dataset best overall). Shared protocol
# lives in scripts/_run_vhr10.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../_run_vhr10.sh" ft200 "$@"
