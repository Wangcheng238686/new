#!/usr/bin/env bash
# VHR-10 large-600: large encoder, 600-epoch cosine; best segm/mAP 0.6533
# @ep506 (2026-09-04). Supports full-state crash resume:
#   RESUME_FROM=<ckpt.pth> bash scripts/ablations/vhr10_large600.sh
# Shared protocol lives in scripts/_run_vhr10.sh.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/../_run_vhr10.sh" large600 "$@"
