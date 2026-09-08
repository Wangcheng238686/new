#!/usr/bin/env bash
# NWPU main ablation-matrix protocol.  Model/arm definitions are deliberately
# shared with dev100 and the historical full600 anchor; only the budget and
# validation/selection policy are set here.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export P2V2_PROTOCOL=matrix300
exec bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" "$@"
