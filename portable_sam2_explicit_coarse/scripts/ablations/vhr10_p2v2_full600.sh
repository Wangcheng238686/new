#!/usr/bin/env bash
# Formal NWPU A-series protocol: same A0/A1/A2/A3 definitions as dev100,
# with a single 600-epoch cosine schedule.  The shared arm script owns every
# model knob; this wrapper owns only protocol duration/validation policy.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export P2V2_PROTOCOL=full600
exec bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" "$@"
