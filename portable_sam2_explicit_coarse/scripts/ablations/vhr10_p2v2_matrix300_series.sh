#!/usr/bin/env bash
# Foreground-only NWPU 300-epoch main matrix.  Each arm consumes all four GPUs.
# The default contains the supported main narrative P -> PB -> PBM -> PBM+UDPR;
# P2 arms remain callable explicitly but are not silently included as a main row.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
  ARMS=(p pb a0 a3)
else
  ARMS=("$@")
fi

for arm in "${ARMS[@]}"; do
  echo "===== NWPU P2-v2 matrix300: ${arm} ====="
  bash "${SCRIPT_DIR}/vhr10_p2v2_matrix300.sh" "${arm}"
done
