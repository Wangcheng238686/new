#!/usr/bin/env bash
# Foreground-only formal A-series runner.  Each arm occupies all four GPUs;
# set -e prevents a failed arm from being followed by an incomparable one.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
  ARMS=(a0 a1 a2 a3)
else
  ARMS=("$@")
fi
for arm in "${ARMS[@]}"; do
  echo "===== NWPU P2-v2 full600 series: ${arm} ====="
  bash "${SCRIPT_DIR}/vhr10_p2v2_full600.sh" "${arm}"
done
