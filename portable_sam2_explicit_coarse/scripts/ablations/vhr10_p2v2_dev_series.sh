#!/usr/bin/env bash
# Run the three NWPU P2-v2 dev arms in the only valid order.  Each child is
# foreground-only and occupies all four GPUs; set -e prevents a failed arm
# from being silently followed by an incomparable later arm.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for arm in a0 a1 a2; do
  echo "===== NWPU P2-v2 dev series: ${arm} ====="
  bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" "${arm}"
done
