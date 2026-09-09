#!/usr/bin/env bash
# Run the NWPU A-series dev arms in a caller-selected order.  Each child is
# foreground-only and occupies all four GPUs; set -e prevents a failed arm
# from being silently followed by an incomparable later arm.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# No arguments preserves the complete A0→A1→A2→A3 series.  Passing arm names
# is intentional for append-only work after historical A0/A1/A2 results exist:
# `bash .../vhr10_p2v2_dev_series.sh a4` starts only the new comparable arm.
if [ "$#" -eq 0 ]; then
  ARMS=(a0 a1 a2 a3)
else
  ARMS=("$@")
fi
for arm in "${ARMS[@]}"; do
  echo "===== NWPU P2-v2 dev series: ${arm} ====="
  bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" "${arm}"
done
