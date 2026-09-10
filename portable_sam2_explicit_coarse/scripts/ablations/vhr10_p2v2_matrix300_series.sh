#!/usr/bin/env bash
# Foreground-only NWPU 300-epoch main matrix.  Each arm consumes all four GPUs.
# The default contains the frozen main narrative P -> PB -> PBM -> PBM+UDPR.
# R3 and R3+UDPR are registered production matrix300 arms, but remain explicit
# arguments so an empty invocation never restarts already completed old rows.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
  ARMS=(p pb a0 a3)
else
  ARMS=("$@")
fi

for arm in "${ARMS[@]}"; do
  case "${arm}" in
    p|pb|a0|a1|a2|a2e|a3|a3r|a4|r3|r3_udpr) ;;
    *)
      echo "Unknown matrix300 arm ${arm}; expected p, pb, a0, a1, a2, a2e, a3, a3r, a4, r3, or r3_udpr" >&2
      exit 2
      ;;
  esac
  echo "===== NWPU P2-v2 matrix300: ${arm} ====="
  bash "${SCRIPT_DIR}/vhr10_p2v2_matrix300.sh" "${arm}"
done
