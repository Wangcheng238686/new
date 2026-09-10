#!/usr/bin/env bash
# NWPU main ablation-matrix protocol.  Model/arm definitions are deliberately
# shared with dev100 and the historical full600 anchor; only the budget and
# validation/selection policy are set here.  Keep the matrix300 arm allowlist
# explicit: this is the production 300-epoch entry, including R3/R3+UDPR.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${1:?usage: vhr10_p2v2_matrix300.sh <p|pb|a0|a1|a2|a2e|a3|a3r|a4|r3|r3_udpr>}"
case "${ARM}" in
  p|pb|a0|a1|a2|a2e|a3|a3r|a4|r3|r3_udpr) ;;
  *)
    echo "Unknown matrix300 arm ${ARM}; expected p, pb, a0, a1, a2, a2e, a3, a3r, a4, r3, or r3_udpr" >&2
    exit 2
    ;;
esac
if [ "$#" -ne 1 ]; then
  echo "matrix300 accepts exactly one arm" >&2
  exit 2
fi
export P2V2_PROTOCOL=matrix300
exec bash "${SCRIPT_DIR}/vhr10_p2v2_dev.sh" "${ARM}"
