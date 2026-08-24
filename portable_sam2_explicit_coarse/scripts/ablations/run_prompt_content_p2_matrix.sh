#!/usr/bin/env bash
# Run the six point/box/mask x P2 off/on experiments serially.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for content in point box mask; do
  for p2 in 0 1; do
    echo "matrix_start content=${content} p2=${p2}"
    RUN_IN_BACKGROUND=0 bash \
      "${SCRIPT_DIR}/prompt_content_p2_from_r1_best.sh" \
      "${content}" "${p2}" "$@"
    echo "matrix_done content=${content} p2=${p2}"
  done
done
