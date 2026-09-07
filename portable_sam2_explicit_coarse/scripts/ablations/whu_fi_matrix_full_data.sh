#!/usr/bin/env bash
# Formal four-row Prompt/P2 matrix under the full_image spatial contract.
# All protocol knobs (4 GPUs x batch 1 x accum 2, 150 epochs, full WHU data,
# EMA tracking, early stopping) come from whu_p2_matrix_common.sh defaults —
# identical to the legacy whu_p2_matrix_* entries; ONLY the final-mask
# coordinate/loss contract differs (see whu_fullimage_overlay.sh).  Rows run
# serially: Point -> +Box -> +Mask -> Full(P2BR).  The whu_fi_matrix_* run
# tags keep results disjoint from the roi_local matrix.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Serial contract: the shared runner defaults to RUN_IN_BACKGROUND=1 (launch
# and return), which would start all four rows CONCURRENTLY on the same GPUs.
# Force foreground so each row starts only after the previous one exits 0;
# to run the whole series in background, wrap THIS script in nohup instead.
export RUN_IN_BACKGROUND=0

source "${SCRIPT_DIR}/whu_fullimage_overlay.sh"

export RUN_TAG="whu_fi_matrix_point"
bash "${SCRIPT_DIR}/whu_p2_matrix_point.sh" "$@"

export RUN_TAG="whu_fi_matrix_point_box"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box.sh" "$@"

export RUN_TAG="whu_fi_matrix_point_box_mask"
bash "${SCRIPT_DIR}/whu_p2_matrix_point_box_mask.sh" "$@"

export RUN_TAG="whu_fi_matrix_full"
bash "${SCRIPT_DIR}/whu_p2_matrix_full.sh" "$@"
