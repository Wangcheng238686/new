#!/usr/bin/env bash
set -Eeuo pipefail

# NWPU VHR-10 checkpoint inference / final evaluation (val split doubles as
# the test split). Mirrors scripts/infer_whu_checkpoint.sh but wires the NWPU
# data root, val json, shared "positive image set" image dir, and the 10-class
# category mapping/eval names.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

if [ "$#" -lt 1 ]; then
  echo "usage: bash scripts/infer_nwpu_checkpoint.sh CHECKPOINT [extra inference args]" >&2
  exit 2
fi

: "${VHR10_DATA_ROOT:?load_environment.sh must export VHR10_DATA_ROOT (configs/environment2.sh)}"

CHECKPOINT="$1"
shift

# 10-class protocol: COCO ids 1-10 -> labels 0-9 (must match training).
export DATASET_CATEGORY_MAPPING="1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9"
export DATASET_CATEGORY_NAMES="airplane,ship,storage_tank,baseball_diamond,tennis_court,basketball_court,ground_track_field,harbor,bridge,vehicle"

exec "${PYTHON}" inference/infer_from_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --split custom \
  --data-root "${VHR10_DATA_ROOT}" \
  --ann-file "coco_split/NWPU_instances_val.json" \
  --image-subdir "positive image set" \
  "$@"
