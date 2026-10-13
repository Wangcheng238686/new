#!/usr/bin/env bash
# New lineage only; this wrapper is never launched by the migration itself.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${SAM2_CKPT:?Set the local SAM2 Base+ checkpoint}"
: "${UE_DATA_ROOT:?Set the registered raw pairset root}"
: "${WORK_DIR:?Use a fresh output directory}"
: "${RUN_TAG:?Set a new experiment tag; do not reuse EXP-0006}"
if [[ "$RUN_TAG" == *EXP-0006* || -e "$WORK_DIR" ]]; then
    echo "New WHU baseline requires a new run tag and output directory" >&2
    exit 2
fi
export SAM2_REPO="${SAM2_REPO:-$ROOT/../sam2}"
export PYTHONPATH="$ROOT:$SAM2_REPO${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "${PYTHON:-python3}" train/train_rsprompter_fusion.py \
    --config "$ROOT/configs/ue_whu_a3sgm.py" --dataset ue_raw_surface \
    --data-root "$UE_DATA_ROOT" --image-size 1024 1024 \
    --raw-split-manifest surfbev_b0_split_v2.json \
    --checkpoint-dir "$WORK_DIR" --run-tag "$RUN_TAG" --seed 44 "$@"
