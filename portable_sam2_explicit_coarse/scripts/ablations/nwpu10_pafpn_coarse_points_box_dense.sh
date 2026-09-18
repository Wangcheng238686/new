#!/usr/bin/env bash
set -Eeuo pipefail
# NWPU VHR-10 (10-class) counterpart of c4_pafpn_coarse_points_box_dense.sh.
# Full PromptMiner-SAM2 (Ours): PAFPN neck + explicit coarse prompt route,
# points+box+dense mode. Reuses _run_ablation.sh and only overrides the
# dataset plumbing (data root, COCO split files, multi-class mapping) and the
# config via CONFIG_OVERRIDE.
SCRIPT_DIR_NWPU10="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_NWPU10="$(cd "${SCRIPT_DIR_NWPU10}/../.." && pwd)"
cd "${PROJECT_ROOT_NWPU10}"
# shellcheck source=../load_environment.sh
source "${PROJECT_ROOT_NWPU10}/scripts/load_environment.sh"

export ABLATION_ID="nwpu10_c4_pafpn_coarse_points_box_dense"
# architecture_contract is class-count agnostic: the NWPU config resolves to
# the same architecture id as the WHU c4 winner (identical model, 10 classes).
export EXPECTED_ARCHITECTURE_ID="c4_pafpn_coarse_points_box_dense"
export RUN_TAG="${RUN_TAG:-nwpu10_c4_pafpn_coarse_points_box_dense}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0

# ---- NWPU VHR-10 dataset wiring (val doubles as the test split) ----
export TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${VHR10_DATA_ROOT:?VHR10_DATA_ROOT must be set (configs/environment2.sh)}}"
export WHU_TRAIN_ANN_FILE="coco_split/NWPU_instances_train.json"
export WHU_VAL_ANN_FILE="coco_split/NWPU_instances_val.json"
export WHU_TEST_ANN_FILE="coco_split/NWPU_instances_val.json"
export WHU_TRAIN_IMG_SUBDIR="positive image set"
export WHU_VAL_IMG_SUBDIR="positive image set"
export WHU_TEST_IMG_SUBDIR="positive image set"
# 10-class protocol: COCO category ids 1-10 -> contiguous labels 0-9.
export DATASET_SINGLE_CLASS="0"
export DATASET_CATEGORY_MAPPING="1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9"
export DATASET_CATEGORY_NAMES="airplane,ship,storage_tank,baseball_diamond,tennis_court,basketball_court,ground_track_field,harbor,bridge,vehicle"
# Architecture identical to the WHU c4 winner; only num_classes=10 differs
# (bbox head only; SAM2 mask head is class-agnostic).
export CONFIG_OVERRIDE="configs/nwpu10_baseplus_explicit_coarse.py"

exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
