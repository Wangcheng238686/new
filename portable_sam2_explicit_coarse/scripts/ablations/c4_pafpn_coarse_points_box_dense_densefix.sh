#!/usr/bin/env bash
# C4 dense-gate repair (densefix): identical to C4 (points_box_dense, roi_only,
# no P2 refiner) except the learnable global-sigmoid dense gate is replaced by a
# fixed, untrained interpolation coefficient of 0.5.
#
# Motivation: in the C4 run `shape_dense_alpha_raw` (global_sigmoid mode) stayed
# pinned near its 0.25 init across 8 epochs, so the dense prompt's
# applied_delta_ratio never exceeded ~0.13 and C4 tracked C3 on segm/mAP. This
# wrapper isolates "the gate never opened" as the single changed variable: the
# dense embedding is injected at a constant strength instead of a gate that
# failed to learn.
#
# ABLATION_ID stays the C4 architecture identity; RUN_TAG carries the densefix
# suffix so logs/checkpoints do not collide with the original C4 run.
set -Eeuo pipefail
export ABLATION_ID="c4_pafpn_coarse_points_box_dense"
export RUN_TAG="${RUN_TAG:-c4_pafpn_coarse_points_box_dense_densefix}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0
# Select the densefix config variant (fixed gate at 0.5) instead of the
# committed global-sigmoid config. Override SHAPE_DENSE_ALPHA_INIT to retune.
export CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-configs/whu1024_baseplus_explicit_coarse_densefix.py}"
# Pin to GPU 1,2: C2-r's former slot, now shared with the running C3 (each 4090
# has ~39G free after C3's ~10G). Override from the outer shell if needed.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
