#!/usr/bin/env bash
# R1-C4-RD with ROI-local final-mask coordinates: identical to R1-C4-RD (PAFPN,
# stride-16/64x64, coarse points_box_dense, detached raw-logits dense prompt,
# no refiner) except the final-mask contract switches from full_image to
# roi_local.
#
# Motivation: the full_image mask-supervision collapse hypothesis (see README)
# was confirmed by C3-roi_local — switching to roi_local restored healthy mask
# supervision in one epoch (loss_mask 0.001 -> 0.63, logit_mean -18 -> +0.2).
# R1-C4-RD (full_image, stride-16) plateaued at best 0.6450. This wrapper tests
# whether the dense prompt + stride-16 route also rises once supervision is
# healthy, and whether detached dense then adds measurable value over
# R1-C3-roi_local (once that also exists). It isolates coordinate-space as the
# single changed variable against R1-C4-RD.
#
# ABLATION_ID stays R1-C4-RD's architecture identity; RUN_TAG adds _roi_local.
set -Eeuo pipefail
export ABLATION_ID="r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64"
export RUN_TAG="${RUN_TAG:-r1_c4_rd_pafpn_coarse_points_box_raw_detach_roi_local_emb64}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM="raw_logits"
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE="roi_local"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
