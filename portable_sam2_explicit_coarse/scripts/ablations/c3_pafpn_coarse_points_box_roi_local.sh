#!/usr/bin/env bash
# C3 with ROI-local final-mask coordinates: identical to C3 (PAFPN, coarse
# points_box, no dense, no refiner, stride-32) except the final-mask contract
# switches from the coarse route's default full_image to roi_local — the same
# coordinate contract B1 (MLP) uses.
#
# Motivation: all coarse-route experiments default to full_image, where a WHU
# building (~0.5% image area) makes the full-image mask target ~99.5% background.
# Standard BCE then collapses to an all-background shortcut (logged
# mask_logit_mean≈-18, prob_mean≈0.005, loss_mask≈0.001) and segm/mAP plateaus.
# B1's roi_local target crops the GT inside each proposal so target_fill≈0.5 and
# the mask is actually learned (loss_mask≈0.12, best 0.69). C2-L tried to fix
# this with an roi_balanced_dice loss trick *inside* full_image coordinates and
# scored worse, because the decoder still predicts on the full grid and the
# rectangular support weakens outside-ROI supervision.
#
# This wrapper changes the coordinate *space* (not the loss): the decoder output
# is supervised against ROI-local targets via standard BCE, exactly as B1 does.
# It isolates "full_image BCE collapse" as the single changed variable against
# C3. If segm/mAP rises toward B1's level, the coarse route's low plateau is
# confirmed to be a supervision-coordinate artefact, not an architecture limit.
#
# ABLATION_ID stays C3's architecture identity; RUN_TAG carries _roi_local.
set -Eeuo pipefail
export ABLATION_ID="c3_pafpn_coarse_points_box"
export RUN_TAG="${RUN_TAG:-c3_pafpn_coarse_points_box_roi_local}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box"
export P2_BOUNDARY_REFINER_ENABLED=0
export FINAL_MASK_COORDINATE_MODE="roi_local"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
