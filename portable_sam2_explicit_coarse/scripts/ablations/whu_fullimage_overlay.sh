#!/usr/bin/env bash
# Spatial-contract overlay for the full-image final-mask protocol.
# Source this BEFORE invoking the whu_p2_matrix_* row wrappers.
#
# Contract being fixed (verified in rsprompter/sam2_mask_head*.py):
#   roi_local supervises the full-image SAM2 decoder grid with ROI-cropped
#   targets (mmdet mask_target) and pastes the whole output through the bbox
#   at inference (FCNMaskHead paste), while points/box/dense-canvas prompts
#   are full-image — the decoder must learn an unimplemented frame remap
#   from the final-mask loss alone.  full_image supervises against native
#   full-grid targets (get_full_image_targets) and resizes the decoder
#   output once at inference (_predict_by_feat_single full_image branch,
#   never through the bbox), so prompts and supervision share one frame.
#
# roi_balanced_dice is PART of this protocol, not a hyperparameter sweep:
# plain BCE on a sparse-foreground full canvas collapses to an
# all-background shortcut (git 6dc6aa4), and roi_balanced_dice is the
# existing full_image loss path (C2-L lineage).  Every arm of a run shares
# this contract so arms stay comparable to each other; comparing across
# protocols (vs roi_local runs) is NOT valid.
export FINAL_MASK_COORDINATE_MODE="full_image"
export FINAL_MASK_LOSS_MODE="roi_balanced_dice"
