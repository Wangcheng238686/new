#!/usr/bin/env bash
# NWPU fi-matrix spatial-contract overlay (prep only, NOT yet authorized to
# launch; see docs/nwpu_fi_matrix_plan.md).  Source BEFORE calling
# scripts/_run_vhr10.sh.  Mirrors whu_fullimage_overlay.sh: the fi contract
# is coordinate+loss TOGETHER (roi_balanced_dice is part of the protocol,
# not a hyperparameter sweep — plain BCE on a sparse full canvas collapses,
# git 6dc6aa4).
export FINAL_MASK_COORDINATE_MODE="full_image"
export FINAL_MASK_LOSS_MODE="roi_balanced_dice"
