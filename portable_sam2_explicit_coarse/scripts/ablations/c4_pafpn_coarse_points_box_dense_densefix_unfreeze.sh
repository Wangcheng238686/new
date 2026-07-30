#!/usr/bin/env bash
# C4 densefix + unfreeze mask_downscaling: builds on the densefix gate repair
# (fixed α=0.5) and additionally allows the frozen PromptEncoder
# mask_downscaling convolution stack to adapt, testing a candidate frozen
# bottleneck between the shape-dense signal and the decoder.
#
# Mask downscaling: 3-layer conv stack (4684 params, 0.005M).  Unfreezing it
# tests whether the PE bridge benefits from adapting to the sparse coarse-mask
# canvas instead of retaining SAM's original mask-prompt transform.
#
# Still a C4 architecture (same ABLATION_ID); RUN_TAG adds _unfreeze.
set -Eeuo pipefail
export ABLATION_ID="c4_pafpn_coarse_points_box_dense"
export RUN_TAG="${RUN_TAG:-c4_pafpn_coarse_points_box_dense_densefix_unfreeze}"
export NECK_TYPE="pafpn"
export PROMPT_ROUTE="coarse"
export EXPLICIT_PROMPT_MODE="points_box_dense"
export P2_BOUNDARY_REFINER_ENABLED=0
# Reuse the densefix config (fixed gate at 0.5).
export CONFIG_OVERRIDE="${CONFIG_OVERRIDE:-configs/whu1024_baseplus_explicit_coarse_densefix.py}"
# Train the mask_downscaling conv stack.  The densefix config resolves this
# launcher value into prompt_encoder_cfg.train_mask_downscaling so checkpoint
# reconstruction and resume do not depend on an unrecorded environment switch.
export PROMPT_ENCODER_TRAIN_MASK_DOWNSCALING=1
# Restore a non-zero LR multiplier for the prompt_encoder group (set to 1.0 so
# mask_downscaling trains at the base LR; the rest of PE stays frozen and does
# not enter the optimizer).
export PROMPT_ENCODER_LR_MULT="${PROMPT_ENCODER_LR_MULT:-1.0}"
# Pin to GPU 1,2 (shared with C3, same as densefix).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2}"
exec bash "$(cd "$(dirname "$0")" && pwd)/_run_ablation.sh" "$@"
