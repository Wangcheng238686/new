"""UBR dual-stream +RGB arm: EXP-0029 dual(D+pose) + the CI-positive RGB path.

Spec: docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md (EXP-0030 /
UBR-DUAL-20261028-REAL). PI resolution 2026-10-28: the remote
candidate_detail matrix test verdict made the compact RGB path the ONLY
test-positive RGB consumption shape (+0.89pp [+0.64, +1.20] CI) - P0
source encoder -> per-cell equal-weight mean -> block4 zero-init residual
injection. The running EXP-0029 dual(D+pose) arm is RETAINED as the
comparison arm; this arm adds the RGB path on top of EXP-0029(real), so

    dual(+RGB real) - dual(D+pose)  =  the RGB injection increment alone.

RESEAL_CONTRACT v4 (dual+RGB, EXP-0030):
  Everything in v3 (EXP-0029) verbatim - port 5 real native512 fields,
  geometry head (202,581 params) + loss_geometry lambda 1.0, candidate
  band inference (r(s) prefix / 0.25*s candidates / 250 cap) - plus the
  rgb block: source LoRA SAM2 P0 encoder (r16/alpha32/dropout .05, ctor
  weights from the SAME generic foundation file as the satellite trunk,
  optimizer backbone-group treatment), rgb_project 256->32, FP64 anchor
  sampling (<=8 candidates/cell, <=2/view, cache order) from the mirrored
  s2db_anchor_metadata_v2_fp64 cache, per-cell equal-weight mean, block4
  zero-initialized residual injection (independent path). Reserved branch
  rgb_content='constant' (train-only per-channel mean descriptor, same
  support mask; source encoder skipped) - fourth-arm zero-cost switch,
  launched only after an RGB_WINS verdict.

Integrity contract (fail-closed, asserted by
scripts/probes/ue_ubr_dual_rgb_gates_20261028.py and the runner):
  The resolved diff of this config against the dual config
  (configs/ablations/ue_ubr_dual_512.py) is EXACTLY
    1. roi_head.mask_head.ubr_integration_cfg.rgb_content: 'real'
       (the only added key; 'constant' reserved for the fourth arm)
  and NOTHING else. uav_port stays 'real' (this arm is EXP-0029 arm 2 +
  RGB). dual - single now enumerates THREE items:
  {port 5, loss_geometry, RGB injection path} - the unified architecture's
  full dual-stream definition. Training protocol verbatim EXP-0029/28-R2:
  seed 44, DDP4 b1xa2, lr 5e-4, wd 0.05, warmup 100, cosine, 100 epochs,
  3090-line photometric-only augmentation, sharded val 459 every epoch,
  last-epoch selection, test never read.
"""

_base_ = './ue_ubr_dual_512.py'

model = dict(
    roi_head=dict(
        mask_head=dict(
            ubr_integration_cfg=dict(
                # The single new key: RGB consumption content. 'real' =
                # the P0/mean/block4 path on the true UAV RGB views (the
                # CI-positive compact shape); 'constant' = the matched
                # content control reserved for the fourth arm (launch only
                # after RGB_WINS; nothing auto-launches it).
                rgb_content='real',
            ),
        ),
    ),
)
