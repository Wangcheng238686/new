"""UBR-RESEAL-20261027 level-3 arm: single-stream unified architecture.

Integrity contract (docs/UBR-RESEAL-20261027_preregistration.md, fail-closed):
  The resolved diff of this config against the EXP-0025 sealed config
  (configs/ue_whu_a3sgm_512.py, fingerprint 5305b315...) is EXACTLY
    1. roi_head.mask_head.decoder_tail_refiner_cfg -> {}   (UDPR not
       instantiated: delete, not a switch; loss_decoder_tail must never
       appear in the loss dict)
    2. roi_head.mask_head.ubr_integration_cfg.*             (UBR wiring
       keys, consumed by rsprompter/uav/ubr_train_integration.py)
  and NOTHING else. Any additional drift is a harness FAIL. The training
  protocol (seed 44, DDP4 b1xa2, lr 5e-4, wd 0.05, warmup 100, cosine,
  100 epochs, hflip 0.5, val 459 every epoch, last-epoch selection, test
  never read) is identical to EXP-0025 and is asserted by the runner
  scripts/probes/ue_ubr_reseal_20261027.py, which launches the SAME
  trainer entry (train/train_rsprompter_fusion.py) as
  scripts/ue/run_exp0025_ddp4.sh.

UBR integration (declared here; wired at runtime without touching any
existing file):
  - rsprompter/uav/ubr.py::UnifiedBoundaryRefiner, default ctor
    (163,537 params), port 5 (UAV geometry) ZERO-FILLED -> UBR(off).
  - loss_ubr: band k=8 focal BCE (gamma 2) + soft Dice vs the matched GT
    on each sampled positive RoI's box+32px window, per-instance equal
    weight, lambda 0.1; neighbors channel = sum of OTHER GT masks in the
    same image during training (predicted neighbors at inference).
  - Predict path: band reclassification applied per instance.
"""

_base_ = '../ue_whu_a3sgm_512.py'

model = dict(
    roi_head=dict(
        mask_head=dict(
            # 1) UDPR tail NOT instantiated (delete, not disable).
            decoder_tail_refiner_cfg=dict(_delete_=True),
            # 2) UBR wiring keys (read by the runner; inert as build kwargs).
            ubr_integration_cfg=dict(
                enabled=True,
                band_k=8,
                loss_weight=0.1,
                window_margin=32,
                focal_gamma=2.0,
                uav_port='zero',          # UBR(off): port 5 all zeros
                neighbors_train='gt',     # documented choice (see docstring)
                neighbors_predict='pred',
            ),
        ),
    ),
)
