"""UBR-RESEAL-20261027 level-3 arm: single-stream unified architecture.

Integrity contract (docs/UBR-RESEAL-20261027_preregistration.md, fail-closed):
  The resolved diff of this config against the EXP-0025 sealed config
  (configs/ue_whu_a3sgm_512.py, fingerprint 5305b315...) is EXACTLY
    1. roi_head.mask_head.decoder_tail_refiner_cfg -> {}   (UDPR not
       instantiated: delete, not a switch; loss_decoder_tail must never
       appear in the loss dict)
    2. roi_head.mask_head.ubr_integration_cfg.*             (UBR wiring
       keys, consumed by rsprompter/uav/ubr_train_integration.py)
  3. Dead-config deletions (PI instruction 2026-10-27, journal hygiene):
     point_warmup_cfg (idle enabled=False block), box_prompt_cfg (guard of
     a removed jitter arm), dense_prompt_cfg.gaussian_{omega,gamma}
     (accepted-but-unused, see dense_prompt_utils docstring).
  and NOTHING else. Any additional drift is a harness FAIL. Zero model
  impact is proven by a seeded dual-build state_dict bit-compare. The training
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
            # 3) Config hygiene (PI instruction 2026-10-27, zero model
            #    impact, proven by bit-exact dual-build state_dict compare):
            #    dead keys left by past arm deletions.
            point_warmup_cfg=dict(_delete_=True),   # idle block, enabled=False
            box_prompt_cfg=dict(_delete_=True),     # removed jitter arm guard
            dense_prompt_cfg=dict(                  # drop unused omega/gamma
                _delete_=True,
                transform='raw_logits', detach_input=True, temperature=1.0,
                outside_fill_logit=0.0, clamp_range=(-8.0, 8.0),
                foreground_threshold=0.5),
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
                # 4) Augmentation = 3090-line protocol (PI instruction
                #    2026-10-27): NO flip (dual-stream flip-consistency);
                #    train-only ColorJitter(.15,.15,.10,.02)@p=.5 and
                #    Gaussian noise p=.5 std=.02, applied via class patch
                #    in the runner (semantics copied verbatim from
                #    data/ue_minimal_dataset.py MinimalSatelliteDataset).
                aug_gaussian='0.5,0.02',
                aug_jitter='0.15,0.15,0.10,0.02,0.5',
            ),
        ),
    ),
)
