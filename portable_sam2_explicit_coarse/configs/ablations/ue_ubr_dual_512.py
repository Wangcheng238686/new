"""UBR level-4 dual-stream arms: unified architecture with live port 5.

Preregistration: docs/UBR-RESEAL-20261027_preregistration.md section
"级4 双流臂预注册" (2026-10-27 late-night PI instruction, co-tenant with
EXP-0028-R2); architecture docs/UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md
§2 (UBR ports) and §3 (unified architecture); capacity contract
docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md (32ch x 6 IS the unified
definition - the single-stream R2 arm trains it; the dual arms are
bit-exact in structure).

RESEAL_CONTRACT v3 (dual, EXP-0029):
  arm ② REAL     - uav_port='real':     port 5 = native512 cache fields.
  arm ③ CONSTANT - uav_port='constant': port 5 = ubr.build_constant_geometry
                    (train-only mean_fields where support==1, zeros in
                    holes; support distribution identical to the real arm).
  Both arms: geometry head (pilot network + direct center-box supervision,
  loss_geometry weight 1.0, training-phase loss ONLY - candidates never
  enter detector losses = firewall FW1); the head input is the REAL map in
  BOTH arms, so dual-arm diff collapses to the port-5 content alone.
  Inference (frozen interface, bit-exact): satellite prefix r(s) band +
  geometry candidates 0.25*s band (dedup + NMS vs prefix, <=250 cap),
  all decoded by the model's mask path, UBR band reclassification on every
  instance.

Integrity contract (fail-closed, asserted by the gates
scripts/probes/ue_ubr_dual_gates_20261027.py and the runner):
  The resolved diff of this config against the level-3 single-stream config
  (configs/ablations/ue_ubr_single_512.py) is EXACTLY
    1. roi_head.mask_head.ubr_integration_cfg.uav_port:
         'zero' -> 'real' | 'constant'   (the dual arms' port-5 state; the
         launched arm is selected by UBR_DUAL_ARM / --arm, fail-closed)
    2. roi_head.mask_head.ubr_integration_cfg.geometry_loss_weight: 1.0
         (lambda_geo, design §3.3; the only added key)
  and NOTHING else. Every other key keeps the sealed v1.1 value (checked
  by delegating to the single-stream validator with uav_port normalized).
  dual - single = {port 5, loss_geometry} - the two enumerable items of the
  unified architecture. Training protocol verbatim EXP-0028-R2: seed 44,
  DDP4 b1xa2, lr 5e-4, wd 0.05, warmup 100, cosine, 100 epochs, no flip
  (3090 line: ColorJitter(.15,.15,.10,.02)@p.5 + Gauss(.5,.02) train-only),
  sharded val 459 every epoch, last-epoch selection, test never read.
"""

_base_ = './ue_ubr_single_512.py'

model = dict(
    roi_head=dict(
        mask_head=dict(
            ubr_integration_cfg=dict(
                # 1) port-5 state: the dual arms' ONLY structural difference
                #    vs the single-stream arm (besides loss_geometry). The
                #    launch-time arm is selected by env UBR_DUAL_ARM
                #    (real|constant) / runner --arm; the file default is the
                #    REAL arm (arm ②, the co-tenant first launch).
                uav_port='real',
                # 2) lambda_geo = 1.0 (design §3.3, remote-verified pilot
                #    value). loss_geometry = pilot center-box loss on the
                #    full-512 real geometry map, satellite GT supervision;
                #    candidates never enter detector losses (FW1).
                geometry_loss_weight=1.0,
            ),
        ),
    ),
)
