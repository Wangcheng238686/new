"""VHR-10 variant with prompt-consumption robustification (plan D).

Same recipe as vhr10_baseplus_explicit_coarse (10 classes, RSPrompter 520/130
split); enables the two training-time prompt jitter sources so the SAM2
decoder co-adapts to consumed-but-perturbed prompts:

- box prompt jitter (rsprompter/box_jitter.py, IoU>=0.5 guarded, previously
  compiled in but disabled at jitter_prob=0.0);
- positive-point jitter (ShapePointMiner point_jitter_prob: P1/P2 perturbed
  by up to 1 canvas cell — safe because positive candidates sit >= ~3 cells
  inside the object; negatives keep hard placements to preserve ring
  semantics).

Inference is unchanged (both jitters are training-only); the goal is to open
the decoder's consumption of moved points, which the E0/E1 probes showed is
the binding constraint for any prompt-placement refinement (incl. the P2
refined-canvas point mining, plan C).
"""

_base_ = ["./vhr10_baseplus_explicit_coarse.py"]

model = dict(
    roi_head=dict(
        mask_head=dict(
            box_prompt_cfg=dict(
                jitter_prob=0.3,
                jitter_scale=0.05,
            ),
            shape_point_miner_cfg=dict(
                point_jitter_prob=0.5,
                point_jitter_radius=1,
            ),
        ),
    ),
)
