#!/usr/bin/env python
"""Regression gate: every paper a3sgm checkpoint must rebuild and strict-load.

This is the primary guard for the a3sgm-only cleanup: any change that breaks
checkpoint reconstruction (architecture contract, registry names, module
attribute names, ``__init__`` signatures) fails here before anything else
runs. It exercises the real CLI path (``infer_from_checkpoint.py --build-only``)
so argparse, contract validation, and strict state-dict loading are all
covered, on CPU so it is safe to run next to training jobs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MAINLINE_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = MAINLINE_ROOT.parent

# Machine-local defaults (lthpc); override via environment when porting.
DEFAULT_PYTHON = "/data/wangcheng/envs/cvt2/bin/python"
DEFAULT_SAM2_CKPT = "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt"
DEFAULT_CKPT_ROOT = (
    "/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations"
)

# The two paper-reporting checkpoints.
# - NWPU best-composite (E270) is the paper's NWPU row (mask AP 68.3); it
#   predates the decoder-tail ID suffix, so it also regression-tests the
#   legacy pre-tail ID compat rule in infer_from_checkpoint.
# - WHU best (E118, raw + EMA) is the a3sgm WHU full run.
# The four ablation arms guard the paper's NWPU ablation table (P/B/P+B/P+M)
# and the pruned architecture_id() branches (r1_c3 + pafpn_coarse_* IDs).
CHECKPOINTS = [
    ("nwpu_a3sgm_best_composite", "vhr10_p2v2_matrix300_a3sgm_pbm_sg_udprk64tm_tr1.0_va1.0/best_composite_model_epoch270.pth", "model"),
    ("whu_a3sgm_best_e118", "whu1024_a3sgm_roilocal_full_fast_tr1.0_va1.0/best_model_epoch118.pth", "model"),
    ("whu_a3sgm_best_e118_ema", "whu1024_a3sgm_roilocal_full_fast_tr1.0_va1.0/best_model_epoch118.pth", "ema"),
    ("nwpu_ablation_p", "vhr10_p2v2_matrix300_p_point_tr1.0_va1.0/best_model_epoch155.pth", "model"),
    ("nwpu_ablation_b", "vhr10_p2v2_matrix300_b_box_tr1.0_va1.0/best_model_epoch185.pth", "model"),
    ("nwpu_ablation_pb", "vhr10_p2v2_matrix300_pb_tr1.0_va1.0/best_model_epoch155.pth", "model"),
    ("nwpu_ablation_a0_sg", "vhr10_p2v2_matrix300_a0_sg_tr1.0_va1.0/best_model_epoch155.pth", "model"),
]


def run_case(python: str, sam2_ckpt: str, ckpt_path: Path, weights: str) -> None:
    command = [
        python,
        str(MAINLINE_ROOT / "inference" / "infer_from_checkpoint.py"),
        "--checkpoint",
        str(ckpt_path),
        "--weights",
        weights,
        "--device",
        "cpu",
        "--build-only",
    ]
    env = dict(os.environ)
    env.setdefault("SAM2_REPO", str(REPOSITORY_ROOT / "sam2"))
    env["SAM2_CKPT"] = sam2_ckpt
    result = subprocess.run(
        command,
        cwd=str(MAINLINE_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"FAIL {ckpt_path.name} (weights={weights})")
    if "checkpoint model build/load: OK" not in result.stdout:
        raise SystemExit(
            f"FAIL {ckpt_path.name} (weights={weights}): build-only banner missing"
        )
    print(f"PASS {ckpt_path.name} (weights={weights})")


def main() -> None:
    python = os.environ.get("PYTHON", DEFAULT_PYTHON)
    sam2_ckpt = os.environ.get("SAM2_CKPT", DEFAULT_SAM2_CKPT)
    ckpt_root = Path(os.environ.get("PORTABLE_SAM2_CHECKPOINT_ROOT", DEFAULT_CKPT_ROOT))
    if not Path(sam2_ckpt).is_file():
        raise SystemExit(f"SAM2 base checkpoint not found: {sam2_ckpt}")
    for _label, relative_path, weights in CHECKPOINTS:
        run_case(python, sam2_ckpt, ckpt_root / relative_path, weights)
    print("all a3sgm checkpoint load checks passed")


if __name__ == "__main__":
    main()
