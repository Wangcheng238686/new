#!/usr/bin/env python
"""Resume gate: a saved checkpoint must reload and continue training.

Phase 1 trains 3 optimizer steps and saves last_checkpoint.pth; phase 2
resumes from it for another epoch.  Exercises the save/load/restore chain
(architecture assertion, optimizer/scheduler/scaler restore, config_snapshot
merge) that single forward passes never touch.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MAINLINE_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PYTHON = "/data/wangcheng/envs/cvt2/bin/python"
DEFAULT_SAM2_CKPT = "/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt"
WHU_DATA_ROOT = os.environ.get("WHU1024_DATA_ROOT", "/data/wangcheng/dataset/WHU")


def _freest_gpu() -> str:
    rows = subprocess.run(
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits".split(),
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    pairs = sorted(
        (int(used), int(index)) for index, used in (r.split(",") for r in rows)
    )
    return str(pairs[0][1])


def _run_phase(python: str, gpu: str, out_dir: Path, extra: list) -> str:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["SAM2_CKPT"] = os.environ.get("SAM2_CKPT", DEFAULT_SAM2_CKPT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["RCNN_SAMPLER_NUM"] = "32"
    env["SAVE_LAST_CHECKPOINT"] = "1"
    dataset_args = os.environ.get(
        "RESUME_GATE_DATASET_ARGS", "--dataset whu"
    ).split()
    command = [
        python, "train/train_rsprompter_fusion.py",
        "--config", "configs/a3sgm_whu1024.py",
        "--data-root", WHU_DATA_ROOT,
        *dataset_args,
        "--image-size", "1024", "1024",
        "--batch-size", "1", "--grad-accum-steps", "2",
        "--train-subset-ratio", "0.01", "--val-subset-ratio", "0.01",
        "--val-every-n-epochs", "1", "--val-batch-size", "2",
        "--compute-val-loss", "0",
        "--seed", "44", "--amp", "1", "--lr", "5e-4", "--warmup-iters", "20",
        "--ema-enabled", "0",
        "--checkpoint-dir", str(out_dir),
        *extra,
    ]
    result = subprocess.run(
        command, cwd=str(MAINLINE_ROOT), env=env,
        capture_output=True, text=True, timeout=1800,
    )
    combined = result.stdout + result.stderr
    if result.returncode != 0 or "Traceback" in combined:
        tail = "\n".join(combined.splitlines()[-25:])
        raise SystemExit(f"FAIL resume gate phase {extra}:\n{tail}")
    return combined


def main() -> None:
    python = os.environ.get("PYTHON", DEFAULT_PYTHON)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES") or _freest_gpu()
    out_dir = Path("/tmp/a3sgm_resume_smoke")
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)

    # last_checkpoint.pth is only written on NON-final epochs (the final one
    # saves last_model instead), so phase 1 runs two epochs of three steps.
    phase1 = _run_phase(
        python, gpu, out_dir,
        ["--epochs", "2", "--max-train-batches", "6"],
    )
    assert "Epoch 2 detailed losses" in phase1, "phase 1: epoch summary missing"
    last_ckpt = out_dir / "last_checkpoint.pth"
    assert last_ckpt.is_file(), f"phase 1: {last_ckpt} not written"

    phase2 = _run_phase(
        python, gpu, out_dir,
        ["--epochs", "3", "--max-train-batches", "3",
         "--resume-from", str(last_ckpt)],
    )
    assert "Resumed from" in phase2, "phase 2: resume banner missing"
    assert "Epoch 3 detailed losses" in phase2, "phase 2: continued epoch missing"
    print(f"PASS resume gate (gpu={gpu}, save->resume 3 steps)")


if __name__ == "__main__":
    main()
