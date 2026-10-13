#!/usr/bin/env python
"""DDP launch gate: torchrun must bring the a3sgm trainer through a few
optimizer steps on two ranks.

Single-process smokes cannot see distributed-launch breakage (an orphaned
argparse flag, a rank-dependent code path, a collective-count mismatch).
This gate runs 2 ranks x 3 optimizer steps on a tiny WHU subset and fails on
any traceback or on a missing end-of-epoch log line. AMP + a small RCNN
sampler cap keep each rank inside a fraction of one GPU.
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


def _pick_two_freest_gpus() -> str:
    query = (
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    )
    rows = subprocess.run(
        query.split(), capture_output=True, text=True, check=True
    ).stdout.strip().splitlines()
    used = sorted(
        (int(used), index) for index, used in (r.split(",") for r in rows)
    )
    return f"{used[0][1]},{used[1][1]}"


def main() -> None:
    python = os.environ.get("PYTHON", DEFAULT_PYTHON)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES") or _pick_two_freest_gpus()
    out_dir = Path("/tmp/a3sgm_ddp_smoke")
    subprocess.run(["rm", "-rf", str(out_dir)], check=False)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("/tmp/a3sgm_ddp_smoke.log")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env["SAM2_CKPT"] = os.environ.get("SAM2_CKPT", DEFAULT_SAM2_CKPT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Static config carries the model form; the sampler cap only shrinks the
    # train-time RoI budget (train_cfg is not part of the deployed contract).
    env["GATE_CONFIG"] = os.environ.get(
        "GATE_CONFIG",
        "/tmp/a3sgm_gate_cfg.py",  # sampler-num=16 delta to fit contended GPUs
    )
    # Dataset selection flag (renamed from the dual --use-*-coco switches).
    dataset_args = os.environ.get(
        "DDP_GATE_DATASET_ARGS", "--dataset whu"
    ).split()

    command = [
        python, "-m", "torch.distributed.run",
        "--nproc_per_node=2", "--master_port=39731",
        "train/train_rsprompter_fusion.py",
        "--config", env["GATE_CONFIG"],
        "--data-root", WHU_DATA_ROOT,
        *dataset_args,
        "--image-size", "1024", "1024",
        "--batch-size", "1", "--grad-accum-steps", "2",
        "--epochs", "1", "--max-train-batches", "6",
        "--train-subset-ratio", "0.01", "--val-subset-ratio", "0.01",
        "--val-every-n-epochs", "1", "--val-batch-size", "2",
        "--compute-val-loss", "0",
        "--seed", "44", "--amp", "1", "--lr", "5e-4", "--warmup-iters", "20",
        "--ema-enabled", "0",
        "--checkpoint-dir", str(out_dir),
    ]
    result = subprocess.run(
        command, cwd=str(MAINLINE_ROOT), env=env,
        capture_output=True, text=True, timeout=1800,
    )
    log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    combined = result.stdout + result.stderr
    if result.returncode != 0 or "Traceback" in combined:
        tail = "\n".join(combined.splitlines()[-25:])
        raise SystemExit(f"FAIL DDP launch (gpus={gpus}):\n{tail}")
    if "Epoch 1 detailed losses" not in combined:
        raise SystemExit("FAIL DDP launch: epoch summary line missing")
    print(f"PASS DDP launch 2-rank x 3 steps (gpus={gpus}, log={log_path})")


if __name__ == "__main__":
    main()
