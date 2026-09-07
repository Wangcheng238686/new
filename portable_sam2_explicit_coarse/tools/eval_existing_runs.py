#!/usr/bin/env python
"""Unified full-validation evaluation of existing D-series checkpoints.

Review directive (2026-09-06): pause new training; evaluate the PRE-SELECTED
best weights and the final-epoch weights of the existing arms on the FULL
627-image validation set, without re-scanning epochs.  This tests whether
the dev-subset (62-image) ranking survives full-val evaluation.

Per-run training env is replayed from the launch log's
resolved_hyperparameters block (config-affecting vars only) so the embedded
checkpoint contracts resolve; infra/runtime vars are blocked and the GPU is
assigned per worker.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = Path("/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/refactor_golden")
CKPT_ROOT = Path("/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations")

RUNS = [
    ("d0_point", "whu_d0fi_point_10p_2gpu", "d0_launch.log"),
    ("pb_run1", "whu_d1fi_pb_control_10p_2gpu", "d1fi_launch.log"),
    ("pb_run2", "whu_d6fi_pb_control_10p_2gpu", "d6pb_launch.log"),
    ("d6_mask_run1", "whu_d6fi_mask_gate010_pe010_10p_2gpu", "d6mask_launch.log"),
    ("d6_mask_run2", "whu_d6fi_mask_gate010_pe010_10p_2gpu_rep2", "d6mask_rep2_launch.log"),
    ("d5b_mask_run1", "whu_d5b_mask_gate050_pe010_10p_2gpu", "d5b_launch.log"),
    ("d5b_mask_run2", "whu_d5b_mask_gate050_pe010_10p_2gpu_rep2", "d5b_rep2_launch.log"),
    # D5-A pair added 2026-09-06 (user-approved): the PE-frozen control for
    # the D5-B arms, enabling the full-val PE-adaptation paired comparison.
    ("d5a_mask_run1", "whu_d5a_mask_gate050_10p_2gpu", "d5a_launch.log"),
    ("d5a_mask_run2", "whu_d5a_mask_gate050_10p_2gpu_rep2", "d5a_rep2_launch.log"),
]

_BLOCKLIST = {
    "CUDA_VISIBLE_DEVICES", "NPROC_PER_NODE", "BATCH_SIZE", "GRAD_ACCUM_STEPS",
    "MASTER_PORT", "MASTER_PORT_SOURCE", "RUN_IN_BACKGROUND", "RUN_TIMESTAMP",
    "RUN_SUFFIX", "LOG_FILE", "LOG_DIR", "CHECKPOINT_DIR", "DRY_RUN",
    "CHECK_DATA", "PREFLIGHT_MODEL", "DEV_GPUS", "EFFECTIVE_GLOBAL_BATCH_SIZE",
    "GIT_COMMIT", "GIT_DIRTY", "GIT_DIRTY_FILES", "GIT_DIFF_SHA",
    "MAX_EPOCHS", "TRAIN_SUBSET_RATIO", "VAL_SUBSET_RATIO", "SUBSET_TAG",
    "VAL_EVERY_N_EPOCHS", "VAL_BATCH_SIZE", "COMPUTE_VAL_LOSS",
    "EARLY_STOPPING_PATIENCE", "EARLY_STOPPING_START_EPOCH",
    "EARLY_STOPPING_MIN_DELTA", "EARLY_STOPPING_SMOOTH_WINDOW",
    "MAX_TRAIN_BATCHES", "MAX_VAL_BATCHES", "INIT_FROM", "RESUME_FROM",
    "INIT_EXCLUDE_PREFIXES", "ALLOW_CROSS_ARCH_INIT", "PYTHON",
    "WARMUP_ITERS", "LEARNING_RATE", "WEIGHT_DECAY",
}
_ENV_KEY = re.compile(r"^[A-Z][A-Z0-9_]*$")
_ECHO_ENV = re.compile(r'echo "([a-z0-9_]+)=\$\{([A-Z0-9_]+)\}"')
# Runner echoes derived names for a few labels; the config re-resolution
# needs the ORIGINAL env names the wrappers exported.
_ENV_ALIASES = {"EXPECTED_EXPLICIT_MODE": "EXPLICIT_PROMPT_MODE"}
_BEGIN = "resolved_hyperparameters_begin"
_END = "resolved_hyperparameters_end"


def build_label_map(runner: Path) -> dict:
    """Map resolved-block lowercase labels to their source env vars, parsed
    from the runner's own echo statements (exact, self-maintaining)."""
    return {label: env for label, env in _ECHO_ENV.findall(runner.read_text())}


def extract_env(log_path: Path, run_tag: str, label_map: dict) -> dict:
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    env, inside, current = {}, False, None
    for line in lines:
        if _BEGIN in line:
            inside, current, env = True, None, {}
            continue
        if _END in line:
            if current == run_tag:
                return env
            inside = False
            continue
        if not inside:
            continue
        if line.startswith("run_tag="):
            current = line.split("=", 1)[1].strip()
            continue
        if current != run_tag or "=" not in line:
            continue
        label, value = line.split("=", 1)
        env_name = label_map.get(label.strip())
        if env_name:
            env_name = _ENV_ALIASES.get(env_name, env_name)
            if env_name not in _BLOCKLIST:
                env[env_name] = value.strip()
    raise SystemExit(f"run_tag {run_tag!r} not found in {log_path}")


def run_job(job, gpu: int) -> dict:
    name, weights = job
    out_dir = GOLDEN / "fulleval" / f"{name}_{weights}"
    manifest = out_dir / "run_manifest.json"
    if manifest.is_file():
        return summarize(name, weights, json.loads(manifest.read_text()))
    if out_dir.exists():
        # Stale output from an aborted/duplicated scheduler run (no manifest
        # = incomplete).  Remove to avoid mixing files across attempts.
        shutil.rmtree(out_dir)
        (out_dir.parent / f"{name}_{weights}.log").unlink(missing_ok=True)
    env = dict(os.environ)
    env.update(job_env_cache[name])
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable, "inference/infer_from_checkpoint.py",
        "--checkpoint", str(job_ckpt[name] / f"{weights}.pth"),
        "--weights", "model",
        "--split", "validation",
        "--batch-size", "2",
        "--output-dir", str(out_dir),
    ]
    log = (out_dir.parent / f"{name}_{weights}.log")
    out_dir.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        return {"name": name, "weights": weights, "error": f"exit {proc.returncode}, see {log}"}
    return summarize(name, weights, json.loads(manifest.read_text()))


def summarize(name, weights, manifest) -> dict:
    metrics = manifest.get("metrics", {})
    return {
        "name": name,
        "weights": weights,
        "images": manifest.get("processed_images"),
        "segm/mAP": metrics.get("segm/mAP"),
        "segm/mAP_50": metrics.get("segm/mAP_50"),
        "segm/AR@100": metrics.get("segm/AR@100"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--smoke", action="store_true",
                        help="one run, few batches, single GPU")
    args = parser.parse_args()

    out_root = GOLDEN / "fulleval"
    out_root.mkdir(parents=True, exist_ok=True)
    # Exclusive running lock: a second scheduler instance (the 2026-09-06
    # incident ran a 4-GPU and a 2-GPU instance concurrently over the same
    # output dirs) must refuse to start.
    lock_handle = (out_root / ".scheduler.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            "another eval_existing_runs scheduler holds "
            f"{out_root / '.scheduler.lock'}; refusing to double-start"
        )
    lock_handle.write(f"{os.getpid()}\n")
    lock_handle.flush()

    global job_env_cache, job_ckpt
    label_map = build_label_map(ROOT / "scripts" / "_run_ablation.sh")
    job_env_cache, job_ckpt = {}, {}
    for name, run_tag, log_name in RUNS:
        job_env_cache[name] = extract_env(GOLDEN / log_name, run_tag, label_map)
        job_ckpt[name] = CKPT_ROOT / f"{run_tag}_tr0.1_va0.1"

    if args.smoke:
        name, run_tag, _ = RUNS[0]
        env = dict(os.environ)
        env.update(job_env_cache[name])
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        out = GOLDEN / "fulleval_smoke"
        cmd = [
            sys.executable, "inference/infer_from_checkpoint.py",
            "--checkpoint", str(job_ckpt[name] / "best_model.pth"),
            "--weights", "model", "--split", "validation",
            "--batch-size", "2", "--max-batches", "4",
            "--output-dir", str(out),
        ]
        subprocess.run(cmd, cwd=ROOT, env=env, check=False)
        return

    jobs = [(name, w) for name, _, _ in RUNS for w in ("best_model", "last_checkpoint")]
    results = []
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = {}
        for index, job in enumerate(jobs):
            gpu = args.gpus[index % len(args.gpus)]
            futures[pool.submit(run_job, job, gpu)] = job
        for future in futures:
            results.append(future.result())

    results.sort(key=lambda r: (r["name"], r["weights"]))
    table = {"generated": str(Path().resolve()), "runs": results}
    out = GOLDEN / "fulleval" / "summary.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
