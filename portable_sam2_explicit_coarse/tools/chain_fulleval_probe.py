#!/usr/bin/env python
"""Chain: wait for the full-eval batch, rerun stragglers, then run the
prompt-switch probe on the D6-mask best checkpoint with env replay."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from eval_existing_runs import CKPT_ROOT, GOLDEN, RUNS, build_label_map, extract_env  # noqa: E402

EXPECTED_CELLS = 2 * len(RUNS)  # best + last per run; RUNS may grow

PY = sys.executable
BATCH_PID = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def wait_pid(pid: int) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(30)


def main() -> int:
    if BATCH_PID:
        wait_pid(BATCH_PID)
    # Rerun the orchestrator: completed manifests are skipped, only the
    # failed/missing jobs (e.g. pb_run1 pair after the compat fix) rerun.
    # TWO concurrent jobs max: each full-val evaluation holds ~25-30GB of
    # dense GT/DT masks; four concurrent runs exhausted RAM and swap-thrashed
    # (2026-09-06 incident).
    orchestrator = subprocess.run(
        [PY, str(ROOT / "tools/eval_existing_runs.py"), "--gpus", "0", "1"],
        cwd=ROOT, check=False,
    )
    if orchestrator.returncode != 0:
        print("CHAIN FAILED: orchestrator rerun exited "
              f"{orchestrator.returncode}")
        return 1

    # Verify the summary before continuing: all 14 cells present, none in
    # error.  A missing/failed cell must abort the chain, not be papered
    # over by CHAIN COMPLETE.
    summary_path = GOLDEN / "fulleval" / "summary.json"
    if not summary_path.is_file():
        print("CHAIN FAILED: summary.json missing")
        return 1
    summary = json.loads(summary_path.read_text())
    runs = summary.get("runs", [])
    errors = [r for r in runs if r.get("error") or r.get("segm/mAP") is None]
    if len(runs) != EXPECTED_CELLS or errors:
        print(f"CHAIN FAILED: {len(runs)}/{EXPECTED_CELLS} cells, {len(errors)} in error")
        for r in errors:
            print("  -", r.get("name"), r.get("weights"), r.get("error"))
        return 1

    # Prompt-switch probe on D6-mask run1 best weights (strongest Mask arm),
    # full validation, with that run's training env replayed.
    name, run_tag, log_name = RUNS[3]
    label_map = build_label_map(ROOT / "scripts" / "_run_ablation.sh")
    env = dict(os.environ)
    env.update(extract_env(GOLDEN / log_name, run_tag, label_map))
    env["CUDA_VISIBLE_DEVICES"] = "0"
    ckpt = CKPT_ROOT / f"{run_tag}_tr0.1_va0.1" / "best_model.pth"
    out = GOLDEN / "fulleval" / "prompt_switch_d6_mask_run1.json"
    probe = subprocess.run(
        [PY, str(ROOT / "inference/probes/prompt_switch_probe.py"),
         "--checkpoint", str(ckpt), "--weights", "model",
         "--output", str(out)],
        cwd=ROOT, env=env, check=False,
    )
    if probe.returncode != 0 or not out.is_file():
        print(f"CHAIN FAILED: prompt-switch probe exited {probe.returncode}")
        return 1
    print("CHAIN COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
