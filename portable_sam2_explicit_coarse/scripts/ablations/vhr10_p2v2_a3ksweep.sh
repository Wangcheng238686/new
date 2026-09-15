#!/usr/bin/env bash
# A3 K-sensitivity sweep runner (matrix300 protocol, FROM SCRATCH).
#
# One full-budget A3 (v1 UDPR, attached lineage) training per K value, then
# best/last evaluation and a summary table.  K=64 is the existing a3 matrix
# row — its checkpoint dir already exists and is SKIPPED (never re-run).
#
# Usage (repo root):
#   bash scripts/ablations/vhr10_p2v2_a3ksweep.sh 32 16 128 256
#
# Full-budget discipline (EXPERIMENT_GOVERNANCE.md):
# - every run is ~9.5h on 4 GPUs; check nvidia-smi BEFORE launching
# - launch requires a clean committed tree; the commit carries the EXP-ID
# - interruptible: best checkpoints save continuously; a killed K can be
#   resumed via the RESUME_OWN channel or redone by deleting its dir
# - within-machine canonical numbers; compare K-vs-K deltas
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CKPT_ROOT="${PORTABLE_SAM2_CHECKPOINT_ROOT:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse}"

if [ "$#" -eq 0 ]; then
  echo "usage: vhr10_p2v2_a3ksweep.sh <K> [<K> ...]   e.g. 32 16 128 256 (64 = existing a3 row, auto-skipped)" >&2
  exit 2
fi
if [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>512{c++} END{print c+0}')" -gt 0 ]; then
  echo "ERROR: GPUs are occupied (a run is active?) — refusing to launch. nvidia-smi:" >&2
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >&2
  exit 1
fi

for K in "$@"; do
  case "${K}" in
    ''|*[!0-9]*|0) echo "ERROR: K must be a positive integer, got '${K}'" >&2; exit 2 ;;
  esac
  TAG="vhr10_p2v2_matrix300_a3_pbm_udprk${K}_tr1.0_va1.0"
  if [ -n "$(ls -A "${CKPT_ROOT}/ablations/${TAG}" 2>/dev/null)" ]; then
    echo "=== K=${K}: checkpoint dir already non-empty, SKIPPING (existing row; delete to redo) ==="
    continue
  fi
  echo "===== a3 K=${K}: matrix300 from-scratch train start $(date) ====="
  ( cd "${REPO_ROOT}" && P2V2_PROTOCOL=matrix300 A3_NUM_POINTS="${K}" \
      bash scripts/ablations/vhr10_p2v2_dev.sh a3 )
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "K=${K} TRAIN FAILED rc=${rc} — ABORT sweep" >&2; exit 1; fi
  echo "===== a3 K=${K}: eval $(date) ====="
  for kind in best last; do
    ( cd "${REPO_ROOT}" && P2V2_EVAL_PROTOCOL=matrix300 EVAL_CUDA_VISIBLE_DEVICES=0 \
        A3_SUFFIX="a3_pbm_udprk${K}" \
        bash scripts/ablations/vhr10_p2v2_eval.sh a3 "${kind}" ) \
      || echo "K=${K} eval ${kind} FAILED (non-fatal)"
  done
done

echo "===== A3 K-SWEEP COMPLETE $(date) — summary ====="
python3 - <<'PY'
import glob, json, os
root = os.environ.get("PORTABLE_SAM2_CHECKPOINT_ROOT",
                      "/data/wangcheng/checkpoint/portable_sam2_explicit_coarse")
rows = []
import glob as _g
for d in sorted(glob.glob(f"{root}/ablations/vhr10_p2v2_matrix300_a3_pbm_udprk*_tr1.0_va1.0")):
    if "_seed" in os.path.basename(d):
        continue  # seed replications are not sweep rows
    try:
        k = int(d.split("udprk")[1].split("_")[0])
    except (IndexError, ValueError):
        continue
    entry = {"K": k}
    for kind in ("best", "last"):
        mfiles = sorted(_g.glob(f"{d}/inference_*{kind}*_vhr10/metrics.json")
                        or _g.glob(f"{d}/inference_{kind}_vhr10/metrics.json"))
        if mfiles:
            m = json.load(open(mfiles[-1]))
            entry[kind] = (m["segm/mAP"], m["bbox/mAP"], (m["segm/mAP"] + m["bbox/mAP"]) / 2)
    if entry.get("best") or entry.get("last"):
        rows.append(entry)
rows.sort(key=lambda r: r["K"])
print(f"{'K':>4} | {'best segm/bbox/comp':>23} | {'last segm/bbox/comp':>23}")
for r in rows:
    b = "%.4f/%.4f/%.4f" % r["best"] if r.get("best") else "(pending)"
    l = "%.4f/%.4f/%.4f" % r["last"] if r.get("last") else "(pending)"
    print(f"{r['K']:>4} | {b:>23} | {l:>23}")
print("K=64 row = the existing a3 matrix300 run (0.6689 plateau anchor).")
PY
