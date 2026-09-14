#!/usr/bin/env bash
# K-sensitivity sweep runner (protocol ksweep40, arm a3m heat-start).
#
# Trains one frozen-A0-base tail-only run per K value, then evaluates the
# best and last checkpoints, and finally prints a summary table.  Designed to
# be portable across machines: the A0 matrix300 E300 base checkpoint path is
# overridable via A3M_INIT_FROM and its presence is checked up front.
#
# Usage (from the portable_sam2_explicit_coarse repo root):
#   bash scripts/ablations/vhr10_p2v2_ksweep.sh 16 32 64 128 256
#
# Environment:
#   A3M_INIT_FROM   path to the frozen A0 base checkpoint
#                   (default: canonical-machine path below)
#   NPROC_PER_NODE / GPU visibility are honoured exactly as in vhr10_p2v2_dev.sh.
#
# Notes:
# - Each run consumes all visible GPUs (same entry as every other arm).
# - The queue is interruptible: best checkpoints save continuously; killing
#   this script plus the active training is safe.  Re-running a K whose
#   output dir already exists will refuse (delete the dir to redo).
# - Cross-machine numbers carry that machine's systematic offset; compare
#   WITHIN this sweep only (deltas across K), not against canonical-machine
#   rows without calibration.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_INIT="/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_matrix300_a0_pbm_d5b_tr1.0_va1.0/last_model_epoch300.pth"
INIT_FROM="${A3M_INIT_FROM:-${DEFAULT_INIT}}"

if [ "$#" -eq 0 ]; then
  echo "usage: vhr10_p2v2_ksweep.sh <K> [<K> ...]   e.g. 16 32 64 128 256" >&2
  exit 2
fi
if [ ! -f "${INIT_FROM}" ]; then
  echo "ERROR: frozen A0 base checkpoint not found: ${INIT_FROM}" >&2
  echo "       Transfer it (or point A3M_INIT_FROM at this machine's copy)." >&2
  exit 1
fi
export A3M_INIT_FROM="${INIT_FROM}"
export P2V2_PROTOCOL=ksweep40

for K in "$@"; do
  case "${K}" in
    ''|*[!0-9]*) echo "ERROR: K must be a positive integer, got '${K}'" >&2; exit 2 ;;
  esac
  TAG="vhr10_p2v2_ksweep40_a3m_pbm_udprk${K}tsm_a0_e300init_tr1.0_va1.0"
  if [ -n "$(ls -A "${PORTABLE_SAM2_CHECKPOINT_ROOT:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse}/ablations/${TAG}" 2>/dev/null)" ]; then
    echo "=== K=${K}: checkpoint dir already non-empty, SKIPPING (delete to redo) ==="
    continue
  fi
  echo "===== a3m K=${K}: train start $(date) ====="
  ( cd "${REPO_ROOT}" && A3M_HEATSTART=1 A3M_NUM_POINTS="${K}" \
      bash scripts/ablations/vhr10_p2v2_dev.sh a3m )
  rc=$?
  if [ ${rc} -ne 0 ]; then echo "K=${K} TRAIN FAILED rc=${rc} — ABORT sweep" >&2; exit 1; fi
  echo "===== a3m K=${K}: eval $(date) ====="
  for kind in best last; do
    ( cd "${REPO_ROOT}" && P2V2_EVAL_PROTOCOL=ksweep40 \
        A3M_SUFFIX="a3m_pbm_udprk${K}tsm_a0_e300init" \
        bash scripts/ablations/vhr10_p2v2_eval.sh a3m "${kind}" ) \
      || echo "K=${K} eval ${kind} FAILED (non-fatal)"
  done
done

echo "===== K-SWEEP COMPLETE $(date) — summary ====="
python3 - <<'PY'
import glob, json, os
root = os.environ.get("PORTABLE_SAM2_CHECKPOINT_ROOT",
                      "/data/wangcheng/checkpoint/portable_sam2_explicit_coarse")
rows = []
for d in sorted(glob.glob(f"{root}/ablations/vhr10_p2v2_ksweep40_a3m_pbm_udprk*tsm_a0_e300init_tr1.0_va1.0")):
    k = d.split("udprk")[1].split("tsm")[0]
    entry = {"K": int(k)}
    for kind in ("best", "last"):
        mfile = f"{d}/inference_{kind}_vhr10/metrics.json"
        if os.path.exists(mfile):
            m = json.load(open(mfile))
            entry[kind] = (m["segm/mAP"], m["bbox/mAP"], m.get("bbox/mAP_75", float("nan")))
    if entry.get("best") or entry.get("last"):
        rows.append(entry)
rows.sort(key=lambda r: r["K"])
print(f"{'K':>4} | {'best segm/bbox':>17} | {'last segm/bbox':>17}")
for r in rows:
    b = "%+.4f/%+.4f" % r["best"][:2] if r.get("best") else "(pending)"
    l = "%+.4f/%+.4f" % r["last"][:2] if r.get("last") else "(pending)"
    print(f"{r['K']:>4} | {b:>17} | {l:>17}")
print("NOTE: within-machine deltas only (K vs K); cross-machine offsets apply.")
PY
