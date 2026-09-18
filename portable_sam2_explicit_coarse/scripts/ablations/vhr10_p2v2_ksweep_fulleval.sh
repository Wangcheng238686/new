#!/usr/bin/env bash
# Single-GPU full-COCO re-eval for the 15-run three-base a3m K sweep.
#
# Re-runs inference for every (base, K, best|last) checkpoint — 30 evals —
# through the canonical protocol entry (same as the sweep queue used), pinned
# to ONE GPU so the exports share one uniform device context.  Overwrites the
# existing inference_{kind}_vhr10 dirs (idempotent; dt/gt records and manifests
# are re-exported too).  Afterwards prints the complete COCO table.
#
# Usage:
#   EVAL_CUDA_VISIBLE_DEVICES=1 nohup bash scripts/ablations/vhr10_p2v2_ksweep_fulleval.sh \
#     > /tmp/ksweep_fulleval.log 2>&1 &
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-1}"
export PORTABLE_SAM2_CHECKPOINT_ROOT="${PORTABLE_SAM2_CHECKPOINT_ROOT:-/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse}"

LABELS=(a0_e300init a0_bests_e105init a0_bestc_e265init)
KS=(16 32 64 128 256)
FAIL=0
for label in "${LABELS[@]}"; do
  for K in "${KS[@]}"; do
    for kind in best last; do
      echo "===== eval base=${label} K=${K} ${kind} $(date +%H:%M:%S) ====="
      if ! ( cd "${REPO_ROOT}" && P2V2_EVAL_PROTOCOL=ksweep40 \
          A3M_SUFFIX="a3m_pbm_udprk${K}tsm_${label}" \
          EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
          bash scripts/ablations/vhr10_p2v2_eval.sh a3m "${kind}" ) \
          > "/tmp/ksweep_eval_${label}_k${K}_${kind}.out" 2>&1; then
        echo "EVAL FAILED base=${label} K=${K} ${kind} — see /tmp/ksweep_eval_${label}_k${K}_${kind}.out" >&2
        FAIL=$((FAIL+1))
      fi
    done
  done
done
echo "===== fulleval done $(date); failures=${FAIL} ====="

python3 - <<'PY'
import glob, json, os, re
root = os.environ.get("PORTABLE_SAM2_CHECKPOINT_ROOT",
                      "/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse")
SEGM = ["segm/mAP","segm/mAP_50","segm/mAP_75","segm/mAP_m","segm/mAP_l",
        "segm/AR@1","segm/AR@10","segm/AR@100","segm/AR_m","segm/AR_l"]
BBOX = ["bbox/mAP","bbox/mAP_50","bbox/mAP_75","bbox/mAP_m","bbox/mAP_l",
        "bbox/AR@1","bbox/AR@10","bbox/AR@100","bbox/AR_m","bbox/AR_l"]
rows = []
for d in sorted(glob.glob(f"{root}/ablations/vhr10_p2v2_ksweep40_a3m_*_tr1.0_va1.0")):
    m = re.match(r'.*udprk(\d+)tsm_(.+)_tr1.0', os.path.basename(d))
    k, base = int(m.group(1)), m.group(2)
    for kind in ("best", "last"):
        f = f"{d}/inference_{kind}_vhr10/metrics.json"
        if os.path.exists(f):
            j = json.load(open(f))
            rows.append((base, k, kind, [j.get(x) for x in SEGM+BBOX]))
order = {"a0_e300init":0, "a0_bests_e105init":1, "a0_bestc_e265init":2}
rows.sort(key=lambda r: (order.get(r[0],9), r[1], r[2]))
out = "/tmp/ksweep_fulleval_coco.tsv"
with open(out, "w") as fh:
    hdr = ["base","K","ckpt"] + [x.replace("/","_") for x in SEGM+BBOX]
    fh.write("\t".join(hdr)+"\n")
    for base, k, kind, vals in rows:
        cells = [f"{v:.4f}" if isinstance(v,(int,float)) else "--" for v in vals]
        fh.write("\t".join([base,str(k),kind]+cells)+"\n")
print(f"full COCO table written: {out} ({len(rows)} rows)")
PY
