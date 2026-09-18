#!/usr/bin/env bash
# machine2-local serial launcher: three-base a3m K-sensitivity sweep (ksweep40).
#
# What it queues, in order (each base sweeps K serially; every K run is a
# 40-epoch tail-only heat-start off a FROZEN A0 matrix300 base, then best/last
# eval; the per-K loop, skip-if-done and summary table live in
# vhr10_p2v2_ksweep.sh):
#   last  = imported last_model_epoch300.pth           (main reading caliber)
#   bests = imported best_model_epoch105.pth           (bestS)
#   bestc = imported best_composite_model_epoch265.pth (bestC)
# Bases are the IMPORTED (other-machine) A0 weights the user placed at
# /data1/wangcheng/checkpoint/vhr10_p2v2_matrix300_a0_pbm_d5b_tr1.0_va1.0.
# Outputs still go to THIS machine's default checkpoint root.  Per the ksweep
# protocol: compare WITHIN a base (K vs K) only; absolute numbers carry this
# machine's offset and a foreign base.
#
# GPU plan (user ruling 2026-09-15): four cards even though GPU 1/2 host
# external jobs (~18/21GB used of 48GB; ranks fit alongside).  Effective
# batch stays 8 (4x1x2).  To fall back to the 2-card form export
# CUDA_VISIBLE_DEVICES=0,3 NPROC_PER_NODE=2 GRAD_ACCUM_STEPS=4 before calling.
#
# Usage:
#   bash scripts/ablations/vhr10_p2v2_ksweep_three_bases.sh [run] [last|bests|bestc|all]
#     - no "run": dry mode, prints the plan only
#     - base selector optional, default all (serial: last -> bests -> bestc)
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ksweep.sh's skip-check and summary read this BEFORE the runner loads the
# machine env file; point it at machine2's root explicitly.
export PORTABLE_SAM2_CHECKPOINT_ROOT="${PORTABLE_SAM2_CHECKPOINT_ROOT:-/data1/wangcheng/checkpoint/portable_sam2_explicit_coarse}"
A0DIR="/data1/wangcheng/checkpoint/vhr10_p2v2_matrix300_a0_pbm_d5b_tr1.0_va1.0"

BASE_last="${A0DIR}/last_model_epoch300.pth"
BASE_bests="${A0DIR}/best_model_epoch105.pth"
BASE_bestc="${A0DIR}/best_composite_model_epoch265.pth"
LABEL_last="a0_e300init"
LABEL_bests="a0_bests_e105init"
LABEL_bestc="a0_bestc_e265init"
KS_LIST="${KS_LIST:-16 32 64 128 256}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"

MODE="${1:-plan}"
SEL="${2:-all}"
case "${MODE}" in run|plan) ;; *) echo "usage: $0 [run|plan] [last|bests|bestc|all]" >&2; exit 2 ;; esac
case "${SEL}" in last|bests|bestc|all) ;; *) echo "unknown base '${SEL}'" >&2; exit 2 ;; esac

SELECTED=()
[ "${SEL}" = "all" ] || [ "${SEL}" = "last" ]  && SELECTED+=("last")
[ "${SEL}" = "all" ] || [ "${SEL}" = "bests" ] && SELECTED+=("bests")
[ "${SEL}" = "all" ] || [ "${SEL}" = "bestc" ] && SELECTED+=("bestc")

echo "===== ksweep three-base queue (${MODE}) ====="
echo "  gpus=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} accum=${GRAD_ACCUM_STEPS} (effective $((NPROC_PER_NODE*GRAD_ACCUM_STEPS)))"
echo "  K list: ${KS_LIST}"
FAILED=()
for b in "${SELECTED[@]}"; do
  var_base="BASE_${b}"; var_label="LABEL_${b}"
  path="${!var_base}"; label="${!var_label}"
  echo "--- base ${b}: ${label} <- ${path}"
  if [ ! -f "${path}" ]; then
    echo "    MISSING checkpoint, base ${b} skipped" >&2
    FAILED+=("${b}:missing-base")
    continue
  fi
  if [ "${MODE}" != "run" ]; then continue; fi
  echo "===== base ${b} (${label}) sweep start $(date) ====="
  if ( cd "${REPO_ROOT}" && A3M_INIT_FROM="${path}" A3M_INIT_LABEL="${label}" \
       bash "${SCRIPT_DIR}/vhr10_p2v2_ksweep.sh" ${KS_LIST} ) \
       2>&1 | tee "/tmp/ksweep_${label}.log"; then
    echo "===== base ${b} (${label}) sweep done $(date) ====="
  else
    echo "BASE ${b} SWEEP FAILED — continuing with next base" >&2
    FAILED+=("${b}:sweep-failed")
  fi
done

echo "===== queue ${MODE} complete $(date); failures: ${FAILED[*]:-none} ====="
[ "${#FAILED[@]}" -eq 0 ]
