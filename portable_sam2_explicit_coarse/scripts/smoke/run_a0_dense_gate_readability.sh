#!/usr/bin/env bash
# Frozen A0 train-520 -> val-130 pre-PromptEncoder dense-gate readability gate.
# This script is intentionally NOT a queue and never background-launches: run it
# only after the user has freed one GPU.  It neither trains nor writes checkpoints.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}"
GPU="${DENSE_GATE_GPU:-0}"
CKPT="${A0_CHECKPOINT:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/best_model_epoch36.pth}"
CANONICAL="${A0_CANONICAL_DIR:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/vhr10_p2v2_dev100_a0_pbm_d5b_tr1.0_va1.0/inference_best_vhr10}"
OUT="${DENSE_GATE_OUT:-/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/diagnostics/a0_dense_gate_readability_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$CKPT" || ! -d "$CANONICAL" ]]; then
  echo "missing A0 checkpoint or canonical inference directory" >&2; exit 2
fi
if [[ -e "$OUT" ]]; then echo "refusing existing output: $OUT" >&2; exit 2; fi
mkdir -p "$OUT"
cd "$ROOT"

# Validation keeps the legacy-A0 canonical parity anchor.  Train has no
# canonical inference artifact, so skip only that redundant anchor; the probe
# itself still asserts current/forced detector hashes are identical.
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u inference/probes/canvas_accuracy_probe.py \
  --checkpoint "$CKPT" --device cuda:0 --split validation --canvas-modes learned \
  --gate-alphas current 1.0 --canonical-dir "$CANONICAL" \
  --export-pre-prompt-features --output-dir "$OUT/val"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u inference/probes/canvas_accuracy_probe.py \
  --checkpoint "$CKPT" --device cuda:0 --split train \
  --ann-file coco_split/NWPU_instances_train.json --image-subdir 'positive image set' \
  --canvas-modes learned --gate-alphas current 1.0 --skip-parity-check \
  --export-pre-prompt-features --output-dir "$OUT/train"

"$PYTHON_BIN" -u inference/probes/run_dense_gate_readout.py \
  --train-current "$OUT/train/learned_gate_current" --train-forced "$OUT/train/learned_gate_1.0" \
  --val-current "$OUT/val/learned_gate_current" --val-forced "$OUT/val/learned_gate_1.0" \
  --output-dir "$OUT/readout"

# Full-feature must beat raw and the aligned-appearance row-permutation control.
for arm in raw permuted; do
  OMP_NUM_THREADS=1 "$PYTHON_BIN" -u tools/bootstrap_paired_map.py \
    --baseline-dt "$OUT/readout/$arm/dt_records.json" \
    --baseline-manifest "$OUT/readout/$arm/run_manifest.json" \
    --treatment-dir "$OUT/readout/full" --resamples 500 --seed 44 \
    --output "$OUT/bootstrap_full_vs_${arm}.json"
done
echo "PASS complete artifacts: $OUT"
