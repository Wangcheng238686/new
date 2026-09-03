#!/usr/bin/env bash
# Inference-time ablation matrix for the WHU fast-150 best checkpoint.
# Six configs, two sequential tasks per GPU on GPUs 1/2/3 (parallel across
# GPUs), sharing the cards with the vhr10_large600 training run.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
source "${ROOT}/scripts/load_environment.sh"

CKPT="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/paper_promptminer_rd_p2_whu_full_fast_tr1.0_va1.0/best_model.pth"
OUT_BASE="$(dirname "${CKPT}")"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PORTABLE_SAM2_LOG_ROOT}/ablations"
mkdir -p "${LOG_DIR}"

run_probe () {  # gpu ablation extra_flag out_name
  local gpu="$1" ablation="$2" extra="$3" name="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" inference/probe_decoder_pathways.py \
    --checkpoint "${CKPT}" \
    --split validation \
    --batch-size 1 \
    --ablation "${ablation}" ${extra} \
    --output-dir "${OUT_BASE}/ablation_${name}" \
    > "${LOG_DIR}/ablate_${name}_${TS}.log" 2>&1
}

# GPU 1: baseline + dense-prompt ablation
( run_probe 1 none        ""                       baseline        && \
  run_probe 1 zero_base_dense ""                   no_dense        ) &
P1=$!
# GPU 2: P2 off + sparse-prompt ablation
( run_probe 2 none        "--disable-p2"           no_p2           && \
  run_probe 2 no_sparse   ""                       no_sparse       ) &
P2=$!
# GPU 3: high-res features + combo
( run_probe 3 zero_high_res ""                     no_highres      && \
  run_probe 3 zero_base_dense "--disable-p2"       no_p2_no_dense  ) &
P3=$!

wait ${P1} ${P2} ${P3}
echo "all ablations done"
