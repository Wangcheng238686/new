#!/usr/bin/env bash
# Strict post-run evaluator for NWPU P2-v2 dev arms.  The checkpoint embeds
# the VHR-10 data contract; inference therefore replays its saved 10-class
# split rather than inheriting WHU shell defaults.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/load_environment.sh"

ARM="${1:?usage: vhr10_p2v2_eval.sh <p|pb|a0|a1|a2|a2e|a3> <best|last>}"
KIND="${2:?usage: vhr10_p2v2_eval.sh <p|pb|a0|a1|a2|a2e|a3> <best|last>}"
case "${ARM}" in
  p)  TAG="vhr10_p2v2_dev100_p_point" ;;
  pb) TAG="vhr10_p2v2_dev100_pb" ;;
  a0) TAG="vhr10_p2v2_dev100_a0_pbm_d5b" ;;
  a1) TAG="vhr10_p2v2_dev100_a1_p2v1" ;;
  a2) TAG="vhr10_p2v2_dev100_a2_r1" ;;
  a2e) TAG="vhr10_p2v2_dev100_a2e_r1_esc030" ;;
  a3) TAG="vhr10_p2v2_dev100_a3_pbm_udprk64" ;;
  *) echo "Unknown arm ${ARM}" >&2; exit 2 ;;
esac
case "${KIND}" in
  best) NAME="best_model.pth" ;;
  last) NAME="last_model_epoch100.pth" ;;
  *) echo "Checkpoint kind must be best or last" >&2; exit 2 ;;
esac

RUN_DIR="${PORTABLE_SAM2_CHECKPOINT_ROOT}/ablations/${TAG}_tr1.0_va1.0"
CKPT="${RUN_DIR}/${NAME}"
[[ -f "${CKPT}" ]] || { echo "Missing checkpoint: ${CKPT}" >&2; exit 1; }

# The shared environment owns the training-visible device list.  Evaluation
# may explicitly pin one independent process to a physical GPU without
# changing checkpoint semantics, e.g. EVAL_CUDA_VISIBLE_DEVICES=0.
if [[ -n "${EVAL_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}"
fi

exec "${PYTHON}" "${PROJECT_ROOT}/inference/infer_from_checkpoint.py" \
  --checkpoint "${CKPT}" --split validation --weights model \
  --output-dir "${RUN_DIR}/inference_${KIND}_vhr10" \
  --export-bootstrap-records --device cuda:0
