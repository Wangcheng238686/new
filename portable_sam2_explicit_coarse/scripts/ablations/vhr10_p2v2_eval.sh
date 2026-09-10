#!/usr/bin/env bash
# Strict post-run evaluator for NWPU P2-v2 arms.  The checkpoint embeds the
# VHR-10 data contract; inference therefore replays its saved 10-class split
# rather than inheriting WHU shell defaults.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/load_environment.sh"

ARM="${1:?usage: vhr10_p2v2_eval.sh <p|pb|a0|a1|a2|a2e|a3|a3r|a4|r3|r3_udpr> <best|best_bbox|best_composite|last>}"
KIND="${2:?usage: vhr10_p2v2_eval.sh <p|pb|a0|a1|a2|a2e|a3|a3r|a4|r3|r3_udpr> <best|best_bbox|best_composite|last>}"
# Keep dev100 as the historical default.  Formal protocol callers must set
# P2V2_EVAL_PROTOCOL explicitly, so a matrix300 result cannot silently
# evaluate a same-named dev100 directory.
PROTOCOL="${P2V2_EVAL_PROTOCOL:-dev100}"
case "${PROTOCOL}" in
  dev100) LAST_EPOCH=100 ;;
  matrix300) LAST_EPOCH=300 ;;
  full600) LAST_EPOCH=600 ;;
  *) echo "P2V2_EVAL_PROTOCOL must be dev100, matrix300, or full600" >&2; exit 2 ;;
esac
case "${ARM}" in
  p)  SUFFIX="p_point" ;;
  pb) SUFFIX="pb" ;;
  a0) SUFFIX="a0_pbm_d5b" ;;
  a1) SUFFIX="a1_p2v1" ;;
  a2) SUFFIX="a2_r1" ;;
  a2e) SUFFIX="a2e_r1_esc030" ;;
  a3) SUFFIX="a3_pbm_udprk64" ;;
  a4) SUFFIX="a4_pbm_udprcgk64" ;;
  a3r) SUFFIX="a3_pbm_udprk64_maskramp100" ;;
  r3) SUFFIX="pb_r3" ;;
  r3_udpr) SUFFIX="pb_r3_udprk64" ;;
  *) echo "Unknown arm ${ARM}" >&2; exit 2 ;;
esac
TAG="vhr10_p2v2_${PROTOCOL}_${SUFFIX}"
case "${KIND}" in
  best) NAME="best_model.pth" ;;
  best_bbox) NAME="best_bbox_model.pth" ;;
  best_composite) NAME="best_composite_model.pth" ;;
  last) NAME="last_model_epoch${LAST_EPOCH}.pth" ;;
  *) echo "Checkpoint kind must be best, best_bbox, best_composite, or last" >&2; exit 2 ;;
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
