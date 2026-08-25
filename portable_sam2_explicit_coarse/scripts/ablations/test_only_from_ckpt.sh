#!/usr/bin/env bash
# Test-set inference from an arbitrary training checkpoint.
#
# Usage:
#   bash scripts/ablations/test_only_from_ckpt.sh /path/to/xxx.pth [extra trainer args]
#
# Examples:
#   # maxDet=150 (model cap + eval maxDets both set to 150)
#   TEST_MAX_PER_IMG=150 bash scripts/ablations/test_only_from_ckpt.sh \
#       /data1/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/<run_dir>/best_model_epoch98.pth
#
#   # COCO default maxDet=100 (do NOT set TEST_MAX_PER_IMG)
#   bash scripts/ablations/test_only_from_ckpt.sh \
#       /data1/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/<run_dir>/best_model_epoch98.pth
#
#   # run on GPU 1
#   CUDA_VISIBLE_DEVICES=1 TEST_MAX_PER_IMG=150 \
#       bash scripts/ablations/test_only_from_ckpt.sh /path/to/xxx.pth
#
# The checkpoint's embedded config_snapshot / architecture_contract are parsed
# and the launch environment (NECK_TYPE, PROMPT_ROUTE, EXPLICIT_PROMPT_MODE,
# SAM_IMAGE_EMBED_STRIDE, SHAPE_DENSE_*, P2_BOUNDARY_REFINER_ENABLED, ...) is
# reconstructed from them, so the architecture fingerprint matches the
# checkpoint automatically and no manual env wiring is needed.
#
# Optional env overrides (checked before launch):
#   CUDA_VISIBLE_DEVICES   default 0
#   NPROC_PER_NODE         default 1
#   TEST_MAX_PER_IMG       e.g. 150: caps model detections and eval maxDets;
#                          unset = COCO default maxDets=100
#   VAL_COMPAT_MAX_DETS    training-validation only; no effect on --test-only
#   RUN_TAG                default testonly_<ckpt_basename>
#
# Notes:
#   - Only WHU1024 config family checkpoints are supported (the ablation
#     runner hardcodes the WHU COCO dataset flags).
#   - test-only inference needs ~20 GB GPU memory (dense tiles batch hundreds
#     of instances through the SAM2 decoder); expandable_segments is enabled
#     by default to reduce fragmentation.
#   - Results are appended to logs/ablations/testonly_<ckpt>_tr*.log under
#     "Test bbox/mAP" / "Test segm/mAP" lines.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 /path/to/checkpoint.pth [extra trainer args]" >&2
    exit 2
fi
CKPT="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
shift
if [[ ! -f "${CKPT}" ]]; then
    echo "checkpoint not found: ${CKPT}" >&2
    exit 2
fi

# Capture user intent BEFORE the environment loader exports its own defaults.
_USER_CUDA="${CUDA_VISIBLE_DEVICES:-}"
_USER_NPROC="${NPROC_PER_NODE:-}"
source "${PROJECT_ROOT}/scripts/load_environment.sh"
CUDA_VISIBLE_DEVICES="${_USER_CUDA:-0}"
NPROC_PER_NODE="${_USER_NPROC:-1}"
PYTHON="${PYTHON:-/home/wangcheng/miniconda3/envs/cvt2/bin/python}"

# ---- parse the checkpoint and emit the launch environment ----
PARSED="$("${PYTHON}" - "${CKPT}" <<'PYEOF'
import json, sys, os

import torch

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
cs = ckpt.get("config_snapshot") or {}
ta = cs.get("training_args") or {}
mh = cs.get("mask_head_config") or {}
ac = ckpt.get("architecture_contract") or {}
data_cfg = cs.get("data_config") or {}

config_path = str(ta.get("config") or cs.get("config_path") or "")
if "whu1024" not in os.path.basename(config_path):
    sys.exit(
        f"unsupported config family: {config_path!r} "
        "(test_only_from_ckpt.sh only handles WHU1024 checkpoints)"
    )

env = {}
env["ABLATION_ID"] = ac.get("architecture_id") or ""
env["NECK_TYPE"] = cs.get("neck_type") or "aggregator"
env["PROMPT_ROUTE"] = (
    "coarse" if str(cs.get("prompt_generator_mode")) == "explicit_mask" else "mlp"
)
mode = str(mh.get("explicit_prompt_mode") or "none")
env["EXPLICIT_PROMPT_MODE"] = mode
refiner_cfg = mh.get("p2_boundary_refiner_cfg") or {}
env["P2_BOUNDARY_REFINER_ENABLED"] = 1 if refiner_cfg.get("enabled") else 0
embed_img = int(mh.get("prompt_encoder_image_size") or 1024)
embed_size = int(mh.get("prompt_encoder_embed_size") or embed_img // 16)
env["SAM_IMAGE_EMBED_STRIDE"] = embed_img // embed_size
dense_cfg = mh.get("dense_prompt_cfg") or {}
env["SHAPE_DENSE_TRANSFORM"] = str(dense_cfg.get("transform") or "raw_logits")
env["SHAPE_DENSE_DETACH"] = 1 if dense_cfg.get("detach_input") else 0
env["FINAL_MASK_COORDINATE_MODE"] = str(
    mh.get("final_mask_coordinate_mode") or "roi_local"
)
shape_cfg = mh.get("shape_prior_cfg") or {}
env["SHAPE_CONTEXT_FUSION"] = str(shape_cfg.get("fusion_type") or "roi_only")
env["SHAPE_PRIOR_ENABLED"] = 1 if cs.get("shape_prior_enabled") else 0
env["SHAPE_PRIOR_LOSS_WEIGHT"] = str(mh.get("shape_prior_loss_weight") or 0.10)
roi_sam_cfg = mh.get("roi_sam_cfg") or {}
env["ROI_SAM_ENABLED"] = 1 if roi_sam_cfg.get("enabled") else 0
final_loss_cfg = mh.get("final_mask_loss_cfg") or {}
if str(final_loss_cfg.get("mode") or "standard") != "standard":
    env["FINAL_MASK_LOSS_MODE"] = str(final_loss_cfg["mode"])
env["MAX_EPOCHS"] = int(ta.get("epochs") or cs.get("epochs") or 100)
env["SUBSET_SEED"] = int(ta.get("seed") or cs.get("seed") or 44)
env["TRAIN_SUBSET_RATIO"] = float(cs.get("train_subset_ratio") or 1.0)
env["VAL_SUBSET_RATIO"] = float(cs.get("val_subset_ratio") or 1.0)

print("#begin")
for k, v in env.items():
    print(f"{k}={v}")
print(f"#epoch={ckpt.get('epoch')}")
print(f"#config={config_path}")
print(f"#data_root={ta.get('data_root') or data_cfg.get('data_root')}")
best = ckpt.get("best_metrics") or {}
print(f"#best_segm={best.get('best_segm_map')}")
print(f"#fingerprint={ac.get('model_fingerprint')}")
PYEOF
)"

if [[ "${PARSED}" != "#begin"* ]]; then
    echo "${PARSED}" >&2
    exit 2
fi
eval "$(echo "${PARSED}" | grep -v '^#' | grep -v '^$' | sed 's/^/export /')"
echo "${PARSED}" | grep '^#' | grep -v '^#begin' | sed 's/^#/[ckpt] /'
echo "${PARSED}" | grep -v '^#' | grep -v '^$' | sed 's/^/[arch] /'

export RUN_TAG="${RUN_TAG:-testonly_$(basename "${CKPT}" .pth)}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export RUN_IN_BACKGROUND="${RUN_IN_BACKGROUND:-1}"

echo "[env] CUDA=${CUDA_VISIBLE_DEVICES} nproc=${NPROC_PER_NODE} run_tag=${RUN_TAG}"
[[ -n "${TEST_MAX_PER_IMG:-}" ]] && echo "[env] TEST_MAX_PER_IMG=${TEST_MAX_PER_IMG}"

exec bash "${SCRIPT_DIR}/_run_ablation.sh" -- --resume-from "${CKPT}" --test-only "$@"
