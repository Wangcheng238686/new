#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MATRIX=(
  b0_aggregator_mlp.sh
  b1_pafpn_mlp.sh
  m0_aggregator_mlp_full_image.sh
  m1_pafpn_mlp_full_image.sh
  c1_aggregator_coarse_points.sh
  c2_pafpn_coarse_points.sh
  c3_pafpn_coarse_points_box.sh
  c4_pafpn_coarse_points_box_dense.sh
  c5_pafpn_coarse_densebr.sh
  r0_b0_aggregator_mlp_emb64.sh
  r1_c5_pafpn_coarse_densebr_emb64.sh
)

echo "[1/3] config and command penetration checks"
for script in "${MATRIX[@]}"; do
  echo "---- ${script}"
  DRY_RUN=1 CHECK_DATA=0 PREFLIGHT_MODEL=0 \
    bash "${SCRIPT_DIR}/${script}"
done

echo "[2/3] independent train/val subset data check"
DRY_RUN=1 CHECK_DATA=1 PREFLIGHT_MODEL=0 \
TRAIN_SUBSET_RATIO="${SMOKE_TRAIN_SUBSET_RATIO:-0.01}" \
VAL_SUBSET_RATIO="${SMOKE_VAL_SUBSET_RATIO:-0.02}" \
  bash "${SCRIPT_DIR}/c5_pafpn_coarse_densebr.sh"

if [ "${FULL_MODEL_SMOKE:-1}" = "1" ]; then
  echo "[3/3] representative full model construction checks"
  for script in \
    b0_aggregator_mlp.sh \
    b1_pafpn_mlp.sh \
    m0_aggregator_mlp_full_image.sh \
    m1_pafpn_mlp_full_image.sh \
    c1_aggregator_coarse_points.sh \
    c5_pafpn_coarse_densebr.sh \
    r0_b0_aggregator_mlp_emb64.sh \
    r1_c5_pafpn_coarse_densebr_emb64.sh
  do
    echo "---- model ${script}"
    DRY_RUN=1 CHECK_DATA=0 PREFLIGHT_MODEL=1 \
      bash "${SCRIPT_DIR}/${script}"
  done
else
  echo "[3/3] full model construction skipped (FULL_MODEL_SMOKE=0)"
fi

echo "all ablation smoke checks: OK"
