#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"
SMOKE_LOG_SENTINEL="${PORTABLE_SAM2_TMP_ROOT}/smoke_$$.log"
if [[ -e "${SMOKE_LOG_SENTINEL}" ]]; then
  echo "unexpected pre-existing smoke log sentinel: ${SMOKE_LOG_SENTINEL}" >&2
  exit 1
fi
export LOG_FILE="${SMOKE_LOG_SENTINEL}"
MATRIX=(
  b0_aggregator_mlp.sh
  b1_pafpn_mlp.sh
  m0_aggregator_mlp_full_image.sh
  m1_pafpn_mlp_full_image.sh
  c1_aggregator_coarse_points.sh
  c2_pafpn_coarse_points.sh
  c2l_pafpn_coarse_points_roi_loss.sh
  c2r_pafpn_coarse_points_roi_sam.sh
  c3_pafpn_coarse_points_box.sh
  c4_pafpn_coarse_points_box_dense.sh
  c4_pafpn_coarse_points_box_dense_densefix.sh
  c4_pafpn_coarse_points_box_dense_densefix_unfreeze.sh
  r1_c4_pafpn_coarse_points_box_dense_emb64.sh
  r0_b0_aggregator_mlp_emb64.sh
  c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh
)

echo "[1/4] CPU/Gloo delayed rank-0 control-plane check"
"${PYTHON}" -m torch.distributed.run \
  --standalone --nproc_per_node=2 \
  "${SCRIPT_DIR}/smoke_ddp_control_plane.py"

echo "[2/4] config and command penetration checks"
for script in "${MATRIX[@]}"; do
  echo "---- ${script}"
  smoke_output="$(DRY_RUN=1 CHECK_DATA=0 PREFLIGHT_MODEL=0 \
    bash "${SCRIPT_DIR}/${script}")"
  printf '%s\n' "${smoke_output}"
  grep -q '^ema_enabled=0$' <<<"${smoke_output}"
  grep -q '^ema_eval=0$' <<<"${smoke_output}"
  grep -q '^ema_save_best=0$' <<<"${smoke_output}"
  grep -q '^ddp_control_backend=gloo$' <<<"${smoke_output}"
  grep -q '^ddp_control_timeout_seconds=86400$' <<<"${smoke_output}"
  grep -q '^val_every_n_epochs=1$' <<<"${smoke_output}"
  grep -q '^compute_val_loss=0$' <<<"${smoke_output}"
  grep -q '^nproc_per_node=2$' <<<"${smoke_output}"
  grep -q '^cuda_visible_devices=1,2$' <<<"${smoke_output}"
  grep -q '^grad_accum_steps=4$' <<<"${smoke_output}"
  grep -q '^effective_global_batch_size=8$' <<<"${smoke_output}"
  grep -Fq "environment_config=${PORTABLE_SAM2_ENV_FILE}" <<<"${smoke_output}"
  if [[ "${script}" == "c4_pafpn_coarse_points_box_dense_densefix_unfreeze.sh" ]]; then
    grep -q '^prompt_encoder_train_mask_downscaling=1$' <<<"${smoke_output}"
    grep -q '^prompt_encoder_lr_mult=1.0$' <<<"${smoke_output}"
    grep -q '"prompt_encoder_train_mask_downscaling": true' <<<"${smoke_output}"
  else
    grep -q '^prompt_encoder_train_mask_downscaling=0$' <<<"${smoke_output}"
  fi
done

val_loss_override_output="$(DRY_RUN=1 CHECK_DATA=0 PREFLIGHT_MODEL=0 \
  COMPUTE_VAL_LOSS=1 NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  GRAD_ACCUM_STEPS=2 bash "${SCRIPT_DIR}/b0_aggregator_mlp.sh")"
grep -q '^compute_val_loss=1$' <<<"${val_loss_override_output}"
grep -q -- '--compute-val-loss 1' <<<"${val_loss_override_output}"
grep -q '^nproc_per_node=4$' <<<"${val_loss_override_output}"
grep -q '^cuda_visible_devices=0,1,2,3$' <<<"${val_loss_override_output}"
grep -q '^grad_accum_steps=2$' <<<"${val_loss_override_output}"
grep -q '^effective_global_batch_size=8$' <<<"${val_loss_override_output}"

echo "[3/4] independent train/val subset data check"
DRY_RUN=1 CHECK_DATA=1 PREFLIGHT_MODEL=0 \
TRAIN_SUBSET_RATIO="${SMOKE_TRAIN_SUBSET_RATIO:-0.01}" \
VAL_SUBSET_RATIO="${SMOKE_VAL_SUBSET_RATIO:-0.02}" \
  bash "${SCRIPT_DIR}/c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh"

if [ "${FULL_MODEL_SMOKE:-1}" = "1" ]; then
  echo "[4/4] representative full model construction checks"
  for script in \
    b0_aggregator_mlp.sh \
    b1_pafpn_mlp.sh \
    m0_aggregator_mlp_full_image.sh \
    m1_pafpn_mlp_full_image.sh \
    c1_aggregator_coarse_points.sh \
    c2l_pafpn_coarse_points_roi_loss.sh \
    c2r_pafpn_coarse_points_roi_sam.sh \
    c4_pafpn_coarse_points_box_dense_densefix_unfreeze.sh \
    r1_c4_pafpn_coarse_points_box_dense_emb64.sh \
    r0_b0_aggregator_mlp_emb64.sh \
    c5v2_pafpn_coarse_p2_boundary_refiner_emb64.sh
  do
    echo "---- model ${script}"
    DRY_RUN=1 CHECK_DATA=0 PREFLIGHT_MODEL=1 \
      bash "${SCRIPT_DIR}/${script}"
  done
else
  echo "[4/4] full model construction skipped (FULL_MODEL_SMOKE=0)"
fi

if [[ -e "${SMOKE_LOG_SENTINEL}" ]]; then
  echo "smoke unexpectedly created a log file: ${SMOKE_LOG_SENTINEL}" >&2
  exit 1
fi

echo "all ablation smoke checks: OK"
