#!/usr/bin/env bash
set -Eeuo pipefail

# PromptMiner-SAM2 (Ours) full-method training entry for NWPU VHR-10 (10 cls).
# Mirrors scripts/run_whu1024_explicit_coarse_4gpu.sh: environment loading +
# wrapper selection. WHU-1024 training paths are untouched — this launcher and
# its wrapper only add NWPU wiring (see scripts/ablations/nwpu10_*.sh).

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"
# shellcheck source=load_environment.sh
source "${PROJECT_ROOT}/scripts/load_environment.sh"

: "${VHR10_DATA_ROOT:?load_environment.sh must export VHR10_DATA_ROOT (configs/environment2.sh)}"

export SAM2_MODEL_SIZE=base_plus
export MAX_EPOCHS="${MAX_EPOCHS:-80}"
export EXPLICIT_PROMPT_MODE="${EXPLICIT_PROMPT_MODE:-points_box_dense}"
export P2_BOUNDARY_REFINER_ENABLED="${P2_BOUNDARY_REFINER_ENABLED:-0}"
export TRAIN_SUBSET_RATIO="${TRAIN_SUBSET_RATIO:-1.0}"
export VAL_SUBSET_RATIO="${VAL_SUBSET_RATIO:-1.0}"
# NWPU outputs stay under their own nwpu10_* subtree; never mix with WHU runs.
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PORTABLE_SAM2_CHECKPOINT_ROOT}/nwpu10/${EXPLICIT_PROMPT_MODE}_p2br${P2_BOUNDARY_REFINER_ENABLED}}"
export RUN_TAG="${RUN_TAG:-nwpu10_${EXPLICIT_PROMPT_MODE}_p2br${P2_BOUNDARY_REFINER_ENABLED}}"

case "${EXPLICIT_PROMPT_MODE}:${P2_BOUNDARY_REFINER_ENABLED}:${SAM_IMAGE_EMBED_STRIDE:-32}" in
  points_box_dense:0:32)
    WRAPPER="scripts/ablations/nwpu10_pafpn_coarse_points_box_dense.sh"
    ;;
  *)
    echo "Unsupported NWPU prompt/refiner/stride combination (add a nwpu10 wrapper first): ${EXPLICIT_PROMPT_MODE}/${P2_BOUNDARY_REFINER_ENABLED}/${SAM_IMAGE_EMBED_STRIDE:-32}" >&2
    exit 2
    ;;
esac

exec bash "${WRAPPER}" "$@"
