#!/usr/bin/env bash
# SURFBEV-B0 only.  This script deliberately has no UAV option or path.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-$ROOT/data/ue_raw_hero_uav_pairset_20261011}"
export B0_SEED="${B0_SEED:-44}"
case "$B0_SEED" in
  44) EXPECTED_EXP_ID=EXP-0001 ;;
  45) EXPECTED_EXP_ID=EXP-0002 ;;
  *) echo "B0 seed must be 44 or 45" >&2; exit 2 ;;
esac
if [[ "${B0_EXP_ID:-}" != "$EXPECTED_EXP_ID" ]]; then
  echo "B0_EXP_ID must be $EXPECTED_EXP_ID for seed $B0_SEED" >&2; exit 2
fi
RUN_TAG="surfacebev_b0_raw1024_s${B0_SEED}_${B0_EXP_ID}"
WORK_DIR="/data1/wangcheng/checkpoint/surfbev_b0/${RUN_TAG}"
export RUN_TAG WORK_DIR
export EXPLICIT_PROMPT_MODE=points_box_dense
export P2_BOUNDARY_REFINER_ENABLED=1
export SAM_IMAGE_EMBED_STRIDE=16
export SHAPE_DENSE_TRANSFORM=raw_logits
export SHAPE_DENSE_DETACH=1
export FINAL_MASK_COORDINATE_MODE=roi_local
export SAVE_LAST_MODEL=1
if [[ "$#" -ne 0 ]]; then
  echo "B0 wrapper does not accept unregistered trainer arguments" >&2; exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "B0 full training requires a clean Git worktree" >&2; exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
  echo "B0 output directory already exists: $WORK_DIR" >&2; exit 2
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/accepted_pairs.jsonl" | cut -d ' ' -f 1)" != "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a" ]]; then
  echo "accepted manifest hash mismatch" >&2; exit 2
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/surfbev_b0_split.json" | cut -d ' ' -f 1)" != "894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5" ]]; then
  echo "B0 split hash mismatch" >&2; exit 2
fi
ARCH_FINGERPRINT="$(/data2/wangcheng/envs/cvt2/bin/python -c 'from mmengine.config import Config; from rsprompter.architecture_contract import architecture_contract; print(architecture_contract(Config.fromfile("configs/ue_raw_hero_b0.py").model.to_dict())["model_fingerprint"])')"
if [[ "$ARCH_FINGERPRINT" != "36549a00a8282b2b9903d8c07c7cc9ce6f6e8fba3e93d58212de1999b1aa3dab" ]]; then
  echo "B0 architecture fingerprint drift: $ARCH_FINGERPRINT" >&2; exit 2
fi
echo "B0_CONTRACT exp_id=$B0_EXP_ID seed=$B0_SEED commit=$(git rev-parse HEAD) dirty=0 gpu=$CUDA_VISIBLE_DEVICES run_tag=$RUN_TAG checkpoint_dir=$WORK_DIR"
echo "B0_CONTRACT model_fingerprint=$ARCH_FINGERPRINT data_manifest=f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a split_sha256=894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5 image_size=1024x1024 epochs=100 batch=1 accum=8 lr=5e-4 wd=0.05 save_last=1 early_stop=0"
if [[ "${B0_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi
exec /home/wangcheng/miniconda3/bin/conda run --no-capture-output -n cvt2 python train/train_rsprompter_fusion.py \
  --config configs/ue_raw_hero_b0.py --run-tag "$RUN_TAG" --data-root "$UE_DATA_ROOT" \
  --use-ue-raw-surface --raw-split-manifest surfbev_b0_split.json --image-size 1024 1024 --batch-size 1 --grad-accum-steps 8 \
  --epochs 100 --early-stopping-patience 0 --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed "$B0_SEED" \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --ema-enabled 0 --ema-eval 0 --ema-save-best 0 --checkpoint-dir "$WORK_DIR" --gpu-id 0
