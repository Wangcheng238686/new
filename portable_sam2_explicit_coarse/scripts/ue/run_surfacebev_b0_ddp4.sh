#!/usr/bin/env bash
# SURFBEV-B0 1024, one fresh four-GPU DDP run (seed 44). No UAV input.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-$ROOT/data/ue_raw_hero_uav_pairset_20261011}"
if [[ "${B0_SEED:-44}" != "44" || "${B0_EXP_ID:-}" != "EXP-0005" ]]; then
  echo "B0 DDP4 requires seed 44 and B0_EXP_ID=EXP-0005" >&2; exit 2
fi
export B0_SEED=44
export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_TAG="surfacebev_b0_raw1024_ddp4_s44_EXP-0005"
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
  echo "B0 DDP4 wrapper does not accept unregistered trainer arguments" >&2; exit 2
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
echo "B0_DDP4_CONTRACT exp_id=EXP-0005 seed=44 commit=$(git rev-parse HEAD) dirty=0 gpu=$CUDA_VISIBLE_DEVICES world_size=4 run_tag=$RUN_TAG checkpoint_dir=$WORK_DIR"
echo "B0_DDP4_CONTRACT model_fingerprint=$ARCH_FINGERPRINT data_manifest=f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a split_sha256=894df4561b166d37d06d033a6fc326223062384b402f730591b4c76d41f926d5 image_size=1024x1024 epochs=100 batch_per_rank=1 accum=2 effective_batch=8 lr=5e-4 wd=0.05 save_last=1 early_stop=0 fresh=1"
if [[ "${B0_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi
exec /home/wangcheng/miniconda3/bin/conda run --no-capture-output -n cvt2 torchrun --standalone --nnodes=1 --nproc_per_node=4 train/train_rsprompter_fusion.py \
  --config configs/ue_raw_hero_b0.py --run-tag "$RUN_TAG" --data-root "$UE_DATA_ROOT" \
  --use-ue-raw-surface --raw-split-manifest surfbev_b0_split.json --image-size 1024 1024 --batch-size 1 --grad-accum-steps 2 \
  --epochs 100 --early-stopping-patience 0 --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed 44 \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --ema-enabled 0 --ema-eval 0 --ema-save-best 0 --checkpoint-dir "$WORK_DIR" --gpu-id 0
