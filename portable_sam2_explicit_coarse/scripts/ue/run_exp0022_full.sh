#!/usr/bin/env bash
# EXP-0020/0021: WARM-R1C full runs, dual-seed parallel (one wrapper per run).
# Recipe = the screen-verified R1C arm (R1G + descriptor pathway swap:
# CVT attention descriptors + exact-backprojection stats, PIXEL-CLS A-PASS),
# full 100 epochs on a SINGLE cosine from the EXP-0009 warm init.
# Prereg: docs/EXP-0020-0021_preregistration.md (protocol verbatim from
# EXP-0022/0019: DDP4 co-rent, global batch 8, 87 steps/epoch).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen}"
if [[ "${B0_EXP_ID:-}" != "EXP-0020" && "${B0_EXP_ID:-}" != "EXP-0021" ]]; then
  echo "this wrapper only launches EXP-0020 (seed 44) or EXP-0021 (seed 45)" >&2; exit 2
fi
EXP_ID="${B0_EXP_ID}"
RESUME="${1:-}"
SEED=44
# PI decision 2026-10-17: both seeds run FULL DDP4 (ladder protocol verbatim)
# and co-rent all four GPUs in parallel - world/batch/accum identical to the
# EXP-0009..0017 ladder arms; the only shared-resource effect is wall-clock.
DEVICES="0,1,2,3"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEVICES}"
EPOCHS=100
CONFIG=configs/ue_whu_a3sgm_512_warm_r1c.py
EXPECT_ID=ue-whu-a3sgm-512-warm-r1c-v1
RUN_TAG="surfacebev_v2_512warm_r1c_plateau60_full100_ddp4_s44_EXP-0022"
WORK_DIR="/data2/wangcheng/checkpoint/surfbev_b0/${RUN_TAG}"
INIT_FROM="/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth"
export RUN_TAG WORK_DIR SAVE_LAST_MODEL=1 UE_LOAD_UAV=1

if [[ -n "$(git status --porcelain)" ]]; then
  echo "full training requires a clean Git worktree" >&2; exit 2
fi
if [[ "$RESUME" != "--resume" && -e "$WORK_DIR" ]]; then
  echo "output directory already exists (use --resume to continue): $WORK_DIR" >&2; exit 2
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/accepted_pairs.jsonl" | cut -d ' ' -f 1)" != "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a" ]]; then
  echo "accepted manifest hash mismatch" >&2; exit 2
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/surfbev_b0_split_v2.json" | cut -d ' ' -f 1)" != "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5" ]]; then
  echo "v2 split hash mismatch" >&2; exit 2
fi
BASELINE_ID="$(/data2/wangcheng/envs/cvt2/bin/python -c "import sys; sys.path.insert(0,'.'); from mmengine.config import Config; from ue.contract import baseline_id_for; print(baseline_id_for(Config.fromfile('$CONFIG').model.to_dict()))")"
if [[ "$BASELINE_ID" != "$EXPECT_ID" ]]; then
  echo "identity drift: $BASELINE_ID != $EXPECT_ID" >&2; exit 2
fi
if [[ ! -f "$INIT_FROM" ]]; then
  echo "EXP-0009 init checkpoint missing" >&2; exit 2
fi

RESUME_ARGS=()
INIT_FLAGS=(--init-from "$INIT_FROM" --allow-task-init --allow-cross-arch-init \
            --init-allowed-missing-prefixes "uav_encoder.,proj,gate,bev_head,bev_attn,height_edges")
if [[ "$RESUME" == "--resume" ]]; then
  # crash recovery 2026-10-17: checkpoint carries the warm dual state +
  # optimizer groups; init flags are mutually exclusive with resume and the
  # workdir-existence guard is lifted for the resume path only
  INIT_FLAGS=()
  RESUME_ARGS=(--resume-from "$WORK_DIR/last_checkpoint.pth")
fi
echo "EXP0020_21_CONTRACT exp_id=$EXP_ID seed=$SEED commit=$(git rev-parse HEAD) dirty=0 gpu=$CUDA_VISIBLE_DEVICES world_size=4 epochs=$EPOCHS run_tag=$RUN_TAG"
echo "EXP0020_21_CONTRACT baseline_id=$BASELINE_ID init_from=$INIT_FROM freeze_prefixes=uav_encoder.,proj,gate,bev_head batch=1x4ranks accum=2 global_batch=8 opt_steps_per_epoch=87 cotenant=paired-seed data_manifest=f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a split=17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5 prereg=docs/DUAL-AXIS-SCREEN-20261018_preregistration.md lr_plateau_epoch=60 lr_plateau_mult=0.1"

exec /data2/wangcheng/envs/cvt2/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train/train_rsprompter_fusion.py \
  --config "$CONFIG" --run-tag "$RUN_TAG" --data-root "$UE_DATA_ROOT" \
  --dataset ue_raw_surface --raw-split-manifest surfbev_b0_split_v2.json --image-size 512 512 \
  --batch-size 1 --grad-accum-steps 2 --epochs "$EPOCHS" --early-stopping-patience 0 \
  --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed "$SEED" \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --fusion-no-wd 1 --lr-plateau-epoch 60 --lr-plateau-mult 0.1 \
  "${INIT_FLAGS[@]}" \
  --freeze-base --freeze-allow-prefixes "uav_encoder.,proj,gate,bev_head,bev_attn" \
  --ema-enabled 0 --ema-eval 0 --checkpoint-dir "$WORK_DIR" "${RESUME_ARGS[@]}"
