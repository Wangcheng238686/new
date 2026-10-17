#!/usr/bin/env bash
# EXP-0018/0019: WARM-R1G full runs, dual-seed parallel (one wrapper per run).
# Recipe = the screen-verified R1G arm (EXP-0015 base + gate_init 0.001 +
# warm-frozen satellite stack), trained a full 100 epochs on a SINGLE cosine
# from the EXP-0009 warm initialisation (no restart; screen EXT lesson).
# Prereg: docs/EXP-0018-0019_preregistration.md (DDP2 note: global batch 8
# and 87 optimizer steps/epoch match the DDP4 ladder exactly).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen}"
if [[ "${B0_EXP_ID:-}" != "EXP-0018" && "${B0_EXP_ID:-}" != "EXP-0019" ]]; then
  echo "this wrapper only launches EXP-0018 (seed 44) or EXP-0019 (seed 45)" >&2; exit 2
fi
EXP_ID="${B0_EXP_ID}"
SEED="$([[ "$EXP_ID" == "EXP-0018" ]] && echo 44 || echo 45)"
DEVICES="$([[ "$EXP_ID" == "EXP-0018" ]] && echo "0,1" || echo "2,3")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEVICES}"
EPOCHS=100
CONFIG=configs/ue_whu_a3sgm_512_warm_r1g.py
EXPECT_ID=ue-whu-a3sgm-512-warm-r1g-v1
RUN_TAG="surfacebev_v2_512warm_r1g_full100_ddp2_s${SEED}_${EXP_ID}"
WORK_DIR="/data2/wangcheng/checkpoint/surfbev_b0/${RUN_TAG}"
INIT_FROM="/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth"
export RUN_TAG WORK_DIR SAVE_LAST_MODEL=1 UE_LOAD_UAV=1

if [[ -n "$(git status --porcelain)" ]]; then
  echo "full training requires a clean Git worktree" >&2; exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
  echo "output directory already exists: $WORK_DIR" >&2; exit 2
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

echo "EXP0018_19_CONTRACT exp_id=$EXP_ID seed=$SEED commit=$(git rev-parse HEAD) dirty=0 gpu=$CUDA_VISIBLE_DEVICES world_size=2 epochs=$EPOCHS run_tag=$RUN_TAG"
echo "EXP0018_19_CONTRACT baseline_id=$BASELINE_ID init_from=$INIT_FROM freeze_prefixes=uav_encoder.,proj,gate,bev_head batch=1x2ranks accum=4 global_batch=8 opt_steps_per_epoch=87 data_manifest=f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a split=17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5 prereg=docs/EXP-0018-0019_preregistration.md"

exec /data2/wangcheng/envs/cvt2/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 train/train_rsprompter_fusion.py \
  --config "$CONFIG" --run-tag "$RUN_TAG" --data-root "$UE_DATA_ROOT" \
  --dataset ue_raw_surface --raw-split-manifest surfbev_b0_split_v2.json --image-size 512 512 \
  --batch-size 1 --grad-accum-steps 4 --epochs "$EPOCHS" --early-stopping-patience 0 \
  --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed "$SEED" \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --fusion-no-wd 1 \
  --init-from "$INIT_FROM" --allow-task-init --allow-cross-arch-init \
  --init-allowed-missing-prefixes "uav_encoder.,proj,gate,bev_head,height_edges" \
  --freeze-base --freeze-allow-prefixes "uav_encoder.,proj,gate,bev_head" \
  --ema-enabled 0 --ema-eval 0 --checkpoint-dir "$WORK_DIR"
