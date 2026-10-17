#!/usr/bin/env bash
# DUAL-WARMSTART-SCREEN-20261017 (docs/DUAL-WARMSTART-SCREEN-20261017_preregistration.md)
# Warm-start screening arms on the EXP-0015 recipe: init the whole satellite
# stack from EXP-0009 last and freeze it (only uav_encoder./proj/gate[+bev_head]
# train) - the UAV-off forward is then bit-identical to EXP-0009 by
# construction, so dual-single = on-off. Screening budget 30 epochs, seed 44.
# Usage: run_warm_screen.sh {r1|r1g|r1b} [--resume]
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ARM="${1:?usage: run_warm_screen.sh r1-or-r1g-or-r1b}"
RESUME="${2:-}"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# 30 = preregistered screen budget; 60 = the one-time SCREEN-EXT extension
# (identical warm-restart schedule for every extended arm; see conclusion doc)
EPOCHS="${WARM_EPOCHS:-30}"

case "$ARM" in
  r1)  CONFIG=configs/ue_whu_a3sgm_512_dualv2_l0_lora.py
       EXPECT_ID=ue-whu-a3sgm-512-dualv2-l0-lora-v1 ;;
  r1g) CONFIG=configs/ue_whu_a3sgm_512_warm_r1g.py
       EXPECT_ID=ue-whu-a3sgm-512-warm-r1g-v1 ;;
  r1b) CONFIG=configs/ue_whu_a3sgm_512_warm_r1b.py
       EXPECT_ID=ue-whu-a3sgm-512-warm-r1b-v1 ;;
  r1c) CONFIG=configs/ue_whu_a3sgm_512_warm_r1c.py
       EXPECT_ID=ue-whu-a3sgm-512-warm-r1c-v1 ;;
  r1ch) CONFIG=configs/ue_whu_a3sgm_512_warm_r1ch.py
       EXPECT_ID=ue-whu-a3sgm-512-warm-r1ch-v1 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

RUN_TAG="surfacebev_v2_512warm_${ARM}_scr30_ddp4_s44_WARM-20261017"
WORK_DIR="/data2/wangcheng/checkpoint/surfbev_b0/${RUN_TAG}"
INIT_FROM="/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth"
export RUN_TAG WORK_DIR SAVE_LAST_MODEL=1 UE_LOAD_UAV=1

if [[ "$RESUME" != "--resume" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "screen launch requires a clean Git worktree" >&2; exit 2
  fi
  if [[ -e "$WORK_DIR" ]]; then
    echo "output directory already exists: $WORK_DIR" >&2; exit 2
  fi
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/accepted_pairs.jsonl" | cut -d ' ' -f 1)" != "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a" ]]; then
  echo "accepted manifest hash mismatch" >&2; exit 2
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/surfbev_b0_split_v2.json" | cut -d ' ' -f 1)" != "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5" ]]; then
  echo "v2 split hash mismatch" >&2; exit 2
fi
BASELINE_ID="$(/data2/wangcheng/envs/cvt2/bin/python -c "import sys; sys.path.insert(0,'.'); from mmengine.config import Config; from ue.contract import baseline_id_for; print(baseline_id_for(Config.fromfile('$CONFIG').model.to_dict()))")"
if [[ "$BASELINE_ID" != "$EXPECT_ID" ]]; then
  echo "arm $ARM identity drift: $BASELINE_ID != $EXPECT_ID" >&2; exit 2
fi
if [[ ! -f "$INIT_FROM" ]]; then
  echo "EXP-0009 init checkpoint missing: $INIT_FROM" >&2; exit 2
fi

RESUME_ARGS=()
if [[ "$RESUME" == "--resume" ]]; then
  # SCREEN-EXT: the checkpoint already carries the warm-initialised dual
  # state + optimizer groups; --init-from is mutually exclusive with
  # --resume-from and must be dropped (freeze-base is kept and recomputed
  # identically from the config's LoRA trainable set).
  INIT_FLAGS=()
  RESUME_ARGS=(--resume-from "$WORK_DIR/last_checkpoint.pth")
else
  INIT_FLAGS=(--init-from "$INIT_FROM" --allow-task-init --allow-cross-arch-init \
              --init-allowed-missing-prefixes "uav_encoder.,proj,gate,bev_head,bev_attn,height_edges")
fi

echo "WARMSCREEN_CONTRACT arm=$ARM commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) gpu=$CUDA_VISIBLE_DEVICES epochs=$EPOCHS seed=44 run_tag=$RUN_TAG"
echo "WARMSCREEN_CONTRACT baseline_id=$BASELINE_ID init_from=${INIT_FLAGS[1]:-none(resume)} freeze_prefixes=uav_encoder.,proj,gate,bev_head,bev_attn data_manifest=f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a split=17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5 prereg=docs/DUAL-WARMSTART-SCREEN-20261017_preregistration.md"

exec /data2/wangcheng/envs/cvt2/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train/train_rsprompter_fusion.py \
  --config "$CONFIG" --run-tag "$RUN_TAG" --data-root "$UE_DATA_ROOT" \
  --dataset ue_raw_surface --raw-split-manifest surfbev_b0_split_v2.json --image-size 512 512 \
  --batch-size 1 --grad-accum-steps 2 --epochs "$EPOCHS" --early-stopping-patience 0 \
  --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed 44 \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --fusion-no-wd 1 \
  "${INIT_FLAGS[@]}" \
  --freeze-base --freeze-allow-prefixes "uav_encoder.,proj,gate,bev_head,bev_attn" \
  --ema-enabled 0 --ema-eval 0 --checkpoint-dir "$WORK_DIR" "${RESUME_ARGS[@]}"
