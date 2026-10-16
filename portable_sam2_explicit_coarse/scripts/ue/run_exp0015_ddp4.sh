#!/usr/bin/env bash
# EXP-0015: DUAL-WHY-NULL follow-up arm 1+2 (A/B vs EXP-0009; frozen-snapshot
# protocol identical to EXP-0013): P2-only (L0) zero-init gated fusion with UAV-encoder
# LoRA (use_lora, r=16/a32/qkv+proj/neck recipe) AND fusion weight-decay exemption
# (--fusion-no-wd 1: gate/proj leave the wd-0.05 'other' group; the single trainer-side
# lever named by the DUAL-WHY-NULL diagnosis). Identity ue-whu-a3sgm-512-dualv2-l0-lora-v1.
# Telemetry per docs/EXP-0008_preregistration.md v2/v2.1 (single-level keys).
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export UE_DATA_ROOT="${UE_DATA_ROOT:-/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen}"
if [[ "${B0_EXP_ID:-}" != "EXP-0015" ]]; then
  echo "this wrapper only launches EXP-0015" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_TAG="surfacebev_v2_512dualv2l0lora_sealed_ddp4_s44_EXP-0015"
WORK_DIR="/data2/wangcheng/checkpoint/surfbev_b0/${RUN_TAG}"
export RUN_TAG WORK_DIR
export SAVE_LAST_MODEL=1
export UE_LOAD_UAV=1
if [[ "$#" -ne 0 ]]; then
  echo "EXP-0015 wrapper does not accept unregistered trainer arguments" >&2; exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "EXP-0015 full training requires a clean Git worktree" >&2; exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
  echo "EXP-0015 output directory already exists: $WORK_DIR" >&2; exit 2
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/accepted_pairs.jsonl" | cut -d ' ' -f 1)" != "f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a" ]]; then
  echo "accepted manifest hash mismatch" >&2; exit 2
fi
if [[ "$(sha256sum "$UE_DATA_ROOT/surfbev_b0_split_v2.json" | cut -d ' ' -f 1)" != "17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5" ]]; then
  echo "v2 split hash mismatch" >&2; exit 2
fi
/data2/wangcheng/envs/cvt2/bin/python - "$UE_DATA_ROOT" <<'PYV'
import json, sys, importlib.util
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root.parent / (root.name + '_manifest.json')).read_text())
spec = importlib.util.spec_from_file_location('b', 'scripts/ue/build_surfacebev_b0_split_v2.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
got = m.build(root, verify_paths=False)['source_payload_fingerprint_sha256']
if got != manifest['source_payload_fingerprint_sha256']:
    raise SystemExit('snapshot fingerprint drift: %s' % got)
PYV
ARCH_FINGERPRINT="$(SAM2_CKPT="$SAM2_CKPT" /data2/wangcheng/envs/cvt2/bin/python -c 'from mmengine.config import Config; from rsprompter.architecture_contract import architecture_contract; print(architecture_contract(Config.fromfile("configs/ue_whu_a3sgm_512_dualv2_l0_lora.py").model.to_dict())["model_fingerprint"])')"
if [[ "$ARCH_FINGERPRINT" != "5c7e1afc5877496b6eda4efde652ac786b8f25b7f4e8dc23da61764c6cd8e3bc" ]]; then
  echo "EXP-0015 architecture fingerprint drift: $ARCH_FINGERPRINT" >&2; exit 2
fi
BASELINE_ID="$(/data2/wangcheng/envs/cvt2/bin/python -c 'import sys; sys.path.insert(0,"."); from mmengine.config import Config; from ue.contract import baseline_id_for; print(baseline_id_for(Config.fromfile("configs/ue_whu_a3sgm_512_dualv2_l0_lora.py").model.to_dict()))')"
if [[ "$BASELINE_ID" != "ue-whu-a3sgm-512-dualv2-l0-lora-v1" ]]; then
  echo "EXP-0015 baseline identity drift: $BASELINE_ID" >&2; exit 2
fi
echo "EXP0015_CONTRACT exp_id=EXP-0015 seed=44 commit=$(git rev-parse HEAD) dirty=0 gpu=$CUDA_VISIBLE_DEVICES world_size=4 run_tag=$RUN_TAG checkpoint_dir=$WORK_DIR"
echo "EXP0015_CONTRACT baseline_id=$BASELINE_ID model_fingerprint=$ARCH_FINGERPRINT data_manifest=f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a split_sha256=17f49430b15037c8da25c63a1bf7fe4d510cf5ff153263d2feba62c4393441d5 train=694 val=100 test=151 augmentation=none image_size=512x512_native uav_branch=1 levels=P2-only-L0+uav_lora=1 epochs=100 batch_per_rank=1 accum=2 effective_batch=8 lr=5e-4 wd=0.05 save_last=1 early_stop=0 ema=0 amp=1 fresh=1 fusion_no_wd=1"
if [[ "${B0_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi
exec /data2/wangcheng/envs/cvt2/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train/train_rsprompter_fusion.py \
  --config configs/ue_whu_a3sgm_512_dualv2_l0_lora.py --run-tag "$RUN_TAG" --data-root "$UE_DATA_ROOT" \
  --dataset ue_raw_surface --raw-split-manifest surfbev_b0_split_v2.json --image-size 512 512 \
  --batch-size 1 --grad-accum-steps 2 --epochs 100 --early-stopping-patience 0 \
  --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed 44 \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 --fusion-no-wd 1 \
  --ema-enabled 0 --ema-eval 0 --checkpoint-dir "$WORK_DIR"
