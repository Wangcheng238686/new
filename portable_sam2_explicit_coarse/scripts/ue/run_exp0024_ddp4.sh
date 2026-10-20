#!/usr/bin/env bash
# EXP-0024: sealed single-stream a3sgm @512-native satellite baseline on the
# cleaned-full CitySample archive, ALL augmentation off. Protocol per
# docs/EXP-0024-0025_preregistration.md; model identity ue-whu-a3sgm-512-v1
# (fingerprint 5305b315...) unchanged; data = CitySample_SatUAV_TrainMinimal_v1_20261021
# label revision review99_v3_user_approved_20261021. test never read.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export DS_ROOT="${DS_ROOT:-/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021}"
if [[ "${B0_EXP_ID:-}" != "EXP-0024" ]]; then
  echo "this wrapper only launches EXP-0024" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_TAG="citysample_v1_512single_noaug_ddp4_s44_EXP-0024"
WORK_DIR="/data2/wangcheng/checkpoint/citysample_v1/${RUN_TAG}"
export RUN_TAG WORK_DIR
export SAVE_LAST_MODEL=1
# EXP-0024 data plumbing: whu_coco format pointed at the CitySample annotations
export WHU_TRAIN_ANN_FILE=annotations/train.json
export WHU_VAL_ANN_FILE=annotations/val.json
export WHU_TRAIN_IMG_SUBDIR=
export WHU_VAL_IMG_SUBDIR=
# augmentation contract: flip/vflip/multi-scale/noise/erasing ALL pinned to 0
export TRAIN_FLIP_PROB=0
export TRAIN_VFLIP_PROB=0
export TRAIN_MULTI_SCALE_RESIZE_PROB=0
if [[ "$#" -ne 0 ]]; then
  echo "EXP-0024 wrapper does not accept unregistered trainer arguments" >&2; exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "EXP-0024 full training requires a clean Git worktree" >&2; exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
  echo "EXP-0024 output directory already exists: $WORK_DIR" >&2; exit 2
fi
if [[ "$(sha256sum "$DS_ROOT/dataset.json" | cut -d ' ' -f 1)" != "b42a11edae080f4a31b07a25128bbb8322db84f85769093d8327ba1a45164379" ]]; then echo "pin mismatch: dataset.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/splits/train.json" | cut -d ' ' -f 1)" != "965b17041b669ded9a9ddf6d32e5820c2333133543c81902f0611fd0f854fa5e" ]]; then echo "pin mismatch: splits/train.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/splits/val.json" | cut -d ' ' -f 1)" != "2cb80f5206baefd8b2220a61a5dd75ebc3b3873349f272c5530771ed8cee3c37" ]]; then echo "pin mismatch: splits/val.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/splits/test.json" | cut -d ' ' -f 1)" != "f388cf35703252c33d8a40d769b79452f6ca7c71c594c3cc8bfd217517d856da" ]]; then echo "pin mismatch: splits/test.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/annotations/train.json" | cut -d ' ' -f 1)" != "ce262bf0303ed0ee02355142802f22301f782b2b61e66dd363f6bb74e58e656d" ]]; then echo "pin mismatch: annotations/train.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/annotations/val.json" | cut -d ' ' -f 1)" != "9f9ead36f8c35dda8ff054c510c9da3bb32ffe2e39aed64f70b0c5174402f7f1" ]]; then echo "pin mismatch: annotations/val.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/annotations/test.json" | cut -d ' ' -f 1)" != "6d8f29a148d1a023930a0052e47fa906939a72dc6f9b094ef424ab4c23e619b3" ]]; then echo "pin mismatch: annotations/test.json" >&2; exit 2; fi
/data2/wangcheng/envs/cvt2/bin/python - "$DS_ROOT" <<'PYV'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ds = json.loads((root / 'dataset.json').read_text())
assert ds['label_revision'] == 'review99_v3_user_approved_20261021', ds['label_revision']
assert [ds['splits'][s]['points'] for s in ('train','val','test')] == [2142, 459, 459]
assert [ds['splits'][s]['instances'] for s in ('train','val','test')] == [126214, 14980, 44988]
assert ds['image_size'] == [512, 512]
PYV
ARCH_FINGERPRINT="$(SAM2_CKPT="$SAM2_CKPT" /data2/wangcheng/envs/cvt2/bin/python -c 'from mmengine.config import Config; from rsprompter.architecture_contract import architecture_contract; print(architecture_contract(Config.fromfile("configs/ue_whu_a3sgm_512.py").model.to_dict())["model_fingerprint"])')"
if [[ "$ARCH_FINGERPRINT" != "5305b3155914a2b2e7ecaf24b54957aee86fd6f47db93358daead2cbac5be36c" ]]; then
  echo "EXP-0024 architecture fingerprint drift: $ARCH_FINGERPRINT" >&2; exit 2
fi
BASELINE_ID="$(/data2/wangcheng/envs/cvt2/bin/python -c 'import sys; sys.path.insert(0,"."); from mmengine.config import Config; from ue.contract import baseline_id_for; print(baseline_id_for(Config.fromfile("configs/ue_whu_a3sgm_512.py").model.to_dict()))')"
if [[ "$BASELINE_ID" != "ue-whu-a3sgm-512-v1" ]]; then
  echo "EXP-0024 baseline identity drift: $BASELINE_ID" >&2; exit 2
fi
echo "EXP24_CONTRACT exp_id=EXP-0024 seed=44 commit=$(git rev-parse HEAD) dirty=0 gpu=0,1,2,3 world_size=4 run_tag=$RUN_TAG checkpoint_dir=$WORK_DIR"
echo "EXP24_CONTRACT baseline_id=$BASELINE_ID model_fingerprint=$ARCH_FINGERPRINT dataset=CitySample_SatUAV_TrainMinimal_v1_20261021 label_revision=review99_v3_user_approved_20261021 train=2142 val=459 test=459_sealed augmentation=flip:0,vflip:0,ms:0,noise:0,erasing:0 image_size=512x512_native upsampling=none epochs=100 batch_per_rank=1 accum=2 effective_batch=8 lr=5e-4 wd=0.05 warmup=100 save_last=1 early_stop=0 ema=0 amp=1 fresh=1"
if [[ "${B0_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi
exec /data2/wangcheng/envs/cvt2/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train/train_rsprompter_fusion.py \
  --config configs/ue_whu_a3sgm_512.py --run-tag "$RUN_TAG" --data-root "$DS_ROOT" \
  --dataset whu --image-size 512 512 \
  --batch-size 1 --grad-accum-steps 2 --epochs 100 --early-stopping-patience 0 \
  --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed 44 \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --ema-enabled 0 --ema-eval 0 --checkpoint-dir "$WORK_DIR"
