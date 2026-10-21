#!/usr/bin/env bash
# EXP-0026: sealed single-stream a3sgm @512-native on the v3 rebalanced CitySample
# archive (ALL augmentation off); per-rank batch 2 x accum 1 (effective batch 8 unchanged).
# v3 = PI resplit (2026-10-21) to cure the v1 val/test metric drift
# (docs/VALTEST-DRIFT-20261021.md): 2629 points 1931/285/413, balanced
# distribution, same label authority review99_v3_user_approved_20261021.
# Model identity ue-whu-a3sgm-512-v1 (5305b315...) unchanged; test sealed.
# Prereg: docs/EXP-0026-0027_preregistration.md.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SAM2_CKPT="${SAM2_CKPT:-/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt}"
export DS_ROOT="${DS_ROOT:-/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v3_20261021}"
if [[ "${B0_EXP_ID:-}" != "EXP-0026" ]]; then
  echo "this wrapper only launches EXP-0026" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_TAG="citysample_v1_512single_noaug_ddp4_s44_EXP-0026"
WORK_DIR="/data2/wangcheng/checkpoint/citysample_v1/${RUN_TAG}"
export RUN_TAG WORK_DIR
export SAVE_LAST_MODEL=1
# EXP-0026 data plumbing: whu_coco format pointed at the CitySample annotations
export WHU_TRAIN_ANN_FILE=annotations/train.json
export WHU_VAL_ANN_FILE=annotations/val.json
export WHU_TRAIN_IMG_SUBDIR=
export WHU_VAL_IMG_SUBDIR=
# augmentation contract: flip/vflip/multi-scale/noise/erasing ALL pinned to 0
export TRAIN_FLIP_PROB=0
export TRAIN_VFLIP_PROB=0
export TRAIN_MULTI_SCALE_RESIZE_PROB=0
if [[ "$#" -ne 0 ]]; then
  echo "EXP-0026 wrapper does not accept unregistered trainer arguments" >&2; exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "EXP-0026 full training requires a clean Git worktree" >&2; exit 2
fi
if [[ -e "$WORK_DIR" ]]; then
  echo "EXP-0026 output directory already exists: $WORK_DIR" >&2; exit 2
fi
if [[ "$(sha256sum "$DS_ROOT/dataset.json" | cut -d ' ' -f 1)" != "917c5fe42f2f09db87dcecd21a2e81cecfbe6bb54ffbc6364a240758273e44cc" ]]; then echo "pin mismatch: dataset.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/splits/train.json" | cut -d ' ' -f 1)" != "b186c8734c8f3075f6ae0b3b3feb499107f5f52b423dcc0ff2b503dc2c4ee66d" ]]; then echo "pin mismatch: splits/train.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/splits/val.json" | cut -d ' ' -f 1)" != "617b7429ae31a5503253e0d0b63896f51e24ad334580acf6b497c54f31a78dde" ]]; then echo "pin mismatch: splits/val.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/splits/test.json" | cut -d ' ' -f 1)" != "9822e4a417c24cf6bc146925c3168ca0ac96dbd499d1698ebc52e81494b47964" ]]; then echo "pin mismatch: splits/test.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/annotations/train.json" | cut -d ' ' -f 1)" != "97630d70788e652254cfd4d9e8c7443aa9635a09cad5fd64bfb29a1d59eaa4bd" ]]; then echo "pin mismatch: annotations/train.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/annotations/val.json" | cut -d ' ' -f 1)" != "d59ac33f400c1d91fb77deddbcec1104c78333b45af890e7bc215199d8f2443de" ]]; then echo "pin mismatch: annotations/val.json" >&2; exit 2; fi
if [[ "$(sha256sum "$DS_ROOT/annotations/test.json" | cut -d ' ' -f 1)" != "2ea872f0cbe4976906bb3780e012b352c597a1157f5598d1c73beb1dc6e6914a" ]]; then echo "pin mismatch: annotations/test.json" >&2; exit 2; fi
/data2/wangcheng/envs/cvt2/bin/python - "$DS_ROOT" <<'PYV'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
ds = json.loads((root / 'dataset.json').read_text())
assert ds['label_revision'] == 'review99_v3_user_approved_20261021', ds['label_revision']
assert [ds['splits'][s]['points'] for s in ('train','val','test')] == [1931, 285, 413]
assert [ds['splits'][s]['instances'] for s in ('train','val','test')] == [123788, 18066, 24754]
assert ds['image_size'] == [512, 512]
PYV
ARCH_FINGERPRINT="$(SAM2_CKPT="$SAM2_CKPT" /data2/wangcheng/envs/cvt2/bin/python -c 'from mmengine.config import Config; from rsprompter.architecture_contract import architecture_contract; print(architecture_contract(Config.fromfile("configs/ue_whu_a3sgm_512.py").model.to_dict())["model_fingerprint"])')"
if [[ "$ARCH_FINGERPRINT" != "5305b3155914a2b2e7ecaf24b54957aee86fd6f47db93358daead2cbac5be36c" ]]; then
  echo "EXP-0026 architecture fingerprint drift: $ARCH_FINGERPRINT" >&2; exit 2
fi
BASELINE_ID="$(/data2/wangcheng/envs/cvt2/bin/python -c 'import sys; sys.path.insert(0,"."); from mmengine.config import Config; from ue.contract import baseline_id_for; print(baseline_id_for(Config.fromfile("configs/ue_whu_a3sgm_512.py").model.to_dict()))')"
if [[ "$BASELINE_ID" != "ue-whu-a3sgm-512-v1" ]]; then
  echo "EXP-0026 baseline identity drift: $BASELINE_ID" >&2; exit 2
fi
echo "EXP0026_CONTRACT exp_id=EXP-0026 seed=44 commit=$(git rev-parse HEAD) dirty=0 gpu=0,1,2,3 world_size=4 run_tag=$RUN_TAG checkpoint_dir=$WORK_DIR"
echo "EXP0026_CONTRACT baseline_id=$BASELINE_ID model_fingerprint=$ARCH_FINGERPRINT dataset=CitySample_SatUAV_TrainMinimal_v3_20261021 label_revision=review99_v3_user_approved_20261021 train=1931 val=285 test=413_sealed augmentation=flip:0,vflip:0,ms:0,noise:0,erasing:0 image_size=512x512_native upsampling=none epochs=100 batch_per_rank=2 accum=1 effective_batch=8 lr=5e-4 wd=0.05 warmup=100 save_last=1 early_stop=0 ema=0 amp=1 fresh=1"
if [[ "${B0_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi
exec /data2/wangcheng/envs/cvt2/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 train/train_rsprompter_fusion.py \
  --config configs/ue_whu_a3sgm_512.py --run-tag "$RUN_TAG" --data-root "$DS_ROOT" \
  --dataset whu --image-size 512 512 \
  --batch-size 2 --grad-accum-steps 1 --epochs 100 --early-stopping-patience 0 \
  --lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --seed 44 \
  --val-batch-size 1 --val-every-n-epochs 1 --compute-val-loss 0 --amp 1 \
  --ema-enabled 0 --ema-eval 0 --checkpoint-dir "$WORK_DIR"
