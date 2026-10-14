#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}
ARM=${1:?Use single|dual|fpn|sam followed by train|infer|prepare|verify}
ACTION=${2:?Use train|infer|prepare|verify|dump-env}
shift 2
case "$ARM" in single|dual|fpn|sam) ;; *) exit 2 ;; esac
export SAM2_REPO="$ROOT/../sam2"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export RSPROMPTER_LIGHT_IMPORT=0
unset TEST_MAX_PER_IMG TRAIN_FLIP_PROB TRAIN_VFLIP_PROB TRAIN_MULTI_SCALE_RESIZE_PROB
unset TRAIN_MULTI_SCALE_MODE TRAIN_MULTI_SCALE_IMG_SCALE VAL_COMPAT_MAX_DETS
unset EXPLICIT_PROMPT_MODE NECK_TYPE CUDNN_BENCHMARK RESUME_STRICT
export EVAL_BBOX=1 SAVE_BBOX_BEST_METRIC=bbox/mAP SAVE_COMPOSITE_BEST=0
export EARLY_STOPPING_METRIC=segm/mAP VAL_SHARD_ACROSS_RANKS=0
export SAVE_LAST_MODEL=1 SAVE_LAST_CHECKPOINT=1
export SAM2_CKPT=${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}
cd "$ROOT"
case "$ACTION" in
  dump-env)
    "$PYTHON_BIN" - <<'PY'
import json,os
keys=('SAM2_REPO','SAM2_CKPT','RSPROMPTER_LIGHT_IMPORT','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
      'TEST_MAX_PER_IMG','TRAIN_FLIP_PROB','TRAIN_VFLIP_PROB','TRAIN_MULTI_SCALE_RESIZE_PROB',
      'TRAIN_MULTI_SCALE_MODE','TRAIN_MULTI_SCALE_IMG_SCALE','VAL_COMPAT_MAX_DETS',
      'EXPLICIT_PROMPT_MODE','NECK_TYPE','CUDNN_BENCHMARK','RESUME_STRICT','EVAL_BBOX',
      'SAVE_BBOX_BEST_METRIC','SAVE_COMPOSITE_BEST','EARLY_STOPPING_METRIC','VAL_SHARD_ACROSS_RANKS',
      'SAVE_LAST_MODEL','SAVE_LAST_CHECKPOINT')
print(json.dumps({k:os.environ.get(k) for k in keys},sort_keys=True))
PY
    ;;
  prepare) exec "$PYTHON_BIN" tools/prepare_ue_rgbd.py "$@" ;;
  verify) exec "$PYTHON_BIN" scripts/smoke/check_ue_rgbd.py "$@" ;;
  train)
    for arg in "$@"; do
      case "$arg" in --config|--config=*|--dataset|--dataset=*|--init-from|--init-from=*|--image-size|--image-size=*) echo 'Arm/config/initialization is controlled by this entry' >&2; exit 2 ;; esac
    done
    "$PYTHON_BIN" - "$ARM" <<'PY'
import json,sys
from mmengine import Config
from rsprompter.uav.contract import model_contract
c=Config.fromfile(f'configs/ablations/ue_rgbd_{sys.argv[1]}.py')
print(json.dumps(dict(arm=sys.argv[1],contract=model_contract(c.model.to_dict()),geometry=dict(c.ue_geometry)),indent=2))
PY
    if [[ ${DRY_RUN:-0} == 1 ]]; then exit 0; fi
    COMMAND=(train/train_rsprompter_fusion.py --config "configs/ablations/ue_rgbd_${ARM}.py"
             --dataset ue --image-size 512 512 --batch-size 1 --val-batch-size 1
             --epochs 80 --lr 0.0002 --backbone-lr-mult 0.1 --uav-lr-mult 0.1
             --mask-decoder-lr-mult 0.05 --grad-accum-steps 2 --decoupled-aux-clip
             --amp 1 --amp-init-scale 512 --compute-val-loss 0 --seed 42
             --prompt-debug-stats 1 --prompt-debug-stats-interval 50)
    if [[ ${NPROC_PER_NODE:-1} == 1 ]]; then
      exec "$PYTHON_BIN" "${COMMAND[@]}" "$@"
    fi
    exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" "${COMMAND[@]}" "$@"
    ;;
  infer) exec "$PYTHON_BIN" inference/infer_from_checkpoint.py "$@" ;;
  *) exit 2 ;;
esac
