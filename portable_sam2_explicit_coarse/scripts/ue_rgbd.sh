#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-/data/wangcheng/envs/cvt2/bin/python}
ARM=${1:?Use single|dual|fpn|sam followed by train|infer|prepare|verify}
ACTION=${2:?Use train|infer|prepare|verify|dump-env}
shift 2
case "$ARM" in single|minimal|adapter|nativegeo|dual|fpn|sam) ;; *) exit 2 ;; esac
export SAM2_REPO="$ROOT/../sam2"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export RSPROMPTER_LIGHT_IMPORT=0
unset TEST_MAX_PER_IMG TRAIN_FLIP_PROB TRAIN_VFLIP_PROB TRAIN_MULTI_SCALE_RESIZE_PROB
unset TRAIN_MULTI_SCALE_MODE TRAIN_MULTI_SCALE_IMG_SCALE VAL_COMPAT_MAX_DETS
unset EXPLICIT_PROMPT_MODE NECK_TYPE CUDNN_BENCHMARK RESUME_STRICT
unset TRAIN_GAUSSIAN_NOISE_PROB TRAIN_GAUSSIAN_NOISE_STD TRAIN_COLOR_JITTER_PROB
if [[ "$ARM" == minimal ]]; then export TEST_MAX_PER_IMG=250; fi
export EVAL_BBOX=1 SAVE_BBOX_BEST_METRIC=bbox/mAP SAVE_COMPOSITE_BEST=0
export EARLY_STOPPING_METRIC=segm/mAP VAL_SHARD_ACROSS_RANKS=0
export SAVE_LAST_MODEL=1 SAVE_LAST_CHECKPOINT=1
export SAM2_CKPT=${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}
cd "$ROOT"
if [[ "$ARM" == nativegeo ]]; then
  exec "$PYTHON_BIN" scripts/probes/ue_v1_native_geometry.py "$ACTION" "$@"
fi
if [[ "$ARM" == adapter ]]; then
  exec "$PYTHON_BIN" scripts/probes/ue_v1_adapter.py "$ACTION" "$@"
fi
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
  run)
    [[ "$ARM" == minimal ]] || { echo 'run action is minimal-only' >&2; exit 2; }
    exec "$PYTHON_BIN" scripts/run_ue_minimal_experiment.py "$@"
    ;;
  prepare) exec "$PYTHON_BIN" tools/prepare_ue_rgbd.py "$@" ;;
  verify) exec "$PYTHON_BIN" scripts/smoke/check_ue_rgbd.py "$@" ;;
  train)
    for arg in "$@"; do
      case "$arg" in --config|--config=*|--dataset|--dataset=*|--init-from|--init-from=*|--image-size|--image-size=*) echo 'Arm/config/initialization is controlled by this entry' >&2; exit 2 ;; esac
    done
    if [[ "$ARM" == minimal ]]; then
      for arg in "$@"; do
        if [[ "$arg" == --* ]]; then
          key=${arg%%=*}
          case "$key" in
            --data-root|--checkpoint-dir|--resume-from) ;;
            --epochs|--max-train-batches|--max-val-batches)
              [[ ${UE_MINIMAL_SMOKE:-0} == 1 ]] || { echo 'Smoke-only override rejected' >&2; exit 2; } ;;
            *) echo "Fixed minimal protocol rejects $key" >&2; exit 2 ;;
          esac
        fi
      done
      MINIMAL_COMMAND=(train/train_rsprompter_fusion.py --config configs/ablations/ue_minimal_single.py
        --dataset ue_minimal --image-size 512 512 --batch-size 2 --val-batch-size 1
        --epochs 100 --lr 0.0005 --backbone-lr-mult 1 --mask-decoder-lr-mult 1
        --grad-accum-steps 1 --amp 1 --amp-init-scale 512 --seed 44
        --ema-enabled 0 --ema-eval 0 --compute-val-loss 0 --warmup-iters 100
        --early-stopping-patience 0 --prompt-debug-stats 1 --prompt-debug-stats-interval 50)
      "$PYTHON_BIN" scripts/smoke/check_ue_minimal.py --contract-only
      if [[ ${DRY_RUN:-0} == 1 ]]; then printf '%q ' "$PYTHON_BIN" "${MINIMAL_COMMAND[@]}" "$@"; printf '\n'; exit 0; fi
      if [[ ${NPROC_PER_NODE:-4} == 1 ]]; then exec "$PYTHON_BIN" "${MINIMAL_COMMAND[@]}" "$@"; fi
      exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" "${MINIMAL_COMMAND[@]}" "$@"
    fi
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
