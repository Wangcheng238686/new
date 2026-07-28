#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON=${PYTHON:-/data/wangcheng/envs/cvt2/bin/python}
TRAIN=train/train_rsprompter_fusion.py
CONFIG=configs/rsprompter_anchor_whu1024_base_plus_singlecls.py
DATA=${WHU_DATA_ROOT:-/data/wangcheng/dataset/WHU}
CKPT_BASE=${CKPT_BASE:-/data/wangcheng/checkpoint/ablation_hyperparam_whu1024_baseplus}
LOG_DIR=logs/ablation_hyperparam_whu1024_baseplus

SAM2_CKPT=${SAM2_CKPT:-/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt}
SAM2_BASEPLUS_URL=${SAM2_BASEPLUS_URL:-https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt}
SAM2_BASEPLUS_BYTES=${SAM2_BASEPLUS_BYTES:-323493298}

mkdir -p "$LOG_DIR" "$(dirname "$SAM2_CKPT")"

need_download=0
if [ ! -f "$SAM2_CKPT" ]; then
  need_download=1
else
  actual_bytes=$(stat -c%s "$SAM2_CKPT" || echo 0)
  if [ "$actual_bytes" -ne "$SAM2_BASEPLUS_BYTES" ]; then
    echo "[WARN] checkpoint size mismatch: ${actual_bytes} != ${SAM2_BASEPLUS_BYTES}, re-downloading"
    rm -f "$SAM2_CKPT"
    need_download=1
  fi
fi

if [ "$need_download" -eq 1 ]; then
  echo "[INFO] Downloading SAM2 base_plus checkpoint..."
  wget -c "$SAM2_BASEPLUS_URL" -O "$SAM2_CKPT"
fi

export SAM2_CKPT
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_TIMEOUT=1800
export TORCH_DDP_TIMEOUT_SECONDS=${TORCH_DDP_TIMEOUT_SECONDS:-$NCCL_TIMEOUT}
TS=$(date +%Y%m%d_%H%M%S)

# ---- DDP 配置 ----
# 默认每 GPU bs=2, 4卡有效 bs=8, accum=1。
# 若缩减到 2 卡并行多实验，可将 grad-accum 提到 2 维持有效 batch=8。
# WHU 1024 对 IIMR residual 更敏感，默认使用 2026-04-24 稳定锚点。
GPU_LIST=${GPU_LIST:-0,1,2,3}
NUM_GPUS=${NUM_GPUS:-4}
BEST_BS=${BEST_BS:-2}
BEST_ACCUM=${BEST_ACCUM:-1}
BEST_LR=${BEST_LR:-5e-4}
BEST_MULT=${BEST_MULT:-1.0}
IIMR_NUM_UNCERTAINTY_POINTS=${IIMR_NUM_UNCERTAINTY_POINTS:-4}
IIMR_NUM_ITERATIONS=${IIMR_NUM_ITERATIONS:-2}
IIMR_INTERMEDIATE_LOSS_WEIGHT=${IIMR_INTERMEDIATE_LOSS_WEIGHT:-0.1}
IIMR_USE_LEARNABLE_RESIDUAL=${IIMR_USE_LEARNABLE_RESIDUAL:-0}
IIMR_MEMORY_RESIDUAL_WEIGHT=${IIMR_MEMORY_RESIDUAL_WEIGHT:-0.02}
IIMR_RESIDUAL_WARMUP_EPOCHS=${IIMR_RESIDUAL_WARMUP_EPOCHS:-8}
IIMR_DYNAMIC_FUSION_MODE=${IIMR_DYNAMIC_FUSION_MODE:-gated}
MULTI_SCALE_RESIZE_PROB=${MULTI_SCALE_RESIZE_PROB:-0.5}
MULTI_SCALE_MODE=${MULTI_SCALE_MODE:-range}
MULTI_SCALE_IMG_SCALE=${MULTI_SCALE_IMG_SCALE:-"1344 819 1344 1344"}
RUN_TAG=${RUN_TAG:-iter2_dp${IIMR_NUM_UNCERTAINTY_POINTS}_4gpu}

COMMON_ARGS=(
  --config "$CONFIG"
  --data-root "$DATA"
  --dataset-format whu_coco
  --whu-train-ann-file "2.4 annotation/annotation/train.json"
  --whu-val-ann-file "2.4 annotation/annotation/validation.json"
  --whu-train-img-subdir "2.1 train/train"
  --whu-val-img-subdir "2.3 valid/validation"
  --whu-single-class 1
  --whu-enable-category-mapping 0
  --multi-scale-resize-prob "$MULTI_SCALE_RESIZE_PROB"
  --multi-scale-mode "$MULTI_SCALE_MODE"
  --multi-scale-img-scale $MULTI_SCALE_IMG_SCALE
  --eval-category-name building
  --rcnn-max-per-img "${RCNN_MAX_PER_IMG:-100}"
  --rpn-train-max-per-img "${RPN_TRAIN_MAX_PER_IMG:-1000}"
  --rpn-test-max-per-img "${RPN_TEST_MAX_PER_IMG:-1000}"
  --rpn-pos-iou-thr "${RPN_POS_IOU_THR:--1.0}"
  --rcnn-pos-fraction "${RCNN_POS_FRACTION:-0.25}"
  --fpn-type "${FPN_TYPE:-fused}"
  --topdown-downstream-head "${TOPDOWN_DOWNSTREAM_HEAD:-none}"
  --dense-prompt-mode "${DENSE_PROMPT_MODE:-none}"
  --num-workers 2
  --debug-small-object-stats "${DEBUG_SMALL_OBJECT_STATS:-0}"
  --image-size 1024 1024
  --epochs "${MAX_EPOCHS:-80}"
  --seed 44
  --val-ratio 1.0
  --val-batch-size 1
  --val-every-n-epochs 1
  --early-stopping-patience 10
  --early-stopping-smooth-window 5
  --early-stopping-min-epochs 20
  --early-stopping-min-delta 5e-4
  --early-stopping-metric segm_map
  --sat-other-lr-mult 1.0
  --warmup-iters 100
  --weight-decay 0.05
  --deterministic 1
  --amp 0
  --ema-enabled 1
  --ema-decay 0.999
  --ema-update-every 1
  --ema-eval 1
  --ema-save-best 1
  --ema-eval-start-epoch 5
  --iimr-enabled "${IIMR_ENABLED:-1}"
  --iimr-num-iterations "$IIMR_NUM_ITERATIONS"
  --iimr-use-dynamic-prompting 1
  --iimr-num-uncertainty-points "$IIMR_NUM_UNCERTAINTY_POINTS"
  --iimr-dynamic-balance-ratio "${IIMR_DYNAMIC_BALANCE_RATIO:-0.25}"
  --iimr-dynamic-start-iteration "${IIMR_DYNAMIC_START_ITERATION:-1}"
  --iimr-dynamic-point-mix "${IIMR_DYNAMIC_POINT_MIX:-1.0}"
  --iimr-use-roi-pos-encoding "${IIMR_USE_ROI_POS_ENCODING:-1}"
  --iimr-use-geometry-aware-kv "${IIMR_USE_GEOMETRY_AWARE_KV:-0}"
  --iimr-detach-mask-path "${IIMR_DETACH_MASK_PATH:-0}"
  --iimr-use-topology-memory "${IIMR_USE_TOPOLOGY_MEMORY:-0}"
  --iimr-intermediate-loss-weight "$IIMR_INTERMEDIATE_LOSS_WEIGHT"
  --iimr-use-learnable-residual "$IIMR_USE_LEARNABLE_RESIDUAL"
  --iimr-memory-residual-weight "$IIMR_MEMORY_RESIDUAL_WEIGHT"
  --iimr-enable-after-epochs "${IIMR_ENABLE_AFTER_EPOCHS:-10}"
  --iimr-residual-warmup-epochs "$IIMR_RESIDUAL_WARMUP_EPOCHS"
  --iimr-residual-start-weight "${IIMR_RESIDUAL_START_WEIGHT:-0.0}"
  --iimr-dynamic-fusion-mode "$IIMR_DYNAMIC_FUSION_MODE"
  --iimr-use-boundary-uncertainty "${IIMR_USE_BOUNDARY_UNCERTAINTY:-0}"
  --iimr-adaptive-balance "${IIMR_ADAPTIVE_BALANCE:-0}"
  --max-train-batches "${MAX_TRAIN_BATCHES:-0}"
  --max-val-batches "${MAX_VAL_BATCHES:-0}"
)

LOG_FILE="$LOG_DIR/whu1024_${RUN_TAG}_baseplus_ddp${NUM_GPUS}_${TS}.log"
PARAM_SNAPSHOT="$LOG_DIR/whu1024_${RUN_TAG}_baseplus_ddp${NUM_GPUS}_${TS}.params"
CKPT_DIR="$CKPT_BASE/whu1024_${RUN_TAG}"
BEST_CKPT_PATH="$CKPT_DIR/best_model.pth"

MASTER_PORT=${MASTER_PORT:-29500}
RESUME_FROM=${RESUME_FROM:-}
RUN_IN_BACKGROUND=${RUN_IN_BACKGROUND:-1}
BACKGROUND_NOHUP=${BACKGROUND_NOHUP:-1}
ALLOW_FRESH_OVERWRITE=${ALLOW_FRESH_OVERWRITE:-0}

EXTRA_ARGS=()
if [ -n "$RESUME_FROM" ]; then
  EXTRA_ARGS+=(--resume-from "$RESUME_FROM")
fi

LAUNCH_CMD=(
  "$PYTHON" -u -m torch.distributed.run
  --nproc_per_node "$NUM_GPUS"
  --master_port "$MASTER_PORT"
  "$TRAIN"
  "${COMMON_ARGS[@]}"
  --checkpoint-dir "$CKPT_DIR"
  --batch-size "$BEST_BS"
  --grad-accum-steps "$BEST_ACCUM"
  --lr "$BEST_LR"
  --sat-backbone-lr-mult "$BEST_MULT"
  "${EXTRA_ARGS[@]}"
  "$@"
)

if [ -f "$BEST_CKPT_PATH" ] && [ -z "$RESUME_FROM" ] && [ "$ALLOW_FRESH_OVERWRITE" != "1" ]; then
  echo "[ERROR] Refusing to start a fresh run because an existing checkpoint was found:"
  echo "[ERROR]   $BEST_CKPT_PATH"
  echo "[ERROR] Provide RESUME_FROM=$BEST_CKPT_PATH to continue training,"
  echo "[ERROR] or set ALLOW_FRESH_OVERWRITE=1 only if you intentionally want a new run to reuse this directory."
  exit 1
fi

# ---- 生成参数快照，同时写入 .params 文件、日志文件和终端 ----
_snapshot() {
  cat <<SNAPSHOT
============================================
[EXPERIMENT CONFIG DUMP] $(date)
============================================
[ENV] SAM2_CKPT=$SAM2_CKPT
[ENV] GPU_LIST=$GPU_LIST
[ENV] NUM_GPUS=$NUM_GPUS bs_per_gpu=$BEST_BS accum=$BEST_ACCUM effective_bs=$(( BEST_BS * BEST_ACCUM * NUM_GPUS ))
[ENV] RUN_TAG=$RUN_TAG
[ENV] checkpoint_dir=$CKPT_DIR
[ENV] log_file=$LOG_FILE
[ENV] param_snapshot=$PARAM_SNAPSHOT
[ENV] git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
[ENV] git_commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
[ENV] git_dirty=$(git status --porcelain 2>/dev/null | head -5 | tr '\n' ';' || echo unknown)
$([ -n "$RESUME_FROM" ] && echo "[ENV] RESUME_FROM=$RESUME_FROM" || true)

--- IIMR ---
  iimr_enabled=${IIMR_ENABLED:-1}
  iimr_num_iterations=$IIMR_NUM_ITERATIONS
  iimr_num_uncertainty_points=$IIMR_NUM_UNCERTAINTY_POINTS
  iimr_dynamic_balance_ratio=${IIMR_DYNAMIC_BALANCE_RATIO:-0.25}
  iimr_dynamic_start_iteration=${IIMR_DYNAMIC_START_ITERATION:-1}
  iimr_dynamic_point_mix=${IIMR_DYNAMIC_POINT_MIX:-1.0}
  iimr_use_roi_pos_encoding=${IIMR_USE_ROI_POS_ENCODING:-1}
  iimr_detach_mask_path=${IIMR_DETACH_MASK_PATH:-0}
  iimr_use_topology_memory=${IIMR_USE_TOPOLOGY_MEMORY:-0}
  iimr_use_geometry_aware_kv=${IIMR_USE_GEOMETRY_AWARE_KV:-0}
  iimr_intermediate_loss_weight=$IIMR_INTERMEDIATE_LOSS_WEIGHT
  iimr_use_learnable_residual=$IIMR_USE_LEARNABLE_RESIDUAL
  iimr_memory_residual_weight=$IIMR_MEMORY_RESIDUAL_WEIGHT
  iimr_residual_warmup_epochs=$IIMR_RESIDUAL_WARMUP_EPOCHS
  iimr_residual_start_weight=${IIMR_RESIDUAL_START_WEIGHT:-0.0}
  iimr_enable_after_epochs=${IIMR_ENABLE_AFTER_EPOCHS:-10}
  iimr_dynamic_fusion_mode=$IIMR_DYNAMIC_FUSION_MODE
  iimr_use_boundary_uncertainty=${IIMR_USE_BOUNDARY_UNCERTAINTY:-0}
  iimr_adaptive_balance=${IIMR_ADAPTIVE_BALANCE:-0}

--- Training ---
  lr=$BEST_LR backbone_mult=$BEST_MULT
  epochs=${MAX_EPOCHS:-80}
  seed=${SEED:-44}
  warmup_iters=${WARMUP_ITERS:-100}
  weight_decay=${WEIGHT_DECAY:-0.05}
  amp=${AMP:-0}
  image_size=1024 1024

--- EMA ---
  ema_enabled=${EMA_ENABLED:-1}
  ema_decay=${EMA_DECAY:-0.999}
  ema_update_every=${EMA_UPDATE_EVERY:-1}
  ema_eval=${EMA_EVAL:-1}
  ema_save_best=${EMA_SAVE_BEST:-1}
  ema_eval_start_epoch=${EMA_EVAL_START_EPOCH:-5}

--- Early Stopping ---
  patience=${EARLY_STOPPING_PATIENCE:-10}
  smooth_window=${EARLY_STOPPING_SMOOTH_WINDOW:-5}
  min_epochs=${EARLY_STOPPING_MIN_EPOCHS:-20}
  min_delta=${EARLY_STOPPING_MIN_DELTA:-5e-4}
  metric=${EARLY_STOPPING_METRIC:-segm_map}

--- FPN / Neck ---
  fpn_type=${FPN_TYPE:-fused}
  topdown_downstream_head=${TOPDOWN_DOWNSTREAM_HEAD:-none}
  dense_prompt_mode=${DENSE_PROMPT_MODE:-none}

--- RPN / ROI (small object) ---
  rpn_train_max_per_img=${RPN_TRAIN_MAX_PER_IMG:-1000}
  rpn_test_max_per_img=${RPN_TEST_MAX_PER_IMG:-1000}
  rpn_pos_iou_thr=${RPN_POS_IOU_THR:--1.0}
  rcnn_max_per_img=${RCNN_MAX_PER_IMG:-100}
  rcnn_pos_fraction=${RCNN_POS_FRACTION:-0.25}

--- Multi-Scale Aug ---
  multi_scale_resize_prob=$MULTI_SCALE_RESIZE_PROB
  multi_scale_mode=$MULTI_SCALE_MODE
  multi_scale_img_scale=$MULTI_SCALE_IMG_SCALE

--- Val ---
  val_ratio=${VAL_RATIO:-1.0}
  val_batch_size=${VAL_BATCH_SIZE:-1}
  val_every_n_epochs=${VAL_EVERY_N_EPOCHS:-1}

--- Full CLI args ---
$(printf "  %s\n" "${LAUNCH_CMD[@]}")
============================================
SNAPSHOT
}

_snapshot > "$PARAM_SNAPSHOT"
_snapshot | tee "$LOG_FILE"

if [ "$RUN_IN_BACKGROUND" = "1" ]; then
  if [ "$BACKGROUND_NOHUP" = "1" ]; then
    nohup env CUDA_VISIBLE_DEVICES="$GPU_LIST" "${LAUNCH_CMD[@]}" >> "$LOG_FILE" 2>&1 < /dev/null &
  else
    CUDA_VISIBLE_DEVICES="$GPU_LIST" "${LAUNCH_CMD[@]}" >> "$LOG_FILE" 2>&1 &
  fi
  PID=$!
  disown "$PID" 2>/dev/null || true
  echo "[LAUNCHED] TS=$TS"
  echo "[PID] WHU1024_BASEPLUS_4GPU=$PID log=$LOG_FILE"
  echo "[PARAMS] $PARAM_SNAPSHOT"
else
  echo "[FOREGROUND] TS=$TS"
  echo "[LOG] $LOG_FILE"
  echo "[PARAMS] $PARAM_SNAPSHOT"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" "${LAUNCH_CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
fi
