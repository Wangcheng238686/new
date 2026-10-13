#!/usr/bin/env bash
# H2-D2 live-mechanism A/B watchdog: waits for a GPU with >=22500 MiB free,
# launches arm B (injection) then arm A (control) sequentially.
set -u
ROOT=/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse
PY=/data2/wangcheng/envs/cvt2/bin/python
LOGD=$ROOT/work_dirs/ue
OUTD=/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe
THRESH_MIB=20000  # nounits returns MiB

wait_gpu() {
  while true; do
    for i in 0 1 2 3; do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
      total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i $i)
      free=$((total - used))
      if [ "$free" -ge "$THRESH_MIB" ]; then echo $i; return 0; fi
    done
    sleep 300
  done
}

# arm B
GPU=$(wait_gpu)
echo "$(date +%F' '%T) launching arm B on GPU $GPU" >> $LOGD/AB-watchdog.log
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup $PY scripts/ue/finetune_p2_inject_ab.py --inject 1 \
  --data-root /data1/wangcheng/dataset/ue_coco_with_uav_d2 \
  --single-ckpt work_dirs/ue/ue_d2_single_tr128_va259_s44/best_model_epoch83.pth \
  --out $OUTD/MHBEV-FUS-20260920-D2-AB-B > $LOGD/AB-B-d2.log 2>&1
BPID=$!
echo "$(date +%F' '%T) arm B finished (pid $BPID)" >> $LOGD/AB-watchdog.log

# arm A
GPU=$(wait_gpu)
echo "$(date +%F' '%T) launching arm A on GPU $GPU" >> $LOGD/AB-watchdog.log
CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup $PY scripts/ue/finetune_p2_inject_ab.py --inject 0 \
  --data-root /data1/wangcheng/dataset/ue_coco_with_uav_d2 \
  --single-ckpt work_dirs/ue/ue_d2_single_tr128_va259_s44/best_model_epoch83.pth \
  --out $OUTD/MHBEV-FUS-20260920-D2-AB-A > $LOGD/AB-A-d2.log 2>&1
echo "$(date +%F' '%T) arm A finished" >> $LOGD/AB-watchdog.log
