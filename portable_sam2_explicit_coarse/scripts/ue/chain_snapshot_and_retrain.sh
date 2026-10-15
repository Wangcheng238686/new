#!/usr/bin/env bash
# Autonomous chain: wait for the componentclean_v2 repair pipeline to finish
# and the production set to go quiet, freeze a versioned snapshot, verify it,
# then run EXP-0009 (single) and EXP-0010 (dual) back to back on the snapshot.
set -uo pipefail
PROD=/data1/wangcheng/dataset/Satellite_StrictHeroCoverage_TwoPass_Production_v7_remote_2000_3999
SNAP_ROOT=/data2/wangcheng/dataset_frozen
SNAP=$SNAP_ROOT/ue_raw_hero_uav_pairset_20261014_frozen
PKG=/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse
LOG=/data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/CHAIN-snapshot-retrain.log
STABLE_MIN=20

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "chain started; PI decision: freeze CURRENT state immediately (producer still writing; per-file consistent copy)"

mkdir -p "$SNAP_ROOT"
if [[ ! -f "${SNAP}_manifest.json" ]]; then
  /data2/wangcheng/envs/cvt2/bin/python "$PKG/scripts/ue/freeze_dataset_snapshot.py" \
    --source "$PKG/data/ue_raw_hero_uav_pairset_20261011" --dest "$SNAP" >> "$LOG" 2>&1
  if [[ ! -f "${SNAP}_manifest.json" ]]; then
    log "FATAL: snapshot manifest missing after freeze attempt"; exit 1
  fi
else
  log "snapshot manifest already present; reusing existing snapshot"
fi
log "snapshot frozen: $(sha256sum "${SNAP}_manifest.json" | cut -c1-16)"

cd "$PKG"
if [[ -f /data2/wangcheng/checkpoint/surfbev_b0/surfacebev_snap_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth ]]; then
  log "EXP-0009 last checkpoint already present; skipping to EXP-0010"
  RC9=0
else
  log "launching EXP-0009 (single-stream snapshot arm)"
  B0_EXP_ID=EXP-0009 bash scripts/ue/run_exp0009_ddp4.sh \
    > /data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0009-ddp4.log 2>&1
  RC9=$?
fi
log "EXP-0009 exited rc=$RC9"
if [[ $RC9 -ne 0 || ! -f /data2/wangcheng/checkpoint/surfbev_b0/surfacebev_snap_512_sealed_ddp4_s44_EXP-0009/last_model_epoch100.pth ]]; then
  log "FATAL: EXP-0009 did not complete (rc=$RC9); not launching EXP-0010"; exit 2
fi

log "launching EXP-0010 (dual-stream snapshot arm)"
B0_EXP_ID=EXP-0010 bash scripts/ue/run_exp0010_ddp4.sh \
  > /data2/wangcheng/checkpoint/surfbev_b0/launcher_logs/EXP-0010-ddp4.log 2>&1
RC10=$?
log "EXP-0010 exited rc=$RC10"
if [[ ! -f /data2/wangcheng/checkpoint/surfbev_b0/surfacebev_snap_512dual_sealed_ddp4_s44_EXP-0010/last_model_epoch100.pth ]]; then
  log "FATAL: EXP-0010 last checkpoint missing (rc=$RC10)"; exit 3
fi
log "CHAIN COMPLETE: both snapshot arms trained; ready for A/B analysis"
