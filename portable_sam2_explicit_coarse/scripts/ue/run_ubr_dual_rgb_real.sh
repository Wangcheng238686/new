#!/bin/bash
# EXP-0030 UBR dual-stream +RGB launch wrapper (THE only launch path for
# the +RGB real arm; spec docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md).
# Usage: run_ubr_dual_rgb_real.sh [real|constant]   (default: real)
#   'constant' is the RESERVED fourth arm - launch only after an RGB_WINS
#   verdict; nothing here auto-launches it.
#
# Fail-closed contract echo (RESEAL_CONTRACT v4): commit/tag exp/EXP-0030,
# audit verdict, dataset pins (asserted in the runner), anchor-cache
# manifest, augmentation protocol, arm + rgb content, geometry cache/
# constants SHA, output root. Every launch value is echoed and archived to
# a SIDELINE file (the runner fail-closed asserts the output root does NOT
# exist yet).
#
# QUEUE DISCIPLINE (spec section 6): the launch slot opens only when
# EXP-0028-R2 OR EXP-0029 has finished. Both running = two training
# processes per card; this wrapper REFUSES to add a third (a ~17-19 GB job
# on top of ~29 GB/card would risk OOM during val peaks). The guard below
# requires max_one_training_process_per_card < 2 at launch time.
set -euo pipefail

CONTENT=${1:-real}
case "$CONTENT" in
  real|constant) ;;
  *) echo "FAIL: rgb content must be real|constant (got: $CONTENT)"; exit 1 ;;
esac

export UBR_RGB_OUT_ROOT=/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_dual_rgb_${CONTENT}_20261028
export SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt
export TRAIN_FLIP_PROB=0          # 3090 protocol: NO flip (dual-stream)
export TRAIN_VFLIP_PROB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export UBR_GEOM_CACHE=${UBR_GEOM_CACHE:-/data2/wangcheng/remote-lthpc/cache/ue_v1_native_geometry}
export UBR_GEOM_CONSTANTS=${UBR_GEOM_CONSTANTS:-/data2/wangcheng/remote-lthpc/checkpoint/ue_geometry_mechanism_20261022/common_constants.json}
export UBR_RGB_ANCHOR_CACHE=${UBR_RGB_ANCHOR_CACHE:-/data2/wangcheng/remote-lthpc/cache/ue_v1_s2db_geometry_v2}
export UBR_RGB_DATA_ROOT=${UBR_RGB_DATA_ROOT:-/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021}
PY=/data2/wangcheng/envs/cvt2/bin/python
WORKTREE=/home/wangcheng/project/new/.worktrees/dual-stream-dev
CODE=$WORKTREE/portable_sam2_explicit_coarse
AUDIT=$WORKTREE/docs/reviews/UBR-DUAL-RGB-20261028_audit.md

cd "$CODE"
GIT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)
GIT_TAG=$(git -C "$WORKTREE" describe --tags --exact-match HEAD 2>/dev/null || echo UNTAGGED)
GEOM_CONSTANTS_SHA=$(sha256sum "$GEOM_CONSTANTS" 2>/dev/null | cut -d' ' -f1 || echo MISSING)
ANCHOR_COUNT=$(ls "$UBR_RGB_ANCHOR_CACHE"/*.npz 2>/dev/null | wc -l)

echo "=== EXP-0030 UBR DUAL+RGB WRAPPER CONTRACT (v4) ==="
echo "commit       : $GIT_SHA"
echo "tag          : $GIT_TAG (required: exp/EXP-0030)"
echo "rgb content  : $CONTENT (uav_port: real; geometry head on REAL map)"
echo "audit        : $(grep -m1 VERDICT "$AUDIT" 2>/dev/null || echo MISSING)"
echo "config       : $(sha256sum configs/ablations/ue_ubr_dual_rgb_512.py | cut -c1-16)"
echo "geom cache   : $UBR_GEOM_CACHE (v native512_height_dx_dy_viewcount_v1)"
echo "geom const   : $GEOM_CONSTANTS_SHA"
echo "anchor cache : $UBR_RGB_ANCHOR_CACHE (${ANCHOR_COUNT} npz, v s2db_anchor_metadata_v2_fp64)"
echo "flip         : $TRAIN_FLIP_PROB (3090 protocol: no flip; photometric-only jitter)"
echo "out          : $UBR_RGB_OUT_ROOT"
[ "$(git -C "$WORKTREE" status --porcelain | wc -l)" -eq 0 ] || { echo "FAIL: dirty tree"; exit 1; }
[ "${GIT_TAG#exp/EXP-0030}" != "$GIT_TAG" ] || { echo "FAIL: HEAD not tagged exp/EXP-0030"; exit 1; }
grep -q "VERDICT: PASS" "$AUDIT" 2>/dev/null || { echo "FAIL: audit not PASS ($AUDIT)"; exit 1; }
[ -f "$UBR_GEOM_CACHE/000000.npz" ] || { echo "FAIL: geometry cache missing"; exit 1; }
[ "$ANCHOR_COUNT" -eq 3060 ] || { echo "FAIL: anchor cache must hold 3060 scenes (got $ANCHOR_COUNT)"; exit 1; }
if [ "$CONTENT" = "constant" ]; then
  [ -n "${UBR_RGB_DESCRIPTOR_MEAN:-}" ] && [ -f "$UBR_RGB_DESCRIPTOR_MEAN" ] \
    || { echo "FAIL: the constant arm requires UBR_RGB_DESCRIPTOR_MEAN (train-only mean json)"; exit 1; }
fi

# Queue guard (spec section 6): this launch becomes the SECOND training
# process per card only if exactly one job is currently running. Two jobs
# (EXP-0028-R2 + EXP-0029) means the slot is NOT open yet -> refuse.
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
PER_CARD_MAX=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort | uniq -c | awk '{print $1}' | sort -nr | head -1 || echo 0)
echo "gpu procs    : total=$PROCS max_per_card=${PER_CARD_MAX:-0} (launch requires max_per_card < 2)"
[ "${PER_CARD_MAX:-0}" -lt 2 ] || { echo "FAIL: a card already hosts 2 training processes - the EXP-0030 slot opens when EXP-0028-R2 or EXP-0029 finishes (spec section 6)"; exit 1; }

# Contract archived to a SIDELINE file: the runner fail-closed asserts the
# output root does NOT exist yet, so the wrapper must not create it.
{ echo "=== EXP-0030 UBR DUAL+RGB WRAPPER CONTRACT (v4) ==="; \
  echo "rgb_content $CONTENT commit $GIT_SHA tag $GIT_TAG"; \
  echo "geom_const_sha $GEOM_CONSTANTS_SHA"; env | grep -E "TRAIN_|SAM2_CKPT|UBR_"; \
  sha256sum configs/ablations/ue_ubr_dual_rgb_512.py \
            rsprompter/uav/ubr_dual_integration.py \
            rsprompter/uav/ubr_dual_rgb_integration.py \
            rsprompter/uav/rgb_p0_encoder.py; } \
  | tee "${UBR_RGB_OUT_ROOT}.launch_contract.txt"

exec $PY -u scripts/probes/ue_ubr_dual_rgb_real_20261028.py \
  --rgb-content "$CONTENT" \
  --out-root "$UBR_RGB_OUT_ROOT" \
  --audit-verdict "$AUDIT"
