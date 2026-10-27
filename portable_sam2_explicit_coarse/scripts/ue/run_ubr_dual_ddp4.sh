#!/bin/bash
# EXP-0029 UBR level-4 dual-stream launch wrapper (THE only launch path).
# Usage: run_ubr_dual_ddp4.sh [real|constant]   (default: real)
#
# Fail-closed contract echo (RESEAL_CONTRACT v3): commit/tag exp/EXP-0029,
# audit verdict, dataset pins (asserted in the runner), augmentation
# protocol, arm + port-5 mode, geometry cache/constants SHA, output root.
# Every launch value is echoed and archived to a SIDELINE file (the runner
# fail-closed asserts the output root does NOT exist yet).
#
# Co-tenant discipline (prereg 2026-10-27): max 2 training processes per
# card; EXP-0028-R2 normally occupies all four cards with one process ->
# this launch becomes the second. The wrapper refuses to start if any
# card already hosts 2 training processes.
set -euo pipefail

ARM=${1:-real}
case "$ARM" in
  real|constant) ;;
  *) echo "FAIL: arm must be real|constant (got: $ARM)"; exit 1 ;;
esac

export UBR_DUAL_OUT_ROOT=/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_dual_${ARM}_20261027
export SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt
export TRAIN_FLIP_PROB=0          # 3090 protocol: NO flip (dual-stream)
export TRAIN_VFLIP_PROB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
PY=/data2/wangcheng/envs/cvt2/bin/python
WORKTREE=/home/wangcheng/project/new/.worktrees/dual-stream-dev
CODE=$WORKTREE/portable_sam2_explicit_coarse
GEOM_CACHE=${UBR_GEOM_CACHE:-/data2/wangcheng/remote-lthpc/cache/ue_v1_native_geometry}
GEOM_CONSTANTS=${UBR_GEOM_CONSTANTS:-/data2/wangcheng/remote-lthpc/checkpoint/ue_geometry_mechanism_20261022/common_constants.json}
AUDIT=$WORKTREE/docs/reviews/UBR-DUAL-20261027_audit.md

cd "$CODE"
GIT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)
GIT_TAG=$(git -C "$WORKTREE" describe --tags --exact-match HEAD 2>/dev/null || echo UNTAGGED)
GEOM_CONSTANTS_SHA=$(sha256sum "$GEOM_CONSTANTS" | cut -d' ' -f1)

echo "=== EXP-0029 DUAL WRAPPER CONTRACT (v3) ==="
echo "commit       : $GIT_SHA"
echo "tag          : $GIT_TAG"
echo "arm          : $ARM (port5: $ARM; geometry head trained on REAL map)"
echo "audit        : $(grep -m1 VERDICT "$AUDIT" || echo MISSING)"
echo "config       : $(sha256sum configs/ablations/ue_ubr_dual_512.py | cut -c1-16)"
echo "geom cache   : $GEOM_CACHE ($(ls "$GEOM_CACHE" | wc -l) npz, v native512_height_dx_dy_viewcount_v1)"
echo "geom const   : $GEOM_CONSTANTS_SHA"
echo "flip         : $TRAIN_FLIP_PROB (3090 protocol: no flip; photometric-only jitter)"
echo "out          : $UBR_DUAL_OUT_ROOT"
[ "$(git -C "$WORKTREE" status --porcelain | wc -l)" -eq 0 ] || { echo "FAIL: dirty tree"; exit 1; }
[ "${GIT_TAG#exp/EXP-0029}" != "$GIT_TAG" ] || { echo "FAIL: HEAD not tagged exp/EXP-0029"; exit 1; }
grep -q "VERDICT: PASS" "$AUDIT" || { echo "FAIL: audit not PASS"; exit 1; }
[ -f "$GEOM_CACHE/000000.npz" ] || { echo "FAIL: geometry cache missing"; exit 1; }

# Prereg co-tenant cap: at most 2 training processes per card (our launch
# adds one to each of the 4 cards).
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
PER_CARD_MAX=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | sort | uniq -c | awk '{print $1}' | sort -nr | head -1 || echo 0)
echo "gpu procs    : total=$PROCS max_per_card=${PER_CARD_MAX:-0} (cap: 2 incl. this launch)"
[ "${PER_CARD_MAX:-0}" -lt 2 ] || { echo "FAIL: a card already hosts 2 training processes (prereg cap)"; exit 1; }

# Contract archived to a SIDELINE file: the runner fail-closed asserts the
# output root does NOT exist yet, so the wrapper must not create it.
{ echo "=== EXP-0029 DUAL WRAPPER CONTRACT (v3) ==="; \
  echo "arm $ARM commit $GIT_SHA tag $GIT_TAG"; \
  echo "geom_const_sha $GEOM_CONSTANTS_SHA"; env | grep -E "TRAIN_|SAM2_CKPT|UBR_"; \
  sha256sum configs/ablations/ue_ubr_dual_512.py rsprompter/uav/ubr_dual_integration.py; } \
  | tee "${UBR_DUAL_OUT_ROOT}.launch_contract.txt"

exec $PY -u scripts/probes/ue_ubr_reseal_dual_20261027.py \
  --arm "$ARM" \
  --out-root "$UBR_DUAL_OUT_ROOT" \
  --audit-verdict "$AUDIT"
