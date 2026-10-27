#!/bin/bash
# EXP-0028 UBR re-seal level-3 launch wrapper (THE only launch path).
# Fail-closed contract echo: commit/tag, EXP-ID, audit verdict, dataset
# pins, augmentation protocol, output root. Reproducibility: every launch
# value is echoed and archived into the output dir before training starts.
set -euo pipefail

export RESEAL_ROOT=/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_r2_20261027
export SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt
export SAM2_REPO=${SAM2_REPO:-}
export TRAIN_FLIP_PROB=0          # 3090 protocol: NO flip (dual-stream)
export TRAIN_VFLIP_PROB=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/data2/wangcheng/envs/cvt2/bin/python
WORKTREE=/home/wangcheng/project/new/.worktrees/dual-stream-dev
CODE=$WORKTREE/portable_sam2_explicit_coarse

cd "$CODE"
GIT_SHA=$(git -C "$WORKTREE" rev-parse HEAD)
GIT_TAG=$(git -C "$WORKTREE" describe --tags --exact-match HEAD 2>/dev/null || echo UNTAGGED)
AUDIT=$WORKTREE/docs/reviews/UBR-RESEAL-20261027_audit.md

echo "=== EXP-0028 WRAPPER CONTRACT ==="
echo "commit      : $GIT_SHA"
echo "tag         : $GIT_TAG"
echo "audit       : $(grep -m1 VERDICT "$AUDIT")"
echo "config      : $(sha256sum configs/ablations/ue_ubr_single_512.py | cut -c1-16)"
echo "flip        : $TRAIN_FLIP_PROB (3090 protocol: no flip)"
echo "aug         : ColorJitter(.15,.15,.10,.02)@p.5 + Gauss(.5,.02) train-only"
echo "out         : $RESEAL_ROOT"
[ "$(git -C "$WORKTREE" status --porcelain | wc -l)" -eq 0 ] || { echo "FAIL: dirty tree"; exit 1; }
[ "${GIT_TAG#exp/EXP-0028}" != "$GIT_TAG" ] || { echo "FAIL: HEAD not tagged exp/EXP-0028"; exit 1; }
grep -q "VERDICT: PASS" "$AUDIT" || { echo "FAIL: audit not PASS"; exit 1; }

# Contract archived to a SIDELINE file: the runner fail-closed asserts the
# output root does NOT exist yet, so the wrapper must not create it.
{ echo "=== EXP-0028 WRAPPER CONTRACT ==="; env | grep -E "TRAIN_|SAM2_CKPT"; \
  echo "commit $GIT_SHA tag $GIT_TAG"; sha256sum configs/ablations/ue_ubr_single_512.py; } \
  | tee "${RESEAL_ROOT}.launch_contract.txt"

exec $PY -u scripts/probes/ue_ubr_reseal_r2_20261027.py \
  --out-root "$RESEAL_ROOT" \
  --audit-verdict "$AUDIT"
