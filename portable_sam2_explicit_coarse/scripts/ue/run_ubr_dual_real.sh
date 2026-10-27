#!/bin/bash
# EXP-0029 arm ② (dual-REAL) launcher - the PI-designated launch path
# (scope note 2026-10-27: arm ③ constant is QUEUED for a later manual
# launch via run_ubr_dual_ddp4.sh constant; nothing auto-launches it).
#
# Co-tenant with the running EXP-0028-R2 (GPUs 0-3, ~12.5-15.5 GB/card,
# 48 GB cards): this launch is the SECOND training process per card, per
# the prereg cap (max 2 training processes per card, ~16 GB expected).
# All contract discipline (commit, tag exp/EXP-0029, audit VERDICT: PASS,
# clean tree, co-tenant guard, contract echo v3) lives in - and is
# enforced by - run_ubr_dual_ddp4.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
exec bash "$HERE/run_ubr_dual_ddp4.sh" real
