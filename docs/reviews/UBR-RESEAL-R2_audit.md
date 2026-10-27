# EXP-0028-R2 (UBR v1.1) Independent Pre-Relaunch Audit — 2026-10-27

VERDICT: PASS (round 2 — both blockers fixed and independently re-verified;
one launch precondition remains, see §4, enforced fail-closed by the wrapper)

Auditor scope: read-only review + independent re-runs on GPU3
(CUDA_VISIBLE_DEVICES=3, PY=/data2/wangcheng/envs/cvt2/bin/python — bare
`python` lacks mmengine). Round 1 audited HEAD a8968af (FAIL, 2 blockers);
round 2 re-verified the fixes at HEAD **c251e5a**, tagged exp/EXP-0028-R2.

## 1. Round 2 — the two blocker fixes (both VERIFIED)

- **Blocker 1 (wrapper exec path) — FIXED.**
  `scripts/ue/run_ubr_reseal_ddp4.sh` line 41 now execs
  `scripts/probes/ue_ubr_reseal_20261027.py`, which exists and is the R2
  runner (RUN_TAG/OUT_ROOT_DEFAULT = ue_ubr_reseal_r2_20261027). The
  single-line diff restores the path mangled by the earlier global sed;
  RESEAL_ROOT (line 8) keeps the r2 name, as intended.
- **Blocker 2 (G4 stale at HEAD) — FIXED and independently re-run: PASS.**
  `_git_blob_sha` gained a `ref` argument and the source-hash bump check now
  compares v1.1 files against the dead-arm commit **36b5215** instead of
  HEAD. The diff touches only these two hunks in the gates file — every
  seal-invariance assert is unchanged. Independent G4 re-run at c251e5a
  (GPU3, 13.4 s): **PASS**, and the literal round-1 claims are now fully
  reproduced:
  - `frozen_param_sha` =
    `1dff1b8101a803f5a6ab472f3339b3a4a51de6a63e77cac5c1bcec0bfd372ff0`
    — exactly the claimed `1dff1b8101a803f5…` prefix.
  - `non_ubr_tensors_compared` = **665**, every shared (non-UBR) tensor
    bit-equal between the v1.1-patched build and the sealed EXP-0025 build
    (frozen set 323 params incl. prompt stack 16 + PE 2; LoRA structure 104
    keys identical; UBR-only 28 keys vs sealed-tail 6; capacity 32×6 =
    67,457 params).
  - All 4 CONTRACT_SOURCE_FILES bumped vs 36b5215 (ubr.py,
    ubr_train_integration.py, ue_ubr_reseal_20261027.py,
    ue_ubr_single_512.py); RESEAL_CONTRACT v2 echo verified (version 2,
    tag UBR-RESEAL-20261027-R2, capacity 32×6, val_shard=1).
  - G1 remains valid at c251e5a without re-running: a8968af..c251e5a
    changed only the gates script's two hunks and the wrapper line (plus
    this audit file); the integration sources G1 exercises are identical.

## 2. Round 1 record (superseded FAIL, kept for provenance)

- G1 independently re-run at a8968af on GPU3: PASS, worst_rel_err exactly
  **0.0** on all 3 replayed updates (dead-arm ep1 rank-0 order, indices
  [1696, 121, 591]) for loss_ubr/focal/dice — v1.1-at-old-capacity is
  bit-identical to the frozen v1.0 assembly; magnitude ratios vs dead-arm
  ep1 2.019/1.976/2.038, inside [0.25, 4.0].
- Spot-review of the two highest-risk v1.1 changes in
  `rsprompter/uav/ubr_train_integration.py` — both SOUND:
  (a) bucketed restack: `focal_all[int(j)] = focals[slot]` restores capture
  order with a fail-closed all-filled assert; mean over n_pos preserves
  per-instance equal weight; A4's `λ(Σ)/n_pos` is algebraically identical;
  empty-band instances contribute zeros exactly as v1.0; `& inside`
  reproduces v1.0's pad-excluded band semantics.
  (b) fp16 assembly / fp32 band: band computed from fp32 numpy SDT
  (sdt32) before any fp16 cast and kept bool; gt fp32; feat/logit/sdt/nbr
  fp16 under autocast; focal/dice via `.float()` upcast — as claimed, and
  sealed empirically by G1's 0.0 rel err.
- Reseal runner: VAL_SHARD_ACROSS_RANKS=1 authoritative (direct assign,
  line 209); OUT_ROOT_DEFAULT r2 (line 67); contract v2 echo at launch.
- Registry EXP-0028-R2 row records "PI 接受速度证伪" (G3 11.6 GB < 20 GB
  limit, epoch ~12.5 min) — PI accepts, recorded.
- Failures found: wrapper exec'd a nonexistent `ue_ubr_reseal_r2_20261027.py`;
  G4's bump check compared working tree vs HEAD and could never pass on the
  clean tagged tree. Both fixed in c251e5a (§1).

## 3. Wrapper checks at c251e5a

RESEAL_ROOT = ue_ubr_reseal_r2_20261027; tag check accepts exp/EXP-0028*
and HEAD c251e5a carries exp/EXP-0028-R2; exec target exists; audit gate
reads VERDICT from the R1 file docs/reviews/UBR-RESEAL-20261027_audit.md
(exists, PASS at R1 — cosmetic note only: it does not read this R2 file).

## 4. Launch precondition (fail-closed, not a code defect)

`git status --porcelain` at c251e5a shows ONE untracked file:
**docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md** — the v1.1 design spec
referenced by the gates docstring and the registry row. The wrapper asserts
a fully clean tree before exec, so the relaunch will refuse to start until
this doc is committed (or otherwise resolved). Commit it as part of the
final pre-launch commit; no code change required.

## 5. Side-effects note

Re-running the gates overwrote g1.json/g4.json in
/data2/.../ue_ubr_v11_gates_20261027/ — the current g1.json (round-1 PASS,
0.0 rel err) and g4.json (round-2 PASS, full SHA + counts) are this audit's
independent runs. Recommend archiving both into the repo with the launch
contract.

## Bottom line

All audited claims are confirmed, both round-1 blockers are fixed and
re-verified at the tagged HEAD c251e5a, and G1/G4 are green under
independent re-run. Relaunch is cleared once the untracked v1.1 spec doc is
committed so the wrapper's clean-tree gate passes.
