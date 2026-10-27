# EXP-0028-R2 (UBR v1.1) Independent Pre-Relaunch Audit — 2026-10-27

VERDICT: FAIL (relaunch blocked — 2 mechanical blockers, no numerical or seal-invariance regression found)

Auditor scope: read-only review + independent re-run of G1/G4 on GPU3
(CUDA_VISIBLE_DEVICES=3, PY=/data2/wangcheng/envs/cvt2/bin/python — bare
`python` lacks mmengine). HEAD a8968af, tagged exp/EXP-0028-R2, tree clean.

## 1. Gate re-runs (independent, GPU3)

- **G1 PASS — implementer's claim CONFIRMED.** worst_rel_err exactly **0.0**
  on all 3 replayed updates (dead-arm ep1 rank-0 order, indices [1696, 121,
  591]) for every component (loss_ubr / focal / dice): v1.1-at-old-capacity
  is bit-identical to the frozen v1.0 assembly. Magnitude ratios vs dead-arm
  ep1 epoch means 2.019 / 1.976 / 2.038, inside the prereg [0.25, 4.0] band.
- **G4 FAIL at HEAD — gate is stale, seal property itself PASSED.**
  My re-run trips `assert new != old` at the first CONTRACT_SOURCE_FILES
  entry (`rsprompter/uav/ubr.py`): the source_hashes "bump" check compares
  the working tree against **HEAD** blobs, but v1.1 is now committed at HEAD
  (tag exp/EXP-0028-R2, clean tree), so the hashes are equal by design. G4
  as shipped cannot pass at the very commit it is meant to certify.
  Critically, every seal-relevant assert executed **before** the bump check
  passed in my run: patched-v1.1 build vs sealed EXP-0025 build have
  identical key sets (a_only == ubr keys, b_only == decoder-tail keys),
  **every shared (non-UBR) state_dict tensor bit-equal**, LoRA structure
  identical, frozen set contains the prompt stack and PE. The frozen-SHA
  value claim (1dff1b8101a803f5…) is no longer recoverable from artifacts
  (see §5) and was not re-verified literally; the invariance property it
  fingerprints was independently re-verified PASS. The "665 non-UBR tensors
  bit-equal" count is consistent with the passing `not mismatched` assert
  but the literal count is not re-emitted on a failed gate.

## 2. Spot-review of the two highest-risk v1.1 changes
(`rsprompter/uav/ubr_train_integration.py`) — both SOUND:

- **(a) Bucketed loss restacking in capture order.** `bucket_of =
  digitize(window areas)` over all n_pos windows; per bucket `idx` (ascending
  original positions) → `focal_all[int(j)] = focals[slot]` restacks
  per-instance losses to their original capture slots, with a fail-closed
  `all(v is not None)` assert; final `stack(...).mean()` over n_pos is
  exactly the v1.0 per-instance-equal-weight reduction. The A4 per-bucket
  budget `loss_b = λ·(Σfocals+Σdices)/n_pos` is algebraically identical to
  the unbucketed loss. Empty-band instances contribute zeros exactly as in
  v1.0; `img_of`/`gt_of` ordering and `nbr = all-GT-sum − own` match the
  frozen v1.0 reference. The `& inside` guard reproduces v1.0's
  pad-excluded band semantics.
- **(b) fp16 assembly with fp32 band membership.** SDT is computed in fp32
  numpy (`sdt32`); `band_np = (|sdt32| ≤ k) & inside` is derived from the
  **fp32** SDT **before** any fp16 cast and stays bool to the loss; gt stays
  fp32; feat/logit/sdt/nbr are stored fp16 under autocast and focal/dice are
  computed via `.float()` upcast from the fp16 logits — precisely the
  claimed design. G1's 0.0 rel err seals it empirically.

## 3. Reseal runner (`scripts/probes/ue_ubr_reseal_20261027.py`) — CONFIRMED

`os.environ["VAL_SHARD_ACROSS_RANKS"] = "1"` set authoritatively (direct
assignment, not setdefault); `OUT_ROOT_DEFAULT =
/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_r2_20261027`;
contract v2 (contract_version=2, tag UBR-RESEAL-20261027-R2,
RESEAL_CONTRACT_V2 echoed at launch).

## 4. Wrapper (`scripts/ue/run_ubr_reseal_ddp4.sh`)

- CONFIRMED: `RESEAL_ROOT` = ue_ubr_reseal_r2_20261027 (line 8); tag check
  accepts exp/EXP-0028* and HEAD carries exp/EXP-0028-R2.
- **BLOCKER 1 (relaunch fails immediately):** line 41 execs
  `scripts/probes/ue_ubr_reseal_r2_20261027.py` — **no such file exists**
  under $CODE ($WORKTREE/portable_sam2_explicit_coarse/scripts/probes/; the
  actual R2 runner is `ue_ubr_reseal_20261027.py`). One-line filename fix.
- Minor: the audit gate still reads VERDICT from the R1 file
  docs/reviews/UBR-RESEAL-20261027_audit.md rather than this R2 audit.

## 5. Side effects + observations (non-blocking)

- Re-running the gates overwrote g1.json (now this audit's PASS) and g4.json
  (now this audit's FAIL) in /data2/.../ue_ubr_v11_gates_20261027/ — inherent
  to the fail-closed script; no preserved G4 PASS artifact exists elsewhere,
  which is why the literal frozen-SHA value is not re-verifiable. Recommend
  the fix for BLOCKER 2 below be followed by a fresh full G4 PASS and its
  archival into the repo.
- **BLOCKER 2:** G4's source_hashes check must compare against the dead-arm
  blobs (git 36b5215) instead of HEAD (or treat post-commit equality as the
  expected terminal state). Until then G4 cannot pass at the launch tag.
- Registry EXP-0028-R2 row (docs/EXP_REGISTRY.md) records the G3-speed
  falsification decision — "PI 接受速度证伪" with G3 memory 11.6 GB (limit
  20) and epoch ~12.5 min — CONFIRMED recorded, PI accepts.

## Bottom line

Numerics and seal invariance are in good order (G1 independently
reproduced at 0.0 rel err; non-UBR tensor bit-equality re-verified), but
relaunch must NOT proceed until (1) the wrapper's runner filename is
corrected and (2) G4's source-hash check is fixed to be satisfiable at the
tagged commit and re-run to a full PASS. Both are mechanical fixes that do
not touch the sealed trainer or the v1.1 execution layer.
