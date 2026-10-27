# Independent pre-launch audit — UBR-DUAL-20261027 (EXP-0029, level-4 dual-stream arm ② REAL)

Date: 2026-10-26/27 (audit session) · Commit under review: af71c4934a4502ba92f665b797f5f1f7d899ef94 (= worktree HEAD, "feat(exp): level-4 dual-stream arms — geometry head + port-5 wiring, gates PASS")
Scope: read-only audit; only the gates script was re-run (CUDA_VISIBLE_DEVICES=3, co-tenant with running EXP-0028-R2) and this verdict file written. R2 training on GPUs 0-3 was not touched.

VERDICT: PASS

## 1. Gates re-run (d1/d2/d3) — PASS, reproduces the recorded results

Re-ran `scripts/probes/ue_ubr_dual_gates_20261027.py all` on GPU3 (exit 0; JSONs rewritten to
`/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_dual_gates_20261027/`). Results bit-consistent
with the pre-recorded run (same seed 44; identical loss values to 6 dp):

- d1 config_diff_exact PASS (4.3 s): resolved dual-vs-single diff is exactly
  `{uav_port: 'zero'->'real'}` + added `geometry_loss_weight: 1.0`; nothing removed; validator
  accepts real+constant arms and rejects band_k/loss_weight/refiner_hidden/geometry_loss_weight
  drift and the missing-key variant.
- d2 seal PASS (11.0 s): frozen-parameter SHA `1dff1b8101a803f5…` == pinned v1.1 G4 reference
  `1dff1b8101a803f5a6ab472f3339b3a4a51de6a63e77cac5c1bcec0bfd372ff0` (323 frozen tensors
  incl. prompt stack + no_mask_embed); 665 shared tensors bit-equal; only `ubr_refiner.*` /
  `geometry_head.*` keys added (67,457 / 202,581 params, heat bias −2.19, all trainable); LoRA
  structure identical (104 keys); UDPR tail absent from both builds (6 tail keys in sealed only).
- d3 dual smoke PASS (34.2 s): loss_ubr + loss_geometry present and finite on 3 train batches;
  UBR/* and GEO/* telemetry live; joint fwd+bwd graph healthy (geometry-head AND ubr_refiner
  grads nonzero); FW1 all zeros (see §2c); val pass-1 at deployed cap: 3 scenes × 250 instances,
  prefix band 250 / candidate band 0 (untrained prefix saturates the cap — correct contract
  behavior), val pass-2 with free slots admits 450 candidates; constant-arm predict clean with
  support channel real and holes zeroed; peaks: tier A 2.66 GB, tier B full-backward 4.98 GB
  (< safety cap 26.1 GB AND < hard 6 GB), val 3.336 GB, constant 3.824 GB — all under the 6 GB
  co-tenant budget while R2 held ~14–16.5 GB on each card.

## 2. Highest-risk spot reviews (rsprompter/uav/ubr_dual_integration.py)

### (a) Port-5 per-bucket/per-instance crops — CORRECT

- Train (`_bucket_step`): the port crop `self._port_np[img_of[j]][:, sl[0], sl[1]]` uses the
  exact same slice tuple `sl = (h0:h0+h, w0:w0+w)` as the neighbor crop and the same
  `wins[j]` (h0,w0,h,w) as the mask/logit/gt crops. Boundary safety: `v11.window_around`
  asserts `0 <= h0 < h1 <= 512` (clamped via `min(limit, ceil(x2)+margin)`), so `h0+h <= 512`
  always — slices on the [5,512,512] staged map are always exact-size; a corner window
  (margin clamped at 0) and a far-edge window (clamped at 512) both yield the full window with
  no truncation, and any hypothetical mismatch would raise a shape error into the zero-padded
  `geom_pad[:, :, :h, :w]` (fail-closed by construction).
- Eval (`_refine_inputs`): `port[:, h0:h0+h, w0:w0+w]` uses identical offsets to the
  logit/sdt crops from the same `win = v11.window_around(...)`; the stride-4 `feat_window`
  registration is (correctly) applied ONLY to the feature tap, not the 512-grid port map.
- Padding discipline identical to the other channels: top-left zero-pad (`_pad_batch` /
  preallocated zero pads), zeros outside the crop where `inside`/band is False.
  `ubr.UAV_GEOMETRY_CHANNELS == 5` matches the staged map depth (4 fields + support).

### (b) Val merge band invariant + cap math — CORRECT

- Bands: candidates scored `0.25 · s` with s = sigmoid heat ∈ (0,1] ⇒ candidate band ⊆
  (0, 0.25]; prefix rescored `r(s) = 0.5 + 0.5/(1 − log s)` ⇒ strictly (0.5, 1] for positive
  scores. 0.25 < 0.5 ⇒ the bands can never cross. Both bounds are asserted post-concat in
  `_concat_instances` (`cand > 0`, `cand <= 0.25 + 1e-6`, `prefix > 0.5`), and the d3 gate
  independently re-checks the partition (`cand + pref == n`). Edge: `prefix_band_score` maps
  s <= 0 to exactly 0.5, which would fail the `> 0.5` assert — unreachable for sigmoid-derived
  scores and fail-closed by design.
- Cap: `select_candidates` keeps `keep[:max(0, CANDIDATE_CAP − n_pref)]` = 250 − len(prefix)
  after IoU-dedup vs prefix (≤ nms.iou_threshold) and NMS among candidates; the merge then
  asserts `n_pref + n_cand <= 250`; decode itself is top-250. d3 pass-1 shows the saturated
  case (250 prefix / 0 admitted = correct), pass-2 shows admission into free slots.
- Coordinate-space consistency of the merge (512-grid candidate boxes vs prefix boxes) is
  enforced downstream by the v1.1 paste-parity + window asserts, all exercised in the passing
  d3 val passes.

### (c) FW1 zero-leak — REAL, not vacuous

- Instrumentation: the gate monkeypatches the `ubr_dual_integration.decode_candidates` module
  global — the exact name `_merge_geometry_candidates` resolves at call time — and asserts
  `calls == 0` across the 3 tier-A loss forwards and the tier-B full-gradient loss forward.
  Not vacuous: any decode invocation on a loss path increments and fails.
- Both leak directions are measured on actual autograd grads with a fresh forward:
  `loss_ubr.backward()` → zero nonzero grads on geometry-head params;
  `loss_geometry.backward()` → nonzero geometry-head grads and zero nonzero grads on every
  non-geometry param. Structural backstop: `decode_candidates` is `@torch.no_grad` and only
  reachable from the eval-only predict wrap (`not model.training`); `compute_loss_geometry`
  consumes only the no-grad cached real map and GT; `loss_ubr`'s port-5 input is numpy-sourced
  (no autograd into the head). Confirmed zero leak in the re-run (`fw1` all 0).

## 3. Runner / wrapper (`scripts/ue/run_ubr_dual_ddp4.sh`, `run_ubr_dual_real.sh`, `scripts/probes/ue_ubr_reseal_dual_20261027.py`)

- Gate: wrapper greps this file for literal `VERDICT: PASS`; runner additionally requires
  `--audit-verdict` file containing it, output-root-absent, dataset SHA pins + split counts,
  `decoder_tail_refiner_cfg == {}`, `bucket_backward False`, `grad_checkpoint False`,
  `uav_port == arm`. All present and fail-closed.
- Clean tree: wrapper `git status --porcelain` empty check; runner re-asserts. Present.
- Tag: wrapper requires `git describe --tags --exact-match HEAD` to start with `exp/EXP-0029`.
  Present as a hard gate. NOTE (launch prerequisite, not a finding): af71c49 is HEAD and is
  currently UNTAGGED — the wrapper will (correctly) refuse until the operator tags it.
- Co-tenant guard: wrapper counts compute apps per GPU UUID and refuses if any card already
  hosts >= 2 training processes (prereg cap: max 2/card). Present; current GPU state has
  exactly one R2 process per card (~14–16.5 GB of 48 GB), so a dual launch becomes the second
  per card as preregistered.
- `PYTHONUNBUFFERED=1` exported in the wrapper (plus `python -u` / DDP env passthrough). Yes.
- `TEST_MAX_PER_IMG=250`: confirmed ABSENT from the R2 runner — i.e. it is a registered,
  deliberate deviation in the dual runner, with the deviation note in-source (canonical AP250:
  evaluate the MERGED ≤250 set as prefix r(s) band + 0.25·s candidates at maxDets 250 instead
  of the default 100, otherwise every candidate ranks below the prefix band and the arm's
  candidate stream is invisible in per-epoch telemetry; collection-time R1/R2 readouts use the
  same AP250). Justified; d3 pass-1 mirrors the deployed cap.
- 3090 augmentation patch: functionally byte-identical to the R2 runner
  (`scripts/probes/ue_ubr_reseal_20261027.py`) — only docstring wording and a private
  attribute name differ; same ColorJitter(.15,.15,.10,.02)@p=.5 on BGR→RGB uint8, same
  gaussian noise (.5/.02), train split only, zero flip. Protocol parity with R2 holds.
- Arm ③ discipline: `run_ubr_dual_real.sh` execs `run_ubr_dual_ddp4.sh real` only; constant
  arm requires an explicit manual launch. Nothing auto-launches it.

## 4. R2 safety of commit af71c49 — SAFE

`git diff af71c49^..af71c49 --stat`: 6 files, **2131 insertions, 0 deletions** — all six are
NEW files (config, integration module, two probes, two shell scripts). No file that the
running R2 imports (sealed trainer, `ubr_train_integration`, configs, data/) is modified; the
new module imports v1.1 but nothing in the sealed path imports the new module.

## Non-blocking observations

1. af71c49 untagged at audit time — tag `exp/EXP-0029` before launch (wrapper enforces).
2. Stale conservative comment in the gates script ("a real train batch … needs ~10.8 GB
   peak") vs the measured 4.98 GB full-backward peak; both the hard 6 GB and the
   measured-headroom asserts pass, so no action required.
3. Val telemetry counts `candidates` as the pre-dedup decode top-k (250/img) — labeled as
   such in `drain_val_stats`; readouts should use `accepted_mean`, not `candidates_mean`.
