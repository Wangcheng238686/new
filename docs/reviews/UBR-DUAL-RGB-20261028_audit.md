# Independent pre-launch audit — UBR-DUAL-RGB-20261028 (EXP-0030, dual(D+pose) + RGB real arm)

Date: 2026-10-28 (audit session) · Commit under review: bb230692e964f14799d4cbc1bd947e601ef0e440
(worktree HEAD, "feat(exp): EXP-0030 RGB arm — spec + implementation, G1/G2 PASS").
Scope: gates G3/G4/G5 executed on GPU3 (CUDA_VISIBLE_DEVICES=3, co-tenant with the running
EXP-0028-R2 + EXP-0029 trainings — not touched, verified alive after every gate); G1/G2
reports re-read and their three chief fixes re-derived; full read-only implementation audit
against docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md §1–§4, §7 including live remote SHA
verification over ssh (lthpc). NOTHING was launched (PI order: launch forbidden under current
co-tenancy; the queue guard in scripts/ue/run_ubr_dual_rgb_real.sh correctly refuses while
2 jobs/card run). NOTHING was committed; the three mechanical fixes below are left uncommitted
in the worktree for the chief to commit before launch.

VERDICT: PASS

## 1. Gate results

Reports: /data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_dual_rgb_gates_20261028/{g1..g5}.json

- G1 skeleton seal — PASS (chief's run, 13.2 s): config resolved diff dual→rgb is exactly
  `{rgb_content: 'real'}`; 737 shared tensors bit-equal; only `backbone.rgb_source_branch.*`
  added (46 keys, 419,856 params, 41,264 trainable = LoRA + rgb_project + rgb_output only);
  ctor torch.load paths == {sam2_hiera_base_plus.pt} (task_weights_loaded False); frozen SHA
  `1dff1b81…` == v1.1 G4 reference AND dual build; rgb_output zero-init True; local ported
  file SHA `af32556a…` == PORT_SOURCE_SHAS pin.
- G2 identity — PASS (chief's run, 51.4 s): all 32 loss tensors bitwise-equal to the EXP-0029
  build (2 batches × 16 losses, max_abs_diff 0.0), predict masks bitwise + scores equal,
  reference double-forward bit-stable; one real AdamW update: first backward lights up ONLY
  rgb_output (grad |·|₁ 87.969; lora = project = exactly 0 — the zero-init residual math),
  second backward lights up lora + project (asserted > 0), rgb_output leaves zero, residual
  RMS 0.01233 > 0, RGB telemetry complete; peaks: identity 2.82 GB, full update 13.637 GB
  (< 22 GB spec cap AND < measured free headroom).
- G3 resource smoke — PASS (this audit, 25.9 s): tier A no-grad loss 2.943 GB; ONE
  full-gradient batch 5.061 GB (< safety cap 10.816 GB = free−4 < 22 GB spec; free at run
  time 14.8 GB with both trainings resident); val predict tier 3.07 GB; all losses finite,
  loss_ubr + loss_geometry present; val band contract holds (2 scenes × 250 instances,
  prefix band 250 / candidate band 0 — untrained prefix saturates the cap at init, the same
  correct contract behavior EXP-0029's d3 documented); RGB telemetry live with residual
  RMS exactly 0.0 at zero-init.
  **Epoch budget: slower_than_0029_ratio = 0.0074 (0.74%) vs budget 0.35** — measured from
  one full fwd+bwd batch (4.523 s) minus the staged RGB stage share (0.033 s warm; encoder
  0.0074 s). This is a lower bound (P0 backward share is not staged-separable); even the
  cold-cache stage time (0.177 s ⇒ 3.9%) sits ~9× under budget. Extrapolation across the
  epoch is sound: the RGB stage is per-batch deterministic work (anchor npz + 5 rgb.png
  reads + one merged 5-view P0 forward).
- G4 contract seal — PASS (this audit, 78.4 s incl. the one-time 8.4 GB anchor-cache hash):
  tag `UBR-DUAL-20261028-REAL`, contract_version 4, protocol field-equal to EXP-0029's
  PROTOCOL, dual−single = {port5, loss_geometry, rgb_injection}, geometry block verbatim v3
  (head_params 202,581, λ=1.0, cache native512_height_dx_dy_viewcount_v1, constants SHA
  `761d9795…`, r(s)/0.25/250), rgb block complete (P0 256→32, LoRA config verbatim,
  foundation-only init list, task_weights_loaded False, optimizer group backbone @5e-4/wd.05,
  injection site blocks[4] 64×64 zero-init), anchor cache 3060 scenes / 8.4 GB with manifest
  SHA `ac74a335808b576c…` (64-hex, file exists), port-source SHAs == PORT_SOURCE_SHAS with
  commit 1618908b…, source_hashes include all four +RGB files plus the v3 set, val shard
  inheritance (VAL_SHARD_ACROSS_RANKS=1, TEST_MAX_PER_IMG=250), audit-verdict requirement
  recorded.
- G5 constant drill — PASS (this audit, after fixes F1/F2 below; final run 32.6 s): drill
  descriptor-mean json sha `bc15b83546f64be2…` (train-split-derived gate batches only, with
  the recompute-over-full-train-split note); support masks identical to the real arm;
  encoder skipped (`rgb_encoder_secs == 0.0`, stage 0.6 ms vs 33 ms real); outputs bitwise
  identical to the real arm at zero-init (masks + scores); constant descriptors exactly
  constant (every supported cell bitwise the same 32-d vector); peak 2.621 GB. NOT launched —
  the fourth arm stays queued pending an RGB_WINS verdict (spec §5/§7).

## 2. Remote port fidelity (spec §2) — verified live against lthpc

- `rsprompter/uav/rgb_p0_encoder.py`: local file sha256 `af32556a8d27f349…` == remote file
  sha == PORT_SOURCE_SHAS pin; `diff` of the fetched remote file vs local is EMPTY
  (byte-identical verbatim port).
- Git blobs at the pinned remote commit 1618908b55f96b6fcc324a0a019ddd9938972c01 (which IS
  the remote repo's current HEAD): `rgb_p0_encoder.py` → 017e8339…, `compact_joint.py` →
  b4c049fe…, `attention_ladder.py` → a5e3cf46… — all three match the second element of the
  PORT_SOURCE_SHAS tuples, and the file-level sha256s of compact_joint/attention_ladder
  match the first elements.
- `sample_source_continuous` / `source_uv_grid`: verbatim port (only literal `32` →
  `RGB_PROJECT_DIM` = 32); same valid-mask, same per-view loop `index // 16384 == view`,
  same `(uv + 2.) / 256. − 1.` grid, same `grid_sample(bilinear, border, align_corners=False)`
  and `.T` store — no resampling of candidates, cache order preserved.
- Mean-reduce (compact `InstanceInteraction.fuse_rgb`, the candidate_detail "detail"
  variant): identical expression
  `(selected * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)` — equal
  weight over retained candidates, exact zero where none; per-token `F.layer_norm(features,
  (32,))` before sampling, identical; `pooled.T.reshape(1, 32, 128, 128)` identical;
  RGB_MEAN/RGB_STD buffers verbatim (ImageNet stats, RGB order).
- Block4 injection (compact `attach`/`interact`): trunk discovery (`stage_ends`+`blocks`,
  exactly one trunk), `stage_ends == [1,4,20,23]`, `blocks[4].dim_out == 224`,
  `register_forward_hook` on `blocks[4]`, zero-init `nn.Linear(32, 224)` (weight AND bias —
  the `self.output` discipline), `avg_pool2d(u128, 2, 2)` nominal-lattice 128→64 mapping,
  `max_pool2d(support128, 2, 2) > 0` support mapping, `delta * valid` gating, `base + delta`
  residual — all match. Compact's LN-on-u64 + neighbor/attention stages and the geometry
  CNN are intentionally NOT ported (spec §1 "独立残差路径", §7 explicit non-goals) — the
  ported payload is the per-cell mean descriptor itself, per the spec's arm definition.
- Documented adaptations (not defects): (i) the 5 UAV views pass the P0 encoder as ONE
  merged batch — the spec §6 sentinel explicitly requires this; (ii) the P0 forward runs
  under fp16 autocast with fp32 layer-norm/sampling, matching this line's AMP protocol
  (the satellite trunk gets the same treatment); (iii) the support mask is derived from the
  candidate-anchor index rather than compact's geometry-map channel — stricter and
  self-consistent: injection is gated to cells that actually contributed descriptors, so a
  post-update `rgb_output(0) = bias` cannot leak into candidate-free cells; (iv) minor nit —
  compact attach's `assert trunk.blocks[2].pool.stride == (2,2)` lattice precondition is not
  replicated; it is functionally subsumed by the block4 output-shape assert `(1, 64, 64,
  224)`.
- Encoder ctor: RSSAM2VisionEncoder(base_plus, gradient checkpointing, LoRA r16/α32/
  dropout .05/qkv+proj/bias none) under `fork_rng(seed 2045)` == compact `CompactJoint`
  `source_seed=2045` — the shared init stream is unperturbed (G1 bit-equality proves it);
  `audit_source_state` proves the retained P0 path holds bit-equal ctor state; the encoder
  runs in eval mode with only lora_dropout trainable-mode (RGBP0Encoder.train override,
  same policy as compact).

## 3. Anchor cache decision (spec §2/§3)

Location `/data2/wangcheng/remote-lthpc/cache/ue_v1_s2db_geometry_v2` (mirror of the remote
`/data/wangcheng/cache/ue_v1_s2db_geometry_v2`), version `s2db_anchor_metadata_v2_fp64`,
3060 scenes, 8.4 GB (under the 50 GB mirror cap). Contract records cache_dir, scene_count,
anchor_version, manifest path + sha256 `ac74a335808b576c5ba42c4c15ea569a25c47cac6ce5158070
62977740266b8e`, bytes_total (G4 asserts all of these; the sidecar
`…/cache/ue_s2db_anchor_manifest_v2.jsonl` holds per-file sha256s, 3060 records).
Independently spot-checked this audit: scenes 000000/000163/3059 — version string, index
(16384, 8), anchor (16384, 8, 6) fp64 finite, ≤2 tokens per view per cell and ≤8 per cell,
all 5 views populated; per-scene `report.source_hashes["cameras.json"]` == the LOCAL dataset
tree's cameras.json sha (the SceneRGBAnchorCache.load binding — the remote cache is provably
bound to these local scenes); mirror provenance bit-verified: remote npz sha256 ==
manifest sha for scenes 000000 and 003059.

## 4. FW1-style isolation, optimizer grouping, G2 fixes

- **RGB cannot reach the geometry head or the candidate path.** The injection site is the
  satellite Hiera trunk `blocks[4]` only. The geometry head's input is the native geometry
  npz map (`model.geometry_head(real[None])`) — no trunk feature, no RGB. Geometry-candidate
  decode/admission (`decode_candidates` heat/reg/support → `select_candidates` boxes-only)
  — RGB-free. UBR port 5 crops come from the staged geometry maps (`_port_np`), not RGB.
  RGB reaches the model only through trunk features downstream into the detector (the
  intended consumption), including the mask-decode of accepted geometry candidates via
  `roi_head.predict_mask` — the frozen model interface EXP-0029 already uses; FW1 proper
  (candidates never enter a detector loss; head's train-time forward consumed only by
  loss_geometry) is unchanged — the loss path is byte-identical to EXP-0029 at zero-init
  (G2) and the RGB hook never fires inside loss_geometry.
- **Optimizer grouping correct.** The branch lives at `model.backbone.rgb_source_branch`,
  so train/optimizer.py's `match_name.startswith("backbone.")` routes every RGB parameter
  (encoder LoRA + rgb_project + rgb_output) into the backbone group: lr 5e-4 ×
  backbone_lr_mult (default 1.0), weight_decay 0.05 — the satellite-LoRA same-group rule
  (spec §1) and exactly what the contract's rgb.source_encoder block pins. G1 additionally
  asserts every branch parameter name starts with `backbone.`; the trainable set within the
  branch is exactly {lora_, rgb_project, rgb_output} (41,264 params; the P0 trunk stays
  frozen).
- **The chief's three G2 fixes are correct and minimal.**
  (a) stale-line bypass: both predict extractions now go through `_pred_masks_np`
  symmetrically — coherent, no stale variable.
  (b) zero-init two-step liveness: mathematically right — with W_out ≡ 0 (weight AND bias),
  d(loss)/d(upstream) = W_outᵀ·g is exactly 0, so backward 1 can only light rgb_output
  (observed 87.969; lora = project = 0.0 exactly, asserted); after ONE AdamW step W_out ≠ 0
  (the bias gradient alone guarantees it), backward 2 lights lora + project (asserted > 0,
  passed), `injection_is_zero` flips False, residual RMS 0.0123 > 0.
  (c) MEM_HARD_GB → 22 GB: matches spec §4 G3 (per-card full-arm budget); co-tenant safety
  is preserved by the separate `free_gb` / `safety_cap = free − 4` guards, which are what
  actually bound the gate tiers today (5.06 GB vs 10.8 GB cap).

## 5. Numbered findings

1. **FIXED (mechanical gate bug, G5): `masks_ref` NameError.** cmd_g5 referenced
   `masks_ref` without ever extracting it — the `_pred_masks_np(ref_pred[0])` line was
   dropped (same bug class as the chief's G2 stale-line fix). Fixed by adding the
   extraction after the real-arm reference predict; rerun PASS. No integration change.
2. **FIXED (mechanical gate bug, G5): wrong constancy check.**
   `assert float(sel.std()) == 0.0` can never hold for a meaningful mean vector: std over
   the [N, 32] selection measures the spread ACROSS the mean vector's 32 channel values,
   which legitimately differ (reproduced: bitwise-identical rows → std 0.048). Replaced
   with the exact check `bool((sel == sel[:1]).all())` — every supported cell bitwise the
   same 32-d vector. PASS. No integration change.
3. **FIXED (mechanical launch-wrapper bug): unbound `$GEOM_CONSTANTS`.**
   scripts/ue/run_ubr_dual_rgb_real.sh:46 referenced `$GEOM_CONSTANTS` while the defined
   variable is `UBR_GEOM_CONSTANTS`; under `set -u` this degrades fail-open (bash repro:
   no crash — the `|| echo MISSING` swallows it), so the echoed contract and the archived
   `*.launch_contract.txt` sideline would record `geom_const_sha MISSING` instead of the
   real SHA. Enforced integrity is unaffected (the binding check is
   `_load_channel_means`' CONSTANTS_SHA256 assert + G4's equality assert). Fixed the
   variable reference. No behavior change elsewhere.
4. **NOTE (doc drift, not fixed): gates header comments are stale** — "hard 6 GB cap"
   (now: 22 GB spec cap + tier safety caps) and `MEM_TARGET_GB` "report-only beyond this"
   (tier A/val still hard-assert < 5 GB). Cosmetic; left untouched to keep the diff
   minimal.
5. **NOTE (measurement semantics): G3's epoch-budget ratio is a lower bound** (the P0
   backward share is not staged-separable; the script says so). 0.74% vs 35% budget leaves
   ~47× headroom; even cold-cache stage share (3.9%) is ~9× under. The per-epoch sentinel
   in the runner telemetry (`rgb_encoder_secs_mean`, `stage_secs_mean`) will catch drift
   in the real run (spec §6: >35% over 2 epochs → recheck encoder batching).
6. **NOTE (behavior, correct): val pass at init shows prefix band 250 / candidate band 0**
   (both G3 scenes) — the untrained RPN prefix saturates the 250 cap so 250 − n_pref = 0
   slots remain; identical in kind to EXP-0029's d3 val pass-1. Not a defect.
7. **CONFIRMED (queue discipline): launch correctly blocked today.** The wrapper requires
   tag `exp/EXP-0030` on HEAD (untagged), `VERDICT: PASS` in this file, a CLEAN worktree
   (currently dirty with findings 1–3 fixes — the chief must commit them before launch;
   this audit commits nothing per orders), and `max_per_card < 2` GPU training processes
   (currently 2/card: EXP-0028-R2 + EXP-0029 → refuse). Per PI order nothing was launched.

## 6. Pre-launch checklist state

| Item | State |
|---|---|
| G1–G5 gates | all PASS (reports in ue_ubr_dual_rgb_gates_20261028/) |
| Port fidelity (SHA + semantics) | verified vs live remote, byte/blob-exact |
| Anchor cache binding + manifest | verified (3060 scenes, 8.4 GB, SHA in contract) |
| FW1-style isolation / optimizer grouping / G2 fixes | verified correct |
| Audit verdict (this file) | VERDICT: PASS |
| Chief commits the 3 mechanical fixes | PENDING (worktree dirty by design) |
| Tag exp/EXP-0030 on HEAD | PENDING (at launch) |
| Launch slot (R2 or EXP-0029 finishes) | NOT OPEN — launch forbidden now |
