
## OPS-001 | 2026-10-26 19:57 CST | Baseline sweep + matched-epoch fusion-gate snapshot (read-only)

**Agent role**: OPS (monitor/collect/package). This entry = first monitoring baseline; fusion-gate verdict belongs to main agent.

### 1. Remote health (lthpc, checked 19:54 CST)
- GPUs 0-3 all 94-100% util, 14.4-20.1/24 GB. All compute procs under `/data/wangcheng/envs/cvt2/bin/python`.
- compact_joint owner PID 269383 alive (elapsed 1d20h). Arm trainers: rgbd=269430(GPU3), constant_rgb=269431(GPU0), reference_plane=269432(GPU1). Satellite train+test both `done` per execution_state.json (stage=running overall; pending: rgbd/constant_rgb/reference_plane test). Finish owner v2 PID 291123 alive (1d19h).
- candidate_detail recovery owner PID 969804 alive (3h55m) + FROC-ties watcher PID 970132; 2 concurrent DDP groups (4 GPU each): query+constant and detail+constant, started ~18:43, both `--mode matched --split val` train jobs.

### 2. Per-arm progress + ETA (compact_joint, 100ep target)
| arm | completed epoch (file) | in-progress | mean sec/ep (last10) | ETA train end |
|---|---|---|---|---|
| satellite | 100 (DONE, test done) | - | 1846 | done |
| rgbd | 71 | ep72 | 2793 | ~2026-10-27 18:30 (+22.5h) |
| constant_rgb | 71 | ep72 | 2736 | ~2026-10-27 17:50 (+22.0h) |
| reference_plane | 70 | ep71 | 2794 | ~2026-10-27 19:10 (+23.3h) |
Dual arms ~1.5x satellite per-epoch cost (2790 vs 1846 s). Tests run after per-arm completion.

### 3. Matched-epoch segm/mAP | bbox/mAP (training-time val, NOT final gate)
| epoch | satellite | rgbd | constant_rgb | reference_plane | rgbd−const segm | rgbd−sat segm |
|---|---|---|---|---|---|---|
| 1 | .6767/.6753 | .6647/.7046 | .6849/.6903 | .6547/.6866 | -0.0202 | -0.0120 |
| 10 | .8038/.7798 | .8147/.7824 | .8191/.8000 | .7973/.7801 | -0.0044 | +0.0109 |
| 30 | .8259/.8076 | .8516/.8252 | .8393/.8141 | .8300/.8109 | +0.0123 | +0.0257 |
| 50 | .8477/.8244 | .8663/.8409 | .8650/.8300 | .8487/.8259 | +0.0013 | +0.0186 |
| 70 | .8571/.8371 | .8667/.8431 | .8680/.8392 | .8548/.8310 | -0.0013 | +0.0096 |
| 71 (latest common) | .8590/.8364 | .8657/.8397 | .8635/.8378 | (ep70) | +0.0022 | +0.0067 |
- satellite ep100 final: segm .8607 / bbox .8365; satellite best=ep62 score .8620. rgbd−const bbox deltas: ep30 +0.0111, ep50 +0.0109, ep70 +0.0039, ep71 +0.0019.
- EARLY READING (non-final, no CI, selection policy not applied): (a) both dual arms > satellite at every matched epoch from ep10 on, dual arms reach ~satellite-final level ~30ep earlier; (b) rgbd vs constant_rgb: rgbd ahead mid-training (ep30 segm +0.0123, bbox +0.011), gap compresses to ±0.002-0.004 by ep50-71 — NOT yet a clear real-vs-constant separation at current epochs; needs full 100ep + paired bootstrap per prereg.

### 4. Candidate_detail DDP matrix (10 train arms, 10ep each, 4-GPU DDP, ~951 s/ep)
- Complete: query/correct, detail/correct, both/correct, both/constant (query_correct & detail_correct via verified-receipt recovery).
- Running (ep5/10, ~951s/ep, ETA ~21:30 tonight): query/constant, detail/constant.
- Pending: both/plane, baseline/correct, baseline/constant, satellite/correct trains; then 12 val_matched + 12 test_matched evals + analysis.
- Est. all trains done ~02:45-03:00 Oct 27 (2 concurrent groups), evals after.
- Latest val (segm/mAP, ep10 unless noted): query/correct .8112, detail/correct .8164, both/correct .8067, both/constant .8063; running ep4: query/constant .7825, detail/constant .7857.

### 5. Score-band telemetry (validation_current/records.json, n=459 imgs/arm)
- original_path = final pred dict; length proxy = mask_count. All arms max mask_count=250 (cap hit on every arm); mean 129-166 (compact_joint: sat 142.0, rgbd 138.3, const_rgb 147.8, ref_plane 145.9; cand_detail 131.2-166.0). Prefix-slot pressure present on all arms equally.
- geometry_accepted (count/img): satellite=0 all imgs (sanity: no UAV geometry input). rgbd 111.7 > ref_plane 102.3 ≈ const_rgb 102.2. Real content accepts ~9 more geometries/img than matched controls — mechanism telemetry only, not an outcome.

### 6. Local machine
- 4x RTX 4090. GPUs 0-2 hold 18.9GB each, ALL owned by user liubowen (VideoMME qwen3vl runs) — none of ours. GPU3 idle. No wangcheng training procs. Clean.

### Open questions for main agent
1. rgbd−constant_rgb matched-epoch deltas at ep50-71 (±0.002-0.004) are small vs ep30 peak (+0.012) — compression is either convergence or real-vs-constant genuinely marginal; final ep100 + paired CI needed before any claim.
2. Mask_count hitting 250 cap on all arms incl. satellite: confirm whether AR@250 cap clipping affects cross-arm comparability reading (likely protocol-consistent, but worth noting in any fusion table footnote).
3. candidate_detail correct-vs-constant at ep10 (both/both: .8067 vs .8063 segm) shows near-zero delta — same watch item as #1.

**Artifacts**: remote dirs read only; no files created on remote. Local: this log entry only.

## OPS-002 | 2026-10-28 02:10 CST | Workspace map pass (docs-only, chief-authorized)

**Agent role**: docs organization (this session, authorized by chief). Scope: index documents only — zero file moves/renames/deletes, zero git commits (left for chief).

### 1. Deliverables written
- `docs/WORKSPACE-MAP.md` — master map: active assets (3 arms + code + contract lineage + authoritative docs), archived-verdict ladder (probe phase / interface era / old fleets / EXP-0001..0027), data-disk artifact map (`/data2/wangcheng/remote-lthpc/checkpoint/`, caches, `citysample_v1`, `surfbev_b0`), and launch runbook with wrapper commands.
- `README.md` (worktree root) — rewritten as short entry point; the previous WHU-SurfaceBEV-migration README content is preserved as the "历史" section (nothing discarded).
- This log entry.

### 2. Running-jobs evidence check (not guessed)
- EXP-0028-R2: owner PID 2972162 alive, cwd = worktree `portable_sam2_explicit_coarse/`, ep 21/100 (log 01:53), CUDA_VISIBLE_DEVICES=0,1,2,3, SAM2_CKPT pinned. Imports: `scripts/probes/ue_ubr_reseal_20261027.py` -> `rsprompter/uav/{ubr,ubr_train_integration}.py`, host `train/train_rsprompter_fusion.py`, config `ue_ubr_single_512.py`.
- EXP-0029 arm-2: owner PID 3006086 alive, same cwd/env, ep 7/100, runner `ue_ubr_reseal_dual_20261027.py` -> `ubr_dual_integration.py` (extends, never forks).
- EXP-0030: not launched (PI freeze). NOTE: tag `exp/EXP-0030` does not exist at HEAD (ae26918) — the wrapper will fail-closed until tagged; flagged in map §四.

### 3. Flagged ambiguities (registered in map, not resolved here)
- `WARM-RR-V2_preregistration.md` has no matching conclusion/EXP-ID (arm launched or not, unknown).
- Remote verdict doc `UE_UAV_EFFECTIVE_DIRECTION_VERDICT_20261024.md` referenced by 3 docs but not present in this worktree (original lives on the remote line).
- `run_ubr_reseal_ddp4.sh` asserts audit `UBR-RESEAL-20261027_audit.md` while an `UBR-RESEAL-R2_audit.md` also exists (matches actual running process argv; registered as-is).
- EXP-0025 last-anchor ep98-100 supplementation note vs prereg's `last_model_epoch100.pth` — chief decision at level-3 closeout.

**Artifacts**: 3 files written above; no other path touched. Worktree was clean before this pass (git status 0 lines); these 3 files are the only changes, uncommitted per order.
