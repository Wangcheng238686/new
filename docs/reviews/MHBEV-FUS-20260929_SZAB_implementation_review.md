# MHBEV-FUS-20260929-SZAB — independent implementation review

**Verdict: PASS with concerns — the two-arm single-variable framework, the Appendix V.2 loss reweighting, the small-instance readout and every preregistered gate input are correctly implemented and were verified against the actual mmdet/rsprompter code paths; launch is not blocked, but the two missing Appendix V telemetry records (V.5.2 initial-state hash, V.5.1 per-epoch Σw) should be added before the formal A/B launch and the governance/script state committed as the governance rule requires.**

Reviewed on 2026-09-29 (static, read-only; no feature scan, detector inference or
training was run) against Appendix V of `EXPERIMENT_GOVERNANCE.md`,
`portable_sam2_explicit_coarse/scripts/ue/finetune_p2_inject_szab.py` (the
implementation), `portable_sam2_explicit_coarse/scripts/ue/finetune_p2_inject_ab.py`
(the declared superset baseline, working-tree version), and the read-only upstream
code (`rsprompter/models.py`, `rsprompter/models_sam2.py`, installed mmdet
`cross_entropy_loss.py` / `bbox_head.py` / `data_preprocessor.py`).
Worktree state at review: commit `df2972328ecd45461957140b06cf29874eab794f`, dirty
(script and Appendix V are uncommitted working-tree additions; governance requires
committing them before launch). `python -m py_compile` passes.

## 1. Arm isolation / single-variable key set — PASS (one documented inert deviation)

Full block diff of szab vs ab. Complete list of differences and classification
(szab line numbers):

| Diff | Lines | Category |
|---|---|---|
| Docstring rewrite | 1–16 | documentation |
| `--limit-train-batches` / `--limit-val-scenes` / `--mem-fraction` args, mem cap | 127–133, 137–139 | (c) smoke/mem switches, default 0.0 = no-op |
| Warm-fit early `break` once `n_fit>=80 and n_c>=20` | 240–243 | outside declared set — see CONCERN C4 (verified inert) |
| `sz_engagement` accumulator + training-only telemetry inside `extract_with_inject` | 263, 272–276 | (b) telemetry; `no_grad` + `detach`, does not touch the injected `delta` |
| Appendix V.2 reweighting: `SZWeightedMaskCE`, `sz_gt_area_w`, `sz_pos_roi_weights`, `mask_head.loss_mask`/`loss_and_target` wrap, `bbox_head.get_targets` wrap, stats | 283–362 | (a) two-arm common loss redefinition |
| `gi.sz_gt_w` + `model._sz_gt_w` side channel in `make_batch` | 397, 401–402 | supports (a) |
| `SMALL_AREA_PX`, `small_instance_f1` | 434–471 | (b) small-instance F1 readout |
| eval restructure: `n_val` cap, `kept_masks/kept_scores`, `sf1s`, `small_f1`, 3-tuple return, `imgIds=range(n_val)` | 476–479, 494, 504–510, 516–520, 524, 531 | (b)+(c); the split `pm` computation is numerically identical to ab's conditional expression (ab:354) |
| manifest additions (`experiment`, `arm`, `sz_reweight`, `boot_seed`, `small_area_px`, limits) | 536–541, 546–547 | (b) |
| per-epoch stats reset/accumulation, telemetry fields, `per_scene_smallf1_epoch{e}.json`, print line, `torch.save` metrics filter | 555, 559–564, 569–575, 583–585, 597–608, 610–619 | (b)/(c); the metrics filter change (`k != "per_scene_bf1"` → `not startswith("per_scene")`) is content-identical since `ev` never carries per-scene keys |

Arm-internal (A vs B) differences are only the injection mechanism itself:
injector/γ construction (205–212), warm-fit block (215–260), two optimizer groups
(365–376), `inject["enabled"]` flag and inject-only telemetry — exactly the
registered single variable. The reweighting hooks (283–362) are installed
unconditionally before the `if args.inject` branch, so both arms share them.
No other behavioral difference exists.

**C4 (concern):** the warm-fit `break` (240–243) is an efficiency change not in the
three declared categories. It was verified inert: (i) the calibration counter
`n_c` always reaches 20 before `n_fit` can reach 80, so `p2s`/`sb`/`X`/`T_rows`
are bit-identical to ab's; (ii) the loop consumes no global RNG — the satellite
and drone inner encoders are left in `eval()` by `_freeze_params`
(`rsprompter/models_sam2.py:508–511`) until the first `model.train()` (which
happens only at the epoch loop), the drone encoder is additionally pinned to eval
by `DualStreamRSPrompterAnchor.train()` (`rsprompter/models.py:265–268`), no
dropout/droppath exists in neck/geometry_bev, and `UEDualStreamDataset.__getitem__`
uses no global RNG — so the torch/numpy RNG streams at DataLoader creation are
identical with and without the break. Recommend recording this as an approved
inert deviation in Appendix V, or reverting it.

## 2. Mask loss weighting correctness — PASS

- **Equivalence with `mmdet.mask_cross_entropy`** (`cross_entropy_loss.py:149–199`):
  mmdet computes `pred[arange(N), label]` then an all-pixel BCE `mean` over all
  ROIs. The merged config resolves `rcnn.mask_size=(1024,1024)`, so every ROI's
  target is a uniform 1024×1024 canvas; per-ROI pixel counts are therefore equal
  and the per-ROI-mean weighted mean equals the plain mean when all w are equal.
  Verified numerically on CPU: uniform w reproduces `mask_cross_entropy` with
  rel. diff 7.4e-8 (float reduction order only); non-uniform w reproduces
  `Σ w_i L_i / Σ w_i` exactly (`SZWeightedMaskCE.forward`, szab:292–308;
  `denom = w.sum().clamp_min(1e-12)`).
- **Call-site compatibility:** `models.py:729–735` calls
  `self.mask_head.loss_and_target` entirely by kwargs
  (`mask_preds=, sampling_results=, batch_gt_instances=, rcnn_train_cfg=,
  final_mask_pos_priors=`); the wrapper (szab:327–334) matches and forwards to the
  captured original bound method. `loss_and_target`
  (`models_sam2.py:2812–2892`) calls `self.loss_mask(...)` only at lines
  2878/2880 — both the `class_agnostic` and standard branches route through the
  installed `SZWeightedMaskCE` (instance attribute shadows the module), so no
  branch bypasses the weighting. `quality_head` is disabled and
  `segm_score_mode='detector'`, `final_mask_loss_cfg` unset → `mode='standard'`
  (`models_sam2.py:1228` setdefault), `roi_sam` disabled — verified on the merged
  config. `roi_balanced_dice` (which would bypass `loss_mask`) is not active.
  `loss_mask` config is `CrossEntropyLoss(use_mask=True, loss_weight=1.0)`
  (root config line 190), so replacing the module drops no weight multiplier.
- **w alignment:** `pos_labels` row order == concat order of per-`res` positives ==
  `mask_targets` row order; w rows are `w_gt[res.pos_assigned_gt_inds]` concatenated
  in the same `sampling_results` order (`sz_pos_roi_weights`, szab:314–320), with
  `gt.sz_gt_w` built from the same valid-filtered mask list as `gi.labels`/`gi.masks`
  (szab:391–397). Verified: in the class-agnostic branch the head passes
  `torch.zeros_like(pos_labels)`; `SZWeightedMaskCE` uses it only for channel
  selection and takes w from `_sz_roi_w`, and the numeric probe confirms the
  zeros-label path is identical to the labeled path.
- **n==0 path** (szab:295–296) is defensive only: `mask_loss` early-returns before
  `loss_and_target` when there are no positive ROIs (`models.py:565–597`), and
  `loss_and_target` handles `mask_preds.size(0)==0` itself
  (`models_sam2.py:2856–2857`) without calling `loss_mask`. The path returns a
  grad-carrying tensor; unreachable in practice.
- **Timing:** `_sz_roi_w` is set inside the wrapper before the original call
  (szab:329–330) and every `loss_and_target` call goes through the wrapper
  (single call site, `models.py:729`).
- **Config premise correction (informational, C8):** the audit brief assumed
  `class_agnostic` unset → False. The root config explicitly sets
  `class_agnostic=True` for the mask head
  (`rsprompter_anchor_satS_v11_sam2_large_full.py:186`), confirmed in the merged
  config. This does not weaken the weighting (see above); recorded because the
  premise in the brief was inaccurate.
- **C3 (concern, telemetry only):** `sz_mask_ce._last_stats` persists across
  batches; on a batch where `mask_loss` early-returns (zero positives, e.g. an
  empty-GT scene that is still not `skipped` because its loss terms carry grad),
  the training loop (szab:569–575) re-accumulates the previous batch's stats,
  slightly biasing `sz_mask_weighted_over_plain` / `sz_mean_pos_roi_w`. The loss
  itself is unaffected. Clearing `_last_stats = None` at the top of each batch
  iteration would remove the bias.

## 3. bbox head weighting correctness — PASS (verified on the installed mmdet head)

- `sz_get_targets` (szab:340–357) preserves the four-tuple semantics of
  `BBoxHead.get_targets` and only scales weights; `loss_and_target` uses the
  targets for cascade-rcnn only, and `models.py:424` consumes only `loss_bbox`.
  Both the `bbox_train_fp32` branch (`models.py:424`) and the default
  `StandardRoIHead.bbox_loss` route through `self.bbox_head.loss_and_target` →
  `self.get_targets` (instance attribute), so the wrapper is always in force.
- mmdet `_get_targets_single` (`bbox_head.py:186–199`) orders per-image labels as
  positives-first, sets `label_weights = 1.0` for pos and neg
  (`pos_weight=-1` → 1.0), and `bbox_weights` = 1 for pos / 0 for neg. The
  wrapper's `pos_mask=(labels>=0)&(labels<num_classes)` therefore selects exactly
  the positive entries in `sampling_results` order, matching
  `cat(w_gt[res.pos_assigned_gt_inds])`. Verified on a real
  `Shared2FCBBoxHead(num_classes=1)`: pos `label_weights` == w of assigned GT,
  neg `label_weights` == 1.0 untouched, pos `bbox_weights` == w, neg == 0,
  assertion `int(pos_mask.sum()) == w_roi.numel()` holds and raises on mismatch
  (szab:350–352).
- **avg_factor untouched:** cls `avg_factor = max(Σ(label_weights>0),1)`
  (`bbox_head.py:376–377`) — since `w_k>0` always (no truncation), the count
  remains the total ROI count exactly as in the unweighted baseline (verified:
  256/256). bbox `avg_factor = bbox_targets.size(0)` (`bbox_head.py:418`) is
  untouched. This matches Appendix V.2.2 verbatim.
- **Broadcast:** `scale[:, None]` spans the 4 box coordinates (szab:356);
  negatives' zero regression weights stay zero (scale=1 there).

## 4. Weight definition — PASS

- `sz_gt_area_w` (szab:310–312): `1/sqrt(max(pixel_area,1))` in float64 → float32,
  no upper/lower truncation — exactly V.2's `w_k`. Areas are per-GT pixel counts
  of the 1024²-canvas masks after the same `valid` filter applied to
  labels/bboxes/masks (szab:391–395), so index alignment with
  `pos_assigned_gt_inds` (0-based into the filtered GT list) holds.
- Three-source consistency: `gi.sz_gt_w` (szab:397), `model._sz_gt_w` (szab:401–402,
  same tensor, batch_size=1 always), and the mask hook's `gt.sz_gt_w` read from
  `batch_gt_instances` are the same values; the `DetDataPreprocessor` path is
  transparent to the extra field (`samplelist_boxtype2tensor` touches only
  `BaseBoxes` bboxes; `pad_gt_masks` touches only `masks`; `BatchFixedSizePad`
  early-returns for already-1024² inputs — verified in the installed mmdet).
- Edge cases: empty-GT batch → zero-length w; both consumers no-op with 0
  positives (mask path early-returns upstream; bbox path asserts 0==0). A
  zero-pixel GT mask gets `w_k = 1` — consistent with `max(a_k,1)`.

## 5. Pre-registration conformance — PASS with concerns (C1)

- **Primary metric:** last-epoch 259-val `segm AP_s` — `history.json` carries
  `AP_s` per epoch (COCOeval `stats[3]`, area<32², standard `model.predict`
  inference identical to ab), `epochs` default 6.
- **Companion:** AP/AP50/AP75/AP_s/AP_m/AP_l + boundary-F1 per epoch in
  `history.json`; `per_scene_bf1_epoch{e}.json` and
  `per_scene_smallf1_epoch{e}.json` per scene. ORIG-AB-sz e4 context is external,
  as registered.
- **Significance:** the paired bootstrap (10k draws, seed 20260936) is an offline
  analysis; the artifacts fully support it — `per_scene_smallf1_epoch{e}.json`
  records `scene=i, f1, n_gt_small, n_pred_small, matched` for both arms, and the
  val loop is deterministic (`range(n_val)`, no shuffle). `boot_seed=20260936` is
  archived in the manifest (szab:540).
- **`small_instance_f1` criteria** (szab:436–471) match V.4 exactly: both sides
  `area < 1024 px²` strict (`32*32`), predictions additionally `score>=0.5`
  (caller filter at szab:504), greedy one-to-one matching in descending score
  order with `IoU >= 0.5`, P/R over small-pred/small-GT counts, both-empty → 1,
  one-side-empty → 0.
- **Gate SZAB_INJECT_PASS:** (i) AP_s from both arms' `history.json`; (ii) AP and
  boundary-F1 from the same; (iii) `inject_gamma` (szab:603–604) and
  `inject_engagement` (szab:605–608) in B's history; (iv) both manifests carry
  `git_commit`, `single_ckpt_sha256`, `dual_ckpt_sha256`, and the seed triple
  (env 20260928 / subset 44 / fit 20260925) — all comparable across arms.
  All four gates are evaluable from the produced artifacts.
- **C1 (concern, add before launch):** Appendix V.5.2 requires "两臂初始模型
  state（除 B 臂 injector/γ）hash 相等" to be **written into the manifest**; the
  manifest (szab:534–547) has no such field. The equality itself is provable from
  this audit (identical build+load code path before the `if args.inject` branch,
  same seeds, identical ckpt SHA256s recorded), but the preregistered record
  cannot be produced truthfully after launch. A few additive lines (hash of the
  initial `state_dict` keys/values excluding `injector*`) close this.
- The lineage/training keys match V.1 (config, dual r1 e85 + single e86 overlay
  with `dropped==[]` assert, frozen `drone_encoder./geometry_bev.`, warm fit
  20+80/1500 cells/FIT_SEED 20260925, γ init 0.25, 712/259 splits, 6 epochs,
  AdamW 1e-4 / inj 1e-3 / wd 0.01 / accum 8 / warmup 40 / cosine floor 0.1 /
  AMP / seed 20260928, empty-GT skip logic unchanged).
- Governance state: Appendix V and the script are uncommitted at review time;
  commit them together with this review before launch, per the governance rule.

## 6. Telemetry triad — PARTIAL (otherwise complete, no over-claim)

- γ per epoch (B only): `inject_gamma` (szab:603–604) — supports gate (iii).
- Engagement: accumulated inside `extract_with_inject` only when `model.training`
  (szab:272–276); eval forwards and warm-fit forwards do not contribute.
  **C6 (note):** the exported quantity is mean-of-squares ratio
  `mean|Δ|²/mean|P2|²` rather than a mean of per-forward `|Δ|/|P2|` ratios; the
  registered gate (`> 0`) is unaffected.
- Weighted/plain mask-loss ratio `sz_mask_weighted_over_plain` per epoch
  (szab:597–599) — the V.5.1 加权/未加权比. **C2 (concern):** V.5.1's "逐 epoch
  Σw" is exported only as the mean (`sz_mean_pos_roi_w = Σw/n_roi`,
  szab:600–601); Σw and n_roi are accumulated (szab:359–362) but not written to
  `history.json`.
- Warm-fit R²: printed and archived (`manifest.warm_fit_r2`, szab:544, 260).
- Manifest (szab:534–547): git commit/dirty, experiment/arm, ckpt SHA256s, seed
  triple, `sz_reweight` scope (matches V.2's weighted/unweighted partition
  exactly), `boot_seed`, `small_area_px`, smoke limits. No field over-claims a
  gain; conclusions are deferred to the V.4 gates, as required by V.5.1/V.5.3.
- Missing: the V.5.2 initial-state hash (see C1). With C1/C2 addressed the triad
  is complete.

## 7. Numerical fidelity / baseline regression — PASS

- Injection math in `extract_with_inject` (szab:266–280) is line-identical to ab
  (ab:248–257): `gamma * injector(bev.float())`, bilinear resize to `x[0]`,
  `(x[0].float() + delta).to(x[0].dtype)`; the inserted telemetry runs under
  `no_grad` on detached copies and cannot alter gradients or the injected value.
- Training loop: cosine/warmup/accum/skip logic is identical; at defaults
  `n_batches = len(train_loader)` (712), so the optimizer-step condition and
  `total_steps = epochs * ceil(712/8) = 504` are exactly ab's (ab:404–406 vs
  szab:583–585). `skipped` semantics unchanged.
- Warm-fit early break: verified inert (item 1, C4) — collected fit cells,
  calibration statistics and the RNG stream at training start are identical to
  ab's; the B-arm fitted conv is bit-identical to what the frozen ab code would
  produce.
- `eval_epoch` changes are additive: `coco_dt` and `bf1s` are produced by the
  same statements as ab; the restructured `pm` branch is logically identical;
  COCOeval settings (`imgIds=range(259)`, default maxDets) unchanged.

## 8. Fail-fast — PASS with concern (C5)

- Mask-hook weight/ROI mismatch raises `RuntimeError` (szab:301–302); bbox-hook
  pos/weight mismatch raises (szab:350–352); the single-ckpt overlay asserts
  `dropped == []` (szab:156). Silent misalignment is not reachable through the
  audited call graph (single call sites for both wrapped methods).
- **C5 (concern):** nothing hard-prevents a `--limit-*` smoke run from writing
  into the official `MHBEV-FUS-20260929-SZAB-A/B` directories. Mitigations
  verified: the manifest records `limit_train_batches`/`limit_val_scenes`, and
  `gt_val259.json` is always built over the full `len(val_ds)` scenes
  (szab:410–432), so a truncated-GT contamination of a later formal run is not
  possible; other artifacts are overwritten by the formal run. A guard refusing
  `limit_* > 0` when the out-dir name matches the registered experiment IDs (or
  an `--smoke` subdirectory requirement) would make this airtight.

## Concern summary (all non-blocking; C1/C2 recommended before launch)

- **C1:** Appendix V.5.2 initial-model-state hash (both arms, excl. injector/γ)
  missing from the manifest — preregistered record, not producible post-hoc.
- **C2:** V.5.1 "逐 epoch Σw" exported only as mean w; Σw/n_roi not archived.
- **C3:** stale `_last_stats` re-accumulation on zero-positive batches biases the
  weighted/plain telemetry ratio (loss unaffected).
- **C4:** warm-fit early break lies outside the declared difference categories;
  verified bit-inert (fit math and RNG streams unchanged); record or revert.
- **C5:** no hard smoke/official output-directory guard (mitigated, see item 8).
- **C6:** engagement is a ratio of epoch-mean squared norms, not a mean of
  per-forward norm ratios (gate `>0` unaffected).
- **C7 (trivial):** the epoch print references `ev['AP_s']`, defined only when
  predictions exist; unreachable in practice (inherited pattern from ab).
- **C8 (informational):** the brief's premise "class_agnostic 未设→False" is
  wrong for the final config (`class_agnostic=True` at root config line 186);
  weighting is unaffected because both loss branches route through the installed
  module.

## Launch authorization

Conditional on the governance rule being satisfied — commit Appendix V, the script,
and this review — the run may be launched as preregistered. C1/C2 should be added
first (purely additive manifest/history fields; if added, no re-review beyond a
diff check of those lines is needed). C3–C8 are recorded and do not alter any gate.
This review authorizes only the two preregistered SZAB arms (`--inject 0/1`,
default limit flags, official output directories) — no detector, data, or
hot-start modification.
