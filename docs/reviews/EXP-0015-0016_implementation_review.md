# EXP-0015 / EXP-0016 implementation review (independent audit, pre-launch)

Audit target: HEAD = `cf00f7a` "feat(exp): EXP-0015/0016 in-route arms - fusion wd
exemption + L0-LoRA / image_embeddings injection site" (branch `codex/dual-stream-dev`,
worktree `/home/wangcheng/project/new/.worktrees/dual-stream-dev`, audit date 2026-10-16).
All checklist claims were recomputed independently (python `/data2/wangcheng/envs/cvt2/bin/python`,
`SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt`, cwd
`portable_sam2_explicit_coarse/`); no tracked file was modified except this report.

## 1. Commit scope — OK

`git show --stat cf00f7a` lists exactly the 11 expected files and nothing else:

```
docs/EXP-0015-0016_preregistration.md              |  94 +++++
docs/EXP_REGISTRY.md                               |   2 +
configs/ue_whu_a3sgm_512_dualembed.py              | 262 +++++++++++
configs/ue_whu_a3sgm_512_dualv2_l0_lora.py         | 259 ++++++++++
scripts/ue/run_exp0015_ddp4.sh                     |  67 +++++
scripts/ue/run_exp0016_ddp4.sh                     |  67 +++++
train/cli_args.py                                  |   2 +
train/optimizer.py                                 |  25 +
train/train_rsprompter_fusion.py                   |   1 +
ue/contract.py                                     |   3 +
ue/dual.py                                         |  36 ++-
```

(814 insertions, 4 deletions; the four deletions are the rewritten docstring/parsing lines
inside `ue/dual.py`.) No other file touched.

## 2. No-regression on existing arms — OK (diff argument + empirical check)

### 2a. `ue/dual.py` (`git diff cf00f7a^ cf00f7a -- .../ue/dual.py`)

Four hunks:

1. **Signature**: adds `embed_fusion=False, embed_channels=256` kwargs — new keys, absent
   from every pre-existing config (verified by grep over `configs/`), so all existing builds
   take the defaults.
2. **Parsing** (the only behavioral edit reachable by old configs):
   old `levels = [int(t) for t in (target_levels or [target_level])]` →
   new `levels = [...] if target_levels is not None else [int(target_level)]`.
   Case analysis over every registered config (grep of `target_level` in `configs/`):
   `dual`/`dualnorm` pass `target_level=1` (`target_levels` absent → `None`); `dualv2`/
   `dualv2_lora` pass `[0,1]`; `dualv2_l0`/`dualv2_l0_lora` pass `[0]`; `dualembed` (new)
   passes `[]`. For `None` both expressions yield `[target_level]`; for non-empty lists
   `[int(t) for t in L]` is identical with and without the `or` short-circuit. Only the
   empty list changes meaning (`[] or [target_level]` used to fall back; now stays `[]`),
   and `[]` is passed by the new EXP-0016 config alone. Downstream, `if len==1 / elif
   len>1 / (implicit else: nothing)` preserves the old two-branch behavior for len 1/2;
   len 0 creates no `proj`/`gate` and is rejected unless `embed_fusion` (mutual-exclusion
   `ValueError` if both set).
3. **`extract_feat`**: `image_embeddings = out[1]` is a pure alias read; the `embed_fusion`
   block (proj_e/gate_e path + `_E` telemetry) is inside `if self.embed_fusion:` and only
   appends telemetry when `self.training`. Return changed from `tuple(x), *out[1:]` to
   `tuple(x), image_embeddings, *out[2:]` — with `embed_fusion=False` the local is the
   unmodified `out[1]` object, so the returned tuple is element-for-element the same
   references (bit-identical).
4. No other change in the file.

### 2b. `train/optimizer.py` (`git diff cf00f7a^ cf00f7a`)

The loop adds exactly one guarded branch, placed **first** (immediately after
`match_name` construction, before `roi_head.mask_head.decoder_tail_refiner` and all other
buckets):

```python
if fusion_no_wd and match_name.split(".")[0] in FUSION_ROOTS:   # ("gate","proj","gate_e","proj_e")
    params_fusion.append(param); names_fusion.append(name); continue
```

With `fusion_no_wd=False` the condition is False for every parameter, `params_fusion`
stays empty, the `if params_fusion:` group-append block is skipped, and every parameter
falls through to exactly the parent bucket sequence — flag-off grouping is unchanged by
construction. The new group, when present, is prepended (before `backbone` etc.) and sets
`weight_decay=0.0`, `lr=lr*other_lr_mult` (the same LR the params would have had in
`other`; no LR change — the single preregistered lever).

**Token-collision check (empirical)**: `named_parameters()` top-level tokens of the built
dual model are `['backbone', 'gate', 'neck', 'proj', 'roi_head', 'rpn_head', 'uav_encoder']`
(plus `gate_e`, `proj_e` for 0016; `data_preprocessor`, `height_edges`, `telemetry` carry no
parameters). Exact first-token matching against `FUSION_ROOTS` hits only the intended
modules; notably `neck` and all `uav_encoder.*` nested names (including LoRA's
`...patch_embed.proj.lora_A...`, first token `uav_encoder`) cannot collide.

**Empirical no-regression (beyond the diff argument)**: built all five pre-existing dual
models (`dual`, `dualv2`, `dualv2_l0`, `dualnorm`, `dualv2_lora`) from a temp worktree at
`cf00f7a^` and from HEAD, both at `torch.manual_seed(44)`:

```
ue_whu_a3sgm_512_dual        state_dict bit-identical: True | grouping identical: True | requires_grad identical: True
ue_whu_a3sgm_512_dualv2      state_dict bit-identical: True | grouping identical: True | requires_grad identical: True
ue_whu_a3sgm_512_dualv2_l0   state_dict bit-identical: True | grouping identical: True | requires_grad identical: True
ue_whu_a3sgm_512_dualnorm    state_dict bit-identical: True | grouping identical: True | requires_grad identical: True
ue_whu_a3sgm_512_dualv2_lora state_dict bit-identical: True | grouping identical: True | requires_grad identical: True
```

(grouping = flag-off `_build_optimizer` name→group map; identical to the parent code's
grouping including group membership and order.) `cli_args.py` adds only the
`--fusion-no-wd` flag (default 0); `train_rsprompter_fusion.py` adds only its pass-through.

## 3. Config single-variable + fingerprints — OK

- `git diff cf00f7a^:.../ue_whu_a3sgm_512_dualv2_l0.py cf00f7a:.../ue_whu_a3sgm_512_dualv2_l0_lora.py`:
  exactly the docstring and one token — `'use_gradient_checkpointing': False}` →
  `'use_gradient_checkpointing': False, 'use_lora': True}`.
- `git diff cf00f7a^:.../ue_whu_a3sgm_512_dual.py cf00f7a:.../ue_whu_a3sgm_512_dualembed.py`:
  exactly the docstring and one block — `'target_level': 1` → `'target_levels': [],
  'embed_fusion': True, 'embed_channels': 256`. (wd exemption is deliberately trainer-side,
  not in-config; single-variable is at the config level and the wrapper level separately.)

Fingerprints recomputed via `ue.contract` (`architecture_fingerprint(whu_model_config(id))`,
`SAM2_CKPT` set): all 8 registered ids resolve to themselves; the 6 pre-existing
fingerprints unchanged; the two new ones match the pinned values and are unique in the
registry:

```
0015 ue-whu-a3sgm-512-dualv2-l0-lora-v1  5c7e1afc5877496b6eda4efde652ac786b8f25b7f4e8dc23da61764c6cd8e3bc  UNIQUE
0016 ue-whu-a3sgm-512-dualembed-v1       12a63439a1471832abd414d8347869317bc2ed4e172db220cd40eac87411dee8  UNIQUE
```

Audit note (no action): `MODELS.build(cfg.model)` mutates nested `cfg.model` (mmengine
injects a key deep inside `roi_head`), so `baseline_id_for` must run before build — which
is exactly what the wrappers (fresh subprocess, no build) and the trainer
(`assert_whu_model(cfg.model.to_dict())` at line 183 before `MODELS.build` at 188) do.
A harness that computed the id after build observed `None`; production order is safe.

## 4. Wrappers — OK (two cosmetic notes)

`git diff cf00f7a^:run_exp0013_ddp4.sh cf00f7a:run_exp001{5,6}_ddp4.sh`: differences are
exactly header comments, `B0_EXP_ID` guard id, `RUN_TAG`, config path, pinned fingerprint,
baseline id, the `levels=` CONTRACT-echo annotation (`P2-only-L0+uav_lora=1` /
`embed-only-E16`), `EXP13_` → `EXP001{5,6}_` echo prefixes, `+uav_lora=1`/`fusion_no_wd=1`
echo fields, and the added `--fusion-no-wd 1` in the trainer invocation. Trainer protocol
args are otherwise identical across all three wrappers (normalized diff): seed 44, 100
epochs, batch 1, accum 2, lr 5e-4, wd 0.05, warmup 100, amp 1, no EMA, no early stop,
`SAVE_LAST_MODEL=1`.

Fail-closed guards intact in both: `B0_EXP_ID` match, no-args rule, clean-tree,
`WORK_DIR` absence, accepted-pairs manifest hash `f15cd8f5…`, split hash `17f49430…`,
snapshot PYV `source_payload_fingerprint_sha256` check, arch fingerprint equality,
baseline-id equality. Neither target work dir exists:

```
/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualv2l0_sealed_ddp4_s44_EXP-0015  -> No such file
/data2/wangcheng/checkpoint/surfbev_b0/surfacebev_v2_512dualv2l0_sealed_ddp4_s44_EXP-0016  -> No such file
```

Cosmetic, non-blocking: (i) EXP-0016's header comment block retains the 0013 wording
("L0-only fusion … the only variable vs EXP-0011 is target_levels [0,1] -> [0]"); only the
identity line was updated. (ii) EXP-0016's `RUN_TAG` stem is `512dualv2l0_…` (0013
leftover) rather than `dualembed`. Both are misleading-to-humans only; the executed config
path, fingerprint and baseline-id guards pin the actual arm.

## 5. Prereg adequacy — OK (`docs/EXP-0015-0016_preregistration.md`)

- Four elements present and inherited: (1) primary metric val segm/bbox AP@[.50:.95] vs
  EXP-0009 + AP50/75, UAV-off ablation, full gate telemetry (0015 unsuffixed / 0016 `_E`);
  (2) selection = last epoch 100, protocol enumerated item-by-item (v2 snapshot, seed 44,
  DDP4 batch1×accum2, lr 5e-4, wd 0.05 with fusion group 0, no aug, no early stop, test
  sealed); (3) scene-paired percentile bootstrap, 10,000 draws, seed 20261014; (4) the
  four verdict gates with the v2.3 effective-opening operator (Σ|Δ|/Σ|proj|), 0016's
  operator at the embedding site (`open_E`).
- Composition-arm structure stated honestly for both: 0015 "vs 0013 = +LoRA +wd豁免; vs
  0012 = +wd豁免 −L1"; 0016 "vs 0010 = 注入点 + wd豁免"; main A/B vs EXP-0009 in both
  cases, with attribution dependence on the existing ladder explicitly acknowledged.
- Risks registered: prompt-encoder alignment (frozen PE priors vs perturbed embedding) and
  32×32 coarseness, each with the follow-up decision rule (high_res ablation before any
  route verdict). Interpretation clauses pre-stated for both arms, incl. the inert-branch
  rule (opening < 0.20 + non-significant on−off).
- Smoke-results section matches what a smoke can actually assert: gate-grad at init > 0
  (anti-saddle), LoRA end-to-end grads only after gate opened (0→0.01), step-0 equivalence
  claimed only in eval mode (LoRA dropout 0.05 boundary re-stated), perturbation touching
  only `image_embeddings` with pyramid levels bit-unchanged, fusion group wd=0 + flag-off
  legacy grouping.
- Transparent future-work disclosure: the ladder tooling will need a `_E`/empty-target
  delta_operator branch, registered as a conclusion-phase change.

## 6. Independent numeric spot-checks — ALL PASSED (GPU 3, real frozen val scene)

(a) **EXP-0016 forward** (built from `configs/ue_whu_a3sgm_512_dualembed.py` on cuda;
    `UERawPairDataset(root=/data2/wangcheng/dataset_frozen/ue_raw_hero_uav_pairset_20261014_frozen,
    'val', load_uav=True, split_manifest='surfbev_b0_split_v2.json', image_size=(512,512))`,
    scene `Coord_002670_pt_gx_-7_gy_-3_s_005_…`): `target_levels == []`, `embed_fusion`
    with `proj_e`/`gate_e` present and no `proj`/`gate`; train-mode `mode='loss'` forward +
    backward produced telemetry whose non-empty keys are **exactly**
    `{gate_mean_E, gate_std_E, gate_absmax_E, delta_p3_ratio_E, bev_coverage_E}`;
    `bev_coverage_E = 0.9707 > 0`; `gate_e.grad.norm() = 6.3e+02 > 0`.

(b) **EXP-0015 build** (`configs/ue_whu_a3sgm_512_dualv2_l0_lora.py`):
    `target_levels == [0]`; uav_encoder trainable parameters = **1,492,528** exactly
    (LoRA adapters 1,061,424 + `modules_to_save=['neck']` 431,104; encoder self-audit
    `unexpected_trainable=0`; backbone base fully frozen) — same recipe/number as the
    EXP-0012 registration.

(c) **Optimizer grouping** (`_build_optimizer(..., lr=5e-4, weight_decay=0.05)`, trainer
    default mults): with `fusion_no_wd=True` the `fusion` group contains exactly
    `{gate, proj.weight, proj.bias}` for 0015 and `{gate_e, proj_e.weight, proj_e.bias}`
    for 0016, with `weight_decay == 0.0` and `lr == 5e-4 × other_lr_mult(1.0)`; with
    `fusion_no_wd=False` no `fusion` group exists and those same names land in `other`
    (full flag-off membership verified = every requires-grad param in exactly one group;
    see also the bit-identical parent-vs-HEAD grouping comparison in §2b).

## Verdict

All six checklist items verified; findings are limited to two cosmetic wrapper-naming
notes (stale EXP-0016 header comment and `dualv2l0` RUN_TAG stem), neither of which
affects guards, identity pinning, protocol or the single-variable structure.

VERDICT: PASS
