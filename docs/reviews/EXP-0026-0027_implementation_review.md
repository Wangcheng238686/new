# EXP-0026 / EXP-0027 implementation review (independent launch audit)

- Audit date: 2026-10-21
- Repo / branch: `/home/wangcheng/project/new/.worktrees/dual-stream-dev` @ `codex/dual-stream-dev`
- Implementation commit under audit: `69b4ce4` (HEAD at audit time; worktree clean except this report)
- Auditor instruction: recompute everything; wrappers are generated artifacts — audited as-committed, generation script not trusted.
- Scope of file modifications by this audit: this report only.

---

## 1. Commit scope — PASS

`git show --stat 69b4ce4` touches exactly 4 files, all additions:

| File | Lines |
|---|---|
| `docs/EXP-0026-0027_preregistration.md` | +50 |
| `docs/EXP_REGISTRY.md` | +2 (EXP-0026, EXP-0027 rows) |
| `portable_sam2_explicit_coarse/scripts/ue/run_exp0026_ddp4.sh` | +76 |
| `portable_sam2_explicit_coarse/scripts/ue/run_exp0027_ddp4.sh` | +76 |

No other file touched; no trainer/config/data code changed. As scoped.

## 2. Wrapper diff audit — FAIL (one corrupt SHA pin; everything else within the allowed set)

Method: full `diff run_exp0024_ddp4.sh` vs each new wrapper, then read both wrappers end-to-end, then
programmatically re-verify every pin against the live v3 tree.

**Allowed-difference conformance.** Every hunk in both diffs falls in the allowed set: header comment
block; `DS_ROOT` default v1→v3; EXP-id strings in the `B0_EXP_ID` guard and echo/gate messages;
contract-echo prefix `EXP24_CONTRACT` → `EXP0026_CONTRACT`/`EXP0027_CONTRACT`; the 7 pinned SHA256s;
count assertions `[2142,459,459]/[126214,14980,44988]` → `[1931,285,413]/[123788,18066,24754]`;
dataset name in the contract echo; batch layout in the echo (`batch_per_rank=2 accum=1`) and the
trainer line (`--batch-size 2 --grad-accum-steps 1`); and in 0027 only `TRAIN_FLIP_PROB=0.5` +
`augmentation=flip:0.5`. Guard logic (clean-worktree, no-extra-args, `WORK_DIR` non-existence,
`B0_EXP_ID` match), SAM2_CKPT path, dataset-plumbing exports, fingerprint/baseline gates, and the
trainer invocation are structurally byte-identical apart from the lines above.

**SHA pin verification against the live v3 tree** (`/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v3_20261021`):

| File | Pin in wrappers | Actual sha256 | Verdict |
|---|---|---|---|
| dataset.json | `917c5fe4…44cc` | identical (64) | OK |
| splits/train.json | `b186c873…e66d` | identical (64) | OK |
| splits/val.json | `617b7429…8dde` | identical (64) | OK |
| splits/test.json | `9822e4a4…7964` | identical (64) | OK |
| annotations/train.json | `97630d70…a4bd` | identical (64) | OK |
| **annotations/val.json** | `d59ac33f400c1d91fb77deddbcec1104c7`**`8333`**`b45af890e7bc215199d8f2443de` — **65 chars** | `d59ac33f400c1d91fb77deddbcec1104c7`**`83`**`3b45af890e7bc215199d8f2443de` (64 chars) | **MISMATCH — duplicated `3` at index 37** |
| annotations/test.json | `2ea872f0…914a` | identical (64) | OK |

The val.json pin carries one extra `3` (`…c783` `3` `3b45…` vs actual `…c783` `3b45…`). Same typo in
**both** wrappers (same generation source). Empirical confirmation — preflight of the committed
wrapper as-is:

```
$ B0_EXP_ID=EXP-0026 B0_PREFLIGHT_ONLY=1 bash scripts/ue/run_exp0026_ddp4.sh
pin mismatch: annotations/val.json      # exit 2, line 45 — dataset.json/splits/train..test/annotations/train gates all passed before it
```

The gate fails closed against the *genuine* v3 file, i.e. neither arm can launch at all. Safe
direction, but a hard launch blocker (see Blockers).

**Identity gates (verified live, independently of the blocked preflight):**

- architecture fingerprint command from wrapper line 57 against `configs/ue_whu_a3sgm_512.py`
  returns `5305b3155914a2b2e7ecaf24b54957aee86fd6f47db93358daead2cbac5be36c` — equals the pinned sealed
  single-stream fingerprint. Gate intact (same config file as EXP-0024/25; `--config` unchanged).
- baseline-id command from wrapper line 61 returns `ue-whu-a3sgm-512-v1` — equals the pinned id. Gate intact.

**Cosmetic defects (non-blocking, both wrappers):**

1. `RUN_TAG` = `citysample_v1_512single_noaug_ddp4_s44_EXP-002{6,7}` — says `v1` though the dataset
   is v3, and says `noaug` for **both** arms though EXP-0027 is the default-aug arm. The v1-era
   convention and the registry call 0027 `defaug` (EXP-0025's dir is `…defaug_ddp4_s44_EXP-0025`).
   Directory names stay unique via the EXP-id suffix (no collision with EXP-0024's dir), but the 0027
   tag misidentifies the arm for every downstream artifact lookup keyed on run_tag.
2. `WORK_DIR` parent remains `/data2/wangcheng/checkpoint/citysample_v1/` — anticipated by the audit
   charter; it is merely a directory name; verified unique/non-colliding (EXP-0026/0027 dirs absent).
3. EXP-0027 line 27 stale comment: `# augmentation contract: … ALL pinned to 0` sits directly above
   `export TRAIN_FLIP_PROB=0.5`.

## 3. Batch-layout equivalence — PASS (with corrected arithmetic)

Trainer invocation lines, both eras:

- v1 (`run_exp0024/0025_ddp4.sh`): `torch.distributed.run --nproc_per_node=4` + `--batch-size 1 --grad-accum-steps 2` → effective batch 4×1×2 = **8**.
- v3 (`run_exp0026/0027_ddp4.sh`): `--nproc_per_node=4` + `--batch-size 2 --grad-accum-steps 1` → effective batch 4×2×1 = **8**.

The only trainer-line change in both diffs is the batch/accum line; all other hyperparameters are
byte-identical to run_exp0024: `--lr 5e-4 --weight-decay 0.05 --warmup-iters 100 --epochs 100
--seed 44 --amp 1 --ema-enabled 0 --ema-eval 0 --val-batch-size 1 --val-every-n-epochs 1
--compute-val-loss 0 --early-stopping-patience 0 --dataset whu --image-size 512 512
--config configs/ue_whu_a3sgm_512.py`, `SAVE_LAST_MODEL=1`, same train module, same
`--run-tag/--data-root/--checkpoint-dir` structure. No EMA, val every epoch, save-last, no early stop
— all confirmed.

**Optimizer-steps correction** (charter asked to "confirm steps/epoch doubles" — it does not; with
effective batch 8 unchanged, steps/epoch = N_train/8 in both layouts): v3 ≈ 1931/8 ≈ **242**
steps/epoch vs v1 2142/8 ≈ **268** — ~10% *fewer*, tracking the smaller v3 train set, not a doubling.
The equivalence claim that matters (effective batch 8, DDP gradient math, GroupNorm ⇒ no BN
statistics) holds and is correctly argued in the prereg.

## 4. Data sanity on v3 — PASS (loader-level; one leakage-relevant cross-era number to carry forward)

Using `/data2/wangcheng/envs/cvt2/bin/python`, cwd `portable_sam2_explicit_coarse`:

- `WHUCocoInstanceDataset(ann_file='annotations/val.json', image_subdir='')` → **285 samples**.
- Decoded sample [0]: img (3,512,512); gt_masks (19,512,512), **19/19 masks nonzero**; labels unique = `{0}`;
  bboxes sane (e.g. `[0,418,73,512]`, `[365,0,500,77]`). Decoded sample [142]: 166 instances, labels `{0}`.
- `dataset.json`: `label_revision = review99_v3_user_approved_20261021`; points 1931/285/413;
  instances 123788/18066/24754; `image_size [512,512]`. All match the wrapper assertions.
- v3 splits pairwise disjoint: train∩test = 0, train∩val = 0, val∩test = 0; union 2629 = 1931+285+413. No within-v3 split mixing.
- v1 ↔ v3 crosstab (point IDs):

  | v1 \ v3 | train | val | test |
  |---|---|---|---|
  | train (2142) | 701 | 285 | **413** |
  | val (459) | 459 | 0 | 0 |
  | test (459) | 459 | 0 | 0 |

  743 v1 points are absent from v3 entirely; 312 v3 points are new. **v1-train → v3-test overlap = 413
  points = 100% of v3-test (19.3% of v1-train); v3-val likewise comes exclusively from v1-train.**
  Within the EXP-0026/27 pair there is no leakage (fresh training, disjoint v3 splits). Cross-era
  hazard to record: any checkpoint trained on v1-train (e.g. EXP-0024/25 arms or anything derived)
  must never be evaluated on v3 val/test — that would be train-on-test contamination; and v1-pair vs
  v3-pair number comparisons are across disjoint data pools (prereg already demotes the v1 pair to
  historical reference, consistent with this).

## 5. Preregistration (`docs/EXP-0026-0027_preregistration.md`) — PASS

- Four elements inherited from EXP-0024-25: (1) primary metric = per-epoch COCO segm/bbox
  AP@[.50:.95] on the full 285-scene val, contrast EXP-0027−EXP-0026 at last, with AP50/AP75/size
  strata; (2) selection = last epoch 100; (3) scene-paired percentile bootstrap, 10,000 draws, seed
  **20261014**; (4) AUG_WINS/NULL/HARM decision rule with the new-era satellite baseline = better-last
  of this pair (v1 pair demoted; test unseal per registered cap250+maxDets250 protocol, commit `187c29e`).
- Drift-check clause present: post-training val(285)-vs-test(413, cap250) gap still >5 pts ⇒ problem
  is upstream of the split (back to data generation); ~1-2 pts ⇒ v3 becomes the new-era standard split.
- Batch-layout change registered as a protocol change, explicitly **not** a judged variable
  (effective batch unchanged, GroupNorm, DDP gradient equivalence; recorded in the contract line).
- Runtime fallback documented: GPU1 third-party task ~29G leaves ~19.5G — first-epoch OOM watch, and
  single-arm deferral as the preregistered degradation path that does not touch the judging criteria.

## 6. Launch readiness — PASS on directory state; blocked by item 2

- `WORK_DIR` parent: `/data2/wangcheng/checkpoint/citysample_v1/` (still says `v1` — cosmetic,
  verified to be a directory name only; the EXP-0026/0027 child dirs do not exist, so the
  `[[ -e "$WORK_DIR" ]]` gate passes; no collision with the existing EXP-0024 `noaug` / EXP-0025
  `defaug` dirs).
- GPUs (4× RTX 4090, 49140 MiB each): GPU0/2/3 3239 MiB used (small resident processes, PIDs
  3861064/66/67, 0% util except 70% blip on GPU2); **GPU1 29540 MiB used by third-party task
  (PID 3704048, 26296 MiB + 3216 MiB) ⇒ ~19.6G free** — matches the prereg's memory-constraint
  clause. Four-card co-tenancy as registered.
- Notwithstanding, no launch can proceed: the committed wrappers fail their own val.json pin gate
  (item 2).

---

## Verdict

**VERDICT: BLOCKED**

1. **Corrupt SHA pin for `annotations/val.json` in both wrappers** (`run_exp0026_ddp4.sh` and
   `run_exp0027_ddp4.sh`, line 45): pin is 65 chars — a duplicated `3` at index 37
   (`…c783[3]3b45af…`); the genuine v3 hash is
   `d59ac33f400c1d91fb77deddbcec1104c7833b45af890e7bc215199d8f2443de`. Verified empirically:
   `B0_PREFLIGHT_ONLY=1` run of the committed wrapper exits 2 with `pin mismatch: annotations/val.json`
   against the correct, otherwise fully pin-consistent v3 tree. Both arms are unlaunchable until the
   pin is corrected in a follow-up commit and re-audited (pin fix is one character; re-run preflight
   to green, nothing else in this audit changes).

Non-blocking notes for the fix commit (recommended, cosmetic): 0027 `RUN_TAG` says `noaug` though it
is the defaug arm (registry/v1-era convention: `defaug`; also `v1` in tag vs v3 data), and the line-27
comment above `TRAIN_FLIP_PROB=0.5` still reads "ALL pinned to 0". Carry-forward number: **413/413
(100%) of v3-test points come from v1-train** — never evaluate v1-train-trained checkpoints on v3
val/test.
