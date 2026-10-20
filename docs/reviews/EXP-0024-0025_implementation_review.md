# EXP-0024 / EXP-0025 — independent implementation review (launch audit)

Audited 2026-10-20 against HEAD = `4a653d6` (`4a653d670a4d5c103d2a58412fe109f82f4c8a42`,
branch `codex/dual-stream-dev`, worktree clean at audit start) and the archive
`/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021`. Everything below was
recomputed by the auditor (git, sha256sum, and live python probes with
`/data2/wangcheng/envs/cvt2/bin/python` from `portable_sam2_explicit_coarse/`); nothing was
taken from the commit message or the prereg on trust.

## 1. Commit scope — PASS

`git show --stat 4a653d6` touches exactly 5 files, 221 insertions, 0 deletions:

| file | delta |
| --- | --- |
| `docs/EXP-0024-0025_preregistration.md` | +65 (new) |
| `docs/EXP_REGISTRY.md` | +2 (EXP-0024/0025 rows) |
| `portable_sam2_explicit_coarse/data/whu_instance_dataset.py` | +6 (sole code change) |
| `portable_sam2_explicit_coarse/scripts/ue/run_exp0024_ddp4.sh` | +74 (new) |
| `portable_sam2_explicit_coarse/scripts/ue/run_exp0025_ddp4.sh` | +74 (new) |

No other file is modified; the committed prereg is byte-identical to the working copy.

## 2. Data adapter correctness — PASS

**(a) Diff minimalism.** The change at `data/whu_instance_dataset.py:169-174` is a pure
insertion inside the RLE branch: `counts = segmentation["counts"]; if isinstance(counts, list):
segmentation = mask_utils.frPyObjects(segmentation, h, w)`. The polygon branch (lines 152-160)
and the compressed-RLE path (`decode(segmentation)` on the untouched dict when `counts` is
bytes/str) are unchanged; the outer guard still requires both `"counts"` and `"size"` keys,
and `h, w` are passed in the COCO `(rows, cols)` order `frPyObjects` expects. Uncompressed-RLE
decoding runs through pycocotools exactly as the archive README prescribes ("先
`frPyObjects` 转换再 decode").

**(b) Independent decode verification.** Loaded `WHUCocoInstanceDataset(data_root=<archive>,
ann_file="annotations/val.json", image_subdir="", image_size=(512,512), flip_prob=0.0)` and
compared against the archive's own reader (`sys.path.insert(0, <archive>/tools); from
read_dataset import TrainingArchive`, an independent NumPy column-major `decode_rle`), on 4
val points (positions 0/1/2 plus the max-instance position 408; pids 163/178/179/1725;
instance counts 20/15/8/87):

- instance count equal to the archive on every point (459 samples both sides; `iscrowd` is 0
  for all 14980 val annotations, so no filter divergence);
- **all 130 decoded masks bitwise-equal** to an archive mask under greedy pairing;
- every `gt_labels` entry is 0 (single class; dataset is `single_class=True`);
- every `gt_bboxes[k]` exactly equals the xyxy extent recomputed from its own decoded mask
  (`[cols[0], rows[0], cols[-1]+1, rows[-1]+1]`) — bboxes derive from masks, no residual
  COCO-`bbox` usage.

**(c) Trainer plumbing.** `train/cli_args.py:79-84`: `--dataset` choices are exactly
`("whu", "vhr10", "ue_raw_surface")`. `train/train_rsprompter_fusion.py:348` maps `whu` →
`dataset_format="whu_coco"`. The augmentation force-zero at lines 316-317 fires only
`if args.dataset == "ue_raw_surface"` — the whu path is untouched by it, exactly as the
prereg's note requires. `WHU_{TRAIN,VAL}_{ANN_FILE,IMG_SUBDIR}` env overrides flow through
lines 363-374 into `create_train_loader` (`data/loader.py:177-180`).

## 3. Augmentation single-variable — PASS

**(a) Trainer defaults.** `train_rsprompter_fusion.py:288-292`: `TRAIN_FLIP_PROB` default
`"0.5"`, `TRAIN_VFLIP_PROB` default `"0.0"`, `TRAIN_MULTI_SCALE_RESIZE_PROB` default `"0.0"`.
Gaussian noise and random erasing have **no** CLI or env knobs at all — they are hardcoded
`gaussian_noise_prob=0.0` and `random_erasing_prob=0.0` in the `create_train_loader` call
(lines 329, 331; `cli_args.py` contains no flip/noise/erasing flags). Val dataset is built
with every augmentation explicitly 0 (`loader.py:210-222`) in both arms.

**(b) Wrapper diff.** `diff run_exp0024_ddp4.sh run_exp0025_ddp4.sh` shows differences only
in: EXP id strings (guard/echo/error messages, incl. `EXP24_CONTRACT` vs `EXP25_CONTRACT`
prefixes), `RUN_TAG` (`noaug`/`defaug`), the header comment (lines 2-3) and the one-line
augmentation-contract comment (line 25), `TRAIN_FLIP_PROB=0` vs `0.5` (line 26), and the
`augmentation=flip:0,...` vs `flip:0.5,...` field of the contract echo (line 64). The
`exec torch.distributed.run ... train/train_rsprompter_fusion.py` blocks (lines 68-74) are
**byte-identical**: both `--dataset whu`, both with zero augmentation CLI flags
(augmentation is env-only). Both arms export all three env knobs explicitly, so EXP-0025
does not silently depend on the trainer default.

**(c) Empirical flip check.** Instantiated the train dataset (`annotations/train.json`,
2142 samples) twice with `flip_prob=0.0` and `flip_prob=1.0` (vflip/ms/noise/erasing 0), same
indices, on the first 3 samples with >10 instances (idx 0/1/2; 123/161/170 instances; 454
instance-level checks; flip_prob=1.0 fires deterministically since `rand() < 1.0` a.s. and
prob-0 short-circuits before consuming RNG):

- flipped image tensor == exact horizontal mirror of the unflipped image (bitwise);
- every flipped mask == mirrored original mask (bitwise, 454/454);
- flipped bboxes == `[W-x2, y1, W-x1, y2]` exactly; instance count unchanged.

(The commit message's own smoke numbers 20/20 and 144/144 are consistent with but smaller
than this audit's 130-instance decode and 454-instance flip re-verification.)

## 4. Wrapper guards — PASS

- **7 SHA256 pins recomputed independently** (`sha256sum` on the live archive) — all match
  the values pinned in both wrappers (lines 38-44):
  `dataset.json b42a11ed…64379`, `splits/train.json 965b1704…4fa5e`,
  `splits/val.json 2cb80f52…e3c37`, `splits/test.json f388cf35…856da`,
  `annotations/train.json ce262bf0…e656d`, `annotations/val.json 9f9ead36…2f7f1`,
  `annotations/test.json 6d8f29a1…619b3`.
- **Metadata asserts verified against the live `dataset.json`**: `label_revision ==
  review99_v3_user_approved_20261021`; points 2142/459/459; instances 126214/14980/44988;
  `image_size == [512, 512]` (real PNGs confirmed 512x512 on disk).
- **Runtime gates present in both wrappers and live-tested at HEAD**: `B0_EXP_ID` match
  (lines 12-14), no-unregistered-args (29-31), clean-worktree (32-34), output-dir must not
  exist (35-37), the 7 sha pins, the metadata heredoc, architecture fingerprint (55-58) and
  baseline-id (59-62). I re-ran both identity probes at HEAD with the pinned interpreter:
  fingerprint = `5305b3155914a2b2e7ecaf24b54957aee86fd6f47db93358daead2cbac5be36c` (match),
  baseline id = `ue-whu-a3sgm-512-v1` (match; requires `SAM2_CKPT`, which the wrappers
  export).
- **Work dirs absent**:
  `/data2/wangcheng/checkpoint/citysample_v1/citysample_v1_512single_noaug_ddp4_s44_EXP-0024`
  and `…512single_defaug_ddp4_s44_EXP-0025` do not exist (the `citysample_v1` parent itself
  does not exist yet), so the fresh-start gate will pass.
- **GPU state**: 4x RTX 4090, 15 MiB used of 49140 MiB each, 0% utilization — idle and
  compatible with the planned co-tenant DDP4 layout (both arms on GPUs 0-3, `CUDA_VISIBLE_DEVICES=0,1,2,3`
  in both wrappers, 2 ranks per GPU).
- **Trainer invocation identical between arms** — see item 3(b).

## 5. Seal discipline — PASS

- `train/train_rsprompter_fusion.py` (770 lines) contains **zero** occurrences of the string
  "test" (case-insensitive); `create_test_loader` exists in `data/loader.py:273` but nothing
  in `train/` or `scripts/` invokes it. The config `configs/ue_whu_a3sgm_512.py` mentions
  only mmdet-style `test_cfg`/`test_box_source` RCNN inference settings, not the test split.
- The whu_coco branch of `create_train_loader` consumes only the train and val annotation
  files (`loader.py:177-180`); `raw_split_manifest` is used solely by the `ue_raw_surface`
  branch (lines 149-155). The wrappers export only `WHU_TRAIN_*` / `WHU_VAL_*` env vars.
- The `splits/test.json` sha pin is a byte-level integrity lock; no code path loads it. The
  heredoc reads only aggregate metadata from `dataset.json` (revision/counts/image size).
- Val side: built from `annotations/val.json` only, augmentation-free, `val_subset_ratio`
  default 1.0 (full 459), `--val-every-n-epochs 1`; DDP default rank0 runs the complete
  unsharded val loader. Prereg states "test 全程不读" / test reserved for final evaluation —
  consistent with the code.

## 6. Preregistration — PASS

`docs/EXP-0024-0025_preregistration.md` contains the four elements:

1. **Primary metrics**: val 459 full, per-epoch COCO segm/bbox AP@[.50:.95]; two-arm
   contrast = EXP-0025 − EXP-0024 at last-epoch; companions AP50/AP75, area strata, curves.
2. **Selection rule**: both arms evaluated at last epoch 100 (no best-checkpoint cherry
   pick).
3. **Significance**: post-hoc scene-paired percentile bootstrap on the last checkpoints,
   10,000 draws, seed 20261014; per-epoch training curves explicitly non-decision-grade.
4. **Decision gates**: AUG_WINS (both primary-metric CI lower bounds > 0 → default aug
   becomes standard), AUG_NULL (CI contains 0 → record, higher arm as customary default,
   labelled non-significant), AUG_HARM (any CI upper bound < 0 → augmentation off becomes
   standard); plus the baseline-selection rule — the last-epoch better arm becomes the
   new-era satellite baseline; no absolute single-arm threshold (baseline-establishing
   pair, not a gain claim).

**Protocol constants match the wrappers/trainer invocation exactly**: seed 44 (`--seed 44`
+ contract echo), DDP4 batch 1 x accum 2 (global 8), lr 5e-4, wd 0.05, warmup 100 steps,
cosine (CosineAnnealingLR, `train/loops.py:490`), 100 epochs, AMP on (`--amp 1`), EMA off
(`--ema-enabled 0 --ema-eval 0`), early stopping disabled (`--early-stopping-patience 0`;
`loops.py:994` only triggers when patience > 0), save-last (`SAVE_LAST_MODEL=1` consumed at
`loops.py:916` → `last_model_epoch100.pth`), val every epoch on the full 459
(`--val-every-n-epochs 1`, `--compute-val-loss 0`, `val_subset_ratio` default 1.0), fresh
start (`--init-from` default None; output-dir gate). `docs/EXP_REGISTRY.md` carries both
EXP rows with the correct one-variable description.

## Non-blocking observations

- The trainer's `dataset_format` ternary maps "anything not ue_raw_surface/vhr10" to
  `whu_coco`, but argparse choices already restrict inputs to the three exact strings, so
  no unregistered format can reach that fallback.
- The default `TRAIN_MULTI_SCALE_IMG_SCALE` spec is non-empty, but with
  `TRAIN_MULTI_SCALE_RESIZE_PROB=0` pinned in both wrappers (and the dataset's own prob-0
  short-circuit) multi-scale can never fire in either arm.
- The unflipped/flipped empirical comparison is deterministic (prob-1.0 vs prob-0.0), so it
  is a stronger check than a seeded prob-0.5 sample would be.
- Launch operational note: the wrappers' clean-worktree gate (`git status --porcelain`)
  sees untracked files too, so this review file must be committed (or moved out) before a
  wrapper is launched — standard for this governance flow.

VERDICT: PASS
