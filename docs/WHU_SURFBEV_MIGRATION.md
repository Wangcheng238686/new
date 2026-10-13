# WHU / SurfaceBEV migration

Source: `95026ae677dd0a8d77a3ee4c5fb68abeb4ea01c1`.
Original committed baseline: `27b42d1`.
Archive branch: `codex/archive-dual-stream-pre-whu-20261013`.
Migration branch: `codex/whu-surfacebev-migration`.

The migration was subsequently fast-forwarded into `codex/dual-stream-dev` at
`/home/wangcheng/project/new/.worktrees/dual-stream-dev`. This is the workspace
for subsequent dual-stream development. Integration brought in `9466d7b`; the
satellite model still has no new fusion implementation. The migration worktree
is retained for its original acceptance logs and temporary fixtures.

Integration checks are recorded in
`docs/whu_surfacebev_migration/workspace-integration.json`. The additional
backup at `/home/wangcheng/.codex/backups/whu-surfacebev-integration-20261013`
contains the pre-integration patch/status and the 91 old untracked source and
protocol files moved aside before the fast-forward. Adopted source/protocols
are already in the migration commits; replaying that old overlay would undo
the migration. Local datasets and historical outputs stayed in place and are
excluded from new Git additions; previously tracked records remain tracked.

Recovery snapshot: `/home/wangcheng/.codex/backups/whu-surfacebev-20261013`.
It contains a binary HEAD diff, full untracked-file archive (symlinks preserved),
status and content-hash inventory. Restore into a fresh worktree at the original
HEAD using `git apply tracked.patch`, then extract `untracked.tar` there.
The original worktree was left unchanged during isolated migration acceptance;
the later integration updates its code while preserving datasets and outputs.

Only mainline source/protocol changes are adopted; experiment data/output files
remain external and their inventory is in the recovery manifest. Historical
protocols retain their original experimental identity. Old P2/P3 and A1 routes
are archived; new end-to-end fusion and full training are outside this change.

Full source snapshot including dirty source/protocols: `codex/archive-dual-stream-snapshot-20261013` (`7659eb7`).
Domain model packages are transplanted byte-for-byte; legacy registered models are removed.

## Result and supported entrypoints

The satellite implementation is the sealed WHU a3sgm model, without new fusion.
The model registry still exposes `RSPrompterAnchor`. Model domain packages,
`inference/model_rebuild.py`, and both sealed reference configurations are
byte-identical to source `95026ae`. The vendored SAM2 tree is also unchanged.
The source configuration fingerprint is
`5d5f78ec947fba1c5a92fb28a8486c9b8d7050c66c0034abf8425d1034bce17b`.

WHU semantics: one class, roi_local final masks, standard mask loss, frozen
PromptEncoder mask-downscaling, dense gate initialization 0.25, and UDPR bounded
logit residual 2.0. UE uses the registered spatial/object-disjoint v2 pair split
and no augmentation. A stable numeric image ID was added for COCO evaluation;
images, target masks, boxes, camera metadata and split assignments are preserved.

| Entry | Purpose |
|---|---|
| `configs/ue_whu_a3sgm.py` | Exact WHU model, local SAM2 paths only |
| `train/train_rsprompter_fusion.py --dataset ue_raw_surface` | New raw-pair B0 train/val chain |
| `inference/infer_from_checkpoint.py` | Strict checkpoint reconstruction and UE validation inference |
| `ue/features.py` | Satellite pyramids, SAM embeddings, PE and high-resolution features; no injection |
| `configs/ue_uav_encoder.py` | Independent frozen UAV encoder, same settings as archived route |
| `scripts/ue/create_baseline_identity.py` | Pin a new checkpoint by identity and SHA256 |
| `scripts/ue/probe_surfacebev_p1_h0.py` | Label-free H0 harvesting and passive prediction parity |
| `scripts/ue/probe_surfacebev_p1.py` | Recoverable P1 readout/cache chain on the new baseline |
| `scripts/ue/run_p1_supervised.py` | Supervisor; status file lives beside the output directory |

The public single-stream ablation launchers were replaced by the source-supported
WHU/NWPU launchers; obsolete recipes are available in the archive branch.
Other retained SurfaceBEV data QA/render tools retain their historical protocols.
Historical documents in this repository are evidence of their original runs,
not preregistration or results for the new baseline.

## Runtime and small-batch verification

Local validation environment: `/data2/wangcheng/envs/cvt2/bin/python` (Python 3.9,
PyTorch 2.8.0+cu128, MMCV 2.1.0, MMDetection 3.3.0, MMEngine 0.10.7).
The default Python is not the validated environment. No packages were changed in
this existing environment. Machine-local dataset contents and validation weights
are intentionally excluded from Git.

From the repository root of the current development worktree:

```bash
export PYTHON=/data2/wangcheng/envs/cvt2/bin/python
export SAM2_CKPT=/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt
export SAM2_REPO="$PWD/sam2"
export PYTHONPATH="$PWD/portable_sam2_explicit_coarse:$SAM2_REPO"
export UE_DATA_ROOT="$PWD/portable_sam2_explicit_coarse/data/ue_raw_hero_uav_pairset_20261011"
```

The development worktree retains its original raw-pair directory and scene links.
The isolated migration worktree's scenes symlink points to this directory and
remains valid. On another checkout, provide the existing registered
pairset via `UE_DATA_ROOT`; no dataset rematerialization is needed. SAM2 identity
is recorded in checkpoints and must match the probe's runtime checkpoint.

Bounded training-chain smoke (one optimizer update, one validation batch):

```bash
SAVE_LAST_MODEL=1 "$PYTHON" portable_sam2_explicit_coarse/train/train_rsprompter_fusion.py \
  --config portable_sam2_explicit_coarse/configs/ue_whu_a3sgm.py \
  --dataset ue_raw_surface --data-root "$UE_DATA_ROOT" --image-size 1024 1024 \
  --epochs 1 --batch-size 1 --val-batch-size 1 --num-workers 0 \
  --train-subset-ratio .002 --val-subset-ratio .01 \
  --max-train-batches 1 --max-val-batches 1 --grad-accum-steps 1 \
  --warmup-iters 1 --amp 1 --ema-enabled 0 --seed 44 \
  --run-tag MIGRATION_SMOKE_ONLY --checkpoint-dir /tmp/ue-whu-new-smoke
```

Use a fresh checkpoint directory. Runs limited by batches or subset ratios carry
`ue_baseline.validation_only=true`; they cannot resume a formal baseline run.
Task-weight `--init-from` is rejected for this UE baseline: a different
initialization requires a separate experiment protocol. The new full-run wrapper
`scripts/ue/run_ue_whu_b0.sh` requires explicit local paths, a new run tag and a
fresh directory. Full training must be independently registered; it was not run.
The trainer also supports direct `torchrun` invocation, but DDP was not validated
in this migration.

Strict build/load and validation inference:

```bash
"$PYTHON" portable_sam2_explicit_coarse/inference/infer_from_checkpoint.py \
  --checkpoint /tmp/ue-whu-new-smoke/last_model_epoch1.pth \
  --sam2-ckpt "$SAM2_CKPT" --device cpu --build-only
"$PYTHON" portable_sam2_explicit_coarse/inference/infer_from_checkpoint.py \
  --checkpoint /tmp/ue-whu-new-smoke/last_model_epoch1.pth \
  --sam2-ckpt "$SAM2_CKPT" --data-root "$UE_DATA_ROOT" \
  --split validation --max-batches 1 --num-workers 0 --no-eval \
  --output-dir /tmp/ue-whu-new-inference
```

H0/P1 require `--checkpoint` and `--baseline-contract`. Create the latter with
`create_baseline_identity.py --checkpoint <new-checkpoint> --output <new-json>`.
H0 additionally requires a fresh `--output-dir`; P1 requires `--h0 <summary>` and
`--experiment-id <new-id>`. H0 smoke uses `--train-scenes 1 --val-scenes 1
--height-samples-per-view 256`; P1 smoke uses `--phase harvest --max-train 1
--max-val 1`. Both report `SMOKE_ONLY`; P1 exits successfully without evaluating.
A temporary baseline is rejected from formal evaluation even with `--allow-test`.
The public UE inference entry also rejects `--allow-nonstrict` and test access
for validation-only checkpoints.

New caches bind checkpoint SHA, WHU architecture, SAM2 identity, pair/split hashes,
H0 digest, implementation fingerprints, experiment ID and subset limits. Existing
unidentified caches, old H0 summaries, changed code or a different baseline are
rejected. Resume requires the same cache contract. Historical outputs are never
adopted by changing a checkpoint constant.

## Acceptance evidence and limits

Machine-readable records are in `docs/whu_surfacebev_migration/acceptance.json`.
Full local logs and temporary fixtures are in `.migration-validation/` (ignored).

| Check | Result |
|---|---|
| 24 sealed model/config/rebuild files vs pinned source | Byte-identical |
| Same temporary weights, input, batch, seed, GPU and precision | Strict load passed in both isolated processes |
| Preprocessing, all extracted features, predictions, all loss tensors | Bitwise equal |
| Total loss | Exactly 4.773105144500732 in both processes; smoke only |
| Backward | 334 finite gradient tensors in both; expected backbone/neck/RPN/ROI/CSPM/UDPR groups present |
| Serialization after backward | Strict save/reload passed |
| Actual new trainer | 1 train batch, 1 val batch and checkpoint save passed |
| Saved trainer checkpoint | Public strict rebuild and UE validation inference passed |
| SurfaceBEV H0/P1 | 1 train + 1 val scene; P3 512×128×128; passive parity and correct/sham invariants passed |
| Wrong identity, changed weights/config/split, stale/unidentified cache | 9 adversarial unit cases passed |
| Temporary baseline formal/test evaluation | Rejected before output creation or test access |
| Public UE inference non-strict loading and temporary test access | Both rejected |
| Original workspace | Original Git status and binary diff unchanged; 15,955 untracked files/links unchanged |

Gradient digests are **not** bitwise equal. Repeating the unchanged source also
changed 223 gradient digests while keeping forward outputs and losses exact.
This environment's GPU backward is not bitwise repeatable; finite gradients and
coverage passed, but numerical gradient equivalence is not certified. No
acceptance threshold was relaxed for forward outputs or losses.

The seven original paper checkpoint regression cases were not run: those weights
were not located in the inspected checkpoint roots. This migration proves the
sealed source/candidate equivalence with explicitly temporary task weights; it
does not establish a new B0's performance. No full training, test-image evaluation,
DDP validation or new scientific conclusion was produced.

## Old B0 compatibility and recovery

EXP-0006 uses schema 2 and architecture
`c5v2_pafpn_coarse_p2_boundary_refiner_emb64`; the sealed WHU model uses schema 3
and `r1_c4_rd_pafpn_coarse_points_box_raw_detach_emb64_udprk64`.
Actual strict loading was rejected: **6 missing, 14 unexpected, 0 shape mismatch**.
See `old-b0-audit.json` for exact parameter names. There is no non-strict loading
fallback. The old B0 weight, caches and conclusions remain in their original
lineage; a new B0 requires an independent later training run.

Stage commits:

- `7659eb7`: adopted dirty mainline source/protocols and recovery inventory.
- `807d220`: sealed source runtime transplant.
- `40154d4`: UE raw-pair train/inference chain and checkpoint identity.
- `3a3cc82`: SurfaceBEV observation interface and probe adaptation.
- Final validation commit: supported launch recipes, adversarial checks and evidence.

To recover the exact original dirty workspace, create a fresh detached worktree
at `27b42d1`, apply the snapshot's `tracked.patch` with `git apply`, then extract
`untracked.tar` into that fresh worktree. The archive preserves symlinks without
dereferencing external datasets. Do not restore over an occupied worktree.
For the source/protocol checkpoint alone, use
`codex/archive-dual-stream-snapshot-20261013` (`7659eb7`).
The isolated migration was reviewed on `codex/whu-surfacebev-migration` before
integration into `codex/dual-stream-dev`. Each implementation stage remains a
separate commit. For rollback inspection, create a fresh worktree from an archive
branch; for the exact pre-migration dirty state use the snapshot recipe above.
Keep the migration worktree while its ignored acceptance logs are needed.
