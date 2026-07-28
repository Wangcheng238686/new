# WHU-1024 ablation matrix

| ID | Neck | Prompt route | Box token | Dense prompt | DenseBR |
|---|---|---|---:|---:|---:|
| B0 | aggregator | legacy 5-token MLP | - | - | - |
| B1 | PAFPN | legacy 5-token MLP | - | - | - |
| C1 | aggregator | coarse 2P2N | no | no | no |
| C2 | PAFPN | coarse 2P2N | no | no | no |
| C3 | PAFPN | coarse 2P2N | yes | no | no |
| C4 | PAFPN | coarse 2P2N | yes | yes | no |
| C5 | PAFPN | coarse 2P2N | yes | yes | yes |

B0 is the new-mainline bridge control. The byte-exact historical baseline remains
under `../../legacy_baseline` at repository level and is launched with
`../../../scripts/reproduce_legacy_segm.sh`.

Every wrapper first validates the resolved MMEngine config. Architecture values
are fixed by the wrapper and cannot be inherited from stale environment values.

## Dataset subsets

The ablation wrappers default to 10% of the official training split and the
complete official validation split:

```bash
TRAIN_SUBSET_RATIO=0.1
VAL_SUBSET_RATIO=1.0
VAL_BATCH_SIZE=1
```

Override both explicitly when a different screening ratio is needed:

```bash
TRAIN_SUBSET_RATIO=0.1 VAL_SUBSET_RATIO=0.2 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
```

Use `TRAIN_SUBSET_RATIO=1.0 VAL_SUBSET_RATIO=1.0` for a full-data run.

Subsets are deterministic image-level samples. Train uses `SUBSET_SEED`
(default 44); validation uses `SUBSET_SEED + 10000`. All ablations therefore see
the same sample IDs when the ratio and seed are unchanged.

Validation also follows the historical WHU1024 baseline execution contract:
only global rank 0 iterates the full validation loader, the other DDP ranks
wait at epoch-end synchronization, and bbox/segm COCO evaluation uses detector
scores. Empty-GT images remain in COCO evaluation so false positives are
counted. `VAL_BATCH_SIZE` can be overridden, but `1` is the comparison default.

## Smoke and dry-run

```bash
# Config, command, independent subset and representative full-model checks.
bash scripts/ablations/smoke_all.sh

# Fast checks without loading four complete SAM2 models.
FULL_MODEL_SMOKE=0 bash scripts/ablations/smoke_all.sh

# Inspect one resolved experiment without starting training.
DRY_RUN=1 CHECK_DATA=1 PREFLIGHT_MODEL=1 \
  TRAIN_SUBSET_RATIO=0.01 VAL_SUBSET_RATIO=0.02 \
  bash scripts/ablations/c5_pafpn_coarse_densebr.sh
```

Useful common overrides include `MAX_EPOCHS`, `BATCH_SIZE`,
`GRAD_ACCUM_STEPS`, `LEARNING_RATE`, `NPROC_PER_NODE`, `CUDA_VISIBLE_DEVICES`,
`MAX_TRAIN_BATCHES`, `MAX_VAL_BATCHES`, `INIT_FROM`, `RESUME_FROM`, and
`CHECKPOINT_DIR`.

## Terminal logs and hyperparameter snapshots

Every ablation wrapper automatically mirrors both stdout and stderr to the
terminal and to a project-local log:

```text
logs/ablations/<run_tag>_tr<train_ratio>_va<val_ratio>_<timestamp>_pid<pid>.log
```

The log begins with a `resolved_hyperparameters_begin` /
`resolved_hyperparameters_end` block. It records the resolved architecture
route, dataset ratios, optimizer and EMA settings, effective global batch size,
initialization/resume paths, data/checkpoint locations, git commit, and the
exact torchrun command. This is the resolved execution snapshot, so it should
be used for experiment review rather than relying only on wrapper defaults.

Override the directory or the exact file when needed:

```bash
LOG_DIR=/path/to/logs bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh

LOG_FILE=/path/to/exact.log \
  DRY_RUN=1 bash scripts/ablations/b0_aggregator_mlp.sh
```

Project-local runtime logs are ignored by Git.
