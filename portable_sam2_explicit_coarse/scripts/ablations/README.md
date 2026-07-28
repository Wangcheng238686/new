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

Both default to the full official split:

```bash
TRAIN_SUBSET_RATIO=1.0
VAL_SUBSET_RATIO=1.0
```

Quick comparison example:

```bash
TRAIN_SUBSET_RATIO=0.1 VAL_SUBSET_RATIO=0.2 \
  bash scripts/ablations/c4_pafpn_coarse_points_box_dense.sh
```

Subsets are deterministic image-level samples. Train uses `SUBSET_SEED`
(default 44); validation uses `SUBSET_SEED + 10000`. All ablations therefore see
the same sample IDs when the ratio and seed are unchanged.

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
