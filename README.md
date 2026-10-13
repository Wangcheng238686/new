# UE SurfaceBEV with the sealed WHU satellite baseline

This worktree adopts the single-stream WHU a3sgm implementation from
`fast-whu150@95026ae`, with independent UE raw-pair training and SurfaceBEV
observation/probe adapters. CSPM and UDPR retain the sealed implementation.
New end-to-end fusion and full B0 training are outside this migration.

Start with [migration and runtime instructions](docs/WHU_SURFBEV_MIGRATION.md).
The [acceptance record](docs/whu_surfacebev_migration/acceptance.json) covers
source/candidate exact forward/loss parity, small-batch backward and checkpoint
reload, UE validation inference, SurfaceBEV smoke and lineage/cache rejection.

`portable_sam2_explicit_coarse/rsprompter/` contains the sealed domain packages;
`ue/` contains the perimeter adapters. The new UE configuration is
`configs/ue_whu_a3sgm.py` inside that package. Machine-local data/checkpoint paths
must be supplied explicitly. Historical experiment documents keep their original
identity and are not evidence for the new baseline. Old P2/P3, A1 and C5v2 B0
implementations are recoverable from the archived source branch.
