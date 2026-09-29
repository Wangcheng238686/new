# DEPTH conclusion — UAV depth (height) marginal-information probe

**Experiment:** `MHBEV-FUS-20260921-DEPTH` (frozen read-only diagnostic; no
detector training or UAV mutation)
**Status:** complete (v7 clean run; v1–v6 are software-failure artifacts).
**Registered decision gates:** paired bootstrap, 10,000 draws, seed `20260930`,
gate = 95% CI lower bound > 0, evaluated per readout for the
`(SAT+HEIGHT) − SAT` and `HEIGHT − SAT` marginals.

## Question

Does UAV-rendered depth (per-cell world height, 4 stats: cross-view median /
std / max / valid-view fraction from 5 oblique views) carry marginal
information for satellite building segmentation beyond the deployed
satellite-only features? Scenes are the 159 strict-instance renders
`UAV_DepthBackfill_3502_3664` (disjoint from the `ue_coco_with_uav` labeled
benchmark). First 80 usable scenes fit the probes, remaining 79 evaluate
per-scene. Single-stream baseline, gates zeroed, ckpt
`ue_single_c5v2_control_tr128_va259_s44_r1/best_model_epoch86.pth`.

Readouts:

- **R1** band-error AUC — boundary band cells (9×9-dilated GT gradient, ~11.4k
  cells/scene), label = deployed prediction ≠ GT; logistic probes on SAT
  (P3 512-d) / HEIGHT (4-d) / SAT+HEIGHT (516-d).
- **R2** SDF-field Pearson — all cells, target = signed distance /8; ridge and
  MLP readouts, 1500 cells/scene fit.

## Repair history (v6 → v7)

v6 repeatedly exited `too few band cells in fit set` with `fit=80 val=79`
usable. Root cause: the instance-EXR reader's "best channel" heuristic picked
channel 3 (constant 1.0 alpha) instead of channel 0, collapsing the GT to an
all-full occupancy (fill = 1.0, zero band cells) on every scene. The satellite
instance EXR stores per-instance ids on channel 0 as `gray =
local_instance_id/255` (channels 0–2 identical); the gray→`local_instance_id`
identity mapping was verified by centroid back-projection against
`instances.json` `center_ue_cm` (87/101 exact nearest-anchor matches; every
mismatch was a >20 m centroid offset of a correctly matched large building).
v7 fixes the decode (`ids = rint(ch0·255)`, per-instance masks driven by
EXR-observed ids, plus a fill > 0.95 degeneracy guard) and three latent
call-site bugs downstream of the old exit (stray `device=` kwargs to
`fit_logistic`/`fit_mlp`; `sdf_y[sdf_sel]` double-indexing an already-sampled
target vector). The repair changes neither model, checkpoint, split, probe
definitions, cell sampling, nor bootstrap mathematics — mechanical rerun
repair under the D3 precedent (`docs/MHBEV-DIAG-20260918_D3_conclusion.md`).
v1–v6 logs are retained under
`/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/` as
software-failure artifacts; they supply no evidence.

## Results (79 held-out scenes)

| Measure | SAT | HEIGHT | SAT+HEIGHT |
|---|---:|---:|---:|
| R1 band-error AUC (mean ± std) | 0.6993 ± 0.0653 | 0.5025 ± 0.0052 | **0.7206 ± 0.0579** |
| R2 SDF Pearson, ridge | 0.6434 ± 0.0822 | 0.0130 ± 0.0351 | 0.6430 ± 0.0817 |
| R2 SDF Pearson, MLP | 0.6977 ± 0.0962 | 0.0065 ± 0.0320 | 0.6989 ± 0.0963 |

| Paired marginal (10k bootstrap, seed 20260930) | Δ mean | 95% CI | improved |
|---|---:|---|---:|
| **R1: (SAT+HEIGHT) − SAT** | **+0.0213** | **[+0.0187, +0.0240]** | **79/79** |
| R1: HEIGHT − SAT | −0.1968 | [−0.2116, −0.1820] | 0/79 |
| R2 ridge: (SAT+HEIGHT) − SAT | −0.0004 | [−0.0007, −0.0001] | 33/79 |
| R2 MLP: (SAT+HEIGHT) − SAT | +0.0012 | [−0.0003, +0.0027] | 44/79 |
| R2 ridge: HEIGHT − SAT | −0.6304 | [−0.6491, −0.6118] | 0/79 |
| R2 MLP: HEIGHT − SAT | −0.6912 | [−0.7144, −0.6688] | 0/79 |

Per-scene R1 gain distribution: min +0.0060, median +0.0197, max +0.0471 —
strictly positive on every scene.

## Gate verdicts

- **R1 (band-error AUC): PASS, decisively.** Adding height to SAT raises
  boundary-band error AUC by +0.0213 with CI lower bound +0.0187 > 0 and
  improvement on 79/79 scenes. HEIGHT alone is at chance (0.5025), so the
  signal is a genuine *increment* to SAT, not a substitute.
- **R2 (SDF-field regression): FAIL / NULL.** The ridge marginal is
  significantly negative at trivial magnitude (−0.0004); the MLP marginal is
  +0.0012 with CI crossing zero. No positive SDF gain.

`Registered decision: DEPTH_R1_BAND_ERROR_GAIN_ONLY` — UAV depth carries real
marginal information about *where the satellite-only model errs near
boundaries*, but nothing further about the SDF geometry SAT P3 already
reconstructs (SAT R2 0.64–0.70 vs HEIGHT-alone 0.01).

## Interpretation and scope

The R1-pass / R2-fail split is internally consistent: cross-view height
statistics mark vertical-structure presence, which flags boundary cells whose
single-view satellite prediction disagrees with GT, but they add no
information about the signed-distance field beyond what SAT features encode.
Mechanism note (cannot be proven by this probe): the R1 gain is compatible
with height acting as an occupancy prior that is complementary to SAT
appearance exactly where the deployed AP_small losses concentrate (small
buildings with weak SAT evidence). This probe does not establish that fusing
depth improves AP; it establishes the precondition (nonzero marginal error
information) at the AOC/gate level.

Guards: fit/eval scenes disjoint by construction (first-80 split, seed
20260929); no GT or prediction information enters the HEIGHT features
(UAV-rendered geometry only); band label uses the deployed 0.5-score union
prediction; `usable scenes: fit=80 val=79`; R1 fit sample 40,000 band cells,
`err_rate=0.130`; root-fs-full constraint honored (all artifacts under
`/data1`).

## Artifacts

- Outputs: `/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/MHBEV-FUS-20260921-DEPTH-v7/`
  (`summary.json`, `per_scene.json`)
- Log: `.../MHBEV-FUS-20260921-DEPTH-v7.log`
- Probe script (fixed): `portable_sam2_explicit_coarse/scripts/ue/probe_depth_gain.py`
