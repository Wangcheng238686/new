# MHBEV-FUS-20260919 — UAV marginal-information frozen probes: results, evidence, and pending review

**Status: PROVISIONAL — PENDING INDEPENDENT REVIEW (待复核).** Nothing in this file
authorizes detector training, new fusion implementations, or data changes. Scope:
frozen-feature linear readouts only; no detector weights were trained or modified;
no dataset files were created or altered by the probes.

**Registered as:** Appendix J of `EXPERIMENT_GOVERNANCE.md`.
**Implementation:** `portable_sam2_explicit_coarse/scripts/ue/probe_fusion_information.py`
(git `d40be42`). **Outputs:** `portable_sam2_explicit_coarse/outputs/{probe_fusion_information,
probe_fusion_information_d2, probe_fusion_boundary}/` (summary.json + per_scene_val_ap.csv).
**Companion evidence:** geometry audit `docs/ue_geometry_audit_report.md` (commit `2b01f4a`),
perturbation probe `scripts/ue/probe_geometry.py` (commit `947623a`).

---

## 1. Question and decision context

r2 (P2/P3 zero-gated BEV injection, best 0.7286) and r3 (additional stride-16
image-embedding injection, best 0.7306) finished at parity with the single-stream
control (0.7312). Gate monitors showed injected-to-base norm ratios of 0.05% / 0.014%.
Degraded-satellite runs (D2 512→256→512; C30 30% hard occlusion) also showed parity
(single 0.7035 / 0.4944; dual 0.6991 / 0.4987). Before choosing the next fusion
mechanism, four probes were run to establish, without detector training:

1. does UAV content carry *any* scene information through the calibrated BEV (E1);
2. is that information geometry-dependent, i.e. destroyed by calibration mismatch (E1 control);
3. why the zero-gates never opened — gradient dynamics (E2);
4. whether the marginal UAV information (if any) lives in occupancy or in boundary geometry (E4).

## 2. Setup contract (all arms)

- Features extracted per scene: SAT = satellite P3 (512-d, stride 8, from the named
  single-stream checkpoint); LBEV = learned `GeometrySparseBEV` output (128-d, from the
  r3 checkpoint); DBEV = training-free direct geometric BEV — every satellite cell's
  ground point (dataset contract, z=0, 360 m tile) projected through K·E into the 5 UAV
  views, bilinearly sampled on all 4 frozen UAV-FPN levels, valid-view averaged (1024-d);
  DBEVmis = identical but with ground points rotated 90° about the scene center
  (calibration-content mismatch control).
- Probes: per-cell linear readout (per-dim standardization, single Linear layer, Adam
  500 iters, balanced 1:1 cell subsampling for fit: 80 train scenes, ≤400 pos+400 neg
  per scene, 62,508 fit cells). Evaluation: all 16,384 cells per val scene, first 60
  val scenes, per-scene metric, paired statistics vs SAT.
- Lineage: satellite branch from `ue_single_..._r1/best_model_epoch86.pth`; fusion branch
  from `ue_dual_..._r1/best_model_epoch85.pth`; D2 arm satellite branch from
  `ue_d2_single_.../best_model_epoch83.pth` with data root `..._d2`. Train/val scene
  isolation identical to all prior arms (ratio .1797752808988764, seed 44).

## 3. Results (authoritative numbers from summary.json artifacts)

### E1 — occupancy readout, original data (`outputs/probe_fusion_information/summary.json`, md5 5ed8a6ef...)

| Feature | per-scene AP (mean±std, n=60) |
|---|---|
| SAT | 0.8848 ± 0.0767 |
| LBEV | 0.5392 ± 0.1265 |
| DBEV | 0.5395 ± 0.1325 |
| DBEVmis | 0.3268 ± 0.1257 |
| SAT+LBEV | 0.8851 (Δ +0.0003, 28/60 improved) |
| SAT+DBEV | 0.8867 (Δ +0.0019, 31/60 improved, paired t=1.66, p=0.10) |

### E1-D2 — occupancy readout, degraded satellite (`probe_fusion_information_d2/`, md5 a3feaf5a...)

| Feature | per-scene AP |
|---|---|
| SAT (D2-trained branch, D2 images) | 0.8744 ± 0.0815 |
| DBEV (clean UAV) | 0.5322 ± 0.1332 |
| DBEVmis | 0.3301 ± 0.1200 |
| SAT+DBEV | 0.8761 (Δ +0.0017, t=1.10, p=0.28) |

Note: 2× resolution degradation costs the satellite occupancy readout only 0.010 AP —
the occupancy signal is blur-robust; no exploitable gap appears.

### E2 — gate-gradient SNR (`summary.json: grad_stats`; 10 real train batches, loss
materialization replicated from the trainer incl. `_materialize_p2_boundary_refiner_loss`)

| gate | grad mean | grad std | SNR | sign consistency | wd·|γ| pull |
|---|---|---|---|---|---|
| γ_p2 | +4.8e-5 | 2.0e-4 | 0.24 | 0.6 | 2.7e-3 |
| γ_p3 | −5.4e-5 | 9.2e-5 | 0.59 | 0.6 | 3.9e-4 |
| γ_emb | −1.9e-5 | 1.9e-4 | 0.10 | 0.6 | 3.1e-4 |

Coherent signal is 1–2 orders of magnitude below the weight-decay pull: the zero-gate
cannot bootstrap, and the observed equilibrium (γ_p2 ≈ −0.054) is the noise–decay
balance point, not a learned decision.

### E3 — injection spatial structure (`summary.json: p3/p2`)

Injection-magnitude-to-base ratio: on building cells 6.2e-5 vs background 8.2e-5 (P3);
Spearman(rank(ratio), occupancy) = −0.27 (P3) / −0.21 (P2); cosine(injected delta, base
feature) ≈ −0.04/−0.07. The trained injection is spatially unstructured and directionally
orthogonal — it never encoded object evidence.

### E4 — boundary-distance regression, original data (`probe_fusion_boundary/`, md5 a1a26698...)

Target: per-cell signed distance to the nearest building boundary (GT distance transform,
/8 m-scaling). Metric: per-scene Pearson r between probe prediction and target.

| Feature | per-scene r (mean±std, n=60) |
|---|---|
| SAT | 0.5192 ± 0.1251 |
| LBEV | 0.4930 ± 0.1452 |
| DBEV | 0.4836 ± 0.1644 |
| DBEVmis | 0.0306 ± 0.1769 |
| SAT+LBEV | 0.6055 (Δ +0.0863, 57/60 improved) |
| **SAT+DBEV** | **0.6111 (Δ +0.0919, t=13.27, p=2.3e-19, 58/60 improved)** |

## 4. Provisional conclusions (each tied to evidence above; all pending review)

1. **Geometry/content chain is sound.** DBEV carries real scene information and it is
   calibration-dependent (E1: 0.54 correct vs 0.33 mismatched; E4 control 0.03).
   Independent of the earlier geometry audit (2.4 px GT-footprint alignment).
2. **Occupancy is informationally saturated in the satellite stream.** Marginal UAV
   occupancy info ≈ +0.002 AP, not significant, on original AND degraded data (E1, E1-D2).
   This explains all parity results to date and rules out "injection-site search" as a fix.
3. **The zero-gate is dynamically incapable of opening** (E2), and the trained injection
   is spatially meaningless (E3). "Warm-gate initialization" alone would inject
   unstructured content faster; it is justified only together with a signal-bearing target.
4. **The exploitable UAV margin is boundary geometry, not occupancy** (E4: +0.092 r,
   p=2e-19). Physically consistent with oblique views seeing facade/height structure.
5. **Recommended mechanism (NOT yet authorized):** inject BEV (learned or direct) as an
   additional input to the P2BoundaryRefiner / boundary-prior pathway with a small
   non-zero gate initialization (e.g. 1e-2), optionally with a boundary-distance
   auxiliary pretraining of the BEV branch. All other components frozen; same 128/259
   protocol, seed 44, single-stream control unchanged.

## 5. Known limitations a reviewer should check

- **Linear readouts only.** Nonlinearly-encodable marginal information is unmeasured in
  both directions (satellite could also be stronger nonlinearly). The E4 conclusion rests
  on linear readability, which is what a gated residual injection can initially exploit.
- **Dimension asymmetry.** SAT=512-d, DBEV=1024-d, concat=1536-d. The linear probe is
  cheap, but a reviewer may request a matched-dimension control (e.g. PCA-512 DBEV) to
  exclude a pure capacity artifact. The DBEVmis control (same 1024-d, wrong geometry)
  partially addresses this: it lands at 0.03/0.33, far below DBEV.
- **Boundary target is derived from the same GT masks as occupancy** — the E4 claim is
  about feature content, not label independence; but the proposed mechanism must show
  end-to-end gains, not probe gains.
- **Single checkpoint lineage per arm, single seed, one extraction pass**; no CIs over
  re-extraction noise. Per-scene paired stats (n=60) are the only uncertainty estimate.
- **E2 used n=10 batches with LoRA dropout active (model.train())**; sign-consistency 0.6
  at n=10 is weak evidence on its own — the 10–100× gap to the weight-decay pull is the
  load-bearing number, not the sign statistic.
- **Execution finding (recorded for honesty):** the first probe launch computed gate
  gradients against a contaminated loss (raw `_p2br_local_sum`, unmaterialized; loss
  ~250 instead of ~0.15). That run's gradient numbers are invalid and were discarded;
  the fix (replicating the trainer's materialization + "loss"-key filtering) is in the
  committed script. Reviewers can verify: batch losses in the current log are 0.09–0.22.
- **LBEV weights are near random-init** (historical γ≈−0.054 starved their gradient).
  E1/E4 show this is immaterial for the readout conclusions (LBEV ≈ DBEV), but it means
  the probes say nothing about what a *trained* BEV could become.

## 6. Review ask

Independent review should confirm or refute: (a) artifact integrity (checksums above,
rerun commands below); (b) the E4 statistics and the DBEVmis control; (c) whether the
Section 4.5 mechanism is authorized as the next experiment arm, and under which
pre-registered gates (proposed: same 128/259 protocol, seed 44 first, boundary-F1 and
AP75 as primary endpoints, injected-ratio and gate-value monitors mandatory, single-stream
control frozen at 0.7312).

Rerun commands (from `portable_sam2_explicit_coarse/`, GPU with ~20 GB free):

```
python scripts/ue/probe_fusion_information.py                       # E1+E2+E3
python scripts/ue/probe_fusion_information.py --single-ckpt work_dirs/ue/ue_d2_single_tr128_va259_s44/best_model_epoch83.pth --data-root /data1/wangcheng/dataset/ue_coco_with_uav_d2 --out outputs/probe_fusion_information_d2
python scripts/ue/probe_fusion_information.py --target boundary --out outputs/probe_fusion_boundary
```

Expected runtimes ≈ 6 min each on one RTX 4090; logs at
`work_dirs/ue/probe_fusion_information*.log`.
