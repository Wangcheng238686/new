# SZAB conclusion — small-instance-retasked P2-injection A/B

**Experiment:** `MHBEV-FUS-20260929-SZAB` (live-mechanism training A/B, route A)
**Status:** complete (both arms, 6 epochs, seed 20260928, GPUs 2/3 exclusive-arm).
**Registered verdict:** `SZAB_INJECT_NO_SMALL_GAIN` — the warm-fitted P2
injection does **not** improve small-instance segmentation under the
small-instance-retasked objective; the retasked objective itself (both arms)
delivers the small-instance gain the unweighted baseline lacked.

## Question

Appendix V: does the H2-verified warm-fit P2 injection (UAV-geometry BEV → P2,
E5-style least-squares conv, γ init 0.25) become causally useful when the
training task is redefined to emphasize small buildings (loss ∝ 1/√area,
primary readout segm AP_s)? Arms identical except injection: A = retasked, no
injection; B = retasked + injection. Full co-adaptation on all 712 train
scenes, 259-val every epoch.

## Repair history (registered in Appendix V.7)

The first registered weighting scaled positive cls/box losses by raw `w_k`
(batch mean ≈ 0.010). Against weight-1 negatives in the count-based cls
denominator this crushed the positive-class signal ~100×: both arms collapsed
from the 0.731-AP init to 0.28–0.38 within one epoch and stayed below the
unweighted baseline everywhere. Those arms were stopped at e5/e4, archived
(`*-abandoned-v1weighting-0930195934`, with POSTMORTEM; defunct artifacts, not
gate evidence). Amendment V.7 mean-normalizes the bbox-path positive weights
to 1 per batch (relative 1/√area form kept; mask path mathematically
unchanged). Independent amendment re-review: PASS (`docs/reviews/
MHBEV-FUS-20260929_SZAB_implementation_review.md` addendum). Two earlier
startup attempts also died of co-tenant GPU-memory pressure on 09-29 (root
cause confirmed OOM, archives kept); the registered run started 2026-09-30
evening after the amendment.

Sanity at epoch 1 of the registered run: A AP 0.6926 / AP_s 0.1206,
B 0.6807 / 0.1225 — classifier intact and AP_s at/above the unweighted
baseline from the first epoch, i.e. the retasking now works as intended.

## Results (epoch 6 = last checkpoint, 259-val)

| Measure | A (retask) | B (retask+inj) | B−A |
|---|---:|---:|---:|
| **segm AP_s (primary)** | **0.1827** | 0.1772 | **−0.0055** |
| segm AP | 0.7485 | 0.7510 | +0.0025 |
| AP75 | 0.8640 | 0.8613 | −0.0027 |
| boundary-F1 | 0.6399 | 0.6394 | −0.0005 |
| small-instance mask-F1@0.5 | 0.5240 | 0.5131 | −0.0109 |

Per-scene small-mask-F1 paired bootstrap (10k, seed 20260936):
B−A = −0.0109, 95% CI **[−0.0407, +0.0187]**, improved 19/259.

Mechanism telemetry (B): warm fit R² 0.2272 (identical to V1 — fit path
untouched); γ 0.2500 → 0.1836 (−26.6% deviation, gate iii met);
engagement mean-square |Δ|²/|P2|² positive every epoch but declining
(0.0139 → 0.0085) — the injection stayed active while co-adaptation tuned it
down.

## Gate verdicts (Appendix V.4)

- (i) AP_s(B) > AP_s(A): **FAIL** (−0.0055); small-F1 CI lower > 0: **FAIL**
  (upper bound +0.0187, includes 0).
- (ii) global non-degradation: PASS (AP +0.0025 ≥ −0.003; BF1 −0.0005 ≥ −0.003).
- (iii) injection active: PASS (γ −26.6% ≥ 20%; engagement > 0 every epoch).
- (iv) invariants: PASS (same commit/ckpts/seeds across arms;
  `init_state_hash_shared` identical; no smoke limits).

**`SZAB_INJECT_NO_SMALL_GAIN`.**

## Interpretation and scope

1. **The task redefinition works; the injection does not add on top.** Both
   arms at e6 (AP_s 0.177–0.183, small-F1 0.51–0.52) clearly exceed the
   unweighted ORIG-AB-sz reference (AP_s ≈ 0.16–0.17 at e4, still climbing),
   so the loss retasking itself lifts small-instance performance. The B−A
   difference on the primary metric is negative with a CI that includes zero:
   the warm-fitted P2 field contributes no small-instance gain under full
   co-adaptation.
2. **Consistency with prior evidence.** This closes the route coherently: the
   unweighted ORIG-AB (B ≈ A globally), the frozen probes (T1 semantic
   occupancy negative; DEPTH R1 positive only for *error localization*,
   not for reconstructing geometry SAT already encodes), and now the retasked
   live A/B all point the same way — current fixed-height RGB/geometry BEV
   features carry no instance-level AP-relevant marginal information that the
   satellite stream cannot already extract, even when the objective is
   explicitly retargeted at small buildings.
3. Per Appendix V.6, FAIL closes the "current warm-fit P2 injection form under
   the small-instance-retasked task" route; next work moves to data design
   (UAV render quality / instance separation), not another injection variant.
4. Scope: conclusion holds for the registered configuration (this injection
   form, 6 epochs, single training seed; significance quantifies val-image
   sampling uncertainty only, not seed uncertainty).

## Artifacts

- Gate: `/data1/wangcheng/checkpoint/worktrees_dual-stream-dev_probe/MHBEV-FUS-20260929-SZAB-gate/summary.json`
- Arms: `.../MHBEV-FUS-20260929-SZAB-{A,B}/` (history.json, per-scene bf1/smallF1 per epoch, last.pth, manifest.json)
- Abandoned V1 + postmortems: `.../MHBEV-FUS-20260929-SZAB-{A,B}-abandoned-v1weighting-0930195934/`
- Script: `portable_sam2_explicit_coarse/scripts/ue/finetune_p2_inject_szab.py`
  (gate eval: `scripts/ue/szab_gate_eval.py`); review: `docs/reviews/MHBEV-FUS-20260929_SZAB_implementation_review.md`
