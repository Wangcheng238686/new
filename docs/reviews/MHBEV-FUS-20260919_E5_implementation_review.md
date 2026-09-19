# MHBEV-FUS-20260919-E5 — independent implementation review

**Verdict: HOLD — do not launch the current script.**  The design implements nearly
all Appendix K controls, but a deterministic fit-scene selection defect means this
exact version will fail closed before fitting.  It must be corrected and independently
re-reviewed before an E5 run.  This review was static/read-only; no E5 feature scan,
detector inference, or training was run.

Reviewed on 2026-09-19 against Appendix K in `EXPERIMENT_GOVERNANCE.md` and
`portable_sam2_explicit_coarse/scripts/ue/probe_fusion_boundary_confirmation.py`.

## Confirmed implementation coverage

- **Lineage and split:** lines 376--393 load dual first and overlay single-stream
  matching keys; the dual load is strict and the single overlay fails on any unexpected
  key or missing non-UAV/BEV key.  Lines 395--402 reconstruct the fixed ratio/seed
  subset, require 128/259 scenes, select the front 80 fit scenes, archive all 128
  subset IDs plus the fit/val IDs, and assert no train/val intersection.
- **All required comparison arms:** lines 67, 496--503 and 513--523 fit/evaluate
  `SAT`, `SAT+DBEV_correct`, `SAT+DBEV_mis`, and both 512-component PCA concatenations.
  PCA and standardization fit only from the train arrays; all val transforms are
  read-only.  The two PCA fits are separate per correct/mis descriptor condition, a
  reasonable interpretation of “每一套特征”; both are deterministic
  randomized-SVD fits with a fixed seed and fixed single-thread environment.
- **Geometry/labels:** lines 175--203 use the normalized full-tile 360m, z=0,
  W2C/K FP32, autocast-disabled projection contract; the 90-degree row-vector rotation
  matches the existing E4 convention.  Four 256-channel frozen UAV-FPN levels are
  bilinearly sampled and valid-view averaged.  The 128x128 union-mask signed-distance
  target exactly matches Appendix K.
- **No-intervention parity:** fresh dual-stream samples receive `drone_images`,
  `intrinsics`, `extrinsics`, and singular `scene_center` through the trainer helper
  (lines 146--172).  Each selected scene gets normal `model.predict` before/after
  frozen extraction; typed hashes cover bbox, score, label and mask.  The implementation
  also fails closed on proposals being supplied/written, parameter-state drift,
  eval-state drift, and repeated feature hash drift.
- **Statistics and decision:** lines 528--550 use all 259 per-scene values and a
  10,000-draw paired percentile bootstrap with seed 20260926.  It applies the three
  registered lower-CI gates (`.02`, `.02`, `.01`) and only returns `E5_PASS` when every
  invariant also passes.  Contract failures raise and write an `INVALID` terminal
  summary.
- **Archive:** manifest, environment/command, git state, checkpoint/config/resolved
  config/data fingerprints, scene IDs/seeds, typed hashes, geometry contract, sampled
  index hashes, telemetry, fitted PCA/scaler/ridge coefficients, per-scene CSV,
  bootstrap/summary JSON, and stdout log are all produced.  `py_compile` and
  `git diff --check` pass.

## Blocking defect

Appendix K says the fit set is the first 80 **valid** scenes of the fixed 128-scene
subset.  `UEDualStreamDataset` validity only verifies satellite/annotation/camera/center
files; it does not require a polygon label.  I reconstructed the fixed seed-44 subset
from the actual dataset:

- 712 source train scenes -> 128 fixed subset scenes;
- among `train.ids[:80]`, scene **`002646`** has no usable polygon;
- all 259 val scenes have at least one polygon.

The script takes `train.ids[:80]` at line 400 and calls `signed_boundary_label` for
each at lines 442--482.  That function intentionally raises on an empty union mask at
lines 194--203.  Thus the current script deterministically terminates `INVALID` on
`002646`; it cannot produce the registered 80-scene fit artifact.  The earlier E4
script treated `n_used` as the count of label-bearing scenes, so this is also a lineage
drift from the practical meaning of “有效场景”.

## Required correction and next authorization

Before launch, revise only the fit-scene selector so it deterministically walks the
fixed 128-scene subset in order, retains the first 80 scenes with a non-empty valid
polygon union, records both the skipped ID(s) and the resulting 80 IDs in the manifest,
and hard-asserts exactly 80.  Do **not** skip any val scene; the observed val set is
already 259/259 label-bearing.  The Appendix K wording supports this “valid scene”
interpretation; if the team instead intends literal first 80 input-file-valid scenes,
the pre-registration itself must be amended before any run because the specified
readout target is undefined for `002646`.

After that minimal fix, perform a focused independent re-review of the selection and
manifest branch.  The single authorization then is: **launch the full frozen E5
confirmation only after the re-review is PASS and the code/Appendix K state is
committed under the governance rule.**  No detector, UAV, data, hot-start, or training
action is authorized by this HOLD review.

---

## Focused re-review — selector correction (2026-09-19)

**Updated verdict: PASS — the prior blocking defect is corrected; E5 full frozen
confirmation is authorized to launch once the Appendix K/script/review state is
committed as required by experiment governance.**  This PASS authorizes only the
pre-registered E5 read-only scan, not a detector/UAV/data modification, hot-start, or
training run.

I reviewed the selector-only diff and independently reconstructed the official dataset
selection using the same sorted input IDs, ratio `.1797752808988764`, seed 44, polygon
group rasterization, and 128x128 nearest-neighbor P3 union-mask predicate.  The fixed
128-scene subset has exactly one degenerate label-bearing candidate:

| source index in fixed 128 subset | scene ID | outcome |
|---:|---|---|
| 63 | `002646` | skipped: all-zero P3 union mask |

The first 80 nondegenerate records have source indices ending at 80, with final selected
scene `002746`; the selector therefore returns exactly 80 records.  The full fixed
128 subset contains no additional degenerate P3 union mask scene.  The 259 val scenes
remain 259/259 nondegenerate and are not filtered.

Appendix K now pre-registers this exact “first 80 nondegenerate P3 union-mask scenes”
rule, including archival of the skip ID and original index.  The implementation matches
it: `has_non_degenerate_boundary_label` uses the same P3 union construction as the
actual label function; `fit_records` retains `(source_index, scene_id)`; manifest records
the complete subset, selected IDs, source indices and skips; scan, scene-center archive,
train hashing, sample seed (`20260925 + source_index`) and sample-row provenance all use
the same source index.  Thus the prior index/seed drift risk is closed.

Focused verification also passed `python -m py_compile` and `git diff --check`.  No E5
inference, feature extraction, training, or data write was performed during this
re-review.
