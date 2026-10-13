# UE Raw HERO–UAV Pairset: acceptance record

## Accepted dataset

- **Dataset root:** `portable_sam2_explicit_coarse/data/ue_raw_hero_uav_pairset_20261011/`
- **Accepted scenes:** 1,200.
- **Pair manifest SHA-256:** `f15cd8f5b8d10a69d721546d6cfa1f2cce58af2861d57b1357c5e9ff1e1fe62a`.
- **Payload policy:** this is a non-copying immutable view.  Each accepted scene has a
  `satellite` symlink to the producing satellite payload and a `uav` symlink to the
  producing five-view RGBD payload.  The manifest, not directory enumeration, is the
  sole authority for future probes.

## Sources and full acceptance gate

| Source | Enumerated scene directories | Contract pass |
| --- | ---: | ---: |
| Satellite strict HERO production | 1,499 | 1,208 satellite-valid |
| UAV balanced RGBD production | 4,000 | 1,490 valid among the 1,499 same-name scenes |
| Same point + both contracts | 1,499 named intersection | **1,200 paired-valid** |

Every accepted row passes all of the following: identical `(index, point_id, x, y, z)`
point metadata across sources; satellite producer success; strict EXR validation and
cleanup reports; satellite RGB, clean instance-ID mask, instance manifest, K and W2C;
five fixed UAV yaw views; each view's RGB, packed DeviceZ, EXR camera-Z, K, W2C and
producer quality/depth success state.

The claimed 1,308 satellite count is not the current usable count in the inspected
source.  The current strict satellite contract accepts 1,208.  Of these, eight have an
incomplete UAV counterpart, leaving 1,200 pairs.  No failed payload is included.

## Rejections

- 201 satellite directories lack the strict clean-mask/EXR-cleanup payload family.
- 90 satellite directories explicitly report unsuccessful dataset/mask/strict-instance
  production.
- The remaining satellite rejections are sparse missing RGB, status or camera metadata.
- Nine UAV scenes report incomplete five-view producer success; twenty individual UAV
  views lack camera-Z EXR.  These account for eight satellite-valid pairs excluded from
  the final intersection.

The exhaustive per-scene table and rejection reason counts are in
`portable_sam2_explicit_coarse/outputs/PAIR-AUDIT-20261011-hero-uav-status/`.

## Independent payload check

A deterministic SHA-256 ordered sample of 32 accepted scenes decoded all 384 PNG
payloads required for satellite RGB/instance masks and five-view UAV RGB/DeviceZ at
512×512; zero decode failures.  Full-scene payload existence and producer-side quality
and resolution gates were checked for all 1,200 accepted scenes.

## Use boundary

This is a raw paired RGBD dataset, not the legacy `ue_coco_with_uav` layout.  Its
instance label is `satellite_instance_id_clean.png` plus `instances.json`; future
probe work must either read this contract directly or introduce a separately audited
adapter.  A train/validation split is deliberately not assigned here: it must be
pre-registered with the first experiment, using spatially safe grouping rather than a
random scene split.
