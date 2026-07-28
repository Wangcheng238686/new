# Migration manifest

## Base project

- Source: `portable_sam2_fusion_new`
- Commit: `eee751738d0c4e9f5fa7fdc017cec56bafe11584`
- Frozen copy: `legacy_baseline/portable_sam2_fusion_new`
- Integrity list: `legacy_baseline/SOURCE_SHA256SUMS`

The frozen copy is intentionally not imported by the new implementation.

## Selected reference implementation

Source snapshot:

`/home/wangcheng2021/project/portable_sam2_single_v7_main_cleaned/portable_sam2_single_v7_clean`

Selected code only:

- `RSSAM2PAFPN` and the explicit SAM2 MaskHead wiring from
  `rsprompter/models_sam2.py`;
- proposal-compatible RoIHead wiring from `rsprompter/models.py`;
- `rsprompter/shape_prior.py`;
- `rsprompter/dense_prompt_utils.py`;
- `rsprompter/coarse_mask_loss.py`;
- `rsprompter/densebr.py`;
- strict checkpoint helper `rsprompter/ckpt_utils.py`.

Explicitly excluded:

- SABL bbox head and SABL refined-box dependency;
- IIMR and topology memory;
- UAV/multi-view features;
- experimental DecoderBR, directional feedback and duplicate DenseBR losses;
- reference project datasets, subset protocol and reported training schedule.

## Reproducibility rule

Historical claims are reproduced only from `legacy_baseline`. New-mainline
experiments use distinct configuration and checkpoint directories and must not
overwrite historical checkpoints. Every new run should record:

- config dump;
- environment variables controlling architecture;
- SAM2 checkpoint SHA256;
- git commit;
- random seed, world size, batch size and accumulation;
- validation protocol and COCO evaluator settings.

