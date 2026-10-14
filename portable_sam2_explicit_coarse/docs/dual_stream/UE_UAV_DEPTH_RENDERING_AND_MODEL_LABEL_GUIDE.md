# UE UAV depth rendering and model-label guide

This document is the implementation contract for the UAV depth files under:

```text
/data/wangcheng/ue/CitySample/CitySample_5.5/data/Render/UAV_DepthBackfill_3502_3664
```

For the accepted 159-point model-audit subset, use this guide together with:

```text
HANDOFF_MODEL_AUDIT_UE_RGBD_SATELLITE_159.md
model_audit_ue_rgbd_satellite_159.json
```

## 1. Which depth file the model should use

Each accepted UAV view contains two depth representations:

```text
depth_devicez_packed.0000.png  # UE-native nonlinear depth, archival/intermediate
depth_camera_z.0000.exr        # model-facing metric depth
```

The model should normally read `depth_camera_z.0000.exr`.

Its contract is:

- resolution: 512×512, aligned pixel-for-pixel with `oblique_rgb.0000.png`;
- storage: OpenEXR, one channel named `R`, float32;
- quantity: perspective-camera **camera-space Z**, not Euclidean ray distance;
- unit: Unreal world unit, which is centimeters in this project;
- camera convention: CV axes `x=right`, `y=down`, `z=forward`;
- invalid/background handling: consumers must reject non-finite, non-positive or application-out-of-range values.

Do not train directly from the visible intensity of `depth_devicez_packed.0000.png`, and do not
apply per-image min/max normalisation if metric geometry is required.

## 2. How the depth was rendered in UE

The implementation is in `citysample_render_v10_4_dual_view_pair_dataset.py`.

For every group of five points:

1. One PIE/MRQ sequence renders 5 points × 5 UAV cameras = 25 RGB frames.
2. The 25 `CineCameraActor` objects remain alive after the RGB PIE finishes.
3. The pipeline waits 2 seconds after MRQ finalisation.
4. A `SceneCapture2D` is placed at the exact location and rotation of each RGB camera.
5. The capture uses:
   - `capture_source = SCS_DEVICE_DEPTH`;
   - perspective projection;
   - the same horizontal field of view derived from 55.4 mm focal length and 36 mm sensor width;
   - a 512×512 `RTF_RGBA8` render target;
   - four warm-up captures for every view.
6. Only one view is submitted per editor tick. This is required on Linux/Vulkan; submitting all
   25 synchronously in one callback exhausts temporary RHI allocations.
7. The capture is exported as `depth_devicez_packed.0000.png`.
8. After UE exits, `finalize_uav_depth_backfill.py` invokes
   `decode_ue_device_depth_exr.cpp` to create `depth_camera_z.0000.exr`.

Depth is intentionally captured **after the matching RGB PIE**, rather than before it. Earlier
pre-RGB experiments omitted streamed scene content such as bridges. The accepted path keeps the
same RGB camera actors and captures depth immediately after RGB finalisation.

`depth_scenecapture_metadata.json` records the depth camera location, rotation, capture phase,
FOV and warm-up count. The finaliser compares this pose with `camera_extrinsics.json` using an
absolute tolerance of `1e-4`; a point is accepted only when all five views pass.

## 3. Meaning of DeviceZ and the packed PNG

`SCS_DEVICE_DEPTH` does not directly return centimeters. It returns the nonlinear depth-buffer
quantity DeviceZ. Under the reversed infinite perspective projection used by this pipeline:

```text
camera_z_cm = near_clip_cm / DeviceZ
```

The current project conversion uses:

```text
near_clip_cm = 10.0
```

UE's SceneCapture shader encodes `1 - DeviceZ` into the PNG RGB channels. For 8-bit channel
values `R8`, `G8`, `B8`, first normalise them:

```text
r = R8 / 255
g = G8 / 255
b = B8 / 255
```

Then decode:

```text
packed  = r + g / 255 + b / 65025
DeviceZ = 1 - packed
Z_cm    = 10 / DeviceZ
```

Pixels with `DeviceZ <= 1e-9` are written as infinity by the reference converter. The packed PNG
is retained so EXR files can be regenerated and the conversion can be audited. It is not a
display-ready depth label.

Reference implementations:

```text
decode_ue_device_depth.py       # readable NumPy/Pillow implementation
decode_ue_device_depth_exr.cpp  # production PNG-to-float32-EXR converter
finalize_uav_depth_backfill.py  # batch conversion and contract validation
```

The formula above is specific to the accepted SceneCapture/DeviceZ path. Do not apply it to the
earlier MRQ `WorldDepth` experiments; those are a different render pass and were not adopted for
this dataset.

## 4. Camera-Z versus Euclidean ray distance

For pixel `(u,v)`, define the unnormalised pinhole ray:

```text
r = K^-1 [u, v, 1]^T
```

Because the EXR stores camera-Z:

```text
X_cam = Z * r
```

Do **not** normalise `r`. If the stored value were Euclidean ray distance, the correct equation
would instead be `X_cam = distance * r / ||r||`.

If a model explicitly needs Euclidean range, convert camera-Z per pixel:

```text
range = Z * sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)
```

Conversely, if range is available and camera-Z is required:

```text
Z = range / sqrt(((u-cx)/fx)^2 + ((v-cy)/fy)^2 + 1)
```

## 5. Reading the EXR

Recommended Python dependency:

```bash
pip install OpenEXR
```

Minimal loader using the modern OpenEXR Python binding:

```python
from pathlib import Path
import OpenEXR
import numpy as np


def read_camera_z_exr(path: str | Path):
    exr = OpenEXR.File(str(path))
    channels = exr.channels()
    if "R" not in channels:
        raise ValueError(f"EXR has no R channel: {path}; channels={list(channels)}")
    z_cm = np.asarray(channels["R"].pixels, dtype=np.float32)
    if z_cm.ndim != 2:
        z_cm = np.squeeze(z_cm)
    if z_cm.shape != (512, 512):
        raise ValueError(f"unexpected depth shape {z_cm.shape}: {path}")
    valid = np.isfinite(z_cm) & (z_cm > 0.0)
    z_m = z_cm / 100.0
    return z_cm, z_m, valid
```

Do not assume a generic image library preserves named EXR channels or float32 precision. If a
different EXR library is used, verify all of the following on one file before batch conversion:

- channel name is `R`;
- dtype is float32 (or is converted without quantisation);
- shape is `(512,512)` with no transpose or vertical flip;
- center value and min/max agree with the production conversion report;
- no gamma or colour-management transform is applied.

## 6. Reading K and camera-to-world

`camera_intrinsics.json` contains `fx`, `fy`, `cx`, `cy`, `width` and `height`. Build:

```python
import json
import numpy as np


def load_camera(intrinsics_json, extrinsics_json):
    with open(intrinsics_json, "r", encoding="utf-8") as f:
        intr = json.load(f)
    with open(extrinsics_json, "r", encoding="utf-8") as f:
        extr = json.load(f)
    K = np.array([
        [intr["fx"], 0.0, intr["cx"]],
        [0.0, intr["fy"], intr["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    world_to_camera = np.asarray(extr["world_to_camera_cv"], dtype=np.float32)
    camera_to_world = np.asarray(extr["camera_to_world_cv"], dtype=np.float32)
    return K, world_to_camera, camera_to_world
```

The extrinsics contract is version 2 and uses column vectors:

```text
X_cam_h   = world_to_camera_cv @ X_world_h
X_world_h = camera_to_world_cv @ X_cam_h
```

No image flip is part of this contract.

## 7. Vectorised backprojection to world coordinates

```python
import numpy as np


def depth_to_world(z_cm, valid, K, camera_to_world):
    h, w = z_cm.shape
    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        indexing="xy",
    )
    x = (u - K[0, 2]) / K[0, 0] * z_cm
    y = (v - K[1, 2]) / K[1, 1] * z_cm
    points_cam = np.stack([x, y, z_cm], axis=-1)
    points_world = (
        points_cam @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    )
    return points_world[valid]  # UE world coordinates, centimeters
```

For a BEV grid, convert centimeters to meters once, document the world origin, and splat only
valid points. Use the same transformation for features and labels.

## 8. Producing model labels safely

A recommended preprocessing product per UAV view is:

```text
depth_z_m:  float32 [H,W]
valid_mask: bool    [H,W]
K:          float32 [3,3]
c2w:        float32 [4,4]
```

Rules:

1. Preserve the raw EXR; generate model labels into a separate cache.
2. Convert centimeters to meters exactly once.
3. Keep an explicit validity mask. Do not replace infinity with zero without retaining the mask.
4. If resizing RGB/depth, update `fx`, `fy`, `cx`, `cy` by the same pixel scale.
5. Avoid ordinary bilinear interpolation across depth discontinuities. Use nearest-neighbour or
   validity-aware interpolation/downsampling.
6. Do not perform independent random crops or flips on RGB, depth and K/extrinsics.
7. If using inverse depth or log depth as a training target, derive it only on valid pixels and
   retain metric camera-Z for geometry reconstruction and audit.
8. Confirm one sample by projecting reconstructed world points back through
   `world_to_camera_cv` and `K`; valid pixels should return to their original coordinates.

## 9. Dataset acceptance and known exclusions

The authoritative accepted set is the `samples` array in
`model_audit_ue_rgbd_satellite_159.json`. It contains 159 points and 795 UAV views.

Do not discover samples by blindly scanning the root. The root also contains four rejected point
directories: `3575`, `3577`, `3578`, `3579`. Their `view_yaw_288` RGB is obstructed/invalid in
both the old render and the backfill, so they are excluded from five-view training.

For every accepted point, `finalize_uav_depth_backfill.py` verified:

- five RGB PNGs;
- five packed DeviceZ PNGs;
- five float32 camera-Z EXRs;
- five intrinsics and five extrinsics files;
- post-RGB depth capture metadata;
- matching RGB/depth camera locations and rotations;
- successful point and per-view status contracts.

## 10. Satellite-alignment warning

UAV views have explicit per-view K and extrinsics. The corresponding strict satellite run does
not have per-point camera matrices (`camera_params_written=false`); it has documented common
camera settings and point centers. Therefore a model-side agent must not claim exact
UAV-BEV-to-satellite projection until the canonical top-down satellite matrix and UE/CV axis
conventions have been independently validated. This limitation is separate from the UAV depth
contract described here.
