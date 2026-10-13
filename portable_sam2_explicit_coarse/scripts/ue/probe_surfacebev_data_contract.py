#!/usr/bin/env python3
"""SURFBEV-D0-Data-R1: read-only contract probe for the raw Hero/UAV pair set.

This is deliberately *not* a model experiment and it does not certify metric
depth.  It establishes three narrower facts before a model is designed:

1. ``satellite_instance_id_clean.png`` can be read as raster local-instance
   masks whose IDs agree with ``instances.json``.
2. The five supplied K / W2C / C2W camera contracts are mutually inverse.
3. The packed DeviceZ has a reproducible cross-view geometric scale candidate.

For (3) we test the common UE infinite-far reversed-Z form ``z_camera=A/z_d``
over an explicitly supplied grid of A values.  The winning A is only a
*cross-view consistency candidate*.  It is not a substitute for the controlled
plane render required by SURFBEV-D0, and the report says so prominently.

No model weights, labels, or source data are modified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_ROOT = "portable_sam2_explicit_coarse/data/ue_raw_hero_uav_pairset_20261011"
DEFAULT_OUT = "portable_sam2_explicit_coarse/outputs/SURFBEV-D0-Data-R1"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_clean_ids(path):
    """Read a grayscale/RGB/RGBA local-ID PNG without silently mixing channels."""
    a = np.asarray(Image.open(path))
    if a.ndim == 2:
        return a.astype(np.int64)
    rgb = a[..., :3]
    if not (np.array_equal(rgb[..., 0], rgb[..., 1]) and
            np.array_equal(rgb[..., 0], rgb[..., 2])):
        raise ValueError(f"clean ID image is not grayscale-coded: {path}")
    return rgb[..., 0].astype(np.int64)


def packed_device_z(path):
    """Decode the producer's R+G/255+B/65025 packed one-minus-DeviceZ."""
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    packed = (a[..., 0] + a[..., 1] / 255.0 + a[..., 2] / 65025.0) / 255.0
    device_z = 1.0 - packed
    return device_z


def camera(view_dir):
    intr = load_json(view_dir / "camera_intrinsics.json")
    ext = load_json(view_dir / "camera_extrinsics.json")
    k = np.array([[intr["fx"], 0.0, intr["cx"]],
                  [0.0, intr["fy"], intr["cy"]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return k, np.asarray(ext["world_to_camera_cv"], dtype=np.float64), \
        np.asarray(ext["camera_to_world_cv"], dtype=np.float64)


def geometry_for_scene(scene_dir, scales_cm, stride):
    views = sorted((scene_dir / "uav").glob("view_yaw_*"))
    cams = []
    inv_err = []
    for vd in views:
        k, w2c, c2w = camera(vd)
        inv_err.append(float(np.max(np.abs(w2c @ c2w - np.eye(4)))) )
        cams.append((vd, k, w2c, c2w, packed_device_z(vd / "depth_devicez_packed.0000.png")))
    if len(cams) != 5:
        raise ValueError(f"expected five UAV views, got {len(cams)}")

    # Per candidate scale: log depth residuals on cross-view mutually visible
    # samples.  Occluded projections are expected, so report support too.
    result = {str(a): [] for a in scales_cm}
    support = {str(a): 0 for a in scales_cm}
    for si, (_, k, _, c2w, dz) in enumerate(cams):
        h, w = dz.shape
        vv, uu = np.mgrid[stride // 2:h:stride, stride // 2:w:stride]
        u = uu.reshape(-1).astype(np.float64)
        v = vv.reshape(-1).astype(np.float64)
        src_d = dz[v.astype(np.int64), u.astype(np.int64)]
        valid = np.isfinite(src_d) & (src_d > 1e-8) & (src_d < 1.0)
        u, v, src_d = u[valid], v[valid], src_d[valid]
        rays = np.stack([(u - k[0, 2]) / k[0, 0],
                         (v - k[1, 2]) / k[1, 1], np.ones_like(u)], axis=1)
        for a in scales_cm:
            z = a / src_d
            pc = rays * z[:, None]
            pw = pc @ c2w[:3, :3].T + c2w[:3, 3]
            for ti, (_, tk, tw2c, _, tdz) in enumerate(cams):
                if ti == si:
                    continue
                tc = pw @ tw2c[:3, :3].T + tw2c[:3, 3]
                good = tc[:, 2] > 1e-5
                tu = np.rint(tk[0, 0] * tc[:, 0] / tc[:, 2] + tk[0, 2]).astype(np.int64)
                tv = np.rint(tk[1, 1] * tc[:, 1] / tc[:, 2] + tk[1, 2]).astype(np.int64)
                good &= (tu >= 0) & (tu < tdz.shape[1]) & (tv >= 0) & (tv < tdz.shape[0])
                if not np.any(good):
                    continue
                observed = tdz[tv[good], tu[good]]
                good2 = np.isfinite(observed) & (observed > 1e-8) & (observed < 1.0)
                # log ratio is scale-neutral as an error unit and robust across range.
                residual = np.abs(np.log(tc[good, 2][good2] / (a / observed[good2])))
                result[str(a)].append(residual)
                support[str(a)] += int(residual.size)
    summary = {}
    for a in scales_cm:
        x = np.concatenate(result[str(a)]) if result[str(a)] else np.empty(0)
        summary[str(a)] = dict(
            n_cross_view_samples=int(x.size),
            median_abs_log_depth_ratio=(float(np.median(x)) if x.size else None),
            p90_abs_log_depth_ratio=(float(np.quantile(x, .9)) if x.size else None),
            frac_within_2pct=(float(np.mean(x <= math.log(1.02))) if x.size else None),
        )
    return dict(max_w2c_c2w_identity_error=max(inv_err), scale_candidates=summary)


def label_for_scene(scene_dir):
    sat = scene_dir / "satellite"
    ids = read_clean_ids(sat / "satellite_instance_id_clean.png")
    j = load_json(sat / "instances.json")
    declared = {int(x["local_instance_id"]) for x in j["instances"]}
    present = set(np.unique(ids[ids > 0]).astype(int).tolist())
    unexpected = sorted(present - declared)
    missing = sorted(declared - present)
    # An instance can be selected but completely occluded; missing is reported,
    # not called a failure.  Any unexpected raster ID is a hard contract error.
    return dict(shape=list(ids.shape), declared_count=len(declared), present_count=len(present),
                unexpected_ids=unexpected, declared_but_not_visible=missing,
                foreground_pixels=int((ids > 0).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-scenes", type=int, default=1200,
                    help="0 means all accepted scenes")
    ap.add_argument("--geometry-scenes", type=int, default=120,
                    help="deterministic prefix used for cross-view test")
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--scales-cm", default="1,2,5,10,20,50,100")
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    manifest = root / "accepted_pairs.jsonl"
    records = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines() if x]
    # The materialized-pair manifest deliberately records source paths and its
    # stable local key is ``scene_dir`` (not the legacy dataset's ``scene``).
    records = sorted(records, key=lambda x: x["scene_dir"])
    if args.max_scenes:
        records = records[:args.max_scenes]
    scales = [float(x) for x in args.scales_cm.split(",")]
    label_rows, geometry_rows, bad = [], [], []
    for n, rec in enumerate(records):
        scene_name = rec["scene_dir"]
        scene = root / "scenes" / scene_name
        try:
            lr = label_for_scene(scene)
            lr["scene"] = scene_name
            label_rows.append(lr)
            if n < args.geometry_scenes:
                gr = geometry_for_scene(scene, scales, args.stride)
                gr["scene"] = scene_name
                geometry_rows.append(gr)
        except Exception as exc:  # keep an auditable rejection rather than silently skipping
            bad.append(dict(scene=scene_name, error=repr(exc)))
    unexpected = [r for r in label_rows if r["unexpected_ids"]]
    # Aggregate one score per scene per candidate, so long/short samples cannot dominate.
    candidates = {}
    for a in scales:
        vals = [r["scale_candidates"][str(a)]["median_abs_log_depth_ratio"] for r in geometry_rows]
        vals = [x for x in vals if x is not None]
        candidates[str(a)] = dict(
            scenes_with_support=len(vals),
            median_of_scene_medians=(float(np.median(vals)) if vals else None),
            p90_of_scene_medians=(float(np.quantile(vals, .9)) if vals else None),
        )
    usable = [(float(v["median_of_scene_medians"]), k) for k, v in candidates.items()
              if v["median_of_scene_medians"] is not None]
    winner = min(usable)[1] if usable else None
    report = dict(
        probe="SURFBEV-D0-Data-R1",
        scope="read-only raw-data contract; not an AP experiment",
        accepted_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        n_manifest_scenes=len(records), n_label_rows=len(label_rows),
        n_geometry_rows=len(geometry_rows), errors=bad,
        label_contract=dict(unexpected_id_scenes=len(unexpected),
                            all_id_shapes=sorted({tuple(x["shape"]) for x in label_rows}),
                            median_foreground_pixels=float(np.median([x["foreground_pixels"] for x in label_rows])),
                            note="declared-but-not-visible is informational; unexpected raster IDs are failures"),
        camera_contract=dict(max_w2c_c2w_identity_error=max([x["max_w2c_c2w_identity_error"] for x in geometry_rows], default=None),
                             threshold=1e-9),
        devicez_cross_view_candidate=dict(form="z_camera_cm=A/device_z (candidate only)",
                                          scales_cm=scales, per_scale=candidates,
                                          winning_scale_cm=winner,
                                          non_claim="This cannot validate absolute metric depth. A controlled-plane render remains the D0 blocking gate."),
        verdict=("PASS_FOR_READER_AND_RELATIVE_GEOMETRY_ONLY" if not bad and not unexpected else "FAIL"),
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "per_scene_labels.jsonl").write_text("".join(json.dumps(x) + "\n" for x in label_rows), encoding="utf-8")
    (out / "per_scene_geometry.jsonl").write_text("".join(json.dumps(x) + "\n" for x in geometry_rows), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
