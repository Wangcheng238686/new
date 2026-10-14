#!/usr/bin/env python
"""Inspect observed BEV support and satellite projection, without using labels.

The ground/height difference is a projection correction, not an accuracy
measurement: satellite depth or independent correspondences are unavailable.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mmengine import Config
from data.ue_rgbd_dataset import UERGBDInstanceDataset
from rsprompter.uav.cameras import load_camera, localize_camera, project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--cache", default="/tmp/uav_dual_geometry")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--qa-helper", type=Path, help="Optional standalone Matplotlib layout auditor"
    )
    args = parser.parse_args()
    config = Config.fromfile(str(ROOT / "configs/ablations/ue_rgbd_dual.py"))
    dataset = UERGBDInstanceDataset(
        args.manifest, args.split, config.ue_geometry, cache_root=args.cache
    )
    item = dataset[args.sample_index]
    payload = item["uav_payload"]
    row = dataset.samples[args.sample_index]
    n, size = dataset.geometry["grid_size"], dataset.geometry["image_size"]
    reference = payload["reference_xyz"].numpy()
    observed = payload["observation_support"].numpy()
    valid = payload["height_valid"].numpy()
    available = (payload["candidate_index"].numpy() >= 0).any(1)
    supported = observed & valid & available
    k, e, t, _ = load_camera(dataset.satellite_root / row["satellite_dir"])
    e, t = localize_camera(e, t, np.asarray(row["origin_m"]))
    height_uv, z = project(reference, k, e)
    ground = reference.copy()
    ground[:, 2] = 0
    ground_uv, _ = project(ground, k, e)
    inside = supported & (z > 0) & (height_uv >= 0).all(1) & (height_uv < size).all(1)
    shift = np.linalg.norm(height_uv - ground_uv, axis=1)
    rgb = item["img"].permute(1, 2, 0).numpy()[..., ::-1].astype(np.uint8)
    pixel = payload["project_pixel"].numpy()
    mass = payload["project_mass"].numpy()
    quality = payload["project_quality"].numpy()
    raw_mass = np.bincount(pixel, weights=mass, minlength=size * size).reshape(
        size, size
    )
    mean_quality = np.bincount(
        pixel, weights=mass * quality, minlength=size * size
    ).reshape(size, size) / np.maximum(raw_mass, 1e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    confidence = payload["bev_confidence"].numpy()
    table = np.column_stack(
        (
            reference[inside],
            confidence[inside, 1],
            confidence[inside, 2] * dataset.geometry["height_bin_m"],
            confidence[inside, 3] * dataset.geometry["radius_m"],
            shift[inside],
        )
    )
    np.savetxt(
        args.output_dir / "cell_profile.csv",
        table,
        delimiter=",",
        comments="",
        header="local_x_m,local_y_m,height_m,upper_fraction,selected_bucket_sigma_m,nearest_xy_m,projection_correction_px",
    )
    summary = dict(
        point_index=row["index"],
        cells=int(n * n),
        observed_fraction=float(observed.mean()),
        height_valid_fraction=float(valid.mean()),
        candidate_fraction=float(available.mean()),
        projected_fine_pixel_fraction=float((raw_mass > 0).mean()),
        profile_cells=int(inside.sum()),
        projection_correction_p95_px=(
            float(np.percentile(shift[inside], 95)) if inside.any() else None
        ),
        ground_reference="local origin Z, not true ground truth",
        labels_used=False,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    axes[0, 0].imshow(rgb)
    axes[0, 0].contour(raw_mass > 0, levels=[0.5], colors=["#0072B2"], linewidths=0.6)
    axes[0, 0].set_title("a  Satellite + observed footprint boundary")
    axes[0, 0].set(xlabel="u (px)", ylabel="v (px)")
    extent = [-dataset.geometry["side_m"] / 2, dataset.geometry["side_m"] / 2] * 2
    height = np.ma.masked_where(~supported.reshape(n, n), reference[:, 2].reshape(n, n))
    image = axes[0, 1].imshow(height, origin="lower", extent=extent, cmap="viridis")
    fig.colorbar(
        image,
        ax=axes[0, 1],
        label="Observed representative height (local m)",
        shrink=0.85,
    )
    axes[0, 1].set(
        title="b  BEV: observed, height-valid, candidate available",
        xlabel="Local X (m)",
        ylabel="Local Y (m)",
    )
    chosen = np.flatnonzero(inside)
    chosen = chosen[np.argsort(shift[chosen], kind="stable")[-64:]]
    axes[1, 0].imshow(rgb)
    axes[1, 0].scatter(
        ground_uv[chosen, 0],
        ground_uv[chosen, 1],
        marker="+",
        s=22,
        c="#0072B2",
        label="Z = local origin",
    )
    axes[1, 0].scatter(
        height_uv[chosen, 0],
        height_uv[chosen, 1],
        marker="o",
        s=22,
        facecolors="none",
        edgecolors="#D55E00",
        label="Observed Z",
    )
    axes[1, 0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=9,
        ncol=2,
        frameon=False,
    )
    axes[1, 0].set(
        title="c  64 largest centroid projection corrections",
        xlabel="u (px)",
        ylabel="v (px)",
        xlim=(0, size - 1),
        ylim=(size - 1, 0),
    )
    image = axes[1, 1].imshow(
        np.ma.masked_where(raw_mass == 0, mean_quality), vmin=0, vmax=1, cmap="cividis"
    )
    fig.colorbar(
        image,
        ax=axes[1, 1],
        label="Geometric confidence (not label accuracy)",
        shrink=0.85,
    )
    axes[1, 1].set(
        title="d  Confidence on satellite fine pixels", xlabel="u (px)", ylabel="v (px)"
    )
    fig.suptitle(
        f"Point {row['index']}: geometric inspection; no satellite-depth visibility proof",
        fontsize=12,
    )
    fig.canvas.draw()
    if args.qa_helper:
        import importlib.util

        spec = importlib.util.spec_from_file_location("layout_auditor", args.qa_helper)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        issues = helper.audit_layout(fig)
        (args.output_dir / "layout_qa.json").write_text(json.dumps(issues, indent=2))
        if issues:
            raise ValueError(f"Figure layout audit failed: {issues}")
    fig.savefig(args.output_dir / "alignment.png", dpi=300)
    fig.savefig(args.output_dir / "alignment.pdf", dpi=300)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
