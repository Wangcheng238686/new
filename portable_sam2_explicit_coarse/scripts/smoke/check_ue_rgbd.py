#!/usr/bin/env python
"""Static isolation and discriminating geometric checks for UE RGB-D."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("RSPROMPTER_LIGHT_IMPORT", "1")
import numpy as np
import torch
from mmengine import Config
from rsprompter.uav.contract import geometry_config, model_contract
from rsprompter.uav.geometry import build_geometry
from rsprompter.uav.footprints import satellite_footprints
from rsprompter.uav.cvt import DepthCVT
from rsprompter.uav.projection import BEVToSatellite
from rsprompter.uav.candidates import sparse_candidates
from rsprompter.uav.contract import geometry_signature
from prepare_ue_rgbd import footprint
from data.ue_geometry_cache import validate_depth_floor


def check_static():
    configs = {
        arm: Config.fromfile(str(ROOT / f"configs/ablations/ue_rgbd_{arm}.py"))
        for arm in ("single", "dual", "fpn", "sam")
    }
    models = {arm: cfg.model.to_dict() for arm, cfg in configs.items()}
    assert "uav_backbone" not in models["single"]
    assert "sam_fusion" not in models["fpn"]
    assert "fpn_fusion" not in models["sam"]
    for arm, omitted in (("fpn", "sam_fusion"), ("sam", "fpn_fusion")):
        dual = {k: v for k, v in models["dual"].items() if k != omitted}
        assert models[arm] == dual
    common = {
        k: v
        for k, v in models["dual"].items()
        if not k.startswith("uav_") and k not in ("fpn_fusion", "sam_fusion")
    }
    common["type"] = models["single"]["type"]
    assert common == models["single"]
    return {arm: model_contract(model) for arm, model in models.items()}


def check_geometry():
    config = geometry_config(
        dict(
            side_m=64.0,
            grid_size=16,
            image_size=64,
            source_stride=8,
            depth_sample_stride=4,
            candidates_per_view=4,
            radius_m=12.0,
        )
    )
    k = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1.0]])
    e = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 100.0], [0, 0, 0, 1.0]])
    camera = (k, e, np.linalg.inv(e), (64, 64))
    depth = np.full((64, 64), 100.0, dtype=np.float64)
    arrays, stats = build_geometry(
        [depth] * 5, [camera] * 5, camera, np.zeros(3), config
    )
    assert stats["projected_pixels"] == 64 * 64
    for indices in arrays["candidate_index"]:
        valid = indices[indices >= 0]
        tokens = arrays["anchor_token"][valid]
        assert len(tokens) == len(np.unique(tokens))
    pixel_mass = np.bincount(
        arrays["project_pixel"], weights=arrays["project_mass"], minlength=64 * 64
    )
    assert np.allclose(pixel_mass, 1.0, atol=1e-6)
    from mmengine.structures import BaseDataElement

    payload = BaseDataElement(
        metainfo=dict(geometry_signature=geometry_signature(config))
    )
    for key, value in arrays.items():
        setattr(payload, key, torch.from_numpy(value.copy()))
    model = DepthCVT(config, layers=1, query_chunk=32)
    features = torch.randn(1, 5, 256, 8, 8, requires_grad=True)
    bev = model(features, [payload])
    out, quality, coverage = BEVToSatellite(64)(bev, [payload], (8, 8))
    out.square().mean().backward()
    assert torch.isfinite(out).all() and features.grad.abs().sum() > 0
    assert torch.allclose(coverage, torch.ones_like(coverage), atol=1e-6)
    empty, _ = build_geometry(
        [np.full_like(depth, np.inf)] * 5, [camera] * 5, camera, np.zeros(3), config
    )
    assert len(empty["project_source"]) == 0 and not empty["height_valid"].any()
    for key, value in empty.items():
        setattr(payload, key, torch.from_numpy(value.copy()))
    blank = model(features, [payload])
    assert torch.isfinite(blank).all() and torch.count_nonzero(blank) == 0
    # Different fine pixels in one coarse cell are not an occlusion.
    config2 = {**config, "image_size": 32, "side_m": 64.0}
    k2 = np.array([[32.0, 0, 16], [0, 32.0, 16], [0, 0, 1.0]])
    ref = np.array([[0.0, 0, 0], [4.0, 0, 20.0]])
    confidence = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    projected, _ = satellite_footprints(
        ref, np.ones(2, dtype=bool), confidence, k2, e, np.linalg.inv(e), config2
    )
    assert set(projected["project_source"]) == {0, 1}
    overlapping = ref.copy()
    overlapping[1, :2] = 0
    projected, _ = satellite_footprints(
        overlapping,
        np.ones(2, dtype=bool),
        confidence,
        k2,
        e,
        np.linalg.inv(e),
        config2,
    )
    central = projected["project_pixel"] == 16 * 32 + 16
    assert set(projected["project_source"][central]) == {1}
    # A near high surface on a tilted ray precedes the floor section. The
    # conservative footprint must include its camera apex, not just ground.
    t_tilted = np.eye(4)
    half = np.sqrt(0.5)
    t_tilted[:3, :3] = np.array([[0, half, half], [1, 0, 0], [0, half, -half]])
    t_tilted[:3, 3] = [0, 0, 100]
    camera_tilted = (k, np.linalg.inv(t_tilted), t_tilted, (64, 64))
    bounds = footprint(camera_tilted, 0.0)
    near = t_tilted[:3, 3] + t_tilted[:3, 2] * 10
    assert (near[:2] >= bounds[:2]).all() and (near[:2] <= bounds[2:]).all()
    validate_depth_floor(np.full((64, 64), 10.0), camera_tilted, 0.0)
    try:
        validate_depth_floor(np.full((64, 64), 1000.0), camera_tilted, 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("Spatial split floor violation accepted")
    # Dense nearby facades must not hide a compatible roof token before ranking.
    reference = np.array([[0.0, 0.0, 20.0]])
    facade = np.column_stack((np.linspace(0.01, 0.1, 20), np.zeros(20), np.zeros(20)))
    anchors = np.vstack((facade, [[1.0, 0.0, 20.0], [1.1, 0.0, 20.0]]))
    tokens = np.r_[np.arange(20), 20, 20]
    groups = [anchors] + [np.zeros((0, 3))] * 4
    identities = [tokens] + [np.zeros(0, dtype=np.int64)] * 4
    chosen, _ = sparse_candidates(reference, groups, identities, config)
    assert tokens[chosen[0, 0]] == 20
    assert len(np.unique(tokens[chosen[0, :4]])) == 4
    return dict(
        plane_coverage=1.0,
        candidate_unique=True,
        empty_safe=True,
        coarse_non_occlusion=True,
        fine_near_surface=True,
        feature_gradient=True,
        roof_priority=True,
        frustum_apex_bound=True,
        split_floor_checked=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = dict(contracts=check_static(), geometry=check_geometry())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
