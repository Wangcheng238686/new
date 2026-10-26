"""Additive model contracts; legacy IDs and compatibility remain frozen."""

import hashlib
import json
import math

from ..architecture_contract import architecture_contract as legacy_contract
from ..architecture_contract import architecture_fingerprint

GEOMETRY_SCHEMA = "uav_depth_xy_surface_v2"
PAYLOAD_SCHEMA = "ue_rgbd_instances_v1"
DEFAULT_GEOMETRY = dict(
    side_m=512.0,
    grid_size=128,
    image_size=512,
    source_stride=8,
    radius_m=8.0,
    candidates_per_view=8,
    depth_sample_stride=4,
    height_bin_m=2.0,
    min_height_points=3,
    collision_band_m=2.0,
)


def geometry_config(config):
    unknown = set(config) - set(DEFAULT_GEOMETRY)
    if unknown:
        raise ValueError(f"Unknown geometry settings: {sorted(unknown)}")
    result = {**DEFAULT_GEOMETRY, **dict(config)}
    for key, value in result.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"Geometry {key} must be positive")
    for key in (
        "grid_size",
        "image_size",
        "source_stride",
        "candidates_per_view",
        "depth_sample_stride",
        "min_height_points",
    ):
        if int(result[key]) != result[key]:
            raise ValueError(f"Geometry {key} must be an integer")
        result[key] = int(result[key])
    if result["image_size"] % result["source_stride"]:
        raise ValueError("Source stride must divide the UAV image size")
    return result


def geometry_signature(config):
    payload = {"schema": GEOMETRY_SCHEMA, **geometry_config(config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def model_contract(config):
    if config.get("type") != "RSPrompterUAVDual":
        return legacy_contract(config)
    geo = geometry_config(config["uav_bev"]["geometry"])
    routes = [name for name in ("fpn", "sam") if f"{name}_fusion" in config]
    if not routes:
        raise ValueError("A dual model requires at least one fusion route")
    return dict(
        schema_version=4,
        architecture_id=f"uav_cvt_{geo['grid_size']}_{'_'.join(routes)}_roi_local",
        model_fingerprint=architecture_fingerprint(config),
    )
