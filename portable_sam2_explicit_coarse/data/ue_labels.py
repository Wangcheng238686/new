"""Validate visible local IDs against the per-image instance table."""

import json
from pathlib import Path

import cv2
import numpy as np


def validate_instance_map(satellite, ids=None):
    satellite = Path(satellite)
    if ids is None:
        ids = cv2.imread(
            str(satellite / "satellite_instance_id_clean.png"), cv2.IMREAD_UNCHANGED
        )
    if ids is None or ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"Invalid satellite instance map: {satellite}")
    instances = json.loads((satellite / "instances.json").read_text())["instances"]
    mapping = {
        int(item["local_instance_id"]): int(item["global_instance_id"])
        for item in instances
    }
    if len(mapping) != len(instances) or any(key <= 0 for key in mapping):
        raise ValueError(f"Invalid local/global instance mapping: {satellite}")
    unknown = set(map(int, np.unique(ids))) - {0} - mapping.keys()
    if unknown:
        raise ValueError(
            f"Visible instance IDs have no mapping: {satellite}; ids={sorted(unknown)}"
        )
    return mapping
