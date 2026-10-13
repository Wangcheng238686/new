"""Explicit UE lineage; never reinterpret an old B0 as the sealed baseline."""
import ast
import json
from pathlib import Path

from rsprompter.architecture_contract import architecture_contract, file_sha256

SOURCE_COMMIT = "95026ae677dd0a8d77a3ee4c5fb68abeb4ea01c1"
BASELINE_ID = "ue-whu-a3sgm-v1"
ROOT = Path(__file__).resolve().parents[1]


def whu_model_config():
    tree = ast.parse((ROOT / "configs/a3sgm_whu1024.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "model" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("sealed WHU model config missing")


def assert_whu_model(config):
    expected = architecture_contract(whu_model_config())
    actual = architecture_contract(config)
    if actual != expected:
        raise RuntimeError("UE baseline must use the exact sealed WHU model contract")
    return actual


def raw_data_contract(root, split_manifest="surfbev_b0_split_v2.json"):
    from data.ue_raw_pair_dataset import UERawPairDataset, PAIR_SHA256, SPLIT_SHA256

    if split_manifest != "surfbev_b0_split_v2.json":
        raise ValueError("new UE baseline requires the leakage-controlled v2 split")
    # Validate complete manifests and spatial rules without opening test samples.
    UERawPairDataset(root, "train", split_manifest=split_manifest)
    UERawPairDataset(root, "val", split_manifest=split_manifest)
    return {
        "dataset_format": "ue_raw_surface", "data_root": str(Path(root).resolve()),
        "image_size": [1024, 1024], "single_class": True, "num_classes": 1,
        "augmentation_contract": "none", "split_manifest": split_manifest,
        "pair_manifest_sha256": PAIR_SHA256,
        "split_manifest_sha256": SPLIT_SHA256[split_manifest],
        "validation": {"split": "val"}, "test": {"split": "test"},
    }


def baseline_metadata(model_config, validation_only=False):
    return {"baseline_id": BASELINE_ID, "source_commit": SOURCE_COMMIT,
            "architecture_contract": assert_whu_model(model_config),
            "validation_only": bool(validation_only)}


def validate_snapshot(snapshot):
    meta = snapshot.get("ue_baseline", {})
    expected = architecture_contract(whu_model_config())
    if (meta.get("baseline_id") != BASELINE_ID or meta.get("source_commit") != SOURCE_COMMIT
            or meta.get("architecture_contract") != expected
            or snapshot.get("architecture_contract") != expected):
        raise RuntimeError("checkpoint does not identify the sealed UE WHU baseline")
    data = snapshot.get("data_config", {})
    from data.ue_raw_pair_dataset import PAIR_SHA256, SPLIT_SHA256
    if (data.get("dataset_format") != "ue_raw_surface"
            or data.get("split_manifest") != "surfbev_b0_split_v2.json"
            or data.get("augmentation_contract") != "none"
            or data.get("pair_manifest_sha256") != PAIR_SHA256
            or data.get("split_manifest_sha256") != SPLIT_SHA256["surfbev_b0_split_v2.json"]):
        raise RuntimeError("UE checkpoint data contract mismatch")
    return meta


def checkpoint_identity(path, checkpoint):
    snapshot = checkpoint.get("config_snapshot", {})
    meta = validate_snapshot(snapshot)
    return {"schema": "ue-baseline-identity-v1", **meta,
            "checkpoint_sha256": file_sha256(str(path)),
            "data_contract": {key: snapshot["data_config"][key] for key in (
                "pair_manifest_sha256", "split_manifest_sha256", "augmentation_contract")}}


def verify_identity(path, checkpoint, manifest):
    expected = json.loads(Path(manifest).read_text())
    actual = checkpoint_identity(path, checkpoint)
    if actual != expected:
        raise RuntimeError("baseline identity manifest mismatch")
    return actual


def guard_cache(path, contract):
    """A populated cache without provenance is never adopted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise RuntimeError("cache contract mismatch")
    else:
        if any(path.parent.iterdir()):
            raise RuntimeError("cache contains artifacts without a contract")
        with path.open("x") as handle:
            json.dump(contract, handle, indent=2)
