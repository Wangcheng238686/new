"""Explicit UE lineage; never reinterpret an old B0 as the sealed baseline."""
import ast
import json
from pathlib import Path

from rsprompter.architecture_contract import architecture_contract, file_sha256

SOURCE_COMMIT = "95026ae677dd0a8d77a3ee4c5fb68abeb4ea01c1"
BASELINE_ID = "ue-whu-a3sgm-v1"
# The 1024 seal stays canonical; the 512 entry is a declared input-protocol
# variant (EXP-0007, native-GSD UE) with its own architecture fingerprint
# because the fingerprint hashes every model-config key including sizes.
BASELINE_CONFIGS = {
    "ue-whu-a3sgm-v1": "configs/a3sgm_whu1024.py",
    "ue-whu-a3sgm-512-v1": "configs/ue_whu_a3sgm_512.py",
}
ROOT = Path(__file__).resolve().parents[1]


def whu_model_config(baseline_id=BASELINE_ID):
    tree = ast.parse((ROOT / BASELINE_CONFIGS[baseline_id]).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "model" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("sealed WHU model config missing")


def baseline_id_for(config):
    contract = architecture_contract(config)
    for baseline_id in BASELINE_CONFIGS:
        if contract == architecture_contract(whu_model_config(baseline_id)):
            return baseline_id
    return None


def assert_whu_model(config):
    matched = baseline_id_for(config)
    if matched is None:
        raise RuntimeError("model does not match any registered sealed WHU baseline contract")
    return architecture_contract(config), matched


def raw_data_contract(root, split_manifest="surfbev_b0_split_v2.json", image_size=(1024, 1024)):
    from data.ue_raw_pair_dataset import UERawPairDataset, PAIR_SHA256, SPLIT_SHA256

    if split_manifest != "surfbev_b0_split_v2.json":
        raise ValueError("new UE baseline requires the leakage-controlled v2 split")
    # Validate complete manifests and spatial rules without opening test samples.
    UERawPairDataset(root, "train", split_manifest=split_manifest)
    UERawPairDataset(root, "val", split_manifest=split_manifest)
    return {
        "dataset_format": "ue_raw_surface", "data_root": str(Path(root).resolve()),
        "image_size": [int(image_size[0]), int(image_size[1])], "single_class": True, "num_classes": 1,
        "augmentation_contract": "none", "split_manifest": split_manifest,
        "pair_manifest_sha256": PAIR_SHA256,
        "split_manifest_sha256": SPLIT_SHA256[split_manifest],
        "validation": {"split": "val"}, "test": {"split": "test"},
    }


def baseline_metadata(model_config, validation_only=False):
    contract, matched = assert_whu_model(model_config)
    return {"baseline_id": matched, "source_commit": SOURCE_COMMIT,
            "architecture_contract": contract,
            "validation_only": bool(validation_only)}


def validate_snapshot(snapshot):
    meta = snapshot.get("ue_baseline", {})
    baseline_id = meta.get("baseline_id")
    if baseline_id not in BASELINE_CONFIGS:
        raise RuntimeError("checkpoint does not identify a registered sealed UE WHU baseline")
    expected = architecture_contract(whu_model_config(baseline_id))
    if (meta.get("source_commit") != SOURCE_COMMIT
            or meta.get("architecture_contract") != expected
            or snapshot.get("architecture_contract") != expected
            or type(meta.get("validation_only")) is not bool):
        raise RuntimeError("checkpoint does not identify the sealed UE WHU baseline")
    assert_whu_model(snapshot.get("model_config", {}))
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
            "sam2_checkpoint_sha256": snapshot.get("runtime_config", {}).get("sam2_checkpoint_sha256"),
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
