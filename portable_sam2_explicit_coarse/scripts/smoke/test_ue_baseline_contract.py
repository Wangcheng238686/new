"""Adversarial lineage/cache checks; no GPU or training required."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["RSPROMPTER_LIGHT_IMPORT"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ue.contract import (
    baseline_metadata, checkpoint_identity, verify_identity, validate_snapshot,
    guard_cache, whu_model_config,
)
from data.ue_raw_pair_dataset import PAIR_SHA256, SPLIT_SHA256


class BaselineContractTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "model.pth"
        self.checkpoint.write_bytes(b"fixture identity, not a trained model")
        config = whu_model_config()
        meta = baseline_metadata(config, validation_only=True)
        self.snapshot = {
            "model_config": config, "ue_baseline": meta,
            "architecture_contract": meta["architecture_contract"],
            "runtime_config": {"sam2_checkpoint_sha256": "a" * 64},
            "data_config": {"dataset_format": "ue_raw_surface",
                            "split_manifest": "surfbev_b0_split_v2.json",
                            "augmentation_contract": "none",
                            "pair_manifest_sha256": PAIR_SHA256,
                            "split_manifest_sha256": SPLIT_SHA256["surfbev_b0_split_v2.json"]},
        }
        self.payload = {"config_snapshot": self.snapshot}

    def test_new_identity_roundtrip(self):
        identity = checkpoint_identity(self.checkpoint, self.payload)
        path = self.root / "identity.json"
        path.write_text(json.dumps(identity))
        self.assertEqual(verify_identity(self.checkpoint, self.payload, path), identity)

    def test_old_checkpoint_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_snapshot({"model_config": self.snapshot["model_config"]})

    def test_wrong_source_rejected(self):
        self.snapshot["ue_baseline"]["source_commit"] = "wrong"
        with self.assertRaises(RuntimeError):
            validate_snapshot(self.snapshot)

    def test_temporary_role_must_be_boolean(self):
        self.snapshot["ue_baseline"]["validation_only"] = "false"
        with self.assertRaises(RuntimeError):
            validate_snapshot(self.snapshot)

    def test_model_drift_rejected_even_with_forged_metadata(self):
        self.snapshot["model_config"]["roi_head"]["mask_head"]["final_mask_coordinate_mode"] = "full_image"
        with self.assertRaises(RuntimeError):
            validate_snapshot(self.snapshot)

    def test_split_or_augmentation_drift_rejected(self):
        for key, value in (("split_manifest_sha256", "wrong"), ("augmentation_contract", "flip")):
            snapshot = copy.deepcopy(self.snapshot)
            snapshot["data_config"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                validate_snapshot(snapshot)

    def test_modified_weights_rejected(self):
        path = self.root / "identity.json"
        path.write_text(json.dumps(checkpoint_identity(self.checkpoint, self.payload)))
        self.checkpoint.write_bytes(b"different weights")
        with self.assertRaises(RuntimeError):
            verify_identity(self.checkpoint, self.payload, path)

    def test_unidentified_cache_rejected_without_adoption(self):
        cache = self.root / "cache"
        cache.mkdir()
        (cache / "old.npz").write_bytes(b"old")
        with self.assertRaises(RuntimeError):
            guard_cache(cache / "contract.json", {"new": True})
        self.assertFalse((cache / "contract.json").exists())

    def test_cache_drift_rejected(self):
        path = self.root / "cache/contract.json"
        contract = {"baseline": "new", "fingerprint": "v1"}
        guard_cache(path, contract)
        guard_cache(path, contract)
        with self.assertRaises(RuntimeError):
            guard_cache(path, {**contract, "fingerprint": "v2"})
        self.assertEqual(json.loads(path.read_text()), contract)


if __name__ == "__main__":
    unittest.main()
