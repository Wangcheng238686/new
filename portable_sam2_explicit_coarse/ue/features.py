"""Read-only satellite interface for SurfaceBEV and future fusion adapters."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import torch

from .contract import ROOT, assert_whu_model, verify_identity, file_sha256


@dataclass
class SatelliteFeatures:
    pyramids: tuple
    image_embeddings: torch.Tensor
    positional_embeddings: object
    high_res_features: object


def extract_satellite_features(model, inputs):
    features = SatelliteFeatures(*model.extract_feat(inputs))
    if inputs.shape[-2:] != (1024, 1024):
        raise RuntimeError("UE feature interface requires 1024x1024 inputs")
    p3 = features.pyramids[1]
    if tuple(p3.shape[1:]) != (512, 128, 128) or not torch.isfinite(p3).all():
        raise RuntimeError("nonfinite or incompatible satellite P3")
    return features


def model_input(model, item, device):
    from mmdet.structures import DetDataSample
    sample = DetDataSample()
    sample.set_metainfo(dict(item["img_metas"]))
    return model.data_preprocessor({"inputs": [item["img"].to(device)],
                                   "data_samples": [sample.to(device)]}, training=False)


def digest_predictions(outputs):
    result = {}
    for key in ("bboxes", "scores", "labels", "masks"):
        value = getattr(outputs[0].pred_instances, key)
        if hasattr(value, "to_ndarray"):
            value = value.to_ndarray()
        elif hasattr(value, "masks"):
            value = value.masks
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        result[key] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return result


@torch.inference_mode()
def observe_satellite(model, item, device):
    if model.training:
        raise RuntimeError("passive harvesting requires eval mode")
    before = model_input(model, item, device)
    outputs = model.predict(before["inputs"], before["data_samples"], rescale=False)
    digest = digest_predictions(outputs)
    data = model_input(model, item, device)
    features = extract_satellite_features(model, data["inputs"])
    after = model_input(model, item, device)
    check = model.predict(after["inputs"], after["data_samples"], rescale=False)
    if digest != digest_predictions(check):
        raise RuntimeError("passive feature extraction changed satellite predictions")
    return features, outputs, digest


def build_baseline(checkpoint_path, manifest, sam2_ckpt, device, allow_validation=False):
    from inference.model_rebuild import (
        _snapshot, _resolve_sam2_repo, _resolve_model_config,
        _register_and_build, _load_model_state, _select_state_dict,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity = verify_identity(checkpoint_path, checkpoint, manifest)
    if not sam2_ckpt or file_sha256(str(sam2_ckpt)) != identity["sam2_checkpoint_sha256"]:
        raise RuntimeError("SAM2 checkpoint differs from the recorded UE baseline")
    if identity["validation_only"] and not allow_validation:
        raise RuntimeError("validation-only baseline cannot produce formal probe results")
    snapshot = _snapshot(checkpoint)
    args = SimpleNamespace(config=None, sam2_ckpt=sam2_ckpt, sam2_repo=None)
    _resolve_sam2_repo(args, snapshot)
    cfg, source = _resolve_model_config(args, snapshot)
    assert_whu_model(cfg)
    model = _register_and_build(cfg).to(device).eval()
    report = _load_model_state(model, _select_state_dict(checkpoint, "model"), False)
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError("UE baseline must strict-load")
    return model, checkpoint, identity, source


def implementation_fingerprints():
    paths = list((ROOT / "rsprompter").rglob("*.py"))
    paths += list((ROOT / "ue").glob("*.py"))
    paths += [ROOT / "data/ue_raw_pair_dataset.py", ROOT / "inference/model_rebuild.py",
              ROOT / "scripts/ue/probe_surfacebev_p1.py",
              ROOT / "scripts/ue/probe_surfacebev_p1_h0.py", ROOT / "configs/ue_uav_encoder.py"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}
