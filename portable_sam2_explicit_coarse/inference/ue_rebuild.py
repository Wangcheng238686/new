"""Strict UE checkpoint resolution without changing historical compat rules."""

import copy
import os
from pathlib import Path

from rsprompter.architecture_contract import file_sha256
from rsprompter.uav.contract import model_contract


def override_pretrained_paths(config, path):
    for name in ("backbone", "uav_backbone"):
        if name not in config:
            continue
        config[name]["checkpoint_path"] = path
        if config[name].get("init_cfg"):
            config[name]["init_cfg"]["checkpoint"] = path
    head = config["roi_head"]["mask_head"]
    head["sam2_mask_decoder"]["checkpoint_path"] = path
    if head.get("prompt_encoder_cfg", {}).get("ckpt_path"):
        head["prompt_encoder_cfg"]["ckpt_path"] = path


def resolve_ue_model(args, snapshot):
    config = copy.deepcopy(snapshot.get("model_config"))
    if not config:
        raise ValueError("UE checkpoint must carry the resolved model config")
    if model_contract(config) != snapshot.get("architecture_contract"):
        raise ValueError("UE checkpoint model fingerprint/architecture mismatch")
    runtime = snapshot.get("runtime_config", {})
    if not runtime.get("sam2_checkpoint_sha256") or not snapshot.get(
        "data_config", {}
    ).get("manifest_sha256"):
        raise ValueError(
            "UE checkpoint is missing required pretrained/data identity hashes"
        )
    path = getattr(args, "sam2_ckpt", None) or runtime.get("sam2_checkpoint")
    if not path or not Path(path).is_file():
        raise FileNotFoundError(
            "Provide the recorded generic SAM2 checkpoint or --sam2-ckpt"
        )
    if (
        runtime.get("sam2_checkpoint_sha256")
        and file_sha256(path) != runtime["sam2_checkpoint_sha256"]
    ):
        raise ValueError("Generic SAM2 weights differ from the training identity")
    os.environ["SAM2_CKPT"] = str(path)
    override_pretrained_paths(config, str(path))
    return config, "embedded UE model_config"


def resolve_ue_data(args, snapshot):
    data = copy.deepcopy(snapshot["data_config"])
    manifest = getattr(args, "ue_manifest", None)
    if manifest:
        data["manifest"] = manifest
        data.pop("manifest_sha256", None)
    elif args.data_root:
        data["manifest"] = str(Path(args.data_root) / Path(data["manifest"]).name)
    for name in ("satellite_root", "uav_root"):
        value = getattr(args, f"ue_{name}", None)
        if value:
            data[name] = value
    data.update(
        split=args.split,
        batch_size=args.batch_size or 1,
        cache_root=getattr(args, "ue_geometry_cache", None),
    )
    return data


def build_ue_loader(contract, num_workers):
    from data.ue_rgbd_dataset import UERGBDInstanceDataset, ue_loader

    dataset = UERGBDInstanceDataset(
        manifest=contract["manifest"],
        split=contract["split"],
        geometry=contract["geometry"],
        load_uav=contract["load_uav"],
        cache_root=contract["cache_root"],
        satellite_root=contract["satellite_root"],
        uav_root=contract["uav_root"],
    )
    if (
        contract.get("manifest_sha256")
        and dataset.manifest_sha256 != contract["manifest_sha256"]
    ):
        raise ValueError(
            "Recorded evaluation manifest changed; explicit --ue-manifest required for new data"
        )
    # An explicitly new evaluation manifest must also be traceable in the
    # exported run manifest, rather than silently losing its hash.
    contract["manifest_sha256"] = dataset.manifest_sha256
    return ue_loader(dataset, contract["batch_size"], num_workers=num_workers)
