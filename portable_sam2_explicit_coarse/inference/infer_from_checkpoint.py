#!/usr/bin/env python
"""Rebuild a model from a self-describing checkpoint and run WHU inference."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Tuple

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from torch.utils.data import DataLoader


MAINLINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MAINLINE_ROOT.parent
sys.path.insert(0, str(MAINLINE_ROOT))

logger = logging.getLogger("checkpoint_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a portable_sam2_explicit_coarse checkpoint, rebuild its model, "
            "and run inference on an annotated WHU COCO split."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="Training checkpoint (.pth).")
    parser.add_argument(
        "--config",
        default=None,
        help="Fallback MMEngine config for old checkpoints without model_config.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test", "custom"),
        default="validation",
    )
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--image-subdir", default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cpu', 'cuda', or a concrete device such as cuda:1.",
    )
    parser.add_argument(
        "--weights",
        choices=("model", "ema"),
        default="model",
        help="Load checkpoint['model'] or checkpoint['ema_state']['ema_state'].",
    )
    parser.add_argument("--sam2-repo", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--allow-nonstrict",
        action="store_true",
        help="Allow missing/unexpected model keys. Strict loading is the default.",
    )
    return parser.parse_args()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        handle.write("\n")


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cuda":
        value = "cuda:0"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({value}) but is not available")
    return device


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a mapping, got {type(checkpoint).__name__}")
    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain the required 'model' state dict")
    return checkpoint


def _snapshot(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    value = checkpoint.get("config_snapshot", {})
    return dict(value) if isinstance(value, Mapping) else {}


def _restore_architecture_environment(snapshot: Mapping[str, Any]) -> None:
    mapping = {
        "prompt_generator_mode": "PROMPT_GENERATOR_MODE",
        "explicit_prompt_mode": "EXPLICIT_PROMPT_MODE",
        "prompt_sparse_mode": "PROMPT_SPARSE_MODE",
        "prompt_encoder_enabled": "PROMPT_ENCODER_ENABLED",
        "shape_prior_enabled": "SHAPE_PRIOR_ENABLED",
        "explicit_use_box_prompt": "EXPLICIT_USE_BOX_PROMPT",
        "use_shape_dense": "USE_SHAPE_DENSE",
        "prompt_mode": "PROMPT_MODE",
        "neck_type": "NECK_TYPE",
        "prompt_dense_mode": "PROMPT_DENSE_MODE",
        "p2_boundary_refiner_enabled": "P2_BOUNDARY_REFINER_ENABLED",
    }
    for snapshot_key, env_key in mapping.items():
        value = snapshot.get(snapshot_key)
        if value not in (None, ""):
            os.environ[env_key] = str(value)


def _restore_embedded_architecture_environment(
    model_config: Mapping[str, Any],
) -> None:
    """Restore config-factory switches omitted by early schema-v2 snapshots."""
    stride = model_config.get("sam_image_embedding_stride")
    if stride not in (None, ""):
        os.environ["SAM_IMAGE_EMBED_STRIDE"] = str(stride)

    roi_head = model_config.get("roi_head", {})
    head = roi_head.get("mask_head", {}) if isinstance(roi_head, Mapping) else {}
    if not isinstance(head, Mapping):
        return
    final_mode = head.get("final_mask_coordinate_mode")
    if final_mode not in (None, ""):
        os.environ["FINAL_MASK_COORDINATE_MODE"] = str(final_mode)
    dense_cfg = head.get("dense_prompt_cfg", {})
    if isinstance(dense_cfg, Mapping):
        transform = dense_cfg.get("transform")
        if transform not in (None, ""):
            os.environ["SHAPE_DENSE_TRANSFORM"] = str(transform)
        detach = dense_cfg.get("detach_input")
        if detach is not None:
            os.environ["SHAPE_DENSE_DETACH"] = "1" if bool(detach) else "0"


def _resolve_sam2_repo(args: argparse.Namespace, snapshot: Mapping[str, Any]) -> Path:
    runtime = snapshot.get("runtime_config", {})
    saved = runtime.get("sam2_repo") if isinstance(runtime, Mapping) else None
    candidate = (
        args.sam2_repo
        or saved
        or os.environ.get("SAM2_REPO")
        or str(REPOSITORY_ROOT / "sam2")
    )
    path = Path(candidate).expanduser().resolve()
    if not (path / "sam2" / "__init__.py").is_file():
        raise FileNotFoundError(
            f"SAM2 repository is invalid: {path}; expected {path / 'sam2/__init__.py'}"
        )
    os.environ["SAM2_REPO"] = str(path)
    sys.path.insert(0, str(path))
    return path


def _targeted_sam2_checkpoint_override(
    model_config: MutableMapping[str, Any],
    checkpoint_path: str,
) -> None:
    backbone = model_config.get("backbone")
    if isinstance(backbone, MutableMapping):
        backbone["checkpoint_path"] = checkpoint_path
        init_cfg = backbone.get("init_cfg")
        if isinstance(init_cfg, MutableMapping):
            init_cfg["checkpoint"] = checkpoint_path

    roi_head = model_config.get("roi_head")
    if not isinstance(roi_head, MutableMapping):
        return
    mask_head = roi_head.get("mask_head")
    if not isinstance(mask_head, MutableMapping):
        return
    decoder_cfg = mask_head.get("sam2_mask_decoder")
    if isinstance(decoder_cfg, MutableMapping):
        decoder_cfg["checkpoint_path"] = checkpoint_path


def _resolve_model_config(
    args: argparse.Namespace,
    snapshot: Mapping[str, Any],
) -> Tuple[Dict[str, Any], str]:
    _restore_architecture_environment(snapshot)
    saved = snapshot.get("model_config")
    if isinstance(saved, Mapping) and saved:
        model_config = copy.deepcopy(dict(saved))
        source = "checkpoint.config_snapshot.model_config"
        _restore_embedded_architecture_environment(model_config)
    else:
        fallback = args.config or snapshot.get("config_path")
        if not fallback:
            raise RuntimeError(
                "Checkpoint has no embedded model_config. Pass --config for this old checkpoint."
            )
        from mmengine.config import Config

        config_path = Path(str(fallback)).expanduser()
        if not config_path.is_absolute():
            config_path = MAINLINE_ROOT / config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Fallback config not found: {config_path}")
        cfg = Config.fromfile(str(config_path))
        model_config = cfg.model.to_dict()
        source = str(config_path)

    roi_head = model_config.get("roi_head", {})
    mask_head = roi_head.get("mask_head", {}) if isinstance(roi_head, Mapping) else {}
    if isinstance(mask_head, MutableMapping) and "densebr_cfg" in mask_head:
        legacy_densebr = mask_head.pop("densebr_cfg") or {}
        if bool(legacy_densebr.get("enabled", False)):
            raise RuntimeError(
                "Legacy checkpoint enabled DenseBR and cannot be migrated to "
                "P2BoundaryRefiner. Use the historical code revision explicitly."
            )
        mask_head.setdefault("p2_boundary_refiner_cfg", {"enabled": False})

    saved_contract = snapshot.get("architecture_contract", {})
    if saved_contract:
        from mmengine.config import Config
        from rsprompter.architecture_contract import architecture_contract

        resolved_contract = architecture_contract(model_config)
        if (
            saved_contract.get("architecture_id")
            != resolved_contract.get("architecture_id")
            or saved_contract.get("model_fingerprint")
            != resolved_contract.get("model_fingerprint")
        ):
            # Older schema-v2 training runs calculated the contract before
            # MODELS.build(), but captured cfg.model afterwards. MMEngine
            # registries may consume nested config fields in place, so that
            # embedded post-build copy cannot reproduce the pre-build
            # fingerprint. Recover only from the recorded source config and
            # only when it exactly reproduces the saved contract.
            recorded = args.config or snapshot.get("config_path")
            recorded_path = Path(str(recorded)).expanduser() if recorded else None
            if recorded_path is not None and not recorded_path.is_absolute():
                recorded_path = MAINLINE_ROOT / recorded_path
            if recorded_path is not None and recorded_path.is_file():
                candidate = Config.fromfile(str(recorded_path)).model.to_dict()
                candidate_contract = architecture_contract(candidate)
            else:
                candidate = None
                candidate_contract = {}
            if candidate is None or (
                saved_contract.get("architecture_id")
                != candidate_contract.get("architecture_id")
                or saved_contract.get("model_fingerprint")
                != candidate_contract.get("model_fingerprint")
            ):
                raise RuntimeError(
                    "Embedded model_config does not match its saved architecture "
                    f"contract: saved={dict(saved_contract)} "
                    f"resolved={resolved_contract} "
                    f"recorded_config={candidate_contract}"
                )
            logger.warning(
                "Embedded post-build model_config changed fingerprint; using "
                "contract-matching recorded config %s",
                recorded_path,
            )
            model_config = candidate
            source = str(recorded_path)

    runtime = snapshot.get("runtime_config", {})
    saved_sam2_ckpt = (
        runtime.get("sam2_checkpoint") if isinstance(runtime, Mapping) else None
    )
    sam2_checkpoint = (
        args.sam2_ckpt or saved_sam2_ckpt or os.environ.get("SAM2_CKPT")
    )
    if sam2_checkpoint:
        sam2_checkpoint = str(Path(sam2_checkpoint).expanduser().resolve())
        if not Path(sam2_checkpoint).is_file():
            raise FileNotFoundError(f"SAM2 base checkpoint not found: {sam2_checkpoint}")
        os.environ["SAM2_CKPT"] = sam2_checkpoint
        saved_sha256 = (
            runtime.get("sam2_checkpoint_sha256")
            if isinstance(runtime, Mapping)
            else None
        )
        if saved_sha256:
            from rsprompter.architecture_contract import file_sha256

            actual_sha256 = file_sha256(sam2_checkpoint)
            if actual_sha256 != saved_sha256:
                raise RuntimeError(
                    "SAM2 base checkpoint SHA256 mismatch: "
                    f"saved={saved_sha256} actual={actual_sha256}"
                )
        _targeted_sam2_checkpoint_override(model_config, sam2_checkpoint)
    return model_config, source


def _register_and_build(model_config: Mapping[str, Any]):
    import mmdet.models  # noqa: F401
    from mmengine.config import ConfigDict
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmengine.registry import init_default_scope
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from mmdet.registry import MODELS

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor",
            module=DetDataPreprocessor,
        )
    import data  # noqa: F401
    import rsprompter  # noqa: F401

    init_default_scope("mmdet")
    return MODELS.build(ConfigDict(copy.deepcopy(dict(model_config))))


def _select_state_dict(
    checkpoint: Mapping[str, Any],
    weights: str,
) -> Mapping[str, torch.Tensor]:
    if weights == "model":
        state = checkpoint["model"]
    else:
        ema = checkpoint.get("ema_state")
        if not isinstance(ema, Mapping) or not isinstance(ema.get("ema_state"), Mapping):
            raise KeyError("--weights ema requested but checkpoint has no EMA state")
        state = ema["ema_state"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Selected {weights} state is not a mapping")
    return state


def _load_model_state(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    allow_nonstrict: bool,
) -> Dict[str, List[str]]:
    normalized = dict(state)
    if normalized and all(key.startswith("module.") for key in normalized):
        normalized = {key[len("module.") :]: value for key, value in normalized.items()}
    result = model.load_state_dict(normalized, strict=not allow_nonstrict)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _resolve_dataset_contract(
    args: argparse.Namespace,
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    data_config = snapshot.get("data_config", {})
    if not isinstance(data_config, Mapping):
        data_config = {}
    training_args = snapshot.get("training_args", {})
    if not isinstance(training_args, Mapping):
        training_args = {}

    defaults = {
        "validation": (
            "2.4 annotation/annotation/validation.json",
            "2.3 valid/validation",
        ),
        "test": (
            "2.4 annotation/annotation/test.json",
            "2.2 test/test",
        ),
    }
    saved_split = data_config.get(args.split, {})
    if not isinstance(saved_split, Mapping):
        saved_split = {}
    fallback_ann, fallback_images = defaults.get(args.split, ("", ""))

    data_root = (
        args.data_root
        or data_config.get("data_root")
        or training_args.get("data_root")
        or "/data/wangcheng/dataset/WHU"
    )
    ann_file = args.ann_file or saved_split.get("ann_file") or fallback_ann
    image_subdir = (
        args.image_subdir or saved_split.get("image_subdir") or fallback_images
    )
    if args.split == "custom" and (not ann_file or not image_subdir):
        raise ValueError("custom split requires --ann-file and --image-subdir")

    image_size = (
        args.image_size
        or data_config.get("image_size")
        or snapshot.get("image_size")
        or [1024, 1024]
    )
    if len(image_size) != 2:
        raise ValueError(f"image_size must contain two integers, got {image_size!r}")
    batch_size = args.batch_size or training_args.get("val_batch_size") or 1
    return {
        "data_root": str(data_root),
        "ann_file": str(ann_file),
        "image_subdir": str(image_subdir),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "batch_size": int(batch_size),
        "single_class": bool(data_config.get("single_class", True)),
    }


def _build_loader(contract: Mapping[str, Any], num_workers: int) -> DataLoader:
    from data.loader import rtmdet_collate_fn
    from data.whu_instance_dataset import WHUCocoInstanceDataset

    single_class = bool(contract["single_class"])
    # Multi-class runs (iSAID) train with canonical ids 1..15 -> labels 0..14;
    # the loader must apply the same mapping so GT labels match the head.
    category_mapping = None if single_class else {i: i - 1 for i in range(1, 16)}
    dataset = WHUCocoInstanceDataset(
        data_root=contract["data_root"],
        ann_file=contract["ann_file"],
        image_subdir=contract["image_subdir"],
        image_size=tuple(contract["image_size"]),
        single_class=single_class,
        enable_category_mapping=not single_class,
        category_mapping=category_mapping,
        flip_prob=0.0,
        vflip_prob=0.0,
        gaussian_noise_prob=0.0,
        random_erasing_prob=0.0,
        multi_scale_resize_prob=0.0,
    )
    return DataLoader(
        dataset,
        batch_size=int(contract["batch_size"]),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=rtmdet_collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=int(num_workers) > 0,
    )


def _build_data_samples(
    batch: Mapping[str, Any],
    device: torch.device,
) -> Tuple[torch.Tensor, List[DetDataSample]]:
    imgs = batch["imgs"].to(device)
    img_metas = batch["img_metas"]
    gt_bboxes = [value.to(device) for value in batch["gt_bboxes"]]
    gt_labels = [value.to(device) for value in batch["gt_labels"]]
    gt_masks = batch["gt_masks"]
    data_samples: List[DetDataSample] = []

    for index in range(len(imgs)):
        sample = DetDataSample()
        sample.set_metainfo(img_metas[index])
        instances = InstanceData()
        valid = gt_labels[index] >= 0
        valid_indices = valid.nonzero().squeeze(-1).cpu()
        height, width = img_metas[index]["img_shape"][:2]
        if valid_indices.numel() > 0:
            instances.bboxes = gt_bboxes[index][valid_indices]
            instances.labels = gt_labels[index][valid_indices]
            masks = gt_masks[index][valid_indices.numpy()]
            instances.masks = BitmapMasks(masks, height, width)
        else:
            instances.bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)
            instances.labels = torch.zeros((0,), dtype=torch.int64, device=device)
            instances.masks = BitmapMasks(
                np.zeros((0, height, width), dtype=np.uint8),
                height,
                width,
            )
        sample.gt_instances = instances
        data_samples.append(sample.to(device))
    return imgs, data_samples


def _extract_instances_numpy(instances: InstanceData, img_shape: Tuple[int, int]) -> dict:
    height, width = img_shape[:2]

    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return value

    bboxes = as_numpy(instances.bboxes)
    labels = as_numpy(instances.labels)
    scores = as_numpy(instances.scores) if hasattr(instances, "scores") else None
    mask_scores = (
        as_numpy(instances.mask_scores) if hasattr(instances, "mask_scores") else None
    )
    masks = getattr(instances, "masks", None)
    if hasattr(masks, "masks"):
        masks = masks.masks
    elif hasattr(masks, "to_ndarray"):
        masks = masks.to_ndarray()
    else:
        masks = as_numpy(masks)
    if masks is None or len(masks) == 0:
        masks = np.zeros((0, height, width), dtype=np.uint8)
    elif masks.shape[1:] != (height, width):
        masks = np.stack(
            [
                cv2.resize(
                    mask.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in masks
            ],
            axis=0,
        )
    output = {"bboxes": bboxes, "labels": labels, "masks": masks}
    if scores is not None:
        output["scores"] = scores
    if mask_scores is not None:
        output["mask_scores"] = mask_scores
    return output


def _filter_predictions(predictions: dict, threshold: float) -> dict:
    if threshold <= 0 or "scores" not in predictions:
        return predictions
    keep = np.asarray(predictions["scores"]) >= float(threshold)
    return {
        key: value[keep] if isinstance(value, np.ndarray) else value
        for key, value in predictions.items()
    }


def _mask_to_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mask_utils

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle.get("counts"), bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _prediction_records(
    predictions: Mapping[str, Any],
    meta: Mapping[str, Any],
) -> Iterable[dict]:
    bboxes = predictions["bboxes"]
    labels = predictions["labels"]
    scores = predictions.get("scores", np.ones(len(bboxes), dtype=np.float32))
    mask_scores = predictions.get("mask_scores", scores)
    masks = predictions["masks"]
    for index in range(len(bboxes)):
        x1, y1, x2, y2 = [float(value) for value in bboxes[index]]
        yield {
            "image_id": int(meta.get("image_id", meta.get("scene_id", 0))),
            "category_id": int(labels[index]) + 1,
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(scores[index]),
            "mask_score": float(mask_scores[index]),
            "segmentation": _mask_to_rle(masks[index]),
            "file_name": str(meta.get("filename", "")),
        }


def _checkpoint_report(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_schema_version": int(
            checkpoint.get(
                "checkpoint_schema_version",
                snapshot.get("checkpoint_schema_version", 0),
            )
        ),
        "epoch": checkpoint.get("epoch"),
        "best_metrics": checkpoint.get("best_metrics", {}),
        "has_model_config": bool(snapshot.get("model_config")),
        "has_training_args": bool(snapshot.get("training_args")),
        "has_data_config": bool(snapshot.get("data_config")),
        "has_ema_state": bool(checkpoint.get("ema_state")),
        "config_path": snapshot.get("config_path"),
        "neck_type": snapshot.get("neck_type"),
        "explicit_prompt_mode": snapshot.get("explicit_prompt_mode"),
        "p2_boundary_refiner": snapshot.get("p2_boundary_refiner", {}),
        "architecture_contract": snapshot.get("architecture_contract", {}),
        "image_size": snapshot.get("image_size"),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    if any(".densebr." in str(name) for name in checkpoint["model"]):
        raise RuntimeError(
            "Checkpoint contains enabled legacy DenseBR parameters; this architecture "
            "is intentionally unsupported by the v2 inference path."
        )
    snapshot = _snapshot(checkpoint)
    report = _checkpoint_report(checkpoint_path, checkpoint, snapshot)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    if args.inspect_only:
        return

    sam2_repo = _resolve_sam2_repo(args, snapshot)
    model_config, config_source = _resolve_model_config(args, snapshot)
    device = _resolve_device(args.device)
    model = _register_and_build(model_config)
    mask_head = model.roi_head.mask_head
    final_mask_coordinate_mode = getattr(
        mask_head, "final_mask_coordinate_mode", None
    )
    if final_mask_coordinate_mode not in {"roi_local", "full_image"}:
        raise RuntimeError(
            "Checkpoint inference requires an explicit final-mask coordinate "
            "contract; resolved final_mask_coordinate_mode="
            f"{final_mask_coordinate_mode!r}"
        )
    segm_score_mode = getattr(mask_head, "segm_score_mode", "detector")
    segm_score_key = getattr(mask_head, "segm_score_key", "scores")
    if segm_score_mode not in {"detector", "mask_quality"}:
        raise RuntimeError(
            f"Checkpoint has invalid segm_score_mode={segm_score_mode!r}"
        )
    if segm_score_key not in {"scores", "mask_scores"}:
        raise RuntimeError(
            f"Checkpoint has invalid segm_score_key={segm_score_key!r}"
        )
    load_report = _load_model_state(
        model,
        _select_state_dict(checkpoint, args.weights),
        allow_nonstrict=args.allow_nonstrict,
    )
    no_mask_validator = getattr(
        mask_head, "assert_no_mask_embedding_contract", None
    )
    if no_mask_validator is not None:
        no_mask_validator(f"checkpoint inference ({args.weights})")
    model.to(device)
    model.eval()
    logger.info(
        "Model rebuilt from %s; weights=%s strict=%s device=%s",
        config_source,
        args.weights,
        not args.allow_nonstrict,
        device,
    )
    logger.info(
        "Evaluation score contract: bbox=scores segm=%s mode=%s",
        segm_score_key,
        segm_score_mode,
    )
    if args.build_only:
        print("checkpoint model build/load: OK")
        return

    contract = _resolve_dataset_contract(args, snapshot)
    loader = _build_loader(contract, args.num_workers)
    data_preprocessor = model.data_preprocessor
    all_gt: List[dict] = []
    all_dt: List[dict] = []
    all_metas: List[dict] = []
    prediction_records: List[dict] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break
            imgs, data_samples = _build_data_samples(batch, device)
            processed = data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples},
                training=False,
            )
            outputs = model.predict(
                processed["inputs"],
                processed["data_samples"],
                rescale=False,
            )
            for index, output in enumerate(outputs):
                meta = batch["img_metas"][index]
                shape = meta["img_shape"]
                gt = _extract_instances_numpy(
                    processed["data_samples"][index].gt_instances,
                    shape,
                )
                pred_instances = (
                    output.pred_instances
                    if hasattr(output, "pred_instances")
                    else output
                )
                prediction = _filter_predictions(
                    _extract_instances_numpy(pred_instances, shape),
                    args.score_thr,
                )
                all_gt.append(gt)
                all_dt.append(prediction)
                all_metas.append(meta)
                prediction_records.extend(_prediction_records(prediction, meta))
            logger.info(
                "Processed batch %d/%d (%d images)",
                batch_index + 1,
                len(loader),
                len(all_metas),
            )

    if not all_metas:
        raise RuntimeError("No test images were processed")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.parent / f"inference_{args.split}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.json"
    _write_json(predictions_path, prediction_records)

    metrics: Dict[str, float] = {}
    total_gt = sum(len(item["labels"]) for item in all_gt)
    if not args.no_eval and total_gt > 0:
        from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

        coco_gt, coco_segm_dt = build_coco_gt_and_dt(
            all_gt,
            all_dt,
            all_metas,
            score_key=segm_score_key,
        )
        metrics.update(run_coco_eval(coco_gt, coco_segm_dt, iou_type="segm"))
        _, coco_bbox_dt = build_coco_gt_and_dt(
            all_gt,
            all_dt,
            all_metas,
            score_key="scores",
        )
        metrics.update(run_coco_eval(coco_gt, coco_bbox_dt, iou_type="bbox"))
    elif not args.no_eval:
        logger.warning("The selected split has no valid GT; metrics were skipped")
    _write_json(output_dir / "metrics.json", metrics)

    manifest = {
        **report,
        "config_source": config_source,
        "sam2_repo": str(sam2_repo),
        "weights": args.weights,
        "strict": not args.allow_nonstrict,
        "load_report": load_report,
        "dataset": contract,
        "device": str(device),
        "score_thr": float(args.score_thr),
        "bbox_score_key": "scores",
        "segm_score_key": segm_score_key,
        "segm_score_mode": segm_score_mode,
        "mask_postprocess": (
            "roi_local_bbox_paste"
            if final_mask_coordinate_mode == "roi_local"
            else "full_image_resize"
        ),
        "processed_images": len(all_metas),
        "ground_truth_instances": total_gt,
        "predicted_instances": len(prediction_records),
        "predictions": str(predictions_path),
        "metrics": metrics,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
    print(f"inference complete: {output_dir}")


if __name__ == "__main__":
    main()
