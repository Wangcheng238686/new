#!/usr/bin/env python
"""Rebuild a model from a self-describing checkpoint and run WHU inference."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch


MAINLINE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = MAINLINE_ROOT.parent
sys.path.insert(0, str(MAINLINE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    parser.add_argument(
        "--export-bootstrap-records", action="store_true",
        help="Export model-space GT/DT/images records beside predictions for "
        "paired image-bootstrap; includes zero-detection images in images.json.",
    )
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument(
        "--allow-nonstrict",
        action="store_true",
        help="Allow missing/unexpected model keys. Strict loading is the default.",
    )
    parser.add_argument(
        "--max-per-img",
        type=int,
        default=None,
        help="Override test_cfg max_per_img (e.g. 150 for the dense-image cap study).",
    )
    parser.add_argument(
        "--nms-thr",
        type=float,
        default=None,
        help="Override test_cfg rcnn NMS iou_threshold (e.g. 0.6/0.7 for dense-adjacent buildings).",
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

from model_rebuild import (  # noqa: E402
    _load_model_state,
    _register_and_build,
    _resolve_model_config,
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)
from eval_records import (  # noqa: E402
    _build_data_samples,
    _build_loader,
    _checkpoint_report,
    _extract_instances_numpy,
    _filter_predictions,
    _prediction_records,
    _resolve_dataset_contract,
)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
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
    if args.max_per_img:
        for cfg_holder in (getattr(model.roi_head, "test_cfg", None),):
            if cfg_holder is not None and "max_per_img" in cfg_holder:
                cfg_holder["max_per_img"] = int(args.max_per_img)
        logger.info("test_cfg max_per_img overridden to %d", int(args.max_per_img))
    if args.nms_thr is not None:
        _nms_cfg = getattr(getattr(model.roi_head, "test_cfg", None), "get", lambda *_: None)("nms")
        if _nms_cfg is not None and "iou_threshold" in _nms_cfg:
            _old_thr = float(_nms_cfg["iou_threshold"])
            _nms_cfg["iou_threshold"] = float(args.nms_thr)
            logger.info(
                "test_cfg rcnn NMS iou_threshold overridden: %.2f -> %.2f",
                _old_thr,
                float(args.nms_thr),
            )
        else:
            logger.warning("--nms-thr given but rcnn test_cfg has no nms.iou_threshold")
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

    if args.export_bootstrap_records:
        # Keep this export in the evaluator's model-space contract: GT comes
        # from the same resized data samples as prediction RLEs, not from the
        # original-scale VHR annotation file.  That makes both sides directly
        # consumable by bootstrap_paired_map for WHU and 10-class VHR-10.
        from utils.coco_eval_utils import _mask_to_rle, _xyxy_to_xywh

        gt_records = []
        dt_records = []
        images = []
        for gt, prediction, meta in zip(all_gt, all_dt, all_metas):
            image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
            images.append({
                "image_id": image_id,
                "height": int(meta["img_shape"][0]),
                "width": int(meta["img_shape"][1]),
            })
            for index, label in enumerate(np.asarray(gt["labels"])):
                gt_records.append({
                    "image_id": image_id,
                    "category_id": int(label) + 1,
                    "bbox": [float(v) for v in _xyxy_to_xywh(gt["bboxes"][index])],
                    "segmentation": _mask_to_rle(gt["masks"][index]),
                    "iscrowd": 0,
                })
            scores = prediction.get("scores", np.ones(len(prediction["bboxes"])))
            mask_scores = prediction.get("mask_scores", scores)
            for index, box in enumerate(np.asarray(prediction["bboxes"])):
                x1, y1, x2, y2 = (float(v) for v in box)
                dt_records.append({
                    "image_id": image_id,
                    "category_id": int(prediction["labels"][index]) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(scores[index]),
                    "mask_score": float(mask_scores[index]),
                    "segmentation": _mask_to_rle(prediction["masks"][index]),
                })
        _write_json(output_dir / "gt_records.json", gt_records)
        _write_json(output_dir / "dt_records.json", dt_records)
        _write_json(output_dir / "images.json", images)

    metrics: Dict[str, float] = {}
    total_gt = sum(len(item["labels"]) for item in all_gt)
    if not args.no_eval and total_gt > 0:
        from utils.coco_eval_utils import (
            VHR10_CATEGORIES,
            build_coco_gt_and_dt,
            run_coco_eval,
        )

        _num_classes = int(contract.get("num_classes", 0) or 0)
        _eval_categories = VHR10_CATEGORIES if _num_classes == 10 else None
        coco_gt, coco_segm_dt = build_coco_gt_and_dt(
            all_gt,
            all_dt,
            all_metas,
            score_key=segm_score_key,
            categories=_eval_categories,
        )
        metrics.update(run_coco_eval(coco_gt, coco_segm_dt, iou_type="segm"))
        _, coco_bbox_dt = build_coco_gt_and_dt(
            all_gt,
            all_dt,
            all_metas,
            score_key="scores",
            categories=_eval_categories,
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
        # Identity evidence for downstream paired comparisons (bootstrap):
        # the exact image ids this run executed, including zero-detection
        # images, so coverage can be verified as sets rather than counts.
        "processed_image_ids": sorted({
            int(meta.get("image_id", meta.get("scene_id", 0)))
            for meta in all_metas
        }),
        "ground_truth_instances": total_gt,
        "predicted_instances": len(prediction_records),
        "predictions": str(predictions_path),
        "bootstrap_records": bool(args.export_bootstrap_records),
        "metrics": metrics,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default))
    print(f"inference complete: {output_dir}")


if __name__ == "__main__":
    main()
