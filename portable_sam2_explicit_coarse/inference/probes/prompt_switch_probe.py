#!/usr/bin/env python
"""Frozen-weight prompt-switch diagnostic on one Mask checkpoint.

Review directive (2026-09-06): with points, detection boxes and ranking
held fixed, evaluate the same checkpoint under prompt configurations:
  baseline   — points + box token + dense canvas (as trained)
  drop_box   — points + dense            (box token removed at the PE input)
  drop_dense — points + box              (dense canvas removed; PE falls
               back to its no-mask base embedding)
  drop_both  — points only
  p2_off     — P2 residual disabled; point mining and dense canvas consume
               the raw coarse logits (P2-enabled checkpoints only)
This characterises the WEIGHT's dependence on each prompt modality.  It
cannot replace training ablations (the weight was trained with all three
modalities); it distinguishes redundancy from active harm at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch

MAINLINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAINLINE_ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples,
    _build_loader,
    _extract_instances_numpy,
    _load_checkpoint,
    _load_model_state,
    _register_and_build,
    _resolve_dataset_contract,
    _resolve_model_config,
    _select_state_dict,
    _snapshot,
    _write_json,
)
from utils.coco_eval_utils import (  # noqa: E402
    VHR10_CATEGORIES,
    _mask_to_rle,
    _xyxy_to_xywh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = full validation pass")
    parser.add_argument("--output-dir", default=None,
                        help="Optional audit artifact root. Each prompt mode writes "
                             "COCO records, manifest, and hashes below this directory.")
    parser.add_argument("--canonical-dir", default=None,
                        help="Optional production inference directory. On a full pass, "
                             "unhooked standard records must match it exactly.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _hash_update(hasher: "hashlib._Hash", value: Any) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str((array.dtype.str, array.shape)).encode("ascii"))
    hasher.update(array.tobytes())


def _records(
    all_gt: List[dict], all_dt: List[dict], all_metas: List[dict]
) -> tuple[List[dict], List[dict], List[dict]]:
    """Export COCO xywh records consumable by tools/bootstrap_paired_map.py."""
    gt_records, dt_records, images = [], [], []
    for gt, dt, meta in zip(all_gt, all_dt, all_metas):
        image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
        height, width = (int(value) for value in meta["img_shape"][:2])
        images.append({"image_id": image_id, "height": height, "width": width})
        for index, label in enumerate(np.asarray(gt["labels"])):
            gt_records.append({
                "image_id": image_id,
                "category_id": int(label) + 1,
                "bbox": [float(value) for value in _xyxy_to_xywh(gt["bboxes"][index])],
                "segmentation": _mask_to_rle(gt["masks"][index]),
                "iscrowd": 0,
            })
        for index, box in enumerate(np.asarray(dt["bboxes"])):
            x1, y1, x2, y2 = (float(value) for value in box)
            dt_records.append({
                "image_id": image_id,
                "category_id": int(dt["labels"][index]) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(dt["scores"][index]),
                "mask_score": float(dt.get("mask_scores", dt["scores"])[index]),
                "segmentation": _mask_to_rle(dt["masks"][index]),
            })
    return gt_records, dt_records, images


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    ns = SimpleNamespace(
        checkpoint=str(checkpoint_path), config=None, split=args.split,
        data_root=None, ann_file=None, image_subdir=None, image_size=None,
        batch_size=None, sam2_repo=None, sam2_ckpt=None,
    )
    model_config, config_source = _resolve_model_config(ns, snapshot)
    model = _register_and_build(model_config)
    load_report = _load_model_state(
        model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False
    )
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"strict load failed: {load_report}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    head = model.roi_head.mask_head

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else None
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def run_pass(mode, *, unhooked=False):
        state = {"mode": mode}

        def pe_pre_hook(module, fn_args, kwargs):
            mode = state["mode"]
            if mode == "drop_box":
                kwargs["boxes"] = None
            elif mode == "drop_dense":
                kwargs["masks"] = None
            elif mode == "drop_both":
                kwargs["boxes"] = None
                kwargs["masks"] = None
            return fn_args, kwargs

        # p2_off (P2-v2 §4): return raw itself downstream, rather than merely
        # setting beta=0.  This makes the raw-source assertion testable.
        refiner = None
        orig_forward = None
        orig_miner_forward = None
        orig_canvas = None
        if mode == "p2_off":
            refiner = head.p2_boundary_refiner
            if refiner is None:
                raise SystemExit("p2_off requires a P2-enabled checkpoint")
            orig_forward = refiner.forward

            def raw_passthrough(*fn_args, **kwargs):
                out = orig_forward(*fn_args, **kwargs)
                raw = kwargs.get("raw_logits")
                if raw is None and len(fn_args) >= 3:
                    raw = fn_args[2]
                assert raw is not None, "p2_off: raw_logits not captured"
                patched = dict(out)
                patched["refined_logits"] = raw
                patched["delta_logits"] = torch.zeros_like(raw)
                assert torch.equal(patched["refined_logits"], raw)
                assert torch.count_nonzero(patched["delta_logits"]).item() == 0
                state["p2_calls"] = state.get("p2_calls", 0) + 1
                state["raw_logits"] = raw
                return patched

            refiner.forward = raw_passthrough

            orig_miner_forward = head.shape_point_miner.forward

            def raw_miner(logits, *fn_args, **kwargs):
                assert torch.equal(logits, state["raw_logits"]), (
                    "p2_off: point miner did not receive raw coarse logits"
                )
                state["raw_miner_calls"] = state.get("raw_miner_calls", 0) + 1
                return orig_miner_forward(logits, *fn_args, **kwargs)

            head.shape_point_miner.forward = raw_miner
            if head.use_shape_dense:
                orig_canvas = head._shape_prior_to_prompt_mask

                def raw_canvas(logits, *fn_args, **kwargs):
                    assert torch.equal(logits, state["raw_logits"]), (
                        "p2_off: dense canvas did not receive raw coarse logits"
                    )
                    state["raw_canvas_calls"] = state.get("raw_canvas_calls", 0) + 1
                    return orig_canvas(logits, *fn_args, **kwargs)

                head._shape_prior_to_prompt_mask = raw_canvas

        handle = None if unhooked else head.prompt_encoder.register_forward_pre_hook(
            pe_pre_hook, with_kwargs=True
        )
        all_gt, all_dt, all_metas = [], [], []
        detector_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        n_batches = 0
        try:
            with torch.inference_mode():
                for index, batch in enumerate(loader):
                    if args.max_batches and index >= args.max_batches:
                        break
                    imgs, data_samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    outputs = model.predict(
                        processed["inputs"], processed["data_samples"], rescale=False
                    )
                    for position, output in enumerate(outputs):
                        meta = dict(batch["img_metas"][position])
                        shape = meta["img_shape"]
                        gt = _extract_instances_numpy(
                            processed["data_samples"][position].gt_instances, shape
                        )
                        pred_instances = (
                            output.pred_instances
                            if hasattr(output, "pred_instances") else output
                        )
                        dt = _extract_instances_numpy(pred_instances, shape)
                        image_id = int(meta.get("image_id", meta.get("scene_id", position)))
                        _hash_update(detector_hash, np.asarray([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _hash_update(detector_hash, dt[key])
                            _hash_update(output_hash, dt[key])
                        _hash_update(output_hash, dt["masks"])
                        all_gt.append(gt)
                        all_dt.append(dt)
                        all_metas.append(meta)
                    n_batches += 1
        finally:
            if handle is not None:
                handle.remove()
            if refiner is not None:
                refiner.forward = orig_forward
                head.shape_point_miner.forward = orig_miner_forward
                if orig_canvas is not None:
                    head._shape_prior_to_prompt_mask = orig_canvas

        if mode == "p2_off" and not state.get("p2_calls", 0):
            raise RuntimeError("p2_off completed without executing P2BoundaryRefiner")
        if mode == "p2_off" and not state.get("raw_miner_calls", 0):
            raise RuntimeError("p2_off did not verify raw point-miner input")
        if mode == "p2_off" and head.use_shape_dense and not state.get("raw_canvas_calls", 0):
            raise RuntimeError("p2_off did not verify raw dense-canvas input")

        from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

        coco_gt, coco_dt = build_coco_gt_and_dt(
            all_gt, all_dt, all_metas, categories=eval_categories
        )
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        keep = ("segm/mAP", "segm/mAP_50", "segm/mAP_75", "segm/AR@100")
        gt_records, dt_records, images = _records(all_gt, all_dt, all_metas)
        cell = {
            "mode": "standard" if unhooked else mode,
            "unhooked": bool(unhooked),
            "batches": n_batches,
            "images": len(all_metas),
            "metrics": {k: metrics[k] for k in keep if k in metrics},
            "detector_sha256": detector_hash.hexdigest(),
            "output_sha256": output_hash.hexdigest(),
        }
        if output_dir is not None:
            artifact_dir = output_dir / cell["mode"]
            artifact_dir.mkdir(parents=True, exist_ok=False)
            manifest = {
                "checkpoint": str(checkpoint_path), "weights": args.weights,
                "dataset": dict(contract), "processed_images": len(images),
                "processed_image_ids": sorted(item["image_id"] for item in images),
                "ground_truth_instances": len(gt_records),
                "predicted_instances": len(dt_records), "frozen_weights": True,
                "prompt_switch_mode": cell["mode"],
                "detector_sha256": cell["detector_sha256"],
                "output_sha256": cell["output_sha256"], "metrics": metrics,
            }
            _write_json(artifact_dir / "gt_records.json", gt_records)
            _write_json(artifact_dir / "dt_records.json", dt_records)
            _write_json(artifact_dir / "images.json", images)
            _write_json(artifact_dir / "metrics.json", metrics)
            _write_json(artifact_dir / "run_manifest.json", manifest)
            cell["artifact_dir"] = str(artifact_dir)
        return cell

    contract = _resolve_dataset_contract(ns, snapshot)
    eval_categories = None
    if int(contract.get("num_classes", 0) or 0) == 10:
        from utils.coco_eval_utils import VHR10_CATEGORIES
        eval_categories = VHR10_CATEGORIES
    loader = _build_loader(contract, 0)
    standard = run_pass("baseline", unhooked=True)
    if args.canonical_dir is not None:
        if args.max_batches:
            raise SystemExit("--canonical-dir requires a full pass")
        canonical_dir = Path(args.canonical_dir).expanduser().resolve()
        if output_dir is None:
            raise SystemExit("--canonical-dir requires --output-dir")
        canonical_dt = canonical_dir / "dt_records.json"
        if not canonical_dt.is_file():
            # infer_from_checkpoint exports COCO predictions with an optional
            # file_name field; audit records intentionally omit that pathname.
            canonical_dt = canonical_dir / "predictions.json"
        expected_dt = json.loads(canonical_dt.read_text())
        if canonical_dt.name == "predictions.json":
            expected_dt = [
                {key: value for key, value in row.items() if key != "file_name"}
                for row in expected_dt
            ]
        actual_dt = json.loads((output_dir / "standard" / "dt_records.json").read_text())
        if json.dumps(expected_dt, sort_keys=True, separators=(",", ":")) != json.dumps(actual_dt, sort_keys=True, separators=(",", ":")):
            raise RuntimeError("unhooked standard predictions differ from canonical inference")
        canonical_manifest = json.loads((canonical_dir / "run_manifest.json").read_text())
        canonical_ids = canonical_manifest.get("processed_image_ids")
        actual_ids = json.loads((output_dir / "standard" / "images.json").read_text())
        actual_ids = sorted(int(row["image_id"]) for row in actual_ids)
        if canonical_ids is not None and sorted(int(value) for value in canonical_ids) != actual_ids:
            raise RuntimeError("unhooked standard image ids differ from canonical inference")
        canonical_metrics = json.loads((canonical_dir / "metrics.json").read_text())
        for key, value in standard["metrics"].items():
            if key in canonical_metrics and canonical_metrics[key] != value:
                raise RuntimeError(f"unhooked standard {key} differs from canonical inference")
        standard["canonical_records_exact"] = True
        standard["canonical_dir"] = str(canonical_dir)

    results: Dict[str, Dict[str, Any]] = {}
    modes = ["baseline", "drop_box", "drop_dense", "drop_both"]
    if head.p2_boundary_refiner is not None:
        modes.append("p2_off")
    for mode in modes:
        results[mode] = run_pass(mode)
        if results[mode]["detector_sha256"] != standard["detector_sha256"]:
            raise RuntimeError(f"detector changed under prompt mode={mode}")
        if mode == "baseline" and results[mode]["output_sha256"] != standard["output_sha256"]:
            raise RuntimeError("hooked baseline is not bitwise identical to unhooked standard inference")
        print(f"[{mode}] mAP={results[mode]['metrics'].get('segm/mAP'):.4f}",
              flush=True)
    base = results["baseline"]["metrics"]["segm/mAP"]
    summary = {
        "checkpoint": str(checkpoint_path),
        "weights": args.weights,
        "standard_inference": standard,
        "cells": results,
        "delta_vs_baseline": {
            mode: results[mode]["metrics"]["segm/mAP"] - base
            for mode in results if mode != "baseline"
        },
        "note": (
            "frozen-weight prompt-switch: only PromptEncoder box/mask inputs "
            "are switched; detector_sha256 must remain invariant. It characterises "
            "this weight's prompt dependence, NOT a training ablation. When "
            "--output-dir is supplied, artifacts are COCO xywh records consumable "
            "by tools/bootstrap_paired_map.py."
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
