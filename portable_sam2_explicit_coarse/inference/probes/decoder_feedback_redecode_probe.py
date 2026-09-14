#!/usr/bin/env python3
"""Frozen A0SG decoder-feedback boundary re-decode audit.

The standard first pass is production A0SG.  ``feedback`` keeps its detector,
the already encoded sparse prompt, image embeddings and high-resolution image
features fixed, then re-encodes only its detached native mask logits as the
dense mask prompt for one second MaskDecoder call.  ``feedback_boxed`` uses
the same raw logits inside the proposal but restores the production canvas's
``outside_fill_logit`` before PromptEncoder; its ``sham_boxed`` counterpart
shifts the logits before applying that same boundary condition.  ``sham_shift``
is the historical unboxed spatial sham.

No model weight, point/box mining, loss, or checkpoint is changed.  GT is used
only by COCO evaluation after each frozen forward.  The standard and passive
hook P0 cells must be bitwise identical; all intervention cells must retain
the detector hash.  The exported per-image COCO records are compatible with
the paired bootstrap utility.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples, _build_loader, _extract_instances_numpy,
    _load_checkpoint, _load_model_state, _register_and_build,
    _resolve_dataset_contract, _resolve_model_config, _select_state_dict,
    _snapshot, _write_json,
)
from utils.coco_eval_utils import (  # noqa: E402
    VHR10_CATEGORIES, _mask_to_rle, _xyxy_to_xywh, build_coco_gt_and_dt,
    run_coco_eval,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weights", choices=("model", "ema"), default="model")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="validation")
    p.add_argument("--max-batches", type=int, default=0, help="0 means full split")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _hash_update(hasher: Any, value: Any) -> None:
    arr = np.ascontiguousarray(np.asarray(value))
    hasher.update(str((arr.dtype.str, arr.shape)).encode("ascii"))
    hasher.update(arr.tobytes())


def _records(all_gt: List[dict], all_dt: List[dict], metas: List[dict]):
    gt_records, dt_records, images = [], [], []
    for gt, dt, meta in zip(all_gt, all_dt, metas):
        image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
        h, w = (int(x) for x in meta["img_shape"][:2])
        images.append({"image_id": image_id, "height": h, "width": w})
        for i, label in enumerate(np.asarray(gt["labels"])):
            gt_records.append({"image_id": image_id, "category_id": int(label) + 1,
                               "bbox": [float(x) for x in _xyxy_to_xywh(gt["bboxes"][i])],
                               "segmentation": _mask_to_rle(gt["masks"][i]), "iscrowd": 0})
        for i, box in enumerate(np.asarray(dt["bboxes"])):
            x1, y1, x2, y2 = (float(x) for x in box)
            dt_records.append({"image_id": image_id, "category_id": int(dt["labels"][i]) + 1,
                               "bbox": [x1, y1, x2 - x1, y2 - y1],
                               "score": float(dt["scores"][i]),
                               "mask_score": float(dt.get("mask_scores", dt["scores"])[i]),
                               "segmentation": _mask_to_rle(dt["masks"][i])})
    return gt_records, dt_records, images


def _canvas_box_support(
    boxes: torch.Tensor, canvas_size: tuple[int, int], image_size: int,
) -> torch.Tensor:
    """Match ``paste_roi_to_full_canvas``'s full-image half-open box contract.

    This is intentionally a PE-mask-grid support, not the later decoder-grid
    support: the intervention must restore the A0SG *input-canvas* condition
    before PromptEncoder mask downscaling, where unboxed z0 background could
    otherwise leak across the support boundary.
    """
    height, width = canvas_size
    yy = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + .5) * (float(image_size) / float(height))
    xx = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + .5) * (float(image_size) / float(width))
    x1, y1, x2, y2 = boxes.detach().unbind(dim=1)
    return ((xx[None, None, None, :] >= x1[:, None, None, None])
            & (xx[None, None, None, :] < x2[:, None, None, None])
            & (yy[None, None, :, None] >= y1[:, None, None, None])
            & (yy[None, None, :, None] < y2[:, None, None, None]))


def main() -> int:
    a = parse_args()
    ckpt_path = Path(a.checkpoint).expanduser().resolve()
    ckpt = _load_checkpoint(ckpt_path); snapshot = _snapshot(ckpt)
    ns = SimpleNamespace(checkpoint=str(ckpt_path), config=None, split=a.split,
                         data_root=None, ann_file=None, image_subdir=None,
                         image_size=None, batch_size=None, sam2_repo=None, sam2_ckpt=None)
    cfg, config_source = _resolve_model_config(ns, snapshot)
    model = _register_and_build(cfg)
    strict = _load_model_state(model, _select_state_dict(ckpt, a.weights), allow_nonstrict=False)
    if strict["missing_keys"] or strict["unexpected_keys"]:
        raise RuntimeError(f"strict checkpoint load failed: {strict}")
    device = torch.device(a.device); model.to(device).eval()
    head = model.roi_head.mask_head
    if not (head.prompt_encoder_enabled and head.explicit_use_dense_prompt and head.prompt_encoder is not None):
        raise RuntimeError("DFBR probe requires a native PromptEncoder dense-prompt checkpoint")
    if head.decoder_tail_enabled:
        raise RuntimeError("DFBR probe is defined on tail-off A0SG; use an A0SG checkpoint")
    out = Path(a.output_dir).expanduser().resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)
    categories = VHR10_CATEGORIES if int(contract.get("num_classes", 0) or 0) == 10 else None
    original_decoder_forward = head.mask_decoder.forward
    original_dense_forward = head._forward_dense_embeddings
    decoder_signature = inspect.signature(original_decoder_forward)

    def run_pass(mode: str, *, unhooked: bool = False) -> Dict[str, Any]:
        state: Dict[str, Any] = {"first_calls": 0, "second_calls": 0,
                                 "feedback_shape": None, "feedback_detached": None,
                                 "dense_calls": 0, "feedback_canvas_stats": []}
        def capture_dense(*args: Any, **kwargs: Any):
            dense = original_dense_forward(*args, **kwargs)
            # Save the exact production mixer arguments for this batch.  A
            # feedback PE embedding must still pass through the identical
            # no-mask base, alpha gate and box support; direct injection would
            # silently add three variables to the intervention.
            state["dense_args"] = args
            state["dense_kwargs"] = kwargs
            state["dense_calls"] += 1
            return dense
        def public_calls() -> Dict[str, Any]:
            """Never serialize captured live tensors into the audit manifest."""
            return {key: value for key, value in state.items()
                    if key not in ("dense_args", "dense_kwargs")}
        def wrapped_decoder(*args: Any, **kwargs: Any):
            first = original_decoder_forward(*args, **kwargs)
            state["first_calls"] += 1
            if mode == "p0":
                return first
            if len(first) != 3:
                raise RuntimeError("unexpected decoder return contract; DFBR requires tail-off decoder")
            z0 = first[0]
            if z0.ndim != 4 or z0.shape[1] != 1:
                raise RuntimeError(f"expected one native mask logit per ROI, got {tuple(z0.shape)}")
            if tuple(z0.shape[-2:]) != tuple(head._pe_mask_input_size()):
                raise RuntimeError("native decoder logits do not exactly match PromptEncoder mask_input_size; refusing implicit resize")
            if "dense_args" not in state:
                raise RuntimeError("production dense mixer arguments were not captured before decoder")
            feedback = z0.detach()
            if mode in ("sham_shift", "sham_boxed"):
                feedback = feedback.roll((max(1, feedback.shape[-2] // 2), max(1, feedback.shape[-1] // 2)), dims=(-2, -1))
            if mode in ("feedback_boxed", "sham_boxed"):
                boxes = state["dense_args"][3]
                support = _canvas_box_support(
                    boxes, tuple(feedback.shape[-2:]), head.prompt_encoder_image_size,
                )
                outside = float(head.dense_prompt_cfg.get("outside_fill_logit", 0.0))
                feedback = torch.where(support, feedback, feedback.new_full((), outside))
            # The second PE call intentionally has no points or boxes: sparse
            # tokens below are the first-pass tokens, while only dense feedback
            # is re-encoded.  This prevents re-mining or reordering points.
            _unused_sparse, feedback_dense = head.prompt_encoder(points=None, boxes=None, masks=feedback)
            if feedback_dense is None:
                raise RuntimeError("PromptEncoder did not encode feedback mask")
            mix_args = state["dense_args"]
            # Preserve every production mixer input other than the PE-derived
            # source embedding.  Feedback masks are all valid by definition.
            feedback_final = original_dense_forward(
                mix_args[0], feedback_dense,
                torch.ones((feedback_dense.shape[0],), device=feedback_dense.device, dtype=torch.bool),
                mix_args[3], mix_args[4], mix_args[5], mix_args[6], {},
            )
            bound = decoder_signature.bind(*args, **kwargs)
            bound.arguments["dense_prompt_embeddings"] = feedback_final
            state["second_calls"] += 1
            state["feedback_shape"] = list(feedback.shape)
            state["feedback_detached"] = not feedback.requires_grad
            state["feedback_canvas_stats"].append({"mean": float(feedback.float().mean()), "std": float(feedback.float().std())})
            return original_decoder_forward(*bound.args, **bound.kwargs)

        all_gt: List[dict] = []; all_dt: List[dict] = []; metas: List[dict] = []
        detector_hash = hashlib.sha256(); output_hash = hashlib.sha256(); n_batches = 0
        if not unhooked:
            head.mask_decoder.forward = wrapped_decoder
            head._forward_dense_embeddings = capture_dense
        try:
            with torch.inference_mode():
                for batch_i, batch in enumerate(loader):
                    if a.max_batches and batch_i >= a.max_batches: break
                    inputs, samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, training=False)
                    outputs = model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                    for pos, output in enumerate(outputs):
                        meta = dict(batch["img_metas"][pos]); shape = meta["img_shape"]
                        gt = _extract_instances_numpy(processed["data_samples"][pos].gt_instances, shape)
                        pred = output.pred_instances if hasattr(output, "pred_instances") else output
                        dt = _extract_instances_numpy(pred, shape)
                        image_id = int(meta.get("image_id", meta.get("scene_id", pos)))
                        _hash_update(detector_hash, np.asarray([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _hash_update(detector_hash, dt[key]); _hash_update(output_hash, dt[key])
                        _hash_update(output_hash, dt["masks"])
                        all_gt.append(gt); all_dt.append(dt); metas.append(meta)
                    n_batches += 1
        finally:
            head.mask_decoder.forward = original_decoder_forward
            head._forward_dense_embeddings = original_dense_forward
        if not unhooked and state["first_calls"] == 0:
            raise RuntimeError("hooked pass did not call MaskDecoder")
        if mode in ("feedback", "sham_shift", "feedback_boxed", "sham_boxed") and state["second_calls"] != state["first_calls"]:
            raise RuntimeError(f"second decode count mismatch: {state}")
        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, metas, categories=categories)
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        gt_records, dt_records, images = _records(all_gt, all_dt, metas)
        cell_dir = out / ("standard" if unhooked else mode); cell_dir.mkdir()
        payload = {"checkpoint": str(ckpt_path), "weights": a.weights, "dataset": dict(contract),
                   "mode": "standard" if unhooked else mode, "frozen_weights": True,
                   "processed_images": len(images), "processed_image_ids": sorted(x["image_id"] for x in images),
                   "metrics": metrics, "detector_sha256": detector_hash.hexdigest(),
                   "output_sha256": output_hash.hexdigest(), "decoder_calls": public_calls()}
        _write_json(cell_dir / "gt_records.json", gt_records); _write_json(cell_dir / "dt_records.json", dt_records)
        _write_json(cell_dir / "images.json", images); _write_json(cell_dir / "metrics.json", metrics)
        _write_json(cell_dir / "run_manifest.json", payload)
        return {"metrics": {k: metrics[k] for k in ("segm/mAP", "segm/mAP_50", "segm/mAP_75", "segm/AR@100") if k in metrics},
                "detector_sha256": payload["detector_sha256"], "output_sha256": payload["output_sha256"],
                "artifact_dir": str(cell_dir), "decoder_calls": public_calls(), "images": len(images)}

    standard = run_pass("p0", unhooked=True)
    cells = {
        "p0": run_pass("p0"), "feedback": run_pass("feedback"),
        "sham_shift": run_pass("sham_shift"), "feedback_boxed": run_pass("feedback_boxed"),
        "sham_boxed": run_pass("sham_boxed"),
    }
    if cells["p0"]["output_sha256"] != standard["output_sha256"]:
        raise RuntimeError("passive P0 hook is not bitwise identical to unhooked standard")
    for name, cell in cells.items():
        if cell["detector_sha256"] != standard["detector_sha256"]:
            raise RuntimeError(f"detector changed under {name}")
    base = cells["p0"]["metrics"]["segm/mAP"]
    summary = {"checkpoint": str(ckpt_path), "weights": a.weights, "strict_load": strict,
               "config_source": config_source, "contract": contract, "standard": standard, "cells": cells,
               "delta_vs_p0": {name: cell["metrics"]["segm/mAP"] - base for name, cell in cells.items() if name != "p0"},
               "intervention_contract": "P0 is production A0SG. feedback changes only second-pass dense PE input to detached P0 logits; sparse tokens/image embeddings/high-res features are reused. feedback_boxed additionally restores the production input canvas outside_fill_logit outside each proposal before PE. Each sham uses the same operation with a half-grid ROI shift.",
               "pre_registered_decision": "The raw full-canvas feedback result is descriptive. Advance to a training arm only if the contract-matched feedback_boxed-P0 paired image-bootstrap segm/mAP CI lower bound >0, AP75 is non-negative, and feedback_boxed-sham_boxed is directionally positive."}
    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
