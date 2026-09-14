#!/usr/bin/env python
"""Frozen-gate probe for the signed EDT-confidence dense canvas (design doc
a0_signed_edt_confidence_canvas_design.md §5).  Runs on frozen A0 weights and
switches ONLY the dense-canvas representation between passes:

  standard : unhooked production inference (hash reference; R must match it
             bitwise on detector+output hashes)
  R        : raw_logits transform (production parity, hooks passive)
  B        : signed_binary   — foreground +omega / ROI background -omega
  E        : signed_edt_confidence — main candidate (tau, omega, edt_gamma)
  C        : gaussian_edt    — historical full-image positive (counter-example)
  GT_raw   : matched ROIs' canvas replaced by GT binary +-omega in box support
  GT_edt   : matched ROIs' canvas replaced by GT-derived signed EDT-confidence
             (representation-fidelity cell: GT_edt << GT_raw -> the encoding
             deletes shape information the decoder uses -> stop the route)

Transform switching is done by mutating ``head.dense_prompt_cfg`` per pass
(the dispatch reads the dict on every call); no monkey-patching.  Detection
invariance (bbox/label/score hash) is asserted across all cells.  Per-cell
COCO records are exported in the canonical flat format consumed by
``tools/bootstrap_paired_map.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)
from rsprompter.dense_prompt_utils import signed_edt_confidence_prompt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--omega", type=float, default=8.0,
                        help="Signed magnitude. Default 8.0 matches the "
                             "learned canvas clamp (scale-parity anchor).")
    parser.add_argument("--edt-gamma", type=float, default=4.0)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--cells", nargs="+",
                        choices=("R", "B", "E", "C", "GT_raw", "GT_edt"),
                        default=["R", "B", "E", "C", "GT_raw", "GT_edt"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="Per-cell records/manifests for paired bootstrap.")
    return parser.parse_args()


def _update_hash(hasher, value):
    array = np.ascontiguousarray(np.asarray(value))
    hasher.update(str((array.dtype.str, array.shape)).encode("ascii"))
    hasher.update(array.tobytes())
    return hasher


def _mask_to_rle(mask):
    from pycocotools import mask as mask_utils
    rle = mask_utils.encode(
        np.asfortranarray(np.ascontiguousarray(mask.astype(np.uint8)))
    )
    return {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}


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
    ns.sam2_repo = _resolve_sam2_repo(ns, snapshot)
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
    if not head.explicit_use_dense_prompt:
        raise SystemExit("checkpoint does not consume a dense canvas")
    print(f"[startup] signed-EDT probe ready: {checkpoint_path.name} device={device}",
          flush=True)

    cfg_backup = dict(head.dense_prompt_cfg)
    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)

    from utils.coco_eval_utils import VHR10_CATEGORIES, build_coco_gt_and_dt, run_coco_eval

    def gt_canvas(data_samples):
        out = []
        for sample in data_samples:
            instances = getattr(sample, "gt_instances", None)
            if instances is None or len(getattr(instances, "bboxes", [])) == 0:
                out.append(None)
                continue
            out.append((
                instances.bboxes.detach().cpu(),
                instances.labels.detach().cpu(),
                instances.masks.to_tensor(dtype=torch.float32, device="cpu"),
            ))
        return out

    results = {}
    standard = None
    reference_detector = None

    def run_pass(cell, *, canvas_mode=None, standard_inference=False):
        """One full-val pass.  canvas_mode: None = transform-switch only;
        ('gt', use_edt) = GT intervention on matched ROIs (transform raw).
        standard_inference: hooks stay passive (no stats, no intervention) —
        the unhooked-parity reference."""
        state = {
            "ids": None, "boxes": None, "labels": None, "gts": None,
            "label_queue": None, "label_cursor": 0,
            "pos": 0, "neg": 0, "zero": 0, "nonzero": 0, "pixels": 0,
            "empty_rois": 0, "matched": 0, "unmatched": 0,
        }
        detector_hash = hashlib.sha256()
        output_hash = hashlib.sha256()
        all_gt, all_dt, all_metas = [], [], []
        n_batches = 0

        # per-cell transform settings
        head.dense_prompt_cfg.clear()
        head.dense_prompt_cfg.update(cfg_backup)
        if not standard_inference:
            if cell == "B":
                head.dense_prompt_cfg["transform"] = "signed_binary"
                head.dense_prompt_cfg["signed_omega"] = args.omega
                head.dense_prompt_cfg["signed_tau"] = args.tau
                head.dense_prompt_cfg["clamp_range"] = None
            elif cell == "E":
                head.dense_prompt_cfg["transform"] = "signed_edt_confidence"
                head.dense_prompt_cfg["signed_omega"] = args.omega
                head.dense_prompt_cfg["signed_tau"] = args.tau
                head.dense_prompt_cfg["signed_edt_gamma"] = args.edt_gamma
                head.dense_prompt_cfg["clamp_range"] = None
            elif cell == "C":
                head.dense_prompt_cfg["transform"] = "gaussian_edt"
                head.dense_prompt_cfg["clamp_range"] = None
            # R / GT_* keep the production raw transform from cfg_backup

        def head_pre_hook(module, fn_args, kwargs):
            if standard_inference:
                return
            ids = kwargs.get("roi_img_ids")
            boxes = kwargs.get("boxes")
            state["ids"] = ids.detach().cpu() if ids is not None else None
            state["boxes"] = boxes.detach().cpu() if boxes is not None else None
            queue = state["label_queue"]
            if boxes is not None and queue is not None:
                start = int(state["label_cursor"])
                end = start + int(boxes.shape[0])
                if end > int(queue.numel()):
                    raise RuntimeError("proposal-label queue underflow")
                state["labels"] = queue[start:end].detach().cpu()
                state["label_cursor"] = end

        original_transform = head._shape_prior_to_prompt_mask

        def wrapped_transform(roi_mask_logits, boxes):
            prompt_out = original_transform(roi_mask_logits, boxes)
            # dispatch contract: (prompt, valid, stats); stats carries
            # signed_empty_rois for the signed cells
            stats = prompt_out[2] if len(prompt_out) > 2 else {}
            state["empty_rois"] += int(stats.get("signed_empty_rois", 0))
            return prompt_out

        def pe_pre_hook(module, fn_args, kwargs):
            if standard_inference:
                return fn_args, kwargs
            masks = kwargs.get("masks")
            if masks is None:
                return fn_args, kwargs
            arr = masks.detach()
            state["pos"] += int((arr > 0).sum())
            state["neg"] += int((arr < 0).sum())
            state["zero"] += int((arr == 0).sum())
            state["nonzero"] += int((arr != 0).sum())
            state["pixels"] += int(arr.numel())
            ids, boxes, labels, gts = (
                state["ids"], state["boxes"], state["labels"], state["gts"]
            )
            if canvas_mode is None:
                return fn_args, kwargs
            if len(ids) != masks.shape[0]:
                return fn_args, kwargs
            from torchvision.ops import box_iou
            canvas = masks.clone()
            mask_hw = masks.shape[-2:]
            for i in range(masks.shape[0]):
                img = int(ids[i]) if ids[i].numel() else -1
                entry = gts[img] if 0 <= img < len(gts) else None
                if entry is None:
                    state["unmatched"] += 1
                    continue
                bboxes, gt_labels, masks_t = entry
                same = gt_labels == labels[i]
                if not bool(same.any()):
                    state["unmatched"] += 1
                    continue
                ious = box_iou(boxes[i:i + 1].float(), bboxes.float())[0]
                ious = torch.where(same, ious, torch.full_like(ious, -1.0))
                best = int(ious.argmax())
                if float(ious[best]) < args.match_iou:
                    state["unmatched"] += 1
                    continue
                state["matched"] += 1
                gt_occ = masks_t[best].to(device=device)
                import torch.nn.functional as F
                gt_occ = F.interpolate(
                    gt_occ[None, None], size=tuple(mask_hw), mode="nearest"
                )[0, 0]
                x1, y1, x2, y2 = [float(v) for v in boxes[i]]
                sx1 = max(0, min(mask_hw[1] - 1, int(np.floor(x1 * mask_hw[1] / float(head.prompt_encoder_image_size)))))
                sy1 = max(0, min(mask_hw[0] - 1, int(np.floor(y1 * mask_hw[0] / float(head.prompt_encoder_image_size)))))
                sx2 = max(sx1 + 1, min(mask_hw[1], int(np.ceil(x2 * mask_hw[1] / float(head.prompt_encoder_image_size)))))
                sy2 = max(sy1 + 1, min(mask_hw[0], int(np.ceil(y2 * mask_hw[0] / float(head.prompt_encoder_image_size)))))
                if canvas_mode == ("gt", False):
                    logits = torch.where(
                        gt_occ[sy1:sy2, sx1:sx2] >= 0.5,
                        torch.ones_like(gt_occ[sy1:sy2, sx1:sx2]) * 20.0,
                        -torch.ones_like(gt_occ[sy1:sy2, sx1:sx2]) * 20.0,
                    )
                else:
                    logits = (gt_occ[sy1:sy2, sx1:sx2] * 2.0 - 1.0) * 20.0
                q, _v = signed_edt_confidence_prompt(
                    logits[None, None].to(device=device),
                    tau=0.5, omega=args.omega, edt_gamma=args.edt_gamma,
                    use_edt=(canvas_mode == ("gt", True)),
                )
                canvas[i, 0, sy1:sy2, sx1:sx2] = q[0, 0]
            kwargs["masks"] = canvas
            return fn_args, kwargs

        original_predict_mask = model.roi_head.predict_mask

        def wrapped_predict_mask(*fn_args, **fn_kwargs):
            results_list = fn_kwargs.get("results_list")
            if results_list is None:
                results_list = fn_args[2]
            state["label_queue"] = torch.cat(
                [r.labels.detach() for r in results_list], dim=0
            ) if results_list else torch.zeros(0, dtype=torch.long)
            state["label_cursor"] = 0
            return original_predict_mask(*fn_args, **fn_kwargs)

        head_handle = head.register_forward_pre_hook(head_pre_hook, with_kwargs=True)
        pe_handle = head.prompt_encoder.register_forward_pre_hook(
            pe_pre_hook, with_kwargs=True
        )
        model.roi_head.predict_mask = wrapped_predict_mask
        if not standard_inference:
            head._shape_prior_to_prompt_mask = wrapped_transform
        try:
            with torch.inference_mode():
                for index, batch in enumerate(loader):
                    if args.max_batches and index >= args.max_batches:
                        break
                    imgs, data_samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor(
                        {"inputs": imgs, "data_samples": data_samples}, training=False
                    )
                    state["gts"] = gt_canvas(processed["data_samples"])
                    outputs = model.predict(
                        processed["inputs"], processed["data_samples"], rescale=False
                    )
                    if not standard_inference and state["label_queue"] is not None \
                            and state["label_cursor"] != int(state["label_queue"].numel()):
                        raise RuntimeError("proposal-label queue not consumed exactly")
                    for position, output in enumerate(outputs):
                        meta = batch["img_metas"][position]
                        shape = meta["img_shape"]
                        gt = _extract_instances_numpy(
                            processed["data_samples"][position].gt_instances, shape
                        )
                        pred = (
                            output.pred_instances
                            if hasattr(output, "pred_instances") else output
                        )
                        dt = _extract_instances_numpy(pred, shape)
                        image_id = int(meta.get("image_id", position))
                        _update_hash(detector_hash, np.asarray([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _update_hash(detector_hash, dt[key])
                            _update_hash(output_hash, dt[key])
                        _update_hash(output_hash, dt["masks"])
                        all_gt.append(gt)
                        all_dt.append(dt)
                        all_metas.append(dict(meta))
                    n_batches += 1
        finally:
            head_handle.remove()
            pe_handle.remove()
            model.roi_head.predict_mask = original_predict_mask
            if not standard_inference:
                head._shape_prior_to_prompt_mask = original_transform
            head.dense_prompt_cfg.clear()
            head.dense_prompt_cfg.update(cfg_backup)

        coco_gt, coco_dt = build_coco_gt_and_dt(
            all_gt, all_dt, all_metas, score_key="scores",
            categories=VHR10_CATEGORIES,
        )
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        cell_name = "standard" if standard_inference else cell
        cell_out = {
            "cell": cell_name,
            "batches": n_batches,
            "segm/mAP": metrics.get("segm/mAP"),
            "segm/mAP_50": metrics.get("segm/mAP_50"),
            "segm/mAP_75": metrics.get("segm/mAP_75"),
            "detector_sha256": detector_hash.hexdigest(),
            "output_sha256": output_hash.hexdigest(),
            "canvas_pos": state["pos"], "canvas_neg": state["neg"],
            "canvas_zero": state["zero"], "canvas_nonzero": state["nonzero"],
            "canvas_pixels": state["pixels"],
            "empty_rois": state["empty_rois"],
            "matched": state["matched"], "unmatched": state["unmatched"],
        }
        if args.output_dir:
            # Canonical flat records (bootstrap_paired_map.py contract):
            # xywh bboxes, category_id, score/mask_score, images height/width.
            artifact_dir = Path(args.output_dir) / cell_name
            artifact_dir.mkdir(parents=True, exist_ok=True)
            gt_records, dt_records, images = [], [], []
            for gt, dt, meta, position in zip(
                    all_gt, all_dt, all_metas, range(len(all_metas))):
                image_id = int(meta.get("image_id", position))
                # records (RLE sizes) live on the img_shape canvas — the
                # exporter must match that frame, NOT ori_shape (bootstrap
                # reads height/width to build per-image areas)
                h, w = meta["img_shape"][:2]
                images.append({"image_id": image_id, "height": int(h), "width": int(w)})
                for b, l, m in zip(gt["bboxes"], gt["labels"], gt["masks"]):
                    gt_records.append({
                        "image_id": image_id,
                        "category_id": int(l) + 1,
                        "bbox": [float(b[0]), float(b[1]),
                                 float(b[2] - b[0]), float(b[3] - b[1])],
                        "segmentation": _mask_to_rle(m),
                        "iscrowd": 0,
                    })
                for b, l, s, m in zip(
                        dt["bboxes"], dt["labels"], dt["scores"], dt["masks"]):
                    dt_records.append({
                        "image_id": image_id,
                        "category_id": int(l) + 1,
                        "bbox": [float(b[0]), float(b[1]),
                                 float(b[2] - b[0]), float(b[3] - b[1])],
                        "score": float(s),
                        "mask_score": float(s),
                        "segmentation": _mask_to_rle(m),
                    })
            manifest = {
                "checkpoint": str(checkpoint_path), "weights": args.weights,
                "dataset": dict(contract), "cell": cell_name,
                "omega": args.omega, "edt_gamma": args.edt_gamma, "tau": args.tau,
                "processed_images": len(images),
                "processed_image_ids": sorted(x["image_id"] for x in images),
                "ground_truth_instances": len(gt_records),
                "predicted_instances": len(dt_records),
                "metrics": metrics,
            }
            (artifact_dir / "gt_records.json").write_text(json.dumps(gt_records))
            (artifact_dir / "dt_records.json").write_text(json.dumps(dt_records))
            (artifact_dir / "images.json").write_text(json.dumps(images))
            (artifact_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            (artifact_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
            cell_out["artifact_dir"] = str(artifact_dir)
        return cell_out

    # Unhooked standard pass first (§5 parity anchor)
    standard = run_pass("standard", standard_inference=True)
    print(f"[standard] mAP={standard['segm/mAP']:.4f}", flush=True)

    cells = []
    ordered = [c for c in ("R", "B", "E", "C", "GT_raw", "GT_edt") if c in args.cells]
    for cell in ordered:
        mode = None
        if cell == "GT_raw":
            mode = ("gt", False)
        elif cell == "GT_edt":
            mode = ("gt", True)
        cell_out = run_pass(cell, canvas_mode=mode)
        if reference_detector is None:
            reference_detector = cell_out["detector_sha256"]
        elif cell_out["detector_sha256"] != reference_detector:
            raise RuntimeError(
                f"detector output changed under cell {cell}: canvas-only "
                "intervention must not alter detections"
            )
        cells.append(cell_out)
        print(f"[{cell}] mAP={cell_out['segm/mAP']:.4f} "
              f"pos={cell_out['canvas_pos']} neg={cell_out['canvas_neg']} "
              f"empty_rois={cell_out['empty_rois']} "
              f"matched={cell_out['matched']}", flush=True)

    r_cell = next((c for c in cells if c["cell"] == "R"), None)
    if r_cell is not None:
        if r_cell["detector_sha256"] != standard["detector_sha256"] or \
                r_cell["output_sha256"] != standard["output_sha256"]:
            raise RuntimeError(
                "R (hooked, passive) does not reproduce the unhooked standard "
                "pass bitwise — hook leakage, refusing to gate on it"
            )
        print("[parity] R == standard (detector+output hashes bitwise)", flush=True)

    base = r_cell["segm/mAP"] if r_cell is not None else None
    summary = {
        "checkpoint": str(checkpoint_path),
        "config_source": config_source,
        "omega": args.omega, "edt_gamma": args.edt_gamma, "tau": args.tau,
        "match_iou": args.match_iou,
        "standard_inference": standard,
        "cells": cells,
        "delta_mAP_vs_R": {
            c["cell"]: (c["segm/mAP"] - base if base is not None else None)
            for c in cells
        },
        "note": (
            "Frozen-weight representation probe. R is asserted bitwise "
            "identical to the unhooked standard pass. GT_raw vs GT_edt is the "
            "representation-fidelity check: GT_edt << GT_raw -> the signed "
            "EDT encoding deletes shape information -> stop the route."
        ),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary["delta_mAP_vs_R"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
