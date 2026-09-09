#!/usr/bin/env python3
"""Frozen-A0 Oracle that decomposes the reachability of P2-style support.

GT is used only after class-aware, IoU-matched proposal matching to make a
*correct-errors-only* coarse counterfactual.  It never decides the support,
proposal, point prompt, or detector.  All interventions affect only the
existing dense-canvas consumer:

``accepted_s4``  A2's raw-boundary S4 with its coverage rejection;
``ungated_s4``   identical raw-boundary S4 but without the rejection;
``full_roi``     all matched ROI error pixels.

This isolates spatial reachability before investing in another P2 feature or
loss design.  It is an Oracle ceiling, not a trainable result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

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
from inference.probes.coarse_to_mining_oracle_probe import (  # noqa: E402
    _gt_cache, _hash_update, _match_gt_index, _records,
)
from inference.probes.p2_coarse_gate_probe import _roi_target_grid  # noqa: E402
from rsprompter.p2_boundary_refiner import P2BoundaryRefiner  # noqa: E402
from utils.coco_eval_utils import VHR10_CATEGORIES, build_coco_gt_and_dt, run_coco_eval  # noqa: E402

MODES = ("raw", "accepted_s4", "ungated_s4", "full_roi")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="P2-off frozen A0 checkpoint")
    p.add_argument("--support-reference-checkpoint", required=True, help="A2/R1 checkpoint whose raw-support geometry is fixed")
    p.add_argument("--weights", choices=("model", "ema"), default="model")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--split", default="validation")
    p.add_argument("--match-iou", type=float, default=.5)
    p.add_argument("--target-logit", type=float, default=8.)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--canonical-dir", default=None)
    p.add_argument("--skip-parity-check", action="store_true")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _support_refiner(path: Path) -> Tuple[P2BoundaryRefiner, P2BoundaryRefiner, Dict[str, Any]]:
    snap = _snapshot(_load_checkpoint(path))
    cfg = dict(snap.get("p2_boundary_refiner") or {})
    if not cfg.get("enabled"):
        raise RuntimeError("support reference checkpoint has no enabled P2BoundaryRefiner")
    cfg.pop("enabled", None)
    accepted = P2BoundaryRefiner(**cfg)
    ungated_cfg = dict(cfg); ungated_cfg["max_search_coverage"] = 1.0
    return accepted, P2BoundaryRefiner(**ungated_cfg), cfg


def _validate_canonical(standard_dir: Path, canonical_dir: Path) -> None:
    for filename in ("gt_records.json", "dt_records.json", "images.json"):
        got = json.loads((standard_dir / filename).read_text())
        expected = json.loads((canonical_dir / filename).read_text())
        if json.dumps(got, sort_keys=True, separators=(",", ":")) != json.dumps(expected, sort_keys=True, separators=(",", ":")):
            raise RuntimeError(f"unhooked standard {filename} differs from canonical A0 inference")
    got = json.loads((standard_dir / "metrics.json").read_text())
    expected = json.loads((canonical_dir / "metrics.json").read_text())
    if {k: v for k, v in got.items() if k.startswith("segm/")} != {k: v for k, v in expected.items() if k.startswith("segm/")}:
        raise RuntimeError("unhooked standard segm metrics differ from canonical A0 inference")


def main() -> int:
    args = parse_args()
    if args.target_logit <= 0 or not 0 < args.match_iou <= 1:
        raise SystemExit("--target-logit must be >0 and --match-iou in (0,1]")
    if not args.max_batches and (args.skip_parity_check or set(args.modes) != set(MODES) or len(args.modes) != len(MODES) or not args.canonical_dir):
        raise SystemExit("full support Oracle requires all four modes, canonical-dir, and parity check")
    ckpt_path = Path(args.checkpoint).resolve(); ckpt = _load_checkpoint(ckpt_path); snap = _snapshot(ckpt)
    ns = SimpleNamespace(checkpoint=str(ckpt_path), config=None, split=args.split, data_root=None, ann_file=None, image_subdir=None, image_size=None, batch_size=None, sam2_repo=None, sam2_ckpt=None)
    cfg, source = _resolve_model_config(ns, snap)
    model = _register_and_build(cfg)
    strict = _load_model_state(model, _select_state_dict(ckpt, args.weights), allow_nonstrict=False)
    if strict["missing_keys"] or strict["unexpected_keys"]: raise RuntimeError(f"strict load failed: {strict}")
    device = torch.device(args.device); model.to(device).eval(); head = model.roi_head.mask_head
    if head.p2_boundary_refiner is not None: raise RuntimeError("support decomposition requires P2-off A0")
    if not (head.use_shape_dense and head.explicit_use_dense_prompt and head.shape_injector is not None): raise RuntimeError("requires A0 ShapePrior -> active dense canvas")
    accepted_refiner, ungated_refiner, support_cfg = _support_refiner(Path(args.support_reference_checkpoint).resolve())
    contract, loader = _resolve_dataset_contract(ns, snap), _build_loader(_resolve_dataset_contract(ns, snap), 0)
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)

    def run(mode: str, unhooked: bool = False) -> Dict[str, Any]:
        state: Dict[str, Any] = {"gts": None, "labels": None, "label_queue": None, "cursor": 0, "ids": None, "boxes": None}
        counts = {"matched_rois": 0, "unmatched_rois": 0, "accepted_error_pixels": 0, "ungated_error_pixels": 0, "full_error_pixels": 0, "accepted_support_pixels": 0, "ungated_support_pixels": 0, "written_pixels": 0}
        d_hash, o_hash = hashlib.sha256(), hashlib.sha256(); all_gt: List[dict] = []; all_dt: List[dict] = []; metas: List[dict] = []
        orig_predict, orig_coarse = model.roi_head.predict_mask, head._forward_coarse_p2
        p_had, p_saved = "predict_mask" in model.roi_head.__dict__, model.roi_head.__dict__.get("predict_mask")
        c_had, c_saved = "_forward_coarse_p2" in head.__dict__, head.__dict__.get("_forward_coarse_p2")

        def predict(*fn_args: Any, **kwargs: Any) -> Any:
            results = kwargs.get("results_list", fn_args[2] if len(fn_args) > 2 else None)
            state["label_queue"] = torch.cat([r.labels.detach() for r in results], dim=0) if results else torch.zeros(0, dtype=torch.long)
            state["cursor"] = 0
            return orig_predict(*fn_args, **kwargs)

        def pre_hook(module: Any, fn_args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            del module, fn_args
            boxes, ids, queue = kwargs.get("boxes"), kwargs.get("roi_img_ids"), state["label_queue"]
            state["boxes"], state["ids"] = (boxes.detach().cpu() if boxes is not None else None), (ids.detach().cpu() if ids is not None else None)
            if boxes is None or queue is None: return
            start, end = int(state["cursor"]), int(state["cursor"]) + int(boxes.shape[0])
            if end > int(queue.numel()): raise RuntimeError("proposal-label queue underflow")
            state["labels"], state["cursor"] = queue[start:end].detach().cpu(), end

        def coarse(*fn_args: Any, **kwargs: Any) -> Any:
            output = orig_coarse(*fn_args, **kwargs)
            if mode == "raw": return output
            raw, coarse_outputs, raw_canvas, raw_valid = output
            boxes = fn_args[2] if len(fn_args) > 2 else kwargs.get("boxes")
            ids, labels, gts = state["ids"], state["labels"], state["gts"]
            if raw is None or boxes is None or ids is None or labels is None or gts is None: raise RuntimeError("cannot align A0 coarse rows to proposal labels/GT")
            candidate = raw.clone()
            _, _, _, accepted_support, accepted_valid = accepted_refiner._raw_support(raw)
            _, _, _, ungated_support, ungated_valid = ungated_refiner._raw_support(raw)
            if not bool((accepted_support <= ungated_support).all()): raise RuntimeError("accepted support must be subset of ungated support")
            for i in range(raw.shape[0]):
                entry = gts[int(ids[i])] if 0 <= int(ids[i]) < len(gts) else None
                index = _match_gt_index(boxes[i].detach().cpu(), labels[i], entry, args.match_iou)
                if index is None: counts["unmatched_rois"] += 1; continue
                target = _roi_target_grid(entry[2][index], boxes[i].detach().cpu(), tuple(raw.shape[-2:]))
                if target is None: counts["unmatched_rois"] += 1; continue
                target = target.to(raw.device); error = raw[i, 0].ge(0).ne(target)
                counts["matched_rois"] += 1; counts["full_error_pixels"] += int(error.sum())
                counts["accepted_error_pixels"] += int((error & accepted_support[i, 0].bool()).sum()); counts["ungated_error_pixels"] += int((error & ungated_support[i, 0].bool()).sum())
                counts["accepted_support_pixels"] += int(accepted_support[i, 0].sum()); counts["ungated_support_pixels"] += int(ungated_support[i, 0].sum())
                support = accepted_support[i, 0].bool() if mode == "accepted_s4" else (ungated_support[i, 0].bool() if mode == "ungated_s4" else torch.ones_like(error))
                write = error & support; counts["written_pixels"] += int(write.sum())
                candidate[i, 0][write] = torch.where(target[write], torch.full_like(candidate[i, 0][write], args.target_logit), torch.full_like(candidate[i, 0][write], -args.target_logit))
            canvas, valid, _ = head._shape_prior_to_prompt_mask(candidate, boxes)
            return raw, coarse_outputs, canvas, valid

        hook = None
        if not unhooked:
            hook = head.register_forward_pre_hook(pre_hook, with_kwargs=True); model.roi_head.predict_mask = predict; head._forward_coarse_p2 = coarse
        try:
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    if args.max_batches and batch_index >= args.max_batches: break
                    inputs, samples = _build_data_samples(batch, device); processed = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, training=False)
                    state["gts"] = _gt_cache(processed["data_samples"]); outputs = model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                    if not unhooked and state["cursor"] != int(state["label_queue"].numel()): raise RuntimeError("proposal-label queue not consumed exactly")
                    for i, output in enumerate(outputs):
                        meta, shape = dict(batch["img_metas"][i]), batch["img_metas"][i]["img_shape"]
                        gt = _extract_instances_numpy(processed["data_samples"][i].gt_instances, shape); pred = output.pred_instances if hasattr(output, "pred_instances") else output; dt = _extract_instances_numpy(pred, shape)
                        image_id = int(meta.get("image_id", meta.get("scene_id", i))); _hash_update(d_hash, np.asarray([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"): _hash_update(d_hash, dt[key]); _hash_update(o_hash, dt[key])
                        _hash_update(o_hash, dt["masks"]); all_gt.append(gt); all_dt.append(dt); metas.append(meta)
        finally:
            if hook is not None:
                hook.remove()
                if p_had: model.roi_head.predict_mask = p_saved
                else: delattr(model.roi_head, "predict_mask")
                if c_had: head._forward_coarse_p2 = c_saved
                else: delattr(head, "_forward_coarse_p2")
        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, metas, categories=VHR10_CATEGORIES); metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        gt_records, dt_records, images = _records(all_gt, all_dt, metas); name = "standard" if unhooked else mode; arm_dir = output_dir / name; arm_dir.mkdir(exist_ok=True)
        _write_json(arm_dir / "gt_records.json", gt_records); _write_json(arm_dir / "dt_records.json", dt_records); _write_json(arm_dir / "images.json", images); _write_json(arm_dir / "metrics.json", metrics)
        _write_json(arm_dir / "run_manifest.json", {"checkpoint": str(ckpt_path), "dataset": dict(contract), "processed_images": len(images), "processed_image_ids": sorted(x["image_id"] for x in images), "ground_truth_instances": len(gt_records), "support_oracle_mode": name, "metrics": metrics})
        return {"mode": name, "segm/mAP": metrics["segm/mAP"], "detector_sha256": d_hash.hexdigest(), "output_sha256": o_hash.hexdigest(), "counts": counts, "artifact_dir": str(arm_dir)}

    standard = None if args.skip_parity_check else run("raw", True)
    if standard is not None and not args.max_batches:
        _validate_canonical(output_dir / "standard", Path(args.canonical_dir).resolve()); standard["canonical_records_exact"] = True
    cells = []
    for mode in args.modes:
        cell = run(mode)
        if standard is not None:
            if cell["detector_sha256"] != standard["detector_sha256"]: raise RuntimeError(f"detector changed under {mode}")
            if mode == "raw" and cell["output_sha256"] != standard["output_sha256"]: raise RuntimeError("raw hook is not bitwise identical")
        cells.append(cell); print(json.dumps(cell), flush=True)
    raw_map = next(x["segm/mAP"] for x in cells if x["mode"] == "raw")
    summary = {"checkpoint": str(ckpt_path), "support_reference_checkpoint": str(Path(args.support_reference_checkpoint).resolve()), "support_geometry": support_cfg, "config_source": source, "strict_load": strict, "dataset_contract": contract, "standard": standard, "cells": cells, "delta_mAP_vs_raw": {x["mode"]: x["segm/mAP"] - raw_map for x in cells}, "note": "Frozen A0 support-decomposition Oracle: GT corrects only raw-binary error pixels in the named raw-derived support; points/detector/unmatched rows stay raw."}
    _write_json(output_dir / "summary.json", summary); print(json.dumps(summary, indent=2), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
