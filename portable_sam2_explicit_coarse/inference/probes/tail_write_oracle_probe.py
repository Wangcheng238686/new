#!/usr/bin/env python3
"""Frozen a3sgt tail-write policy oracle (diagnostic upper bounds).

The a3sgt (residual_v1_stop) tail learned a safe-but-useless write: point BCE
0.407, net flip ~0.  This probe bounds what ANY K64 write policy could have
bought at these frozen weights, separating AUTHORISATION from MAGNITUDE:

  standard / p0   production forward (deploys learned delta); passive hook
                  must be bitwise identical to the unhooked pass.
  no_write        base z only (delta removed) — the frozen-base reference.
  gt_auth         learned delta applied ONLY on selected points whose base
                  sign contradicts the matched-GT label (perfect authorisation,
                  learned magnitudes).
  inv_auth        learned delta applied only on CORRECT points (sham control:
                  same operation, inverted authorisation).
  gt_cap          delta* = clip(margin*y_signed - z0, +/-delta_logit_max) on
                  every selected point of a matched ROI.  PRE-AUDIT cell: its
                  margin is anchored at logit 0 while deployment binarises at
                  mask_thr_binary=0.4 (logit -0.4055), so it cannot delete
                  false-positive area and pulls correct negatives inside; it
                  is NOT an upper bound.  Kept verbatim for the record.
  gt_cap_thr      POST-AUDIT corrective cell: target_logit = boundary
                  + signed*margin with boundary = logit(thr/(1-thr)) read from
                  the live test_cfg; writes ONLY on the wrong side of that
                  target (correct-side points get exactly zero), never
                  overshooting the target within the cap.  Its gain is a
                  demonstrated oracle-achievable value under the fixed
                  constraints (K=64 lowest-|z| selection, matched ROIs,
                  256-rasterised labels, cap, margin 0.10, single write,
                  frozen detector) — a lower bound on oracle value, not an
                  upper bound on all policies.

GT enters only AFTER the frozen decoder/tail forward, to relabel and rebuild
the residual; detector fields (boxes/scores/labels) and proposals are never
touched, and every cell must retain the standard detector hash.  Unmatched
proposals (IoU<0.5 same-class) receive no write in oracle cells.  Exported
per-image COCO records are compatible with tools/bootstrap_paired_map.py.

Pre-registered decision rule (written before any cell was evaluated): if the
gt_cap - no_write paired image-bootstrap segm/mAP 95% CI lower bound is below
+0.005, the K64 tail-write program is closed at these weights; if gt_cap is
>= +0.01 significant while gt_auth - no_write is not, a margin-style training
arm (magnitude + authorisation) is the justified follow-up; if gt_auth -
no_write alone clears +0.005 (CI lower bound > 0), only an authorisation head
(gate) is justified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

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
    _gt_cache, _match_gt_index,
)
from inference.probes.tail_input_readability_probe import (  # noqa: E402
    _target_for_final_mask_contract,
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
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.10)
    p.add_argument("--num-points", type=int, default=0,
                   help="override the tail selection budget K at frozen weights "
                        "(0 = checkpoint value). Selection is a stable |z| rank, "
                        "so smaller K is an exact prefix; larger K evaluates the "
                        "trained MLP on points it never trained on (extrapolation, "
                        "flagged in the payload).")
    p.add_argument("--cells", nargs="+", default=None,
                   help="subset of cells to run (default: all). Useful for K sweeps.")
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
    _tail_mode = str(head._decoder_tail_cfg.get("mode", "residual_v1_stop"))
    if not (head.decoder_tail_enabled
            and getattr(head, "_decoder_tail_train_base_logits", False)
            and _tail_mode in {"residual_v1_stop", "residual_v1_stop_margin"}):
        raise RuntimeError(
            f"probe requires a tailstop-family checkpoint "
            f"(residual_v1_stop[_margin]); got mode={_tail_mode!r}"
        )
    out = Path(a.output_dir).expanduser().resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)
    categories = VHR10_CATEGORIES if int(contract.get("num_classes", 0) or 0) == 10 else None
    original_predict = model.roi_head.predict_mask
    original_tail = head.decoder_tail_refiner.forward
    cap = float(head.decoder_tail_refiner.delta_logit_max)
    trained_k = int(head.decoder_tail_refiner.num_points)
    if a.num_points:
        if a.num_points < 1:
            raise SystemExit("--num-points must be >= 1")
        head.decoder_tail_refiner.num_points = int(a.num_points)
    rcnn_test_cfg = getattr(model.roi_head, "test_cfg", None) or {}
    thr = float(rcnn_test_cfg.get("mask_thr_binary", 0.5))
    if not 0.0 < thr < 1.0:
        raise RuntimeError(f"mask_thr_binary={thr} outside (0,1); threshold-aware cell undefined")
    # The deployed binarisation boundary in logit space. gt_cap (margin around
    # logit 0) is mis-calibrated against this boundary: an audited run showed
    # it cannot delete false-positive area and pulls correct negatives inside.
    # gt_cap_thr writes to boundary +/- margin and only on the wrong side.
    boundary = float(torch.log(torch.tensor(thr / (1.0 - thr))).item())

    def run_pass(mode: str, *, unhooked: bool = False) -> Dict[str, Any]:
        state: Dict[str, Any] = {"queue": None, "cursor": 0, "boxes": None, "ids": None,
                                 "labels": None, "gts": None,
                                 "tail_calls": 0, "matched_rois": 0, "total_rois": 0,
                                 "written_points": 0, "selected_points": 0,
                                 "flip": 0, "destroy": 0, "written_delta_abs_sum": 0.0}

        def predict(*fa: Any, **kw: Any) -> Any:
            results = kw.get("results_list", fa[2] if len(fa) > 2 else None)
            state["queue"] = torch.cat([r.labels.detach() for r in results]) if results else torch.zeros(0, dtype=torch.long, device=device)
            state["cursor"] = 0
            return original_predict(*fa, **kw)

        def pre(_m: Any, _fa: Tuple[Any, ...], kw: Dict[str, Any]) -> None:
            boxes, ids = kw.get("boxes"), kw.get("roi_img_ids")
            state["boxes"], state["ids"] = boxes, ids
            if boxes is not None:
                s, e = state["cursor"], state["cursor"] + boxes.shape[0]
                state["labels"], state["cursor"] = state["queue"][s:e], e

        def tail(logits: torch.Tensor, upscaled: torch.Tensor, tokens: torch.Tensor) -> Any:
            refined, outputs = original_tail(logits, upscaled, tokens)
            state["tail_calls"] += 1
            if mode == "p0":
                return refined, outputs
            n, _, h, w = logits.shape
            flat = logits[:, 0].reshape(n, -1)
            indices = outputs["indices"]
            z0 = flat.gather(1, indices)
            if mode == "no_write":
                state["total_rois"] += n
                return logits, outputs
            delta = torch.zeros_like(z0)
            for i in range(n):
                state["total_rois"] += 1
                entry = state["gts"][int(state["ids"][i])]
                gt_index = _match_gt_index(state["boxes"][i].detach().cpu(),
                                           state["labels"][i].detach().cpu(), entry, a.match_iou)
                if gt_index is None:
                    continue
                state["matched_rois"] += 1
                target = _target_for_final_mask_contract(
                    head, entry[2][gt_index], state["boxes"][i].detach().cpu(), (h, w)
                ).reshape(-1)[indices[i].cpu()].to(device=z0.device, dtype=z0.dtype)
                error = z0[i].ge(0.0) != target.gt(0.5)
                if mode == "gt_auth":
                    delta[i] = outputs["delta"][i] * error.to(z0.dtype)
                elif mode == "inv_auth":
                    delta[i] = outputs["delta"][i] * (~error).to(z0.dtype)
                elif mode == "gt_cap":
                    signed = target.mul(2.0).sub(1.0)
                    delta[i] = (a.margin * signed - z0[i]).clamp(-cap, cap)
                elif mode == "gt_cap_thr":
                    signed = target.mul(2.0).sub(1.0)
                    target_logit = boundary + signed * a.margin
                    delta[i] = signed * (signed * (target_logit - z0[i])).clamp(min=0.0, max=cap)
                else:
                    raise RuntimeError(f"unknown mode {mode!r}")
                state["selected_points"] += int(indices.shape[1])
                state["written_points"] += int((delta[i].abs() > 1e-6).sum())
                state["written_delta_abs_sum"] += float(delta[i].abs().sum())
                flipped = ((z0[i] + delta[i]).ge(0.0)) != z0[i].ge(0.0)
                state["flip"] += int((flipped & error).sum())
                state["destroy"] += int((flipped & ~error).sum())
            new_refined = flat.scatter(1, indices, z0 + delta).reshape(n, 1, h, w)
            return new_refined, outputs

        detector_hash = hashlib.sha256(); output_hash = hashlib.sha256()
        all_gt: List[dict] = []; all_dt: List[dict] = []; metas: List[dict] = []
        hook_handle = None
        if not unhooked:
            model.roi_head.predict_mask = predict
            hook_handle = head.register_forward_pre_hook(pre, with_kwargs=True)
            head.decoder_tail_refiner.forward = tail
        try:
            with torch.inference_mode():
                for batch_i, batch in enumerate(loader):
                    if a.max_batches and batch_i >= a.max_batches: break
                    inputs, samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, training=False)
                    state["gts"] = _gt_cache(processed["data_samples"])
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
        finally:
            model.roi_head.predict_mask = original_predict
            if hook_handle is not None:
                hook_handle.remove()
            head.decoder_tail_refiner.forward = original_tail
        if not unhooked and state["tail_calls"] == 0:
            raise RuntimeError("hooked pass did not call the tail refiner")
        if not unhooked and state["cursor"] != int(state["queue"].numel()):
            raise RuntimeError("proposal label queue mismatch")
        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, metas, categories=categories)
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        gt_records, dt_records, images = _records(all_gt, all_dt, metas)
        cell_dir = out / ("standard" if unhooked else mode); cell_dir.mkdir()
        payload = {"checkpoint": str(ckpt_path), "weights": a.weights, "dataset": dict(contract),
                   "mode": "standard" if unhooked else mode, "frozen_weights": True,
                   "margin": a.margin, "match_iou": a.match_iou, "delta_logit_max": cap,
                   "num_points": int(head.decoder_tail_refiner.num_points),
                   "trained_num_points": trained_k,
                   "selection_is_extrapolation": bool(a.num_points and a.num_points > trained_k),
                   "mask_thr_binary": thr, "boundary_logit": boundary,
                   "processed_images": len(images),
                   "processed_image_ids": sorted(x["image_id"] for x in images),
                   "metrics": metrics, "detector_sha256": detector_hash.hexdigest(),
                   "output_sha256": output_hash.hexdigest(),
                   "mechanism": {k: state[k] for k in ("tail_calls", "matched_rois", "total_rois",
                                                       "written_points", "selected_points",
                                                       "flip", "destroy", "written_delta_abs_sum")}}
        _write_json(cell_dir / "gt_records.json", gt_records); _write_json(cell_dir / "dt_records.json", dt_records)
        _write_json(cell_dir / "images.json", images); _write_json(cell_dir / "metrics.json", metrics)
        _write_json(cell_dir / "run_manifest.json", payload)
        return {"metrics": {k: metrics[k] for k in ("segm/mAP", "segm/mAP_50", "segm/mAP_75", "segm/AR@100") if k in metrics},
                "detector_sha256": payload["detector_sha256"], "output_sha256": payload["output_sha256"],
                "mechanism": payload["mechanism"], "images": len(images)}

    standard = run_pass("p0", unhooked=True)
    all_cells = ["p0", "no_write", "gt_auth", "inv_auth", "gt_cap", "gt_cap_thr"]
    selected = [c for c in (a.cells or all_cells) if c != "standard"]
    unknown = set(a.cells or []) - set(all_cells)
    if unknown:
        raise SystemExit(f"unknown cells {sorted(unknown)}")
    cells = {name: run_pass(name) for name in selected}
    if cells["p0"]["output_sha256"] != standard["output_sha256"]:
        raise RuntimeError("passive P0 hook is not bitwise identical to unhooked standard")
    for name, cell in cells.items():
        if cell["detector_sha256"] != standard["detector_sha256"]:
            raise RuntimeError(f"detector changed under {name}")
    base = cells["no_write"]["metrics"]["segm/mAP"]
    summary = {"checkpoint": str(ckpt_path), "weights": a.weights, "strict_load": strict,
               "config_source": config_source, "contract": contract, "standard": standard,
               "cells": cells,
               "delta_vs_no_write": {name: cell["metrics"]["segm/mAP"] - base
                                     for name, cell in cells.items() if name != "no_write"},
               "intervention_contract": (
                   "no_write = base z only. gt_auth applies the LEARNED delta only on selected "
                   "points whose base sign contradicts the matched-GT label (IoU>=0.5 same-class). "
                   "inv_auth applies the same operation with inverted authorisation (correct points "
                   "only; sham control). gt_cap (pre-audit) writes clip(margin*y - z0, +/-cap) on "
                   "all selected points — mis-calibrated against mask_thr_binary and retained only "
                   "as the recorded negative control. gt_cap_thr (post-audit corrective) writes "
                   "signed*clamp(signed*(boundary + signed*margin - z0), 0, cap): wrong-side-only "
                   "relative to the deployed binarisation boundary. Unmatched proposals receive no "
                   "write. GT is consulted only after the frozen forward; detector hash must be "
                   "identical in every cell."),
               "pre_registered_decision": (
                   "If gt_cap - no_write CI95 lower bound < +0.005: close the K64 tail-write "
                   "program at these weights. If gt_cap significant >= +0.01 while gt_auth - "
                   "no_write is not: margin-style training arm (magnitude+authorisation). If "
                   "gt_auth - no_write alone clears +0.005 (CI lb > 0): authorisation head only."),
               }
    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
