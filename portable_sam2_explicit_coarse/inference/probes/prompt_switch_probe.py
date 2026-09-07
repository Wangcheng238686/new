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
    _select_state_dict,
    _snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = full validation pass")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


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

    def run_pass(mode):
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

        handle = head.prompt_encoder.register_forward_pre_hook(
            pe_pre_hook, with_kwargs=True
        )
        all_gt, all_dt, all_metas = [], [], []
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
                        meta = batch["img_metas"][position]
                        shape = meta["img_shape"]
                        gt = _extract_instances_numpy(
                            processed["data_samples"][position].gt_instances, shape
                        )
                        pred_instances = (
                            output.pred_instances
                            if hasattr(output, "pred_instances") else output
                        )
                        all_gt.append(gt)
                        all_dt.append(_extract_instances_numpy(pred_instances, shape))
                        all_metas.append({"img_shape": shape})
                    n_batches += 1
        finally:
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
        keep = ("segm/mAP", "segm/mAP_50", "segm/AR@100")
        return {
            "batches": n_batches,
            "images": len(all_metas),
            "metrics": {k: metrics[k] for k in keep if k in metrics},
        }

    contract = _resolve_dataset_contract(ns, snapshot)
    eval_categories = None
    if int(contract.get("num_classes", 0) or 0) == 10:
        from utils.coco_eval_utils import VHR10_CATEGORIES
        eval_categories = VHR10_CATEGORIES
    loader = _build_loader(contract, 0)
    results = {}
    modes = ["baseline", "drop_box", "drop_dense", "drop_both"]
    if head.p2_boundary_refiner is not None:
        modes.append("p2_off")
    for mode in modes:
        results[mode] = run_pass(mode)
        print(f"[{mode}] mAP={results[mode]['metrics'].get('segm/mAP'):.4f}",
              flush=True)
    base = results["baseline"]["metrics"]["segm/mAP"]
    summary = {
        "checkpoint": str(checkpoint_path),
        "weights": args.weights,
        "cells": results,
        "delta_vs_baseline": {
            mode: results[mode]["metrics"]["segm/mAP"] - base
            for mode in results if mode != "baseline"
        },
        "note": (
            "frozen-weight prompt-switch: points/detections/ranking unchanged; "
            "characterises this weight's prompt dependence, NOT a training "
            "ablation"
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
