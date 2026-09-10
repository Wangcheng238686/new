#!/usr/bin/env python3
"""Frozen A0 point-reliability Oracle: invalidate semantically incorrect points.

Uses production invalid-point zeroing, NOT attention-token deletion. GT is only
used for class-aware matching and point membership. Coarse/dense/boxes unchanged.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from inference.infer_from_checkpoint import (
    _build_data_samples, _build_loader, _extract_instances_numpy,
    _load_checkpoint, _load_model_state, _register_and_build,
    _resolve_dataset_contract, _resolve_model_config, _select_state_dict,
    _snapshot, _write_json,
)
from inference.probes.coarse_to_mining_oracle_probe import (
    _gt_cache, _hash_update, _match_gt_index, _records,
)
from inference.probes.p2_support_decomposition_oracle import _validate_canonical
from utils.coco_eval_utils import VHR10_CATEGORIES, build_coco_gt_and_dt, run_coco_eval


def unreliable_points(coords, labels, target):
    """Full-image floor sampling; invalid slots stay invalid."""
    h, w = target.shape
    xy = coords.floor().long()
    if not bool(((xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)).all()):
        raise RuntimeError("point outside target image")
    membership = target[xy[:, 1], xy[:, 0]].bool()
    return (labels >= 0) & (labels.eq(1) != membership)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--random-control", action="store_true",
                        help="same per-ROI/sign suppression count; random valid slots, seed44")
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = _load_checkpoint(ckpt_path)
    snap = _snapshot(ckpt)
    ns = SimpleNamespace(checkpoint=str(ckpt_path), config=None, split="validation", data_root=None,
                         ann_file=None, image_subdir=None, image_size=None, batch_size=None,
                         sam2_repo=None, sam2_ckpt=None)
    cfg, source = _resolve_model_config(ns, snap)
    model = _register_and_build(cfg)
    strict = _load_model_state(model, _select_state_dict(ckpt, "model"), allow_nonstrict=False)
    model.cuda().eval()
    head = model.roi_head.mask_head
    if head.p2_boundary_refiner is not None or head.final_mask_coordinate_mode != "full_image":
        raise RuntimeError("requires P2-off full-image A0")
    if getattr(head, "roi_sam_enabled", False) or head.shape_point_miner is None:
        raise RuntimeError("requires global-coordinate ShapePointMiner")
    contract = _resolve_dataset_contract(ns, snap)
    loader = _build_loader(contract, 0)

    def run(mode):
        state = {}
        rng = np.random.default_rng(44)
        counts = dict(proposals=0, matched=0, eligible=0, valid_positive=0, valid_negative=0, wrong_positive=0, wrong_negative=0, suppressed=0)
        all_gt, all_dt, metas, diagnostics = [], [], [], []
        detector, outputs_hash, unchanged_prompt = (hashlib.sha256() for _ in range(3))
        original = model.roi_head.predict_mask
        had = "predict_mask" in model.roi_head.__dict__
        saved = model.roi_head.__dict__.get("predict_mask")

        def predict(*a, **kw):
            results = kw.get("results_list", a[2] if len(a) > 2 else None)
            state["queue"] = torch.cat([r.labels.detach().cpu() for r in results])
            state["cursor"] = 0
            return original(*a, **kw)

        def pre(module, a, kw):
            boxes, ids = kw["boxes"].detach().cpu(), kw["roi_img_ids"].detach().cpu()
            start = state["cursor"]
            state.update(boxes=boxes, ids=ids, labels=state["queue"][start:start+len(boxes)])
            state["cursor"] += len(boxes)
            if len(state["labels"]) != len(boxes):
                raise RuntimeError("proposal queue mismatch")

        def points(module, a, result):
            coords, labels, stats = result
            changed = labels.clone()
            _hash_update(unchanged_prompt, coords.cpu().numpy())
            _hash_update(unchanged_prompt, labels.cpu().numpy())
            for j, box in enumerate(state["boxes"]):
                image_index = int(state["ids"][j])
                entry = state["gts"][image_index]
                target_index = _match_gt_index(box, state["labels"][j], entry, .5)
                counts["proposals"] += 1
                bad = torch.zeros(4, dtype=torch.bool)
                cpu_labels = labels[j].cpu()
                if target_index is not None:
                    counts["matched"] += 1
                    bad = unreliable_points(coords[j].cpu(), cpu_labels, entry[2][target_index])
                    counts["valid_positive"] += int(cpu_labels.eq(1).sum())
                    counts["valid_negative"] += int(cpu_labels.eq(0).sum())
                    counts["wrong_positive"] += int((bad & cpu_labels.eq(1)).sum())
                    counts["wrong_negative"] += int((bad & cpu_labels.eq(0)).sum())
                eligible = bool(bad.any())
                counts["eligible"] += int(eligible)
                state["rows"][image_index].append((box, target_index, eligible))
                remove = bad & cpu_labels.eq(0) if mode == "negative" else bad
                if mode in ("negative", "all"):
                    if args.random_control:
                        sampled = torch.zeros_like(remove)
                        for sign in (0, 1):
                            count = int((remove & cpu_labels.eq(sign)).sum())
                            available = torch.where(cpu_labels.eq(sign))[0].numpy()
                            if count:
                                sampled[rng.choice(available, count, replace=False)] = True
                        remove = sampled
                    changed[j, remove.to(labels.device)] = -1
                    counts["suppressed"] += int(remove.sum())
            assert bool(((changed == labels) | ((changed == -1) & (labels >= 0))).all())
            return coords, changed, stats

        def pe_input(module, a, kw):
            for key in ("boxes", "masks"):
                value = kw[key]
                if value is None:
                    raise RuntimeError("requires active A0 box and dense prompts")
                _hash_update(unchanged_prompt, value.detach().cpu().numpy())

        handles = []
        if mode != "standard":
            model.roi_head.predict_mask = predict
            handles = [head.register_forward_pre_hook(pre, with_kwargs=True),
                       head.shape_point_miner.register_forward_hook(points),
                       head.prompt_encoder.register_forward_pre_hook(pe_input, with_kwargs=True)]
        try:
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    if args.max_batches and batch_index >= args.max_batches:
                        break
                    inputs, samples = _build_data_samples(batch, torch.device("cuda:0"))
                    data = model.data_preprocessor(dict(inputs=inputs, data_samples=samples), training=False)
                    state["gts"] = _gt_cache(data["data_samples"])
                    state["rows"] = [[] for _ in samples]
                    predictions = model.predict(data["inputs"], data["data_samples"], rescale=False)
                    if mode != "standard" and state["cursor"] != len(state["queue"]):
                        raise RuntimeError("unconsumed proposal queue")
                    for i, prediction in enumerate(predictions):
                        meta = dict(batch["img_metas"][i])
                        gt = _extract_instances_numpy(data["data_samples"][i].gt_instances, meta["img_shape"])
                        dt = _extract_instances_numpy(prediction.pred_instances, meta["img_shape"])
                        image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
                        _hash_update(detector, np.array([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _hash_update(detector, dt[key]); _hash_update(outputs_hash, dt[key])
                        _hash_update(outputs_hash, dt["masks"])
                        diag = dict(image_id=image_id, eligible=0, neighbor_pixels=0, neighbor_covered=0,
                                    target_pixels=0, target_covered=0)
                        if mode != "standard":
                            rows = state["rows"][i]
                            assert len(rows) == len(dt["bboxes"])
                            for row, (box, target_index, eligible) in enumerate(rows):
                                assert np.array_equal(box.numpy(), dt["bboxes"][row])
                                if not eligible:
                                    continue
                                masks = state["gts"][i][2].bool().numpy()
                                target = masks[target_index]
                                others = masks.copy(); others[target_index] = False
                                neighbor = others.any(0) & ~target
                                predicted = np.asarray(dt["masks"][row], dtype=bool)
                                assert predicted.shape == target.shape
                                diag["eligible"] += 1
                                diag["neighbor_pixels"] += int(neighbor.sum())
                                diag["neighbor_covered"] += int((neighbor & predicted).sum())
                                diag["target_pixels"] += int(target.sum())
                                diag["target_covered"] += int((target & predicted).sum())
                        diagnostics.append(diag)
                        all_gt.append(gt); all_dt.append(dt); metas.append(meta)
                    print(f"{mode} batch={batch_index+1} counts={counts}", flush=True)
        finally:
            for handle in handles:
                handle.remove()
            if mode != "standard":
                if had: model.roi_head.predict_mask = saved
                else: delattr(model.roi_head, "predict_mask")
        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, metas, categories=VHR10_CATEGORIES)
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        gt_records, dt_records, images = _records(all_gt, all_dt, metas)
        arm = out / mode; arm.mkdir()
        manifest = dict(checkpoint=str(ckpt_path), dataset=dict(contract), processed_images=len(images),
                        processed_image_ids=sorted(x["image_id"] for x in images), ground_truth_instances=len(gt_records))
        for name, value in dict(gt_records=gt_records, dt_records=dt_records, images=images,
                                metrics=metrics, run_manifest=manifest, diagnostics=diagnostics).items():
            _write_json(arm / f"{name}.json", value)
        return dict(mode=mode, metrics=metrics, counts=counts, detector_sha256=detector.hexdigest(),
                    output_sha256=outputs_hash.hexdigest(), unchanged_prompt_sha256=unchanged_prompt.hexdigest())

    standard = run("standard")
    if not args.max_batches:
        _validate_canonical(out / "standard", Path(args.canonical_dir))
    raw = run("raw")
    assert raw["output_sha256"] == standard["output_sha256"], "identity hook altered output"
    cells = [run("negative"), run("all")]
    for cell in cells:
        assert standard["detector_sha256"] == raw["detector_sha256"] == cell["detector_sha256"]
        assert raw["unchanged_prompt_sha256"] == cell["unchanged_prompt_sha256"]
        assert {k:v for k,v in raw["counts"].items() if k != "suppressed"} == {k:v for k,v in cell["counts"].items() if k != "suppressed"}
    summary = dict(strict_load=strict, config_source=source, standard=standard, raw=raw, cells=cells,
                   canonical_verified=not bool(args.max_batches),
                   random_control=args.random_control, random_seed=44,
                   protocol="matched IoU0.5; floor GT membership; invalid label -1 via production zero-token path")
    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
