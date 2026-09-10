#!/usr/bin/env python3
"""Frozen A0 geometric interventions, with immutable detector and other prompts.

box_gt: replace only PE box input on same-class IoU >= .5 matched proposals.
support_expand: expand only dense embedding support, matched proposals only,
about the original centre to 1.1 times total width/height (5% on each side).
Neither intervention changes proposal features, coarse, points or canvas.
GT matching is privileged diagnostic evidence, not a deployable selector.
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


def expanded_boxes(boxes, matched, factor=1.1):
    """No mutation or clipping: support rasterizer naturally clips at image edge."""
    result = boxes.clone()
    centre = (boxes[:, :2] + boxes[:, 2:]) * .5
    half = (boxes[:, 2:] - boxes[:, :2]) * (.5 * factor)
    expanded = torch.cat((centre - half, centre + half), dim=1)
    result[matched] = expanded[matched]
    return result


class GeometryHooks:
    """Temporary hooks; track row order, actual PE/support boxes and all invariants."""

    def __init__(self, model, mode):
        if mode not in ("raw", "box_gt", "support_expand"):
            raise ValueError(mode)
        self.model, self.head, self.mode = model, model.roi_head.mask_head, mode
        self.state = {}
        self.rows = []
        self.counts = dict(proposals=0, matched=0, pe_calls=0, support_calls=0,
                           pe_box_changed=0, support_box_changed=0, support_grid_changed=0)
        self.hashes = {key: hashlib.sha256() for key in (
            "points_coords", "points_labels", "canvas", "pre_support_dense",
            "pe_boxes_before", "pe_boxes_after", "support_boxes_before", "support_boxes_after",
            "support_grid", "roi_boxes",
        )}
        self.handles, self.restores = [], []

    def record(self, key, value):
        if value is None:
            raise RuntimeError(f"required active PBM input absent: {key}")
        _hash_update(self.hashes[key], value.detach().cpu().numpy())

    def patch(self, obj, name, function):
        self.restores.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, function)

    def install(self):
        original_predict = self.model.roi_head.predict_mask

        def predict(*a, **kw):
            results = kw.get("results_list", a[2] if len(a) > 2 else None)
            self.state["queue"] = torch.cat([r.labels.detach().cpu() for r in results])
            self.state["cursor"] = 0
            result = original_predict(*a, **kw)
            if self.state["cursor"] != len(self.state["queue"]):
                raise RuntimeError("unconsumed class queue")
            return result

        self.patch(self.model.roi_head, "predict_mask", predict)
        self.handles.append(self.head.register_forward_pre_hook(self.pre, with_kwargs=True))
        self.handles.append(self.head.prompt_encoder.register_forward_pre_hook(self.pe, with_kwargs=True))
        original_dense = self.head._forward_dense_embeddings

        def dense(*a, **kw):
            value = kw.get("dense_pe", a[1] if len(a) > 1 else None)
            valid = kw.get("shape_dense_valid", a[2] if len(a) > 2 else None)
            if valid is not None and not bool(valid.bool().all()):
                raise RuntimeError("probe requires all-valid raw dense")
            self.record("pre_support_dense", value)
            return original_dense(*a, **kw)

        self.patch(self.head, "_forward_dense_embeddings", dense)
        original_support = self.head._dense_embedding_box_support

        def support(boxes, spatial_size, dtype):
            self.counts["support_calls"] += 1
            if not torch.equal(boxes.detach().cpu(), self.state["boxes"]):
                raise RuntimeError("dense support lost proposal-frame identity")
            self.record("support_boxes_before", boxes)
            selected = torch.tensor([x is not None for x in self.state["targets"]], device=boxes.device)
            changed = expanded_boxes(boxes, selected) if self.mode == "support_expand" else boxes
            value = original_support(changed, spatial_size, dtype)
            baseline = original_support(boxes, spatial_size, dtype) if self.mode == "support_expand" else value
            self.record("support_boxes_after", changed)
            self.record("support_grid", value)
            for j, row in enumerate(self.state["chunk_rows"]):
                row["support_box_before"] = boxes[j].detach().cpu().tolist()
                row["support_box_after"] = changed[j].detach().cpu().tolist()
                row["support_pixels_before"] = int(baseline[j].sum())
                row["support_pixels_after"] = int(value[j].sum())
                did = not torch.equal(boxes[j], changed[j])
                grid_did = not torch.equal(baseline[j], value[j])
                self.counts["support_box_changed"] += int(did)
                self.counts["support_grid_changed"] += int(grid_did)
            return value

        self.patch(self.head, "_dense_embedding_box_support", support)

    def pre(self, module, args, kwargs):
        boxes, ids = kwargs["boxes"].detach().cpu(), kwargs["roi_img_ids"].detach().cpu()
        start = self.state["cursor"]
        labels = self.state["queue"][start:start + len(boxes)]
        if len(labels) != len(boxes):
            raise RuntimeError("proposal class queue mismatch")
        self.state.update(boxes=boxes, ids=ids, targets=[], chunk_rows=[])
        self.state["cursor"] += len(boxes)
        self.record("roi_boxes", boxes)
        for j, box in enumerate(boxes):
            image_index = int(ids[j])
            target = _match_gt_index(box, labels[j], self.state["gts"][image_index], .5)
            row = dict(image_id=self.state["image_ids"][image_index],
                       row=self.state["row_counts"][image_index], label=int(labels[j]),
                       target_index=target, proposal_box=box.tolist())
            self.state["row_counts"][image_index] += 1
            self.state["targets"].append(target)
            self.state["chunk_rows"].append(row)
            self.rows.append(row)
            self.counts["proposals"] += 1
            self.counts["matched"] += int(target is not None)

    def pe(self, module, args, kwargs):
        self.counts["pe_calls"] += 1
        points = kwargs["points"]
        if points is None:
            raise RuntimeError("requires active points")
        self.record("points_coords", points[0]); self.record("points_labels", points[1])
        self.record("canvas", kwargs["masks"])
        boxes = kwargs["boxes"]
        self.record("pe_boxes_before", boxes)
        if not torch.equal(boxes.detach().cpu(), self.state["boxes"]):
            raise RuntimeError("PE box differs from proposal frame")
        changed = boxes.clone()
        for j, target in enumerate(self.state["targets"]):
            if target is not None and self.mode == "box_gt":
                entry = self.state["gts"][int(self.state["ids"][j])]
                changed[j] = entry[0][target].to(boxes)
            row = self.state["chunk_rows"][j]
            row["pe_box_before"] = boxes[j].detach().cpu().tolist()
            row["pe_box_after"] = changed[j].detach().cpu().tolist()
            self.counts["pe_box_changed"] += int(not torch.equal(boxes[j], changed[j]))
        self.record("pe_boxes_after", changed)
        return args, dict(kwargs, boxes=changed)

    def close(self):
        for handle in self.handles:
            handle.remove()
        for obj, name, existed, value in reversed(self.restores):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--canonical-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=0)
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
    if not head.restrict_dense_prompt_to_box or getattr(head, "decoder_tail_refiner", None) is not None:
        raise RuntimeError("requires box-limited dense and tail-off A0")
    if not all((head.explicit_use_point_prompt, head.explicit_use_box_prompt, head.explicit_use_dense_prompt)):
        raise RuntimeError("requires all three active PBM modalities")
    contract = _resolve_dataset_contract(ns, snap)
    if int(contract.get("num_classes", 0)) != len(VHR10_CATEGORIES):
        raise RuntimeError("this diagnostic is registered for VHR-10 only")
    loader = _build_loader(contract, 0)

    def run(mode):
        all_gt, all_dt, metas = [], [], []
        detector, outputs = hashlib.sha256(), hashlib.sha256()
        hooks = None if mode == "standard" else GeometryHooks(model, mode)
        try:
            if hooks is not None:
                hooks.install()
            with torch.inference_mode():
                for batch_index, batch in enumerate(loader):
                    if args.max_batches and batch_index >= args.max_batches:
                        break
                    inputs, samples = _build_data_samples(batch, torch.device("cuda:0"))
                    data = model.data_preprocessor(dict(inputs=inputs, data_samples=samples), training=False)
                    if hooks is not None:
                        hooks.state.update(gts=_gt_cache(data["data_samples"]),
                            image_ids=[int(m.get("image_id", m.get("scene_id", 0))) for m in batch["img_metas"]],
                            row_counts=[0] * len(samples))
                    predictions = model.predict(data["inputs"], data["data_samples"], rescale=False)
                    for i, prediction in enumerate(predictions):
                        meta = dict(batch["img_metas"][i])
                        gt = _extract_instances_numpy(data["data_samples"][i].gt_instances, meta["img_shape"])
                        dt = _extract_instances_numpy(prediction.pred_instances, meta["img_shape"])
                        image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
                        for digest in (detector, outputs):
                            _hash_update(digest, np.array([image_id], dtype=np.int64))
                        for key in ("bboxes", "scores", "labels"):
                            _hash_update(detector, dt[key]); _hash_update(outputs, dt[key])
                        _hash_update(outputs, dt["masks"])
                        if hooks is not None:
                            rows = [r for r in hooks.rows if r["image_id"] == image_id]
                            assert len(rows) == len(dt["bboxes"])
                            for j, row in enumerate(rows):
                                assert row["row"] == j and row["label"] == int(dt["labels"][j])
                                assert np.array_equal(row["proposal_box"], dt["bboxes"][j])
                        all_gt.append(gt); all_dt.append(dt); metas.append(meta)
                    print(f"{mode} batch={batch_index + 1}", flush=True)
        finally:
            if hooks is not None:
                hooks.close()
        coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, metas, categories=VHR10_CATEGORIES)
        metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
        gt_records, dt_records, images = _records(all_gt, all_dt, metas)
        arm = out / mode; arm.mkdir()
        manifest = dict(checkpoint=str(ckpt_path), dataset=dict(contract), processed_images=len(images),
                        processed_image_ids=sorted(x["image_id"] for x in images), ground_truth_instances=len(gt_records),
                        mode=mode, partial=bool(args.max_batches), probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        payloads = dict(gt_records=gt_records, dt_records=dt_records, images=images, metrics=metrics, run_manifest=manifest)
        if hooks is not None:
            assert hooks.counts["pe_calls"] == hooks.counts["support_calls"] > 0
            payloads["interventions"] = hooks.rows
        for name, value in payloads.items():
            _write_json(arm / f"{name}.json", value)
        return dict(mode=mode, metrics=metrics, detector_sha256=detector.hexdigest(), output_sha256=outputs.hexdigest(),
                    counts={} if hooks is None else hooks.counts,
                    hashes={} if hooks is None else {k: v.hexdigest() for k, v in hooks.hashes.items()})

    standard = run("standard")
    if not args.max_batches:
        _validate_canonical(out / "standard", Path(args.canonical_dir))
    raw = run("raw")
    assert raw["output_sha256"] == standard["output_sha256"], "identity hooks altered output"
    arms = {"standard": standard, "raw": raw}
    for mode in ("box_gt", "support_expand"):
        arm = run(mode)
        assert arm["detector_sha256"] == standard["detector_sha256"] == raw["detector_sha256"]
        invariant_keys = ("points_coords", "points_labels", "canvas", "pre_support_dense", "pe_boxes_before", "support_boxes_before", "roi_boxes")
        invariant_keys += ("support_boxes_after", "support_grid") if mode == "box_gt" else ("pe_boxes_after",)
        for key in invariant_keys:
            assert arm["hashes"][key] == raw["hashes"][key], f"{mode} altered {key}"
        for key in ("proposals", "matched", "pe_calls", "support_calls"):
            assert arm["counts"][key] == raw["counts"][key]
        arms[mode] = arm
    summary = dict(strict_load=strict, config_source=source, canonical_verified=not bool(args.max_batches),
                   protocol="same-class IoU>=0.5; unmatched unchanged; B PE-box GT only; S total width/height x1.1 only", **arms)
    _write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
