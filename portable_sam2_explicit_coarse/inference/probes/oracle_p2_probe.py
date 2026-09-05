#!/usr/bin/env python
"""Oracle probe: measure the value ceiling of the P2 refinement route.

Replaces the learned P2BoundaryRefiner residual with the GT-derived optimal
in-band correction at inference time and reports final COCO metrics:

    delta* = clamp(target_logits - raw_logits, +/-cap) * search_support
    target_logits = K * (2*gt - 1)   (gt = box-matched GT mask at 64x64)

If even this oracle correction cannot move segm/mAP, no trained refiner can
help under the current coupling (mined points + gated dense prompt + decoder),
and the coupling itself must be redesigned before spending GPU-days training.

Support mask (band radius, coverage rejection) is inherited from the real
refiner so the probe respects the mechanism's spatial constraints.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MAINLINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAINLINE_ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples,
    _build_loader,
    _extract_instances_numpy,
    _filter_predictions,
    _load_checkpoint,
    _load_model_state,
    _register_and_build,
    _resolve_dataset_contract,
    _resolve_model_config,
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)

logger = logging.getLogger("oracle_probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("validation", "test", "custom"), default="validation")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--image-subdir", default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--oracle-cap", type=float, default=None,
                        help="clamp on the oracle delta (0.4 = trained beta bound, 8 = full override)")
    parser.add_argument("--oracle-points", action="store_true",
                        help="Feed GT crops to the shape-point miner instead of the predicted coarse "
                             "mask (perfect points; dense prompt stays raw). Mutually exclusive with "
                             "--oracle-cap.")
    parser.add_argument("--oracle-points-k", type=int, default=0,
                        help="If >0, ignore the miner and place K GT positives (interior FPS) + K GT "
                             "negatives (outer-ring FPS) as the point prompt. Answers: does the 2P2N "
                             "budget bind?")
    parser.add_argument("--target-logit-k", type=float, default=8.0)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--sam2-repo", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


class OracleP2Wrapper:
    """Swap the refiner's learned delta for the GT-derived optimum.

    Also records the per-chunk global-frame ROIs so the point-miner oracle can
    align GT crops to miner rows (refiner and miner see the same chunk order).
    """

    def __init__(self, refiner, cap: Optional[float], k: float, match_iou: float):
        self.refiner = refiner
        self.cap = None if cap is None else float(cap)
        self.k = float(k)
        self.match_iou = float(match_iou)
        self.last_rois = None
        self.oracle_rois_recorder = None
        self._orig_forward = refiner.forward
        self.gt_cache: List[Optional[Tuple[np.ndarray, torch.Tensor]]] = []
        refiner.forward = self._wrapped  # type: ignore[method-assign]

    def set_batch_gt(self, entries: List[Optional[Tuple[np.ndarray, torch.Tensor]]]) -> None:
        self.gt_cache = entries

    def close(self) -> None:
        self.refiner.forward = self._orig_forward  # type: ignore[method-assign]

    def _wrapped(self, p2_feature, prompt_rois, raw_logits):
        outputs = dict(self._orig_forward(p2_feature, prompt_rois, raw_logits))
        rois_np = prompt_rois.detach().cpu().numpy()
        self.last_rois = rois_np
        if self.oracle_rois_recorder is not None:
            self.oracle_rois_recorder.append(rois_np)
        if self.cap is None or prompt_rois.shape[0] == 0 or not self.gt_cache:
            return outputs
        with torch.no_grad():
            delta = torch.zeros_like(raw_logits)
            matched = 0
            for n in range(raw_logits.shape[0]):
                img = int(rois_np[n, 0])
                if img >= len(self.gt_cache) or self.gt_cache[img] is None:
                    continue
                gt_boxes, gt_masks = self.gt_cache[img]
                if len(gt_boxes) == 0:
                    continue
                ious = _box_iou_one(rois_np[n, 1:5], gt_boxes)
                best = int(ious.argmax())
                if ious[best] < self.match_iou:
                    continue
                target = _crop_to_coarse(gt_masks[best], rois_np[n, 1:5],
                                         raw_logits.shape[-1]).to(raw_logits.device)
                target_logits = self.k * (2.0 * target - 1.0)
                residual = (target_logits - raw_logits[n : n + 1]).clamp_(-self.cap, self.cap)
                delta[n : n + 1] = residual * outputs["search_support"][n : n + 1]
                matched += 1
            outputs["refined_logits"] = raw_logits + delta
            outputs["delta_logits"] = delta
        logger.debug("oracle applied to %d/%d rois", matched, raw_logits.shape[0])
        return outputs


class OraclePointMinerWrapper:
    """Mine prompt points from GT crops instead of the predicted coarse mask.

    The refiner records each chunk's global-frame ROIs right before mining, so
    rows align 1:1 between the two wrappers within one mask-head forward.
    With ``point_budget > 0`` the miner is bypassed entirely: K positives are
    farthest-point-sampled on the GT interior and K negatives on the GT outer
    ring, testing whether the 2P2N budget is the binding constraint.
    """

    def __init__(self, miner, refiner_wrapper: OracleP2Wrapper, k: float, match_iou: float,
                 point_budget: int = 0):
        self.miner = miner
        self.rois_source = refiner_wrapper
        self.k = float(k)
        self.match_iou = float(match_iou)
        self.point_budget = int(point_budget)
        self._orig_forward = miner.forward
        miner.forward = self._wrapped  # type: ignore[method-assign]

    def close(self) -> None:
        self.miner.forward = self._orig_forward  # type: ignore[method-assign]

    def _wrapped(self, mask_logits, boxes, image_size):
        gt_cache = self.rois_source.gt_cache
        rois = self.rois_source.last_rois
        aligned = gt_cache and rois is not None and rois.shape[0] == mask_logits.shape[0]
        if aligned and self.point_budget > 0:
            coords, labels = self._budget_points(mask_logits, rois, gt_cache, image_size)
            _, _, stats = self._orig_forward(mask_logits, boxes, image_size)
            return coords, labels, stats
        if aligned:
            with torch.no_grad():
                oracle_logits = mask_logits.clone()
                size = mask_logits.shape[-1]
                for n in range(mask_logits.shape[0]):
                    img = int(rois[n, 0])
                    if img >= len(gt_cache) or gt_cache[img] is None:
                        continue
                    gt_boxes, gt_masks = gt_cache[img]
                    if len(gt_boxes) == 0:
                        continue
                    ious = _box_iou_one(rois[n, 1:5], gt_boxes)
                    best = int(ious.argmax())
                    if ious[best] < self.match_iou:
                        continue
                    target = _crop_to_coarse(gt_masks[best], rois[n, 1:5], size)
                    oracle_logits[n] = self.k * (2.0 * target - 1.0).to(mask_logits.device)
                mask_logits = oracle_logits
        return self._orig_forward(mask_logits, boxes, image_size)

    def _budget_points(self, mask_logits, rois, gt_cache, image_size):
        from scipy.ndimage import binary_dilation, distance_transform_edt

        n = mask_logits.shape[0]
        budget = self.point_budget
        size = mask_logits.shape[-1]
        coords = torch.zeros((n, 2 * budget, 2), device=mask_logits.device, dtype=mask_logits.dtype)
        labels = torch.full((n, 2 * budget), -1, device=mask_logits.device, dtype=torch.long)

        def fps(cand, seed, count, avoid=()):
            chosen = [seed]
            while len(chosen) < count and len(cand):
                dist = None
                for ref in list(chosen) + list(avoid):
                    d = ((cand[:, None, :] - np.asarray(ref)[None, None, :]) ** 2).sum(-1)
                    dist = d if dist is None else np.minimum(dist, d)
                nxt = int(np.argmax(dist))
                chosen.append(cand[nxt])
            return chosen

        for row in range(n):
            img = int(rois[row, 0])
            if img >= len(gt_cache) or gt_cache[img] is None:
                continue
            gt_boxes, gt_masks = gt_cache[img]
            if len(gt_boxes) == 0:
                continue
            ious = _box_iou_one(rois[row, 1:5], gt_boxes)
            best = int(ious.argmax())
            if ious[best] < self.match_iou:
                continue
            gt = _crop_to_coarse(gt_masks[best], rois[row, 1:5], size).cpu().numpy() >= 0.5
            if not gt.any():
                continue
            edt = distance_transform_edt(gt)
            pos_cand = np.argwhere(gt & (edt >= 1.0))
            if len(pos_cand) == 0:
                pos_cand = np.argwhere(gt)
            seed = pos_cand[int(np.argmax(edt[gt & (edt >= 1.0)] if (gt & (edt >= 1.0)).any() else gt))]
            positives = fps(pos_cand, seed, budget)
            ring = binary_dilation(gt, iterations=2) & ~gt
            neg_cand = np.argwhere(ring) if ring.any() else np.argwhere(~gt)
            negatives = []
            if len(neg_cand):
                d0 = ((neg_cand[:, None, :] - np.asarray(positives)[None, :, :]) ** 2).sum(-1).min(1)
                negatives = fps(neg_cand, neg_cand[int(np.argmax(d0))], budget)
            x1, y1, x2, y2 = [float(v) for v in rois[row, 1:5]]
            labels[row] = -1
            for slot, yx in enumerate(positives[:budget]):
                nx = (float(yx[1]) + 0.5) / size
                ny = (float(yx[0]) + 0.5) / size
                coords[row, slot, 0] = x1 + nx * max(x2 - x1, 2.0)
                coords[row, slot, 1] = y1 + ny * max(y2 - y1, 2.0)
                labels[row, slot] = 1
            for slot, yx in enumerate(negatives[:budget]):
                nx = (float(yx[1]) + 0.5) / size
                ny = (float(yx[0]) + 0.5) / size
                coords[row, budget + slot, 0] = x1 + nx * max(x2 - x1, 2.0)
                coords[row, budget + slot, 1] = y1 + ny * max(y2 - y1, 2.0)
                labels[row, budget + slot] = 0
        coords[..., 0].clamp_(0.0, float(image_size if not isinstance(image_size, (list, tuple)) else image_size[1]) - 1.0)
        coords[..., 1].clamp_(0.0, float(image_size if not isinstance(image_size, (list, tuple)) else image_size[0]) - 1.0)
        return coords, labels


def _box_iou_one(box: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], gt_boxes[:, 0])
    y1 = np.maximum(box[1], gt_boxes[:, 1])
    x2 = np.minimum(box[2], gt_boxes[:, 2])
    y2 = np.minimum(box[3], gt_boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1]) + (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
        gt_boxes[:, 3] - gt_boxes[:, 1]
    ) - inter
    return inter / np.clip(area, 1e-6, None)


def _crop_to_coarse(gt_mask: torch.Tensor, box: np.ndarray, size: int) -> torch.Tensor:
    height, width = gt_mask.shape
    x1 = int(np.floor(box[0]).clip(0, width - 1))
    y1 = int(np.floor(box[1]).clip(0, height - 1))
    x2 = int(np.ceil(box[2]).clip(x1 + 1, width))
    y2 = int(np.ceil(box[3]).clip(y1 + 1, height))
    crop = gt_mask[y1:y2, x1:x2][None, None].float()
    resized = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return (resized[0, 0] >= 0.5).float()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    _resolve_sam2_repo(args, snapshot)
    model_config, _ = _resolve_model_config(args, snapshot)
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model = _register_and_build(model_config)
    _load_model_state(model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False)
    model.to(device)
    model.eval()
    mask_head = model.roi_head.mask_head
    refiner = getattr(mask_head, "p2_boundary_refiner", None)
    if refiner is None:
        raise RuntimeError("This checkpoint has P2BoundaryRefiner disabled.")
    if (args.oracle_points or args.oracle_points_k > 0) and args.oracle_cap is not None:
        raise RuntimeError("point-oracle modes and --oracle-cap are mutually exclusive")
    if not (args.oracle_points or args.oracle_points_k > 0 or args.oracle_cap is not None):
        raise RuntimeError(
            "pick one probe mode: --oracle-cap FLOAT, --oracle-points, or --oracle-points-k K"
        )
    if args.oracle_points_k > 0:
        logger.info(
            "oracle probe: %dP%dN GT budget points (interior FPS + outer-ring FPS); "
            "dense prompt and coarse mask stay raw",
            args.oracle_points_k, args.oracle_points_k,
        )
    elif args.oracle_points:
        logger.info(
            "oracle probe: GT-mined prompt points (dense prompt and coarse mask stay raw); k=%.1f",
            args.target_logit_k,
        )
    else:
        logger.info(
            "oracle probe: cap=%.2f k=%.1f (trained beta bound was %.2f)",
            args.oracle_cap,
            args.target_logit_k,
            float(refiner.beta) * float(refiner.delta_logit_max),
        )

    contract = _resolve_dataset_contract(args, snapshot)
    loader = _build_loader(contract, args.num_workers)
    segm_score_key = getattr(mask_head, "segm_score_key", "scores")
    segm_score_mode = getattr(mask_head, "segm_score_mode", "detector")
    data_preprocessor = model.data_preprocessor

    oracle = OracleP2Wrapper(refiner, args.oracle_cap, args.target_logit_k, args.match_iou)
    point_oracle = None
    if args.oracle_points or args.oracle_points_k > 0:
        miner = getattr(mask_head, "shape_point_miner", None)
        if miner is None:
            raise RuntimeError("point-oracle modes require a shape_point_miner on the mask head")
        point_oracle = OraclePointMinerWrapper(
            miner, oracle, args.target_logit_k, args.match_iou,
            point_budget=args.oracle_points_k,
        )
    all_gt: List[dict] = []
    all_dt: List[dict] = []
    all_metas: List[dict] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                imgs, data_samples = _build_data_samples(batch, device)
                batch_gt = []
                for sample in data_samples:
                    instances = sample.gt_instances
                    if len(getattr(instances, "bboxes", [])) == 0:
                        batch_gt.append(None)
                        continue
                    boxes = instances.bboxes.detach().cpu().numpy()
                    masks = instances.masks.to_tensor(dtype=torch.float32, device="cpu")
                    batch_gt.append((boxes, masks))
                oracle.set_batch_gt(batch_gt)
                processed = data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                outputs = model.predict(
                    processed["inputs"], processed["data_samples"], rescale=False
                )
                for index, output in enumerate(outputs):
                    meta = batch["img_metas"][index]
                    shape = meta["img_shape"]
                    gt = _extract_instances_numpy(
                        processed["data_samples"][index].gt_instances, shape
                    )
                    pred_instances = (
                        output.pred_instances if hasattr(output, "pred_instances") else output
                    )
                    prediction = _filter_predictions(
                        _extract_instances_numpy(pred_instances, shape), 0.0
                    )
                    all_gt.append(gt)
                    all_dt.append(prediction)
                    all_metas.append(meta)
    finally:
        if point_oracle is not None:
            point_oracle.close()
        oracle.close()

    from utils.coco_eval_utils import (
        _mask_to_rle,
        _xyxy_to_xywh,
        build_coco_gt_and_dt,
        run_coco_eval,
    )

    coco_gt, coco_segm_dt = build_coco_gt_and_dt(all_gt, all_dt, all_metas, score_key=segm_score_key)
    metrics = run_coco_eval(coco_gt, coco_segm_dt, iou_type="segm")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-image GT/detection records: image-level paired bootstrap CIs require
    # resampling images and recomputing COCO mAP offline.
    gt_records = []
    dt_records = []
    for gt, prediction, meta in zip(all_gt, all_dt, all_metas):
        image_id = int(meta.get("image_id", meta.get("scene_id", 0)))
        labels = np.asarray(gt["labels"])
        bboxes = np.asarray(gt["bboxes"])
        masks = np.asarray(gt["masks"])
        for i in range(len(labels)):
            gt_records.append({
                "image_id": image_id,
                "category_id": int(labels[i]) + 1,
                "bbox": [float(v) for v in _xyxy_to_xywh(bboxes[i])],
                "segmentation": _mask_to_rle(masks[i]),
                "iscrowd": 0,
            })
        scores = prediction.get("scores", np.ones(len(prediction["bboxes"]), dtype=np.float32))
        mask_scores = prediction.get("mask_scores", scores)
        for i in range(len(prediction["bboxes"])):
            x1, y1, x2, y2 = [float(v) for v in prediction["bboxes"][i]]
            dt_records.append({
                "image_id": image_id,
                "category_id": int(prediction["labels"][i]) + 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[i]),
                "mask_score": float(mask_scores[i]),
                "segmentation": _mask_to_rle(prediction["masks"][i]),
            })
    with (output_dir / "gt_records.json").open("w", encoding="utf-8") as handle:
        json.dump(gt_records, handle)
        handle.write("\n")
    with (output_dir / "dt_records.json").open("w", encoding="utf-8") as handle:
        json.dump(dt_records, handle)
        handle.write("\n")
    images = [
        {
            "image_id": int(meta.get("image_id", meta.get("scene_id", 0))),
            "height": int(meta["img_shape"][0]),
            "width": int(meta["img_shape"][1]),
        }
        for meta in all_metas
    ]
    with (output_dir / "images.json").open("w", encoding="utf-8") as handle:
        json.dump(images, handle)
        handle.write("\n")
    payload = {
        "checkpoint": str(checkpoint_path),
        "oracle_cap": args.oracle_cap,
        "oracle_points": bool(args.oracle_points),
        "oracle_points_k": args.oracle_points_k,
        "target_logit_k": args.target_logit_k,
        "match_iou": args.match_iou,
        "segm_score_key": segm_score_key,
        "segm_score_mode": segm_score_mode,
        "images": len(all_metas),
        "metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: v for k, v in metrics.items() if "mAP" in k}, indent=2))
    print(f"oracle probe complete: {output_dir}")


if __name__ == "__main__":
    main()
