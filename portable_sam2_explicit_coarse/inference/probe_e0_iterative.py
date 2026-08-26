#!/usr/bin/env python
"""E0: untrained PA-SAM-style second decode round with uncertainty-mined points.

Round 1 runs the normal pipeline; per ROI we capture the decoded low-res
logits and the coarse prompt mask. Extra prompt points are mined heuristically
(PA-SAM's hard-point mining without any training):

    E      = pixel entropy of the first-round mask
    pos    = top-k of  E * relu(p_decoded - p_coarse)   (uncertain expansion)
    neg    = top-k of  E * relu(p_coarse - p_decoded)   (uncertain shrinkage)
    fallback: top-k of E on fg / bg alone.

Round 2 re-runs the same forward with 4 extra positive + 4 extra negative
points appended to the miner's 2P2N (invalid slots labelled -1, so unused
slots are neutralised by the PE). Final COCO metrics come from round 2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MAINLINE_ROOT = Path(__file__).resolve().parents[1]
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

logger = logging.getLogger("e0_probe")

EXTRA_PER_SIDE = 4
GRID = 256
SEPARATION = 16


def _crop_gt_to_grid(gt_mask, box, size=256):
    import torch.nn.functional as F

    height, width = gt_mask.shape
    x1, y1, x2, y2 = [float(v) for v in box]
    x1i = int(np.floor(x1).clip(0, width - 1))
    y1i = int(np.floor(y1).clip(0, height - 1))
    x2i = int(np.ceil(x2).clip(x1i + 1, width))
    y2i = int(np.ceil(y2).clip(y1i + 1, height))
    crop = gt_mask[y1i:y2i, x1i:x2i][None, None].float()
    resized = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return (resized[0, 0] >= 0.5).numpy()


def _mine_oracle_extra_points(logits256, gt_mask, box, k, sides="both"):
    """Points on the model's true error regions: FN -> extra positives, FP -> extra negatives."""
    p = 1.0 / (1.0 + np.exp(-logits256))
    pred = p >= 0.5
    gt = _crop_gt_to_grid(gt_mask, box, logits256.shape[0])
    fn_map = (gt & ~pred).astype(np.float32)
    fp_map = (~gt & pred).astype(np.float32)
    x1, y1, x2, y2 = [float(v) for v in box]
    points = []

    def to_img(x, y, label):
        return (x1 + (x + 0.5) / logits256.shape[1] * max(x2 - x1, 2.0),
                y1 + (y + 0.5) / logits256.shape[0] * max(y2 - y1, 2.0), label)

    if sides in ("both", "pos"):
        for x, y in _greedy_topk(fn_map, k, SEPARATION):
            points.append(to_img(x, y, 1))
    if sides in ("both", "neg"):
        for x, y in _greedy_topk(fp_map, k, SEPARATION):
            points.append(to_img(x, y, 0))
    return points or None


class E0Hooks:
    """Capture per-chunk rois/coarse logits (round 1) and append points (round 2)."""

    def __init__(self, refiner, miner, mask_head, oracle=False, sides="both"):
        self.miner = miner
        self.oracle = oracle
        self.sides = sides
        self.batch_gt: List = []
        self.round1 = True
        self.extra: List[Optional[np.ndarray]] = []  # per global row: [K,3] (x, y, label) or None
        self._row_pointer = 0
        self._rois_chunks: List[np.ndarray] = []
        self._raw_chunks: List[torch.Tensor] = []
        self._low_chunks: List[torch.Tensor] = []
        self._orig_refiner = refiner.forward
        self._orig_miner = miner.forward
        self._hook = mask_head.register_forward_hook(self._mask_head_hook)
        refiner.forward = self._refiner_wrapped  # type: ignore[method-assign]
        miner.forward = self._miner_wrapped  # type: ignore[method-assign]

    def close(self) -> None:
        self._hook.remove()
        self.refiner_forward_restore()
        self.miner.forward = self._orig_miner  # type: ignore[method-assign]

    def refiner_forward_restore(self) -> None:
        pass

    def _refiner_wrapped(self, p2_feature, prompt_rois, raw_logits):
        outputs = self._orig_refiner(p2_feature, prompt_rois, raw_logits)
        if self.round1:
            self._rois_chunks.append(prompt_rois.detach().cpu().numpy())
            self._raw_chunks.append(raw_logits.detach().float().cpu())
        return outputs

    def _mask_head_hook(self, module, args, output):
        if self.round1 and isinstance(output, tuple) and torch.is_tensor(output[0]):
            self._low_chunks.append(output[0].detach().float().cpu())

    def _miner_wrapped(self, mask_logits, boxes, image_size):
        coords, labels, stats = self._orig_miner(mask_logits, boxes, image_size)
        if self.round1 or not self.extra:
            return coords, labels, stats
        n = mask_logits.shape[0]
        rows = list(range(self._row_pointer, self._row_pointer + n))
        self._row_pointer += n
        device, dtype = coords.device, coords.dtype
        new_coords = coords.new_zeros((n, labels.shape[1] + 2 * EXTRA_PER_SIDE, 2))
        new_labels = labels.new_full((n, labels.shape[1] + 2 * EXTRA_PER_SIDE), -1)
        new_coords[:, : labels.shape[1]] = coords
        new_labels[:, : labels.shape[1]] = labels
        for i, row in enumerate(rows):
            extra = self.extra[row] if row < len(self.extra) else None
            if extra is None or len(extra) == 0:
                continue
            for j, (x, y, lab) in enumerate(extra[: 2 * EXTRA_PER_SIDE]):
                new_coords[i, labels.shape[1] + j, 0] = float(x)
                new_coords[i, labels.shape[1] + j, 1] = float(y)
                new_labels[i, labels.shape[1] + j] = int(lab)
        return new_coords, new_labels, stats

    def begin_round1(self) -> None:
        self.round1 = True
        self._rois_chunks, self._raw_chunks, self._low_chunks = [], [], []

    def finish_round1(self, k_extra=4) -> None:
        self.round1 = False
        self._row_pointer = 0
        if not self._rois_chunks or not self._low_chunks:
            # batch had no mask ROIs at all; round 2 degenerates to a plain pass
            self.extra = []
            return
        rois = np.concatenate(self._rois_chunks, axis=0)
        raw = torch.cat(self._raw_chunks, dim=0)
        low = torch.cat(self._low_chunks, dim=0)
        self.extra = []
        for row in range(rois.shape[0]):
            if self.oracle:
                extra = None
                img = int(rois[row, 0])
                if img < len(self.batch_gt) and self.batch_gt[img] is not None:
                    from inference.oracle_p2_probe import _box_iou_one

                    gt_boxes, gt_masks = self.batch_gt[img]
                    if len(gt_boxes) > 0:
                        ious = _box_iou_one(rois[row, 1:5], gt_boxes)
                        best = int(ious.argmax())
                        if ious[best] >= 0.3:
                            extra = _mine_oracle_extra_points(
                                low[row, 0].numpy(), gt_masks[best], rois[row, 1:5],
                                k_extra, self.sides,
                            )
                self.extra.append(extra)
            else:
                self.extra.append(
                    _mine_extra_points(low[row, 0].numpy(), raw[row, 0].numpy(), rois[row, 1:5], k_extra)
                )
        logger.info("round1 done: %d rois, %d with extra points", len(self.extra),
                    sum(1 for e in self.extra if e is not None and len(e)))

    def begin_round2(self) -> None:
        self._row_pointer = 0


def _greedy_topk(score_map: np.ndarray, k: int, sep: int):
    picked = []
    work = score_map.copy()
    for _ in range(k):
        idx = np.argmax(work)
        if work.flat[idx] <= 1e-6:
            break
        y, x = np.unravel_index(idx, work.shape)
        picked.append((int(x), int(y)))
        y0, y1 = max(0, y - sep), min(work.shape[0], y + sep + 1)
        x0, x1 = max(0, x - sep), min(work.shape[1], x + sep + 1)
        work[y0:y1, x0:x1] = 0.0
    return picked


def _mine_extra_points(logits256, coarse64, box, k):
    p = 1.0 / (1.0 + np.exp(-logits256))
    eps = 1e-6
    entropy = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
    coarse = np.repeat(np.repeat(
        1.0 / (1.0 + np.exp(-coarse64)), 4, axis=0), 4, axis=1)
    if coarse.shape != p.shape:
        coarse = np.resize(coarse, p.shape)
    pos_map = entropy * np.maximum(p - coarse, 0.0)
    neg_map = entropy * np.maximum(coarse - p, 0.0)
    pos = _greedy_topk(pos_map, k, SEPARATION)
    if not pos:
        pos = _greedy_topk(entropy * (p >= 0.5), k, SEPARATION)
    neg = _greedy_topk(neg_map, k, SEPARATION)
    if not neg:
        neg = _greedy_topk(entropy * (p < 0.5), k, SEPARATION)
    x1, y1, x2, y2 = [float(v) for v in box]
    points = []
    for x, y in pos:
        points.append((x1 + (x + 0.5) / GRID * max(x2 - x1, 2.0),
                       y1 + (y + 0.5) / GRID * max(y2 - y1, 2.0), 1))
    for x, y in neg:
        points.append((x1 + (x + 0.5) / GRID * max(x2 - x1, 2.0),
                       y1 + (y + 0.5) / GRID * max(y2 - y1, 2.0), 0))
    return points or None


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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--sam2-repo", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--oracle-points", action="store_true",
                        help="Mine round-2 points from true error regions (GT vs round-1 mask): "
                             "FN -> extra positives, FP -> extra negatives. Ceiling of a trained selector.")
    parser.add_argument("--sides", choices=("both", "pos", "neg"), default="both",
                        help="which error side to point at in oracle mode")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


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
    hooks = E0Hooks(
        mask_head.p2_boundary_refiner,
        mask_head.shape_point_miner,
        mask_head,
        oracle=args.oracle_points,
        sides=args.sides,
    )

    contract = _resolve_dataset_contract(args, snapshot)
    loader = _build_loader(contract, args.num_workers)
    segm_score_key = getattr(mask_head, "segm_score_key", "scores")

    from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

    all_gt: List[dict] = []
    all_dt: List[dict] = []
    all_metas: List[dict] = []
    try:
        with torch.inference_mode():
            for batch in loader:
                imgs, data_samples = _build_data_samples(batch, device)
                processed = model.data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                if args.oracle_points:
                    batch_gt = []
                    for sample in data_samples:
                        instances = sample.gt_instances
                        if len(getattr(instances, "bboxes", [])) == 0:
                            batch_gt.append(None)
                            continue
                        boxes = instances.bboxes.detach().cpu().numpy()
                        masks = instances.masks.to_tensor(dtype=torch.float32, device="cpu")
                        batch_gt.append((boxes, masks))
                    hooks.batch_gt = batch_gt
                hooks.begin_round1()
                model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                hooks.finish_round1()
                hooks.begin_round2()
                outputs = model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                for index, output in enumerate(outputs):
                    meta = batch["img_metas"][index]
                    shape = meta["img_shape"]
                    gt = _extract_instances_numpy(
                        processed["data_samples"][index].gt_instances, shape
                    )
                    pred = _extract_instances_numpy(
                        output.pred_instances if hasattr(output, "pred_instances") else output, shape
                    )
                    all_gt.append(gt)
                    all_dt.append(pred)
                    all_metas.append(meta)
    finally:
        hooks.close()

    coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, all_metas, score_key=segm_score_key)
    metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "mode": ("e1_oracle_" + args.sides) if args.oracle_points else "e0_iterative_4p4n",
                "metrics": metrics,
            },
            handle, indent=2,
        )
    print(json.dumps({k: v for k, v in metrics.items() if "mAP" in k}, indent=2))
    print(f"E0 probe complete: {output_dir}")


if __name__ == "__main__":
    main()
