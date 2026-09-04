#!/usr/bin/env python
"""Offline learnability probe for the P2 point-refiner design.

Answers, before any training investment, whether stride-4 P2 features contain
the information needed to move mined prompt points toward their miner-on-GT
targets:

  collect  run the model on the TRAIN split, record per-point samples
           (feature@point, context, required offset = gt_mined - mined)
  fit      train the actual tiny MLP head offline; report error reduction
  eval     re-run full validation with the fitted head applying predicted
           offsets (no GT involved), paired against the known baseline

If the fitted head recovers a meaningful share of the +0.008 oracle gain,
Stage 1 is de-risked; if it recovers nothing, the features lack the signal.
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
import torch.nn as nn
import torch.nn.functional as F

MAINLINE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAINLINE_ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples,
    _build_loader,
    _load_checkpoint,
    _load_model_state,
    _register_and_build,
    _resolve_dataset_contract,
    _resolve_model_config,
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)
from inference.probes.oracle_p2_probe import _box_iou_one, _crop_to_coarse  # noqa: E402

logger = logging.getLogger("learn_probe")

OFFSET_BOUND_PX = 8.0


class PointProbeWrapper:
    """Wraps refiner (records ROI P2 features) and miner (samples points)."""

    def __init__(self, refiner, miner):
        self.refiner = refiner
        self.miner = miner
        self.gt_cache: List[Optional[Tuple[np.ndarray, torch.Tensor]]] = []
        self.bank: Dict[str, List[torch.Tensor]] = {"feat": [], "ctx": [], "tgt": [], "valid": [], "patch": []}
        self.collect_patch = True
        self.head: Optional[nn.Module] = None
        self.apply_offsets = False
        self.last_roi_p2 = None
        self.last_rois = None
        self.last_raw = None
        self._orig_refiner = refiner.forward
        self._orig_miner = miner.forward
        refiner.forward = self._refiner_wrapped  # type: ignore[method-assign]
        miner.forward = self._miner_wrapped  # type: ignore[method-assign]

    def close(self) -> None:
        self.refiner.forward = self._orig_refiner  # type: ignore[method-assign]
        self.miner.forward = self._orig_miner  # type: ignore[method-assign]

    def _refiner_wrapped(self, p2_feature, prompt_rois, raw_logits):
        outputs = self._orig_refiner(p2_feature, prompt_rois, raw_logits)
        with torch.no_grad():
            projected = self.refiner.p2_projection(p2_feature.detach())
            self.last_roi_p2 = self.refiner.roi_align(projected, prompt_rois)
        self.last_rois = prompt_rois.detach().cpu().numpy()
        self.last_raw = raw_logits.detach()
        return outputs

    def _mine_gt(self, mask_logits, boxes, image_size):
        n = mask_logits.shape[0]
        size = mask_logits.shape[-1]
        gt_logits = torch.full_like(mask_logits, -8.0)
        row_matched = torch.zeros(n, dtype=torch.bool)
        for row in range(n):
            img = int(self.last_rois[row, 0])
            if img >= len(self.gt_cache) or self.gt_cache[img] is None:
                continue
            gt_boxes, gt_masks = self.gt_cache[img]
            if len(gt_boxes) == 0:
                continue
            ious = _box_iou_one(self.last_rois[row, 1:5], gt_boxes)
            best = int(ious.argmax())
            if ious[best] < 0.3:
                continue
            target = _crop_to_coarse(gt_masks[best], self.last_rois[row, 1:5], size)
            gt_logits[row] = (2.0 * target - 1.0).to(mask_logits.device) * 8.0
            row_matched[row] = True
        if not bool(row_matched.any()):
            return None
        coords, labels, _ = self._orig_miner(gt_logits, boxes, image_size)
        return coords.detach(), labels.detach(), row_matched

    @staticmethod
    def _canon(coords: torch.Tensor, labels: torch.Tensor):
        key = coords[..., 1] * 4096.0 + coords[..., 0]
        order = torch.cat([key[:, :2].argsort(dim=1), key[:, 2:].argsort(dim=1) + 2], dim=1)
        return (
            coords.gather(1, order[..., None].expand(-1, -1, 2)),
            labels.gather(1, order),
            order,
        )

    def _miner_wrapped(self, mask_logits, boxes, image_size):
        coords, labels, stats = self._orig_miner(mask_logits, boxes, image_size)
        rois = self.last_rois
        if rois is None or rois.shape[0] != mask_logits.shape[0]:
            return coords, labels, stats
        roi_p2 = self.last_roi_p2
        raw = self.last_raw
        device = mask_logits.device
        n = mask_logits.shape[0]
        size = mask_logits.shape[-1]

        local_yx, _ = self.miner.get_last_local_points()
        grid = ((local_yx.to(device).float() + 0.5) / float(size)) * 2.0 - 1.0
        feat = F.grid_sample(roi_p2, grid.flip(-1)[:, None], mode="bilinear", align_corners=False)
        feat = feat[:, :, 0, :].transpose(1, 2)  # [N,4,C]
        patch = None
        if self.collect_patch:
            radius = 2
            span = 2.0 / float(roi_p2.shape[-1])
            offsets = torch.linspace(-radius * span, radius * span, 2 * radius + 1, device=device)
            dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
            delta = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)  # [25,2] (x,y)
            patch_grid = grid.flip(-1)[:, :, None, :] + delta[None, None, :, :]
            patch = F.grid_sample(roi_p2, patch_grid, mode="bilinear", align_corners=False)
            patch = patch.transpose(1, 2).reshape(n, 4, 2 * radius + 1, 2 * radius + 1, -1)
        prob = torch.sigmoid(raw).squeeze(1)
        flat = (local_yx[..., 0].to(device) * size + local_yx[..., 1].to(device)).long()
        prob_pt = prob.flatten(1).gather(1, flat)  # [N,4]
        box_wh = torch.as_tensor(
            np.stack([rois[:, 3] - rois[:, 1], rois[:, 4] - rois[:, 2]], axis=1),
            device=device, dtype=feat.dtype,
        )  # [N,2] (w,h)
        ctx = torch.stack(
            [
                prob_pt,
                box_wh[:, None, 0].expand(-1, 4),
                box_wh[:, None, 1].expand(-1, 4),
                (labels >= 0).to(feat.dtype) * (torch.where(labels >= 0, labels.to(device), 0).to(feat.dtype) * 2 - 1),
            ],
            dim=-1,
        )  # [N,4,4]

        if self.apply_offsets and self.head is not None:
            with torch.no_grad():
                offsets = OFFSET_BOUND_PX * torch.tanh(self.head(torch.cat([feat, ctx], dim=-1)))
            valid = (labels >= 0).to(offsets.dtype)[..., None]
            image_w = float(image_size if not isinstance(image_size, (list, tuple)) else image_size[1])
            image_h = float(image_size if not isinstance(image_size, (list, tuple)) else image_size[0])
            coords = coords + offsets * valid
            coords[..., 0].clamp_(0.0, image_w - 1.0)
            coords[..., 1].clamp_(0.0, image_h - 1.0)
            return coords, labels, stats

        gt = self._mine_gt(mask_logits, boxes, image_size)
        if gt is not None and bool((labels >= 0).any()):
            gt_coords, gt_labels, row_matched = gt
            pred_c, _, order = self._canon(coords.detach(), labels)
            gt_c, gt_lab_c, _ = self._canon(gt_coords, gt_labels)
            target = gt_c - pred_c  # [N,4,2]
            valid = (
                row_matched[:, None].to(device)
                & (labels >= 0).to(device)
                & (gt_labels.gather(1, order) >= 0).to(device)
            )
            self.bank["feat"].append(feat.cpu())
            self.bank["ctx"].append(ctx.cpu())
            self.bank["tgt"].append(target.cpu())
            self.bank["valid"].append(valid.cpu())
            if patch is not None:
                self.bank["patch"].append(patch.half().cpu())
        return coords, labels, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=("collect", "fit", "eval"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", choices=("validation", "test", "custom"), default="validation")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--image-subdir", default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sam2-repo", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epochs", type=int, default=400)
    return parser.parse_args()


def _build(args):
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
    _load_model_state(model, _select_state_dict(checkpoint, "model"), allow_nonstrict=False)
    model.to(device)
    model.eval()
    return model, device, checkpoint_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    model, device, checkpoint_path = _build(args)
    mask_head = model.roi_head.mask_head
    refiner = getattr(mask_head, "p2_boundary_refiner", None)
    miner = getattr(mask_head, "shape_point_miner", None)
    if refiner is None or miner is None:
        raise RuntimeError("probe requires p2_boundary_refiner and shape_point_miner")

    probe = PointProbeWrapper(refiner, miner)
    contract = _resolve_dataset_contract(args, _snapshot(_load_checkpoint(checkpoint_path)))

    if args.stage == "fit":
        bank = torch.load(work_dir / "bank.pt", weights_only=False)
        feat = torch.cat(bank["feat"]).float()
        ctx = torch.cat(bank["ctx"]).float()
        tgt = torch.cat(bank["tgt"]).float()
        valid = torch.cat(bank["valid"]).bool()
        x = torch.cat([feat, ctx], dim=-1)[valid]
        y = tgt[valid]
        n = x.shape[0]
        perm = torch.randperm(n)
        n_test = max(int(n * 0.1), 1)
        tr_x, te_x = x[perm[n_test:]], x[perm[:n_test]]
        tr_y, te_y = y[perm[n_test:]], y[perm[:n_test]]
        logger.info("samples: %d (train %d / test %d)", n, tr_x.shape[0], te_x.shape[0])
        base_err = te_y.norm(dim=-1).mean().item()
        head = nn.Sequential(nn.Linear(x.shape[-1], 128), nn.GELU(), nn.Linear(128, 2))
        head.to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        tr_x, tr_y, te_x_d, te_y_d = tr_x.to(device), tr_y.to(device), te_x.to(device), te_y.to(device)
        for epoch in range(args.epochs):
            head.train()
            opt.zero_grad()
            pred = OFFSET_BOUND_PX * torch.tanh(head(tr_x))
            loss = F.smooth_l1_loss(pred, tr_y, beta=2.0)
            loss.backward()
            opt.step()
            if (epoch + 1) % 100 == 0:
                head.eval()
                with torch.no_grad():
                    test_pred = OFFSET_BOUND_PX * torch.tanh(head(te_x_d))
                    test_err = (test_pred - te_y_d).norm(dim=-1).mean().item()
                logger.info(
                    "epoch %d loss=%.4f | test err %.3fpx (zero-offset baseline %.3fpx, reduction %.1f%%)",
                    epoch + 1, loss.item(), test_err, base_err,
                    100.0 * (1.0 - test_err / max(base_err, 1e-6)),
                )
        torch.save(head.state_dict(), work_dir / "head.pt")
        with (work_dir / "fit_report.json").open("w") as handle:
            json.dump(
                {"samples": int(n), "baseline_err_px": base_err, "test_err_px": test_err,
                 "reduction_pct": 100.0 * (1.0 - test_err / max(base_err, 1e-6))},
                handle, indent=2,
            )
        probe.close()
        return

    if args.stage == "collect":
        loader = _build_loader(contract, args.num_workers)
        skipped = 0
        processed = 0
        with torch.inference_mode():
            for batch in loader:
                if processed >= args.limit:
                    break
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
                probe.gt_cache = batch_gt
                prepped = model.data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                model.predict(prepped["inputs"], prepped["data_samples"], rescale=False)
                processed += len(imgs)
                if processed % 50 < len(imgs):
                    n_pts = sum(int(v.sum()) for v in probe.bank["valid"])
                    logger.info("collected after %d images: %d point samples", processed, n_pts)
        torch.save(probe.bank, work_dir / "bank.pt")
        n_pts = sum(int(v.sum()) for v in probe.bank["valid"])
        logger.info("collect done: %d images, %d valid point samples", processed, n_pts)
        probe.close()
        return

    # eval stage
    head = nn.Sequential(nn.Linear(64 + 4, 128), nn.GELU(), nn.Linear(128, 2))
    head.load_state_dict(torch.load(work_dir / "head.pt", weights_only=False))
    head.to(device)
    head.eval()
    probe.head = head
    probe.apply_offsets = True
    loader = _build_loader(contract, args.num_workers)
    from inference.infer_from_checkpoint import _extract_instances_numpy
    from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

    all_gt: List[dict] = []
    all_dt: List[dict] = []
    all_metas: List[dict] = []
    segm_score_key = getattr(mask_head, "segm_score_key", "scores")
    with torch.inference_mode():
        for batch in loader:
            imgs, data_samples = _build_data_samples(batch, device)
            processed = model.data_preprocessor(
                {"inputs": imgs, "data_samples": data_samples}, training=False
            )
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
    coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, all_metas, score_key=segm_score_key)
    metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
    with (work_dir / "metrics.json").open("w") as handle:
        json.dump(
            {"checkpoint": str(checkpoint_path), "head": str(work_dir / "head.pt"), "metrics": metrics},
            handle, indent=2,
        )
    print(json.dumps({k: v for k, v in metrics.items() if "mAP" in k}, indent=2))
    probe.close()


if __name__ == "__main__":
    main()
