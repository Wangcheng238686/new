#!/usr/bin/env python
"""Visualize P2BoundaryRefiner behaviour at inference from a training checkpoint.

Rebuilds the model exactly like inference/infer_from_checkpoint.py, hooks the
mask head's ``p2_boundary_refiner`` forward, and renders per-image panels:

  - overview: input + GT outlines, input + final predictions + selected ROI boxes
  - per selected ROI (top-K by detection score):
      crop | coarse mask BEFORE P2 (sigmoid(raw)) | coarse mask AFTER P2
      (sigmoid(refined), dashed white contour = raw) | delta heatmap
      (seismic, clipped at +-beta*delta_logit_max, lime = search band) |
      final SAM2 mask contour for the same ROI

``--ab-no-p2`` adds a side-by-side final-prediction comparison rendered from a
second forward pass with ``refiner.beta = 0`` (delta exactly 0, i.e. P2 off),
showing the end-to-end effect of the refiner on the decoded masks.

This tool is offline: it only reads checkpoints and never touches a running
training process.

Examples:
    python inference/visualize_p2_refiner.py \
        --checkpoint .../best_model.pth \
        --data-root /data1/wangcheng/dataset/WHU \
        --device cuda:3 --limit 6 --top-k 4 \
        --out-dir logs/viz_p2/run_name
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
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

logger = logging.getLogger("p2_viz")

PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (70, 240, 240), (240, 50, 230), (210, 245, 60),
    (250, 190, 190), (0, 128, 128), (220, 190, 255), (170, 110, 40),
    (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0),
]

MARGIN = 48


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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=6, help="max images to render")
    parser.add_argument("--skip", type=int, default=0, help="skip the first N split images")
    parser.add_argument("--top-k", type=int, default=4, help="ROIs per image to detail")
    parser.add_argument("--margin", type=int, default=48, help="crop margin around a ROI box (px)")
    parser.add_argument("--score-thr", type=float, default=0.0, help="min score for a ROI to be detailed")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", choices=("model", "ema"), default="model")
    parser.add_argument("--sam2-repo", default=None)
    parser.add_argument("--sam2-ckpt", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--ab-no-p2", action="store_true",
                        help="extra pass with beta=0 (P2 disabled) and a side-by-side final-mask figure")
    return parser.parse_args()


class P2Capture:
    """Record every P2BoundaryRefiner forward (inputs + outputs) on CPU."""

    def __init__(self, refiner: torch.nn.Module) -> None:
        self.refiner = refiner
        self.frames: List[Dict[str, torch.Tensor]] = []
        self._orig_forward = refiner.forward
        refiner.forward = self._wrapped_forward  # type: ignore[method-assign]

    def _wrapped_forward(self, p2_feature, prompt_rois, raw_logits):
        outputs = self._orig_forward(p2_feature, prompt_rois, raw_logits)
        self.frames.append(
            {
                "rois": prompt_rois.detach().float().cpu(),
                "raw": raw_logits.detach().float().cpu(),
                "refined": outputs["refined_logits"].detach().float().cpu(),
                "delta": outputs["delta_logits"].detach().float().cpu(),
                "support": outputs["search_support"].detach().float().cpu(),
                "support_valid": outputs["support_valid"].detach().cpu(),
            }
        )
        return outputs

    def close(self) -> None:
        self.refiner.forward = self._orig_forward  # type: ignore[method-assign]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _image_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    """CHW float 0-255 -> HWC float 0-1."""
    array = img_tensor.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(array / 255.0, 0.0, 1.0)


def _masks_of(pred: Dict[str, Any]) -> np.ndarray:
    masks = pred.get("masks")
    if masks is None or len(masks) == 0:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    return masks


def _draw_overview(
    ax: plt.Axes,
    image: np.ndarray,
    pred: Dict[str, Any],
    reference: Optional[Dict[str, Any]],
    selected: Optional[List[int]],
    title: str,
) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    if reference is not None:
        for mask in _masks_of(reference):
            ax.contour(mask, levels=[0.5], colors="white", linewidths=0.6)
    masks = _masks_of(pred)
    scores = pred.get("scores", np.ones(len(pred["bboxes"])))
    for index in range(len(pred["bboxes"])):
        color = PALETTE[index % len(PALETTE)]
        rgb = tuple(value / 255.0 for value in color)
        if index < len(masks):
            ax.contour(masks[index], levels=[0.5], colors=[rgb], linewidths=0.8)
        if selected is not None and index in selected:
            x1, y1, x2, y2 = pred["bboxes"][index]
            ax.add_patch(
                plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor="yellow", linewidth=1.2,
                )
            )
            ax.text(
                x1, max(y1 - 4, 2), f"#{index} {float(scores[index]):.2f}",
                color="yellow", fontsize=7,
            )


def _crop_with_margin(box, shape, margin):
    height, width = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    x0 = max(int(np.floor(x1)) - margin, 0)
    y0 = max(int(np.floor(y1)) - margin, 0)
    x3 = min(int(np.ceil(x2)) + margin, width)
    y3 = min(int(np.ceil(y2)) + margin, height)
    if x3 - x0 < 8 or y3 - y0 < 8:
        x0, y0, x3, y3 = 0, 0, width, height
    return x0, y0, x3, y3


def _render_roi_row(axes, image, box, score, raw64, refined64, delta64, support64,
                    final_mask, delta_cap) -> Dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx0, cy0, cx1, cy1 = _crop_with_margin(box, image.shape, MARGIN)
    crop = image[cy0:cy1, cx0:cx1]
    ch, cw = crop.shape[:2]
    extent = [x1 - cx0, x2 - cx0, y2 - cy0, y1 - cy0]

    axes[0].imshow(crop)
    axes[0].add_patch(
        plt.Rectangle((x1 - cx0, y1 - cy0), x2 - x1, y2 - y1,
                      fill=False, edgecolor="yellow", linewidth=1.0)
    )
    axes[0].set_title(f"ROI crop  score={score:.2f}", fontsize=8)

    raw_prob = _sigmoid(raw64)
    axes[1].imshow(crop)
    axes[1].imshow(raw_prob, extent=extent, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0,
                   interpolation="bilinear")
    axes[1].contour(raw_prob, levels=[0.5], colors="white", linewidths=0.8, extent=extent)
    axes[1].set_title("coarse BEFORE P2  p=\u03c3(raw)", fontsize=8)

    refined_prob = _sigmoid(refined64)
    axes[2].imshow(crop)
    axes[2].imshow(refined_prob, extent=extent, cmap="jet", alpha=0.55, vmin=0.0, vmax=1.0,
                   interpolation="bilinear")
    axes[2].contour(raw_prob, levels=[0.5], colors="white", linewidths=0.7,
                    linestyles="dashed", extent=extent)
    axes[2].contour(refined_prob, levels=[0.5], colors="cyan", linewidths=0.9, extent=extent)
    axes[2].set_title("coarse AFTER P2  (dashed=raw)", fontsize=8)

    axes[3].imshow(crop)
    delta_im = axes[3].imshow(
        delta64, extent=extent, cmap="seismic", vmin=-delta_cap, vmax=delta_cap,
        alpha=0.9, interpolation="nearest",
    )
    if support64.max() > 0:
        axes[3].contour(support64, levels=[0.5], colors="lime", linewidths=0.7, extent=extent)
    delta_mean = float(np.abs(delta64).mean())
    axes[3].set_title(
        f"Δ logits  mean|Δ|={delta_mean:.3f}  lime=search band", fontsize=8
    )
    axes[3].figure.colorbar(delta_im, ax=axes[3], fraction=0.046, pad=0.02)

    axes[4].imshow(crop)
    if final_mask is not None and final_mask.max() > 0:
        fy0 = max(int(np.floor(y1)) - MARGIN, 0)
        fx0 = max(int(np.floor(x1)) - MARGIN, 0)
        fy1 = min(int(np.ceil(y2)) + MARGIN, final_mask.shape[0])
        fx1 = min(int(np.ceil(x2)) + MARGIN, final_mask.shape[1])
        final_crop = final_mask[fy0:fy1, fx0:fx1]
        if final_crop.max() > 0:
            axes[4].contour(
                final_crop, levels=[0.5], colors="springgreen", linewidths=1.0,
                extent=[fx0 - cx0, fx1 - cx0, fy1 - cy0, fy0 - cy0],
            )
    axes[4].set_title("final SAM2 mask", fontsize=8)

    for axis in axes:
        axis.set_xlim(0, cw)
        axis.set_ylim(ch, 0)
        axis.axis("off")
    return {
        "score": float(score),
        "delta_abs_mean": float(np.abs(delta64).mean()),
        "delta_abs_max": float(np.abs(delta64).max()),
        "support_pixels": float(support64.sum()),
        "box": [x1, y1, x2, y2],
    }


def _render_image_figure(out_path, image, gt, pred, roi_rows, delta_cap, stem) -> None:
    rows = 2 + len(roi_rows)
    figure = plt.figure(figsize=(20, 4.1 * rows))
    grid = figure.add_gridspec(rows, 5, width_ratios=[1, 1, 1, 1.25, 1])
    figure.suptitle(f"P2BoundaryRefiner @ {stem}", fontsize=11)
    _draw_overview(figure.add_subplot(grid[0, 0:2]), image, gt, gt, None, "input + GT")
    _draw_overview(
        figure.add_subplot(grid[0, 2:5]), image, pred, gt,
        [row["pred_index"] for row in roi_rows],
        "input + final predictions (yellow = detailed ROIs)",
    )
    for row_index, row in enumerate(roi_rows):
        axes = [figure.add_subplot(grid[1 + row_index, col]) for col in range(5)]
        stats = _render_roi_row(
            axes, image, row["box"], row["score"], row["raw"], row["refined"],
            row["delta"], row["support"], row["final_mask"], delta_cap,
        )
        row["stats"] = stats
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def _render_compare_figure(out_path, image, pred_with, pred_without, stem) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    _draw_overview(axes[0], image, pred_with, pred_with, None, "final masks WITH P2")
    _draw_overview(axes[1], image, pred_without, pred_with, None,
                   "final masks WITHOUT P2 (beta=0)")
    figure.suptitle(f"P2 effect on final masks @ {stem}", fontsize=11)
    figure.tight_layout()
    figure.savefig(out_path, dpi=140)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    _resolve_sam2_repo(args, snapshot)
    model_config, config_source = _resolve_model_config(args, snapshot)
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    model = _register_and_build(model_config)
    _load_model_state(model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False)
    model.to(device)
    model.eval()

    refiner = getattr(model.roi_head.mask_head, "p2_boundary_refiner", None)
    if refiner is None:
        raise RuntimeError("This checkpoint has P2BoundaryRefiner disabled; nothing to visualize.")
    delta_cap = float(refiner.beta) * float(refiner.delta_logit_max)
    logger.info(
        "model from %s | refiner beta=%.3f delta_logit_max=%.2f | device=%s",
        config_source, float(refiner.beta), float(refiner.delta_logit_max), device,
    )

    contract = _resolve_dataset_contract(args, snapshot)
    loader = _build_loader(contract, args.num_workers)
    data_preprocessor = model.data_preprocessor

    global MARGIN
    MARGIN = int(args.margin)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else MAINLINE_ROOT / "logs" / "viz_p2" / f"{checkpoint_path.stem}_{args.split}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    rendered = 0
    skipped = 0
    capture = P2Capture(refiner)
    original_beta = float(refiner.beta)
    try:
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                if rendered >= args.limit:
                    break
                imgs, data_samples = _build_data_samples(batch, device)
                processed = data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                capture.frames.clear()
                outputs = model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                if not capture.frames:
                    logger.warning("batch %d: refiner never fired (no mask ROIs?)", batch_index)
                    continue
                frame = capture.frames[-1]
                rois = frame["rois"].numpy()

                pred_without = None
                if args.ab_no_p2:
                    refiner.beta = 0.0
                    capture.frames.clear()
                    outputs_nop2 = model.predict(
                        processed["inputs"], processed["data_samples"], rescale=False
                    )
                    refiner.beta = original_beta
                    pred_without = [
                        _extract_instances_numpy(
                            (out.pred_instances if hasattr(out, "pred_instances") else out),
                            batch["img_metas"][i]["img_shape"],
                        )
                        for i, out in enumerate(outputs_nop2)
                    ]

                for image_index in range(len(imgs)):
                    if skipped < args.skip:
                        skipped += 1
                        continue
                    if rendered >= args.limit:
                        break
                    meta = batch["img_metas"][image_index]
                    shape = meta["img_shape"]
                    stem = Path(str(meta.get("filename", f"img_{batch_index}_{image_index}"))).stem
                    image = _image_numpy(imgs[image_index])
                    gt = _extract_instances_numpy(
                        processed["data_samples"][image_index].gt_instances, shape
                    )
                    out = outputs[image_index]
                    pred = _extract_instances_numpy(
                        out.pred_instances if hasattr(out, "pred_instances") else out, shape
                    )

                    sel = rois[:, 0] == image_index
                    image_rows = rois[sel]
                    global_indices = np.nonzero(sel)[0]
                    scores = (
                        np.asarray(pred["scores"], dtype=np.float64)
                        if "scores" in pred
                        else np.ones(len(image_rows))
                    )
                    num_match = min(len(image_rows), len(pred["bboxes"]))
                    if num_match and num_match != len(image_rows):
                        logger.warning(
                            "%s: %d coarse ROIs vs %d final preds; trusting first %d",
                            stem, len(image_rows), len(pred["bboxes"]), num_match,
                        )
                    order = (
                        sorted(range(num_match), key=lambda j: float(scores[j]), reverse=True)[: args.top_k]
                        if num_match
                        else []
                    )
                    order = [j for j in order if scores[j] >= args.score_thr]

                    final_masks = _masks_of(pred)
                    roi_rows = []
                    for j in order:
                        roi_rows.append(
                            {
                                "pred_index": j,
                                "box": image_rows[j][1:5],
                                "score": float(scores[j]),
                                "raw": frame["raw"][global_indices[j]][0].numpy(),
                                "refined": frame["refined"][global_indices[j]][0].numpy(),
                                "delta": frame["delta"][global_indices[j]][0].numpy(),
                                "support": frame["support"][global_indices[j]][0].numpy(),
                                "final_mask": final_masks[j] if j < len(final_masks) else None,
                            }
                        )
                    if roi_rows:
                        _render_image_figure(
                            out_dir / f"{stem}_p2.png", image, gt, pred, roi_rows, delta_cap, stem
                        )
                    if args.ab_no_p2 and pred_without is not None:
                        _render_compare_figure(
                            out_dir / f"{stem}_p2_compare.png", image, pred,
                            pred_without[image_index], stem,
                        )
                    summary.append(
                        {
                            "image": stem,
                            "num_pred": int(len(pred["bboxes"])),
                            "detailed_rois": [
                                {
                                    "pred_index": row["pred_index"],
                                    "score": row["score"],
                                    "delta_abs_mean": row["stats"]["delta_abs_mean"],
                                    "delta_abs_max": row["stats"]["delta_abs_max"],
                                    "support_pixels": row["stats"]["support_pixels"],
                                }
                                for row in roi_rows
                            ],
                        }
                    )
                    rendered += 1
                    logger.info(
                        "rendered %s (%d preds, %d detailed)", stem, len(pred["bboxes"]), len(roi_rows)
                    )
                logger.info("batch %d done, %d/%d images rendered", batch_index + 1, rendered, args.limit)
    finally:
        capture.close()

    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(checkpoint_path),
                "split": args.split,
                "dataset": contract,
                "delta_cap": delta_cap,
                "images": summary,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    print(f"P2 visualization complete: {out_dir}")


if __name__ == "__main__":
    main()
