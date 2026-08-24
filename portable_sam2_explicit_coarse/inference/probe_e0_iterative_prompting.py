#!/usr/bin/env python
"""E0: heuristic second-round prompting (PA-SAM hard-point mining, untrained).

Round 1 decodes masks with the normal prompts. Per ROI we then mine
uncertainty points from the decoder's own output:  4 positives and 4
negatives from the top-entropy candidates on each side of 0.5 (greedy
min-distance among top candidates to avoid clustering). These are appended
to the PromptEncoder point set and the mask head decodes again. Round-2
outputs replace round-1 everywhere downstream.

If this zero-training variant already moves segm mAP, the trained selector
(PA-SAM style Gumbel top-k + uncertainty token) is worth Stage-1 training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

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
    parser.add_argument("--k-per-side", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=50)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _mine_uncertainty_points(prob: torch.Tensor, k: int, candidates: int):
    """prob [N,256,256] -> extra coords [N,2k,2] (PE frame) + labels [N,2k]."""
    n, h, w = prob.shape
    unc = 4.0 * prob * (1.0 - prob)
    flat = unc.flatten(1)
    top = flat.topk(candidates, dim=1).indices  # [N,C]
    yy = (top // w).float()
    xx = (top % w).float()
    pos_mask = prob.flatten(1).gather(1, top) >= 0.5
    coords = torch.zeros((n, 2 * k, 2), device=prob.device)
    labels = torch.zeros((n, 2 * k), device=prob.device, dtype=torch.long)

    def greedy(cand_yx: torch.Tensor, count: int) -> List[torch.Tensor]:
        picked: List[torch.Tensor] = []
        picked.append(cand_yx[0])
        while len(picked) < count and len(picked) < cand_yx.shape[0]:
            stack = torch.stack(picked)  # [P,2]
            dist = ((cand_yx[:, None, :] - stack[None, :, :]) ** 2).sum(-1).min(1).values
            picked.append(cand_yx[int(dist.argmax())])
        return picked

    for i in range(n):
        for side, lab in ((pos_mask[i], 1), (~pos_mask[i], 0)):
            cand_idx = top[i][side]
            if cand_idx.numel() == 0:
                continue
            cy = (cand_idx // w).float()
            cx = (cand_idx % w).float()
            cand_yx = torch.stack([cy, cx], dim=-1)
            picked = greedy(cand_yx, k)
            slot0 = 0 if lab == 1 else k
            for j, (py, px) in enumerate(picked):
                coords[i, slot0 + j, 0] = (px + 0.5) / w * 1024.0
                coords[i, slot0 + j, 1] = (py + 0.5) / h * 1024.0
                labels[i, slot0 + j] = lab
    return coords, labels


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

    stash: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None]
    orig_head_forward = mask_head.forward
    orig_pe_forward = mask_head.prompt_encoder.forward

    def pe_wrapped(points=None, boxes=None, masks=None, **kwargs):
        if stash[0] is not None and (points is not None or boxes is not None):
            extra_coords, extra_labels = stash[0]
            if points is None:
                points = (extra_coords, extra_labels)
            else:
                coords, labels = points
                points = (
                    torch.cat([coords, extra_coords.to(coords.dtype)], dim=1),
                    torch.cat([labels, extra_labels], dim=1),
                )
        return orig_pe_forward(points=points, boxes=boxes, masks=masks, **kwargs)

    def head_wrapped(x, image_embeddings, image_positional_embeddings,
                     roi_img_ids=None, high_res_features=None, boxes=None,
                     p2_feature=None, prompt_rois=None):
        out1 = orig_head_forward(
            x, image_embeddings, image_positional_embeddings,
            roi_img_ids=roi_img_ids, high_res_features=high_res_features,
            boxes=boxes, p2_feature=p2_feature, prompt_rois=prompt_rois,
        )
        mask_preds1 = out1[0]
        with torch.no_grad():
            prob = torch.sigmoid(mask_preds1[:, 0].detach().float())
            extra_coords, extra_labels = _mine_uncertainty_points(
                prob, args.k_per_side, args.candidates
            )
        stash[0] = (extra_coords, extra_labels)
        try:
            out2 = orig_head_forward(
                x, image_embeddings, image_positional_embeddings,
                roi_img_ids=roi_img_ids, high_res_features=high_res_features,
                boxes=boxes, p2_feature=p2_feature, prompt_rois=prompt_rois,
            )
        finally:
            stash[0] = None
        return out2

    mask_head.prompt_encoder.forward = pe_wrapped  # type: ignore[method-assign]
    mask_head.forward = head_wrapped  # type: ignore[method-assign]
    logger.info("E0 active: round-2 decode with %d+%d entropy-mined points per ROI",
                args.k_per_side, args.k_per_side)

    contract = _resolve_dataset_contract(args, snapshot)
    loader = _build_loader(contract, args.num_workers)
    segm_score_key = getattr(mask_head, "segm_score_key", "scores")

    from utils.coco_eval_utils import build_coco_gt_and_dt, run_coco_eval

    all_gt: List[dict] = []
    all_dt: List[dict] = []
    all_metas: List[dict] = []
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

    mask_head.forward = orig_head_forward  # type: ignore[method-assign]
    mask_head.prompt_encoder.forward = orig_pe_forward  # type: ignore[method-assign]
    coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, all_metas, score_key=segm_score_key)
    metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(
            {"checkpoint": str(checkpoint_path), "k_per_side": args.k_per_side,
             "candidates": args.candidates, "metrics": metrics},
            handle, indent=2,
        )
    print(json.dumps({k: v for k, v in metrics.items() if "mAP" in k}, indent=2))
    print(f"E0 probe complete: {output_dir}")


if __name__ == "__main__":
    main()
