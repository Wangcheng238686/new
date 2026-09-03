#!/usr/bin/env python
"""Pathway-contribution probe: counterfactual ablations at inference.

Uses the mask head's built-in inference-only diagnostic switches
(_diagnostic_forward_ablation) to measure how much each decoder input
pathway currently contributes to final segm mAP:

  zero_high_res   : zero the decoder's high-res feature path (s0/s1)
  no_sparse       : drop point/box sparse prompt tokens
  zero_base_dense : zero the base dense embedding (no-mask + ROI residual)
  zero_image_pe   : zero the image positional embedding

Run with the standard architecture env (see scripts/visualize_p2_checkpoint.sh).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

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

logger = logging.getLogger("pathway_probe")


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
    parser.add_argument(
        "--ablation",
        choices=("none", "zero_high_res", "no_sparse", "zero_base_dense", "zero_image_pe"),
        required=True,
    )
    parser.add_argument(
        "--disable-p2",
        action="store_true",
        help="Additionally set P2BoundaryRefiner beta=0 (composable with --ablation).",
    )
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
    if args.ablation != "none":
        mask_head._diagnostic_forward_ablation = args.ablation
        logger.info("diagnostic ablation active: %s", args.ablation)
    if args.disable_p2:
        refiner = getattr(mask_head, "p2_boundary_refiner", None)
        if refiner is None:
            logger.warning("--disable-p2 given but this checkpoint has no P2BoundaryRefiner")
        else:
            logger.info(
                "P2BoundaryRefiner beta overridden at inference: %.4f -> 0.0",
                float(refiner.beta),
            )
            refiner.beta = 0.0

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

    coco_gt, coco_dt = build_coco_gt_and_dt(all_gt, all_dt, all_metas, score_key=segm_score_key)
    metrics = run_coco_eval(coco_gt, coco_dt, iou_type="segm")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w") as handle:
        json.dump(
            {"checkpoint": str(checkpoint_path), "ablation": args.ablation, "metrics": metrics},
            handle, indent=2,
        )
    print(json.dumps({k: v for k, v in metrics.items() if "mAP" in k}, indent=2))
    print(f"pathway probe complete: {output_dir}")


if __name__ == "__main__":
    main()
