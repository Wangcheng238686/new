#!/usr/bin/env python
"""Offline |delta| quantile probe for P2BoundaryRefiner.

Design-doc v3 §2.3: |delta| quantiles are NOT training-log fields; this
tool loads a P2-enabled checkpoint the same way
``inference/infer_from_checkpoint.py`` does, hooks the refiner forward,
runs a bounded number of validation batches, and reports the |delta|
distribution RESTRICTED to the search support (delta is identically zero
outside it), compared against arbitrary envelope caps (e.g. the
conservative 0.05 and an escalated 0.30).

Used to judge whether a trained P2 head "wants" more room than its
envelope, instead of misreading saturation telemetry.
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
    parser.add_argument("--max-batches", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--envelopes", type=float, nargs="+", default=[0.05, 0.30],
        help="Envelope caps to compare the |delta| distribution against.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(checkpoint_path)
    snapshot = _snapshot(checkpoint)
    ns = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        config=None,
        split=args.split,
        data_root=None,
        ann_file=None,
        image_subdir=None,
        image_size=None,
        batch_size=None,
        sam2_repo=None,
        sam2_ckpt=None,
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

    refiner = model.roi_head.mask_head.p2_boundary_refiner
    if refiner is None:
        raise SystemExit("checkpoint has no P2 boundary refiner enabled")

    collected: list[np.ndarray] = []

    def hook(module, inputs, output):
        delta = output["delta_logits"].detach().float()
        support = output["search_support"].detach() > 0
        if support.any():
            collected.append(delta[support].abs().cpu().numpy())
        else:
            collected.append(np.zeros(0, dtype=np.float32))

    handle = refiner.register_forward_hook(hook)

    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)
    n_batches = 0
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if index >= args.max_batches:
                    break
                imgs, data_samples = _build_data_samples(batch, device)
                processed = model.data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                model.predict(
                    processed["inputs"], processed["data_samples"], rescale=False
                )
                n_batches += 1
    finally:
        handle.remove()

    values = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    payload = {
        "checkpoint": str(checkpoint_path),
        "weights": args.weights,
        "batches": n_batches,
        "num_support_pixels": int(values.size),
        "abs_delta_mean": float(values.mean()) if values.size else 0.0,
        "abs_delta_quantiles": {
            f"p{q}": float(np.percentile(values, q)) if values.size else 0.0
            for q in (50, 90, 99, 99.9)
        },
        "abs_delta_max": float(values.max()) if values.size else 0.0,
        "fraction_at_or_above_cap": {
            f"{cap:.2f}": float((values >= 0.99 * cap).mean()) if values.size else 0.0
            for cap in args.envelopes
        },
        "envelopes": [float(cap) for cap in args.envelopes],
        "note": (
            "in-support |delta| only (delta is zero outside search_support); "
            "envelope caps are compared with a 0.99 tolerance, matching "
            "P2BR/saturation_threshold; population = predict-time post-NMS "
            "ROIs over the FULL validation split, which differs from "
            "training-proposal telemetry and from a 10% dev val subset; "
            "offline analysis tool, not a training-log field"
        ),
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
