#!/usr/bin/env python
"""Golden forward-equivalence check for refactors.

Builds a model from a self-describing checkpoint exactly the way
``inference/infer_from_checkpoint.py`` does, loads the requested weights
strictly (default: checkpoint["model"], NOT EMA), then

1. fingerprints the loaded state_dict (keys/dtypes/shapes/values SHA256);
2. fingerprints the resolved model_config and the architecture contract;
3. runs a deterministic eval-mode forward on the first validation batch
   (fixed order, no augmentation, fixed seed) and fingerprints the raw
   predictions (bboxes/labels/scores/mask_scores/masks).

Modes:
  --mode create  : compute the fingerprint and write the golden JSON. The
                   forward is executed twice and must agree bitwise, which
                   calibrates the nondeterminism noise floor of this
                   machine/environment.
  --mode check   : recompute and compare against the golden JSON. Exit 0 on
                   bitwise match, exit 1 with a per-field diff otherwise.

Typical use (WHU full C5-v2 golden weights):
  python scripts/smoke/check_forward_equivalence.py \
      --checkpoint <ckpt>.pth --weights model \
      --golden /path/to/golden.json --mode create|check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

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
    _resolve_device,
    _resolve_model_config,
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--weights",
        choices=("model", "ema"),
        default="model",
        help="checkpoint['model'] (default) or EMA state. Golden refactors must use 'model'.",
    )
    parser.add_argument("--golden", required=True, help="Golden JSON path.")
    parser.add_argument(
        "--mode",
        choices=("create", "check"),
        required=True,
    )
    parser.add_argument("--device", default=None, help="Defaults to cuda:0 when available.")
    parser.add_argument("--split", default="validation")
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return _sha256_bytes(np.ascontiguousarray(array).tobytes())


def _state_dict_fingerprint(model: torch.nn.Module) -> Dict[str, Any]:
    state = model.state_dict()
    digest = hashlib.sha256()
    keys = sorted(state.keys())
    dtypes: Dict[str, int] = {}
    for key in keys:
        tensor = state[key]
        dtype_name = str(tensor.dtype)
        dtypes[dtype_name] = dtypes.get(dtype_name, 0) + 1
        digest.update(key.encode("utf-8"))
        digest.update(dtype_name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(_tensor_sha256(tensor).encode("ascii"))
    return {
        "num_keys": len(keys),
        "dtype_counts": dtypes,
        "values_sha256": digest.hexdigest(),
    }


def _predictions_fingerprint(pred_instances: Any) -> Dict[str, Any]:
    def as_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if hasattr(value, "to_ndarray"):
            return value.to_ndarray()
        if hasattr(value, "masks"):
            return value.masks
        return value

    bboxes = np.ascontiguousarray(as_numpy(pred_instances.bboxes), dtype=np.float32)
    labels = np.ascontiguousarray(as_numpy(pred_instances.labels), dtype=np.int64)
    scores = np.ascontiguousarray(as_numpy(pred_instances.scores), dtype=np.float32)
    masks = as_numpy(pred_instances.masks)
    if hasattr(masks, "to_ndarray"):
        masks = masks.to_ndarray()
    elif hasattr(masks, "masks"):
        masks = masks.masks
    masks = np.ascontiguousarray(np.asarray(masks))

    fields = {
        "bboxes_sha256": _sha256_bytes(bboxes.tobytes()),
        "labels_sha256": _sha256_bytes(labels.tobytes()),
        "scores_sha256": _sha256_bytes(scores.tobytes()),
        "masks_sha256": _sha256_bytes(masks.tobytes()),
    }
    stats = {
        "num_instances": int(len(bboxes)),
        "masks_shape": list(masks.shape),
        "score_sum": float(scores.sum()),
        "mask_pixel_sum": int(masks.sum()),
        "first_bbox": [round(float(v), 4) for v in bboxes[0]] if len(bboxes) else [],
        "first_score": float(scores[0]) if len(scores) else None,
    }
    return {"fields": fields, "stats": stats}


def _set_determinism() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def _resolve_device_auto(value: str | None) -> torch.device:
    if value:
        return _resolve_device(value)
    return _resolve_device("auto")


def compute_fingerprint(args: argparse.Namespace) -> Dict[str, Any]:
    """Build + load + forward once; returns the fingerprint dict."""
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
    sam2_repo = _resolve_sam2_repo(ns, snapshot)
    model_config, config_source = _resolve_model_config(ns, snapshot)
    config_sha = _sha256_bytes(
        json.dumps(model_config, sort_keys=True, default=str).encode("utf-8")
    )
    model = _register_and_build(model_config)
    load_report = _load_model_state(
        model, _select_state_dict(checkpoint, args.weights), allow_nonstrict=False
    )
    if load_report["missing_keys"] or load_report["unexpected_keys"]:
        raise RuntimeError(f"Strict load failed: {load_report}")
    model.to(_resolve_device_auto(args.device))
    model.eval()
    state_fp = _state_dict_fingerprint(model)

    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)
    batch = next(iter(loader))
    device = next(model.parameters()).device
    imgs, data_samples = _build_data_samples(batch, device)
    preprocessor = model.data_preprocessor
    with torch.inference_mode():
        processed = preprocessor(
            {"inputs": imgs, "data_samples": data_samples}, training=False
        )
        outputs = model.predict(
            processed["inputs"], processed["data_samples"], rescale=False
        )
    pred_fp = _predictions_fingerprint(outputs[0].pred_instances)

    saved_contract = snapshot.get("architecture_contract", {})
    return {
        "checkpoint": str(checkpoint_path),
        "weights": args.weights,
        "split": args.split,
        "config_source": config_source,
        "model_config_sha256": config_sha,
        "sam2_repo": str(sam2_repo),
        "architecture_contract_saved": {
            "architecture_id": saved_contract.get("architecture_id"),
            "model_fingerprint": saved_contract.get("model_fingerprint"),
        },
        "state_dict": state_fp,
        "predictions": pred_fp,
    }


def compare(golden: Mapping[str, Any], current: Mapping[str, Any]) -> list:
    issues = []
    for field in (
        "model_config_sha256",
        "sam2_repo",
    ):
        if golden.get(field) != current.get(field):
            issues.append(f"{field}: golden={golden.get(field)!r} current={current.get(field)!r}")
    for side in ("architecture_contract_saved",):
        if golden.get(side) != current.get(side):
            issues.append(f"{side}: golden={golden.get(side)} current={current.get(side)}")
    for group in ("state_dict",):
        g, c = golden.get(group, {}), current.get(group, {})
        for key in ("num_keys", "dtype_counts", "values_sha256"):
            if g.get(key) != c.get(key):
                issues.append(f"{group}.{key}: golden={g.get(key)!r} current={c.get(key)!r}")
    g, c = golden.get("predictions", {}), current.get("predictions", {})
    for key, value in g.get("fields", {}).items():
        if c.get("fields", {}).get(key) != value:
            issues.append(f"predictions.{key}: MISMATCH")
    if g.get("stats") != c.get("stats"):
        issues.append(f"predictions.stats: golden={g.get('stats')} current={c.get('stats')}")
    return issues


def main() -> int:
    args = parse_args()
    golden_path = Path(args.golden).expanduser().resolve()
    _set_determinism()
    if args.mode == "create":
        first = compute_fingerprint(args)
        second = compute_fingerprint(args)
        issues = compare(first, second)
        if issues:
            print("NOISE: two consecutive runs disagree — golden is unusable:")
            for item in issues:
                print(f"  - {item}")
            return 2
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(
            json.dumps(first, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"GOLDEN CREATED: {golden_path}")
        print(json.dumps(first, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if not golden_path.is_file():
        print(f"Golden file not found: {golden_path}")
        return 2
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    current = compute_fingerprint(args)
    issues = compare(golden, current)
    if issues:
        print("FORWARD EQUIVALENCE: FAIL")
        for item in issues:
            print(f"  - {item}")
        print(json.dumps(current, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    print("FORWARD EQUIVALENCE: PASS (bitwise)")
    print(
        f"  state_dict: {current['state_dict']['num_keys']} keys, "
        f"values_sha256={current['state_dict']['values_sha256'][:16]}..."
    )
    print(
        f"  predictions: {current['predictions']['stats']['num_instances']} instances, "
        f"masks_sha256={current['predictions']['fields']['masks_sha256'][:16]}..."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
