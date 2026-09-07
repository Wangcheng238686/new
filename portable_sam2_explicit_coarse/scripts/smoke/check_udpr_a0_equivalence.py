"""Compare frozen A0 and zero-initialized UDPR on real validation batches.

This is an integration guard, not an accuracy evaluation.  It proves that the
optional decoder feature capture and zero-initialized scatter path preserve
the exact deployed prediction before UDPR learns.
"""
import argparse
import copy
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples, _build_loader, _load_checkpoint, _load_model_state,
    _register_and_build, _resolve_dataset_contract, _resolve_device, _snapshot,
)


def _digest(outputs) -> str:
    digest = hashlib.sha256()
    for output in outputs:
        inst = output.pred_instances if hasattr(output, "pred_instances") else output
        for name in ("bboxes", "labels", "scores"):
            value = getattr(inst, name)
            if torch.is_tensor(value):
                value = value.detach().cpu().contiguous().numpy().tobytes()
            digest.update(value)
        masks = getattr(inst, "masks", None)
        value = masks.masks if hasattr(masks, "masks") else masks
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous().numpy().tobytes()
        elif value is not None:
            value = value.tobytes()
        digest.update(value or b"")
    return digest.hexdigest()


def _predict_one(model, batch, device):
    model.to(device).eval()
    imgs, data_samples = _build_data_samples(batch, device)
    processed = model.data_preprocessor(
        {"inputs": imgs, "data_samples": data_samples}, training=False
    )
    with torch.inference_mode():
        return model.predict(processed["inputs"], processed["data_samples"], rescale=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--image-subdir", default=None)
    parser.add_argument("--image-size", type=int, nargs=2, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    checkpoint = _load_checkpoint(Path(args.checkpoint))
    snapshot = _snapshot(checkpoint)
    base_cfg = copy.deepcopy(dict(snapshot["model_config"]))
    tail_cfg = copy.deepcopy(base_cfg)
    tail_cfg["roi_head"]["mask_head"]["decoder_tail_refiner_cfg"] = dict(
        enabled=True, num_points=64, hidden_dim=128,
        point_loss_weight=1.0, delta_logit_max=2.0,
    )
    device = _resolve_device(args.device)
    contract = _resolve_dataset_contract(args, snapshot)
    loader = _build_loader(contract, num_workers=0)
    batches = []
    for index, batch in enumerate(loader):
        if index >= args.max_batches:
            break
        batches.append(batch)

    baseline = _register_and_build(base_cfg)
    report = _load_model_state(baseline, checkpoint["model"], allow_nonstrict=False)
    if report["missing_keys"] or report["unexpected_keys"]:
        raise RuntimeError(f"A0 strict load failed: {report}")
    baseline_hashes = [_digest(_predict_one(baseline, batch, device)) for batch in batches]
    del baseline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tail = _register_and_build(tail_cfg)
    report = _load_model_state(tail, checkpoint["model"], allow_nonstrict=True)
    expected = sorted(k for k in tail.state_dict() if ".decoder_tail_refiner." in k)
    if sorted(report["missing_keys"]) != expected or report["unexpected_keys"]:
        raise RuntimeError(f"UDPR heat-init whitelist failed: report={report}, expected={expected}")
    tail_hashes = [_digest(_predict_one(tail, batch, device)) for batch in batches]
    if baseline_hashes != tail_hashes:
        raise RuntimeError(f"zero-init UDPR changed A0 predictions: {baseline_hashes} != {tail_hashes}")
    print("UDPR A0 zero-init full prediction equivalence: PASS")
    for index, digest in enumerate(tail_hashes):
        print(f"batch={index} prediction_sha256={digest}")


if __name__ == "__main__":
    main()
