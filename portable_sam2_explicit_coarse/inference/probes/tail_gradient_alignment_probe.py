#!/usr/bin/env python3
"""Read-only A5 tail-gradient alignment audit.

For each fixed training batch, recover an A5 checkpoint and measure the
gradient of the production final-mask loss and of ``loss_decoder_tail`` on
*tail parameters only*.  No backward accumulation, optimizer step, checkpoint
write, or distributed process group is created.  A negative cosine establishes
an auxiliary/final-loss conflict; a non-negative cosine does not rescue an
error-localisation route whose frozen information gate has failed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples, _build_loader, _load_checkpoint, _load_model_state,
    _register_and_build, _resolve_dataset_contract, _resolve_model_config,
    _select_state_dict, _snapshot, _write_json,
)


def _scalar(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item)]
        return sum(tensors) if tensors else None
    return None


def _metrics(left: Iterable[torch.Tensor | None], right: Iterable[torch.Tensor | None]) -> dict[str, float]:
    lsq = rsq = dot = 0.0
    for lgrad, rgrad in zip(left, right):
        if lgrad is not None:
            lsq += float(lgrad.detach().float().square().sum().item())
        if rgrad is not None:
            rsq += float(rgrad.detach().float().square().sum().item())
        if lgrad is not None and rgrad is not None:
            dot += float((lgrad.detach().float() * rgrad.detach().float()).sum().item())
    return {"final_mask_l2": math.sqrt(lsq), "tail_aux_l2": math.sqrt(rsq),
            "dot": dot, "cosine": dot / math.sqrt(max(lsq * rsq, 1e-30))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-ann-file", required=True)
    p.add_argument("--train-image-subdir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-batches", type=int, default=12)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()
    if a.max_batches <= 0:
        raise SystemExit("--max-batches must be positive")
    out = Path(a.output_dir).resolve()
    if out.exists():
        raise RuntimeError(f"refusing to overwrite {out}")
    ckpt_path = Path(a.checkpoint).resolve(); ckpt = _load_checkpoint(ckpt_path); snap = _snapshot(ckpt)
    ns = SimpleNamespace(checkpoint=str(ckpt_path), config=None, split="validation", data_root=None,
                         ann_file=None, image_subdir=None, image_size=None, batch_size=1,
                         sam2_repo=None, sam2_ckpt=None)
    cfg, source = _resolve_model_config(ns, snap); model = _register_and_build(cfg)
    strict = _load_model_state(model, _select_state_dict(ckpt, "model"), allow_nonstrict=False)
    if strict["missing_keys"] or strict["unexpected_keys"]:
        raise RuntimeError(f"strict load failed: {strict}")
    head = model.roi_head.mask_head
    tail = getattr(head, "decoder_tail_refiner", None)
    if tail is None:
        raise RuntimeError("checkpoint has no decoder tail")
    params = [p for p in tail.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("tail has no trainable parameters")
    # This audit asks only for gradients *on the tail*.  Freezing every other
    # parameter preserves both loss-to-tail derivatives while avoiding the
    # 64-RoI SAM2 activation graph that a one-GPU diagnostic cannot hold.
    model.requires_grad_(False)
    tail.requires_grad_(True)
    params = [p for p in tail.parameters() if p.requires_grad]
    device = torch.device(a.device); model.to(device).train()
    contract = _resolve_dataset_contract(ns, snap)
    contract.update({"ann_file": a.train_ann_file, "image_subdir": a.train_image_subdir, "batch_size": 1})
    rows = []
    for batch_id, batch in enumerate(_build_loader(contract, 0)):
        if batch_id >= a.max_batches:
            break
        imgs, samples = _build_data_samples(batch, device)
        data = model.data_preprocessor({"inputs": imgs, "data_samples": samples}, training=True)
        losses = model(data["inputs"], data["data_samples"], mode="loss")
        final_loss, aux_loss = _scalar(losses.get("loss_mask")), _scalar(losses.get("loss_decoder_tail"))
        if final_loss is None or aux_loss is None or not final_loss.requires_grad or not aux_loss.requires_grad:
            continue
        final_grads = torch.autograd.grad(final_loss, params, retain_graph=True, allow_unused=True)
        aux_grads = torch.autograd.grad(aux_loss, params, retain_graph=False, allow_unused=True)
        row = _metrics(final_grads, aux_grads)
        row.update({"batch": batch_id, "loss_mask": float(final_loss.detach()), "loss_decoder_tail": float(aux_loss.detach())})
        rows.append(row)
        model.zero_grad(set_to_none=True)
    if not rows:
        raise RuntimeError("no finite batch exposed both losses")
    # Energy-weighted aggregate, matching the project convention for DDP probes.
    final_sq = sum(r["final_mask_l2"] ** 2 for r in rows); aux_sq = sum(r["tail_aux_l2"] ** 2 for r in rows)
    dot = sum(r["dot"] for r in rows)
    aggregate = {"final_mask_l2_rms": math.sqrt(final_sq / len(rows)),
                 "tail_aux_l2_rms": math.sqrt(aux_sq / len(rows)),
                 "energy_weighted_cosine": dot / math.sqrt(max(final_sq * aux_sq, 1e-30)),
                 "negative_cosine_batch_fraction": sum(r["cosine"] < 0 for r in rows) / len(rows)}
    verdict = "conflict_observed" if aggregate["energy_weighted_cosine"] < 0 else "no_net_conflict_observed"
    summary = {"checkpoint": str(ckpt_path), "strict_load": strict, "config_source": source,
               "train_contract": contract, "tail_parameter_tensors": len(params), "base_parameters_frozen": True, "batches": len(rows),
               "aggregate": aggregate, "verdict": verdict,
               "interpretation": "tail-only gradient accounting; it cannot overturn a failed frozen error-readability gate."}
    out.mkdir(parents=True); _write_json(out / "summary.json", summary); _write_json(out / "batches.json", rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
