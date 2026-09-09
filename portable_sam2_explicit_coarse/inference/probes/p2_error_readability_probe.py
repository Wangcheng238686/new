#!/usr/bin/env python3
"""Frozen-A0 test of whether spatially aligned P2 can localise coarse errors.

This is a *readability* gate, not a new model and not an Oracle mask result.
The frozen A0 detector/coarse head is run once on the NWPU train split to fit
two linear pixel readouts, then once on the held-out validation split:

``raw``       raw-logit / |raw-logit| only;
``p2``        the same raw features plus aligned frozen P2 channels;
``permuted``  the trained P2 readout with ROI P2 rows cyclically permuted.

The target is a raw-binary error outside A2's *accepted* S4 support.  GT is
used only to train/evaluate that target; it never enters A0 forward.  At a
fixed per-ROI write budget, the diagnostic reports error recall, precision,
and the coarse-IoU effect of flipping the raw sign at selected pixels.  A new
P2-conditioned support module is disallowed unless aligned P2 beats raw on
held-out image-bootstrap recall, precision and IoU, while the permutation
control does not.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.ops import RoIAlign

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples, _build_loader, _load_checkpoint, _load_model_state,
    _register_and_build, _resolve_dataset_contract, _resolve_model_config,
    _select_state_dict, _snapshot, _write_json,
)
from inference.probes.coarse_to_mining_oracle_probe import _gt_cache, _match_gt_index  # noqa: E402
from inference.probes.p2_coarse_gate_probe import _roi_target_grid  # noqa: E402
from inference.probes.p2_support_decomposition_oracle import _support_refiner  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help="frozen P2-off A0 checkpoint")
    p.add_argument("--support-reference-checkpoint", required=True, help="A2 checkpoint supplying accepted-S4 geometry")
    p.add_argument("--train-ann-file", required=True)
    p.add_argument("--train-image-subdir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--match-iou", type=float, default=.5)
    p.add_argument("--roi-output-size", type=int, default=64)
    p.add_argument("--samples-per-class-per-roi", type=int, default=16)
    p.add_argument("--max-train-samples-per-class", type=int, default=64000)
    p.add_argument("--fit-steps", type=int, default=400)
    p.add_argument("--fit-batch-size", type=int, default=2048)
    p.add_argument("--fit-lr", type=float, default=.03)
    p.add_argument("--write-budget", type=float, default=.10)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _target_and_eligible(raw: torch.Tensor, box: torch.Tensor, label: torch.Tensor, entry: Any, refiner: Any, threshold: float) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    index = _match_gt_index(box.detach().cpu(), label.detach().cpu(), entry, threshold)
    if index is None:
        return None
    target = _roi_target_grid(entry[2][index], box.detach().cpu(), tuple(raw.shape[-2:]))
    if target is None:
        return None
    target = target.to(raw.device)
    _, _, _, support, _ = refiner._raw_support(raw[None, None])
    # Candidate domain is exactly the part rejected by the production accepted S4.
    return target, raw.ge(0).ne(target), ~support[0, 0].bool()


def _features(raw: torch.Tensor, p2_roi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    raw2 = torch.stack([raw, raw.abs()], dim=-1)
    p2 = p2_roi.permute(1, 2, 0)
    return raw2, torch.cat([raw2, p2], dim=-1)


def _safe_permutation(p2: torch.Tensor) -> torch.Tensor:
    """ROI permutation negative control; no row may retain its own P2 map."""
    if p2.shape[0] < 2:
        # A one-ROI image has no other ROI.  A spatial roll remains a strict
        # misalignment control rather than silently treating aligned P2 as null.
        return p2.roll(shifts=(p2.shape[-2] // 2, p2.shape[-1] // 2), dims=(-2, -1))
    return p2.roll(shifts=1, dims=0)


def _fit_linear(x: torch.Tensor, y: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, std = x.mean(0), x.std(0).clamp_min(1e-5)
    x = (x - mean) / std
    linear = torch.nn.Linear(x.shape[1], 1, device=x.device)
    torch.nn.init.zeros_(linear.weight); torch.nn.init.zeros_(linear.bias)
    opt = torch.optim.AdamW(linear.parameters(), lr=args.fit_lr, weight_decay=1e-4)
    gen = torch.Generator(device=x.device).manual_seed(args.seed)
    for _ in range(args.fit_steps):
        ids = torch.randint(x.shape[0], (min(args.fit_batch_size, x.shape[0]),), generator=gen, device=x.device)
        loss = F.binary_cross_entropy_with_logits(linear(x[ids]).squeeze(1), y[ids])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return linear.weight.detach()[0], linear.bias.detach()[0], torch.stack([mean, std])


def _score(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(((x - norm[0]) / norm[1]).matmul(weight) + bias)


def _add_counts(row: Dict[str, float], name: str, raw: torch.Tensor, target: torch.Tensor, eligible: torch.Tensor, score: torch.Tensor, budget: float) -> None:
    flat = eligible.flatten().nonzero().flatten()
    error = raw.ge(0).ne(target)
    total_error = int((error & eligible).sum())
    count = min(int(flat.numel()), max(1, int(math.ceil(float(flat.numel()) * budget)))) if flat.numel() else 0
    selected = torch.zeros_like(eligible)
    if count:
        selected.flatten()[flat[score.flatten()[flat].topk(count).indices]] = True
    hits = int((selected & error).sum()); writes = int(selected.sum())
    fixed = raw.ge(0).clone(); fixed[selected] = ~fixed[selected]
    pairs = [(name + "_", fixed)]
    if name == "raw":
        pairs.insert(0, ("raw_", raw.ge(0)))
    for prefix, prediction in pairs:
        inter = int((prediction & target).sum()); pred = int(prediction.sum()); tgt = int(target.sum())
        row[prefix + "inter"] += inter; row[prefix + "union"] += pred + tgt - inter
    row[name + "_hits"] += hits; row[name + "_writes"] += writes; row[name + "_errors"] += total_error


def _ratio(records: Iterable[Mapping[str, float]], numerator: str, denominator: str) -> float:
    return sum(r[numerator] for r in records) / max(sum(r[denominator] for r in records), 1.)


def _bootstrap(rows: List[Dict[str, float]], resamples: int, seed: int) -> Dict[str, Any]:
    names = ("raw", "p2", "permuted")
    def metrics(sample: Iterable[Mapping[str, float]], name: str) -> Dict[str, float]:
        return {"recall": _ratio(sample, name + "_hits", name + "_errors"), "precision": _ratio(sample, name + "_hits", name + "_writes"), "coarse_iou": _ratio(sample, name + "_inter", name + "_union")}
    point = {name: metrics(rows, name) for name in names}
    if not resamples: return {"point": point, "ci": {}, "resamples": 0}
    rng = np.random.default_rng(seed); deltas: Dict[str, List[float]] = defaultdict(list)
    for _ in range(resamples):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        values = {name: metrics(sample, name) for name in names}
        for metric in ("recall", "precision", "coarse_iou"):
            deltas["p2_minus_raw_" + metric].append(values["p2"][metric] - values["raw"][metric])
            deltas["p2_minus_permuted_" + metric].append(values["p2"][metric] - values["permuted"][metric])
    ci = {key: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)), "p_gt_zero": float(np.mean(np.asarray(v) > 0))} for key, v in deltas.items()}
    return {"point": point, "ci": ci, "resamples": resamples, "seed": seed}


def _verdict(boot: Mapping[str, Any]) -> Dict[str, Any]:
    if not boot["ci"]: return {"verdict": "not_decided"}
    metrics = ("recall", "precision", "coarse_iou")
    aligned = all(boot["ci"]["p2_minus_raw_" + m]["low"] > 0 for m in metrics)
    control = all(boot["ci"]["p2_minus_permuted_" + m]["low"] > 0 for m in metrics)
    return {"verdict": "pass" if aligned and control else "fail", "p2_beats_raw_all_metrics": aligned, "p2_beats_permuted_all_metrics": control, "rule": "all three held-out paired-image 95% lower bounds >0 against raw and ROI-permuted P2"}


def main() -> int:
    args = parse_args()
    if not (0 < args.match_iou <= 1 and 0 < args.write_budget <= 1 and args.resamples >= 0): raise SystemExit("invalid gate range")
    ckpt_path = Path(args.checkpoint).resolve(); ckpt = _load_checkpoint(ckpt_path); snapshot = _snapshot(ckpt)
    ns = SimpleNamespace(checkpoint=str(ckpt_path), config=None, split="validation", data_root=None, ann_file=None, image_subdir=None, image_size=None, batch_size=1, sam2_repo=None, sam2_ckpt=None)
    cfg, source = _resolve_model_config(ns, snapshot); model = _register_and_build(cfg)
    strict = _load_model_state(model, _select_state_dict(ckpt, "model"), allow_nonstrict=False)
    if strict["missing_keys"] or strict["unexpected_keys"]: raise RuntimeError(f"strict load failed: {strict}")
    device = torch.device(args.device); model.to(device).eval(); head = model.roi_head.mask_head
    if head.p2_boundary_refiner is not None: raise RuntimeError("readability gate requires P2-off A0")
    accepted, _, support_cfg = _support_refiner(Path(args.support_reference_checkpoint).resolve())
    align = RoIAlign(output_size=(args.roi_output_size, args.roi_output_size), spatial_scale=.25, sampling_ratio=0, pool_mode="avg", aligned=True).to(device)
    val_contract = _resolve_dataset_contract(ns, snapshot); train_contract = dict(val_contract); train_contract.update({"ann_file": args.train_ann_file, "image_subdir": args.train_image_subdir})
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    sample_generator = torch.Generator(device=device).manual_seed(args.seed)

    def collect(contract: Mapping[str, Any], *, train: bool) -> Any:
        loader = _build_loader(contract, 0); features: List[torch.Tensor] = []; labels: List[torch.Tensor] = []; rows: Dict[int, Dict[str, float]] = {}
        state: Dict[str, Any] = {"gts": None, "labels": None, "ids": None, "boxes": None, "queue": None, "cursor": 0, "metas": None}
        original_predict, original_coarse = model.roi_head.predict_mask, head._forward_coarse_p2
        def predict(*fa: Any, **kw: Any) -> Any:
            result = kw.get("results_list", fa[2] if len(fa) > 2 else None); state["queue"] = torch.cat([r.labels.detach() for r in result]) if result else torch.zeros(0, dtype=torch.long, device=device); state["cursor"] = 0; return original_predict(*fa, **kw)
        def pre(_m: Any, _fa: Tuple[Any, ...], kw: Dict[str, Any]) -> None:
            boxes, ids = kw.get("boxes"), kw.get("roi_img_ids"); state["boxes"], state["ids"] = boxes, ids
            if boxes is not None:
                s, e = state["cursor"], state["cursor"] + boxes.shape[0]
                state["labels"], state["cursor"] = state["queue"][s:e], e
        def coarse(*fa: Any, **kw: Any) -> Any:
            output = original_coarse(*fa, **kw); raw = output[0]
            if raw is None or state["boxes"] is None: return output
            p2 = fa[4] if len(fa) > 4 else kw.get("p2_feature"); rois = fa[5] if len(fa) > 5 else kw.get("prompt_rois")
            p2_rows = align(p2.detach(), rois).float(); p2_perm = _safe_permutation(p2_rows)
            for i in range(raw.shape[0]):
                image_index = int(state["ids"][i]); entry = state["gts"][image_index] if 0 <= image_index < len(state["gts"]) else None
                got = _target_and_eligible(raw[i, 0], state["boxes"][i], state["labels"][i], entry, accepted, args.match_iou)
                if got is None: continue
                target, error, eligible = got; raw_x, p2_x = _features(raw[i, 0].float(), p2_rows[i]); _, perm_x = _features(raw[i, 0].float(), p2_perm[i])
                if train:
                    for label in (0, 1):
                        ids = (eligible & error.eq(bool(label))).flatten().nonzero().flatten()
                        if ids.numel():
                            take = ids[torch.randperm(ids.numel(), generator=sample_generator, device=ids.device)[:args.samples_per_class_per_roi]]; features.append(torch.cat([raw_x.flatten(0, 1)[take], p2_x.flatten(0, 1)[take]], 1).cpu()); labels.append(error.flatten()[take].float().cpu())
                else:
                    image_id = int(state["metas"][image_index].get("image_id", state["metas"][image_index].get("scene_id", image_index)))
                    zero_counts = {
                        key: 0.0
                        for name in ("raw", "p2", "permuted")
                        for key in (
                            name + "_inter", name + "_union", name + "_hits",
                            name + "_writes", name + "_errors",
                        )
                    }
                    row = rows.setdefault(image_id, {"image_id": image_id, **zero_counts})
                    # Weights are installed after train collection; closure has them then.
                    for name, x in (("raw", raw_x), ("p2", p2_x), ("permuted", perm_x)):
                        score = _score(x.reshape(-1, x.shape[-1]), *readouts[name]).reshape_as(raw[i, 0]); _add_counts(row, name, raw[i, 0], target, eligible, score, args.write_budget)
            return output
        h = head.register_forward_pre_hook(pre, with_kwargs=True); model.roi_head.predict_mask = predict; head._forward_coarse_p2 = coarse
        try:
            with torch.inference_mode():
                limit = args.max_train_batches if train else args.max_val_batches
                for idx, batch in enumerate(loader):
                    if limit and idx >= limit: break
                    inputs, samples = _build_data_samples(batch, device); processed = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, training=False); state["gts"], state["metas"] = _gt_cache(processed["data_samples"]), batch["img_metas"]; model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                    if state["queue"] is not None and state["cursor"] != int(state["queue"].numel()):
                        raise RuntimeError("proposal-label queue was not consumed exactly")
        finally:
            h.remove(); model.roi_head.predict_mask = original_predict; head._forward_coarse_p2 = original_coarse
        return (features, labels) if train else list(rows.values())

    train_features, train_labels = collect(train_contract, train=True)
    if not train_features: raise RuntimeError("no matched outside-support train pixels")
    packed, y = torch.cat(train_features), torch.cat(train_labels); gen = torch.Generator().manual_seed(args.seed)
    keep = torch.cat([(y == 0).nonzero().flatten()[:args.max_train_samples_per_class], (y == 1).nonzero().flatten()[:args.max_train_samples_per_class]])
    keep = keep[torch.randperm(keep.numel(), generator=gen)]; packed, y = packed[keep].to(device), y[keep].to(device)
    raw_w, raw_b, raw_norm = _fit_linear(packed[:, :2], y, args); p2_w, p2_b, p2_norm = _fit_linear(packed[:, 2:], y, args)
    readouts = {"raw": (raw_w, raw_b, raw_norm), "p2": (p2_w, p2_b, p2_norm), "permuted": (p2_w, p2_b, p2_norm)}
    rows = collect(val_contract, train=False); boot = _bootstrap(rows, args.resamples, args.seed); summary = {"checkpoint": str(ckpt_path), "support_reference_checkpoint": str(Path(args.support_reference_checkpoint).resolve()), "support_geometry": support_cfg, "config_source": source, "strict_load": strict, "train_contract": train_contract, "validation_contract": val_contract, "fit_samples": int(y.numel()), "fit_positive_fraction": float(y.mean()), "write_budget": args.write_budget, "validation_images": len(rows), "bootstrap": boot, "gate": _verdict(boot), "note": "Frozen A0 P2 readability gate; GT defines only train/eval old-support-outside error labels. No GT tensor enters model forward or selected correction."}
    _write_json(out / "summary.json", summary); _write_json(out / "image_records.json", rows); print(json.dumps(summary, indent=2), flush=True); return 0


if __name__ == "__main__": raise SystemExit(main())
