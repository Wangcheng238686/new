#!/usr/bin/env python3
"""Frozen A4 tail-input readability audit.

The probe never updates the checkpointed model.  It fits a *linear* error
readout on selected tail points from the train split, then evaluates on the
held-out validation split.  The question is whether a detached 3x3 native
logit neighbourhood supplies error-readability beyond the actual A4 tail
input.  It is not an Oracle correction experiment: GT is used only after the
frozen forward to form train/evaluation labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.infer_from_checkpoint import (  # noqa: E402
    _build_data_samples, _build_loader, _load_checkpoint, _load_model_state,
    _register_and_build, _resolve_dataset_contract, _resolve_model_config,
    _select_state_dict, _snapshot, _write_json,
)
from inference.probes.coarse_to_mining_oracle_probe import _gt_cache, _match_gt_index  # noqa: E402
from inference.probes.p2_coarse_gate_probe import _roi_target_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train-ann-file", required=True)
    p.add_argument("--train-image-subdir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--match-iou", type=float, default=.5)
    p.add_argument("--max-train-per-class", type=int, default=48000)
    p.add_argument("--fit-steps", type=int, default=500)
    p.add_argument("--fit-batch-size", type=int, default=4096)
    p.add_argument("--fit-lr", type=float, default=.02)
    p.add_argument("--resamples", type=int, default=500)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--attach-zero-tail", action="store_true",
                   help="attach a zero-initialised DCR tail to a tail-off A0 checkpoint for capture only")
    p.add_argument("--write-counts", type=int, nargs="+", default=(4, 8, 16),
                   help="Top-M writes per matched ROI; all choices are fixed before evaluation")
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def _point_inputs(logits: torch.Tensor, upscaled: torch.Tensor, tokens: torch.Tensor,
                  indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct A4's exact point input and a spatially displaced patch."""
    n, _, h, w = logits.shape
    feat = upscaled.detach().flatten(2).transpose(1, 2).gather(
        1, indices[..., None].expand(-1, -1, upscaled.shape[1]))
    token = tokens.detach()[:, 0, :].unsqueeze(1).expand(-1, indices.shape[1], -1)
    flat = logits[:, 0].detach().float().reshape(n, -1)
    z = flat.gather(1, indices)
    yy = (indices // w).float() / max(1, h - 1)
    xx = (indices % w).float() / max(1, w - 1)
    base = torch.cat((feat.float(), token.float(), torch.stack((z, xx, yy, z.abs()), -1)), -1)

    def patch(source: torch.Tensor) -> torch.Tensor:
        unfolded = F.unfold(F.pad(source[:, :1].detach().float(), (1, 1, 1, 1), mode="replicate"), 3)
        return unfolded.transpose(1, 2).gather(1, indices[..., None].expand(-1, -1, 9))

    aligned = patch(logits)
    # Strictly spatially misaligned within each ROI; it preserves the patch
    # marginal distribution and cannot retain a point's own neighbourhood.
    shifted = patch(logits.roll((max(1, h // 2), max(1, w // 2)), dims=(-2, -1)))
    return base, torch.cat((aligned, shifted), -1)


def _target_for_final_mask_contract(head: Any, gt_mask: torch.Tensor, box: torch.Tensor,
                                    target_size: Tuple[int, int]) -> torch.Tensor:
    """Mirror production final-mask target coordinates; never crop full-image logits."""
    mode = str(head.final_mask_coordinate_mode)
    if mode == "full_image":
        return F.interpolate(gt_mask[None, None].float(), size=target_size, mode="nearest")[0, 0]
    if mode == "roi_local":
        target = _roi_target_grid(gt_mask, box, target_size)
        if target is None:
            raise RuntimeError("failed to construct ROI-local target")
        return target
    raise RuntimeError(f"unsupported final-mask coordinate mode {mode!r}")


def _fit_linear(x: torch.Tensor, y: torch.Tensor, a: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
    x = (x - mean) / std
    head = torch.nn.Linear(x.shape[1], 1, device=x.device)
    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=a.fit_lr, weight_decay=1e-4)
    g = torch.Generator(device=x.device).manual_seed(a.seed)
    for _ in range(a.fit_steps):
        ids = torch.randint(x.shape[0], (min(a.fit_batch_size, x.shape[0]),), generator=g, device=x.device)
        loss = F.binary_cross_entropy_with_logits(head(x[ids]).squeeze(1), y[ids])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return head.weight.detach()[0], head.bias.detach()[0], torch.stack((mean, std))


def _score(x: torch.Tensor, readout: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    w, b, norm = readout
    return torch.sigmoid(((x - norm[0]) / norm[1]).matmul(w) + b)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if not pos or not neg:
        return float("nan")
    # Mann-Whitney AUC with average ranks for ties (sham scores can tie).
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64); start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _pooled(rows: Sequence[Mapping[str, Any]], arm: str) -> float:
    return _auc(np.concatenate([np.asarray(r[arm], dtype=np.float64) for r in rows]),
                np.concatenate([np.asarray(r["label"], dtype=np.uint8) for r in rows]))


def _bootstrap(rows: List[Dict[str, Any]], a: argparse.Namespace) -> Dict[str, Any]:
    point = {name: _pooled(rows, name) for name in ("i0", "i1", "sham")}
    rng = np.random.default_rng(a.seed); values: Dict[str, List[float]] = defaultdict(list)
    for _ in range(a.resamples):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        m = {name: _pooled(sample, name) for name in point}
        values["i1_minus_i0"].append(m["i1"] - m["i0"])
        values["i1_minus_sham"].append(m["i1"] - m["sham"])
    ci = {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)),
              "p_gt_zero": float(np.mean(np.asarray(v) > 0))} for k, v in values.items()}
    return {"point_auc": point, "paired_image_bootstrap": ci, "resamples": a.resamples, "seed": a.seed}


def _verdict(boot: Mapping[str, Any]) -> Dict[str, Any]:
    ci = boot["paired_image_bootstrap"]
    passed = ci["i1_minus_i0"]["low"] > .01 and ci["i1_minus_sham"]["low"] > .01
    return {"verdict": "pass" if passed else "fail", "rule": "both held-out paired-image AUC lower bounds exceed +0.01", "patch_beats_i0": ci["i1_minus_i0"]["low"] > .01, "patch_beats_sham": ci["i1_minus_sham"]["low"] > .01}


def _write_selectivity(rows: Sequence[Mapping[str, Any]], a: argparse.Namespace) -> Dict[str, Any]:
    """Held-out Top-M authorisation audit over the exact production Top-K set.

    ``i0`` is the train-only error score.  ``uncertainty`` is the production
    candidate order (lowest |z| first); ``random`` is deterministic per-image
    sampling.  GT labels are consulted only here, after the frozen forward.
    """
    counts = sorted(set(int(m) for m in a.write_counts))
    if not counts or counts[0] <= 0 or counts[-1] > 64:
        raise RuntimeError("--write-counts must be positive and at most K=64")

    def one(sample: Sequence[Mapping[str, Any]], m: int, arm: str) -> Dict[str, float]:
        hits = writes = errors = 0
        for row in sample:
            y = np.asarray(row["label"], dtype=np.uint8)
            n = len(y)
            take = min(m, n)
            if not take:
                continue
            if arm == "score":
                order = np.argsort(-np.asarray(row["i0"], dtype=np.float64), kind="mergesort")
            elif arm == "uncertainty":
                # i0 base convention: z is the fourth scalar from the end.
                z = np.asarray(row["base_z"], dtype=np.float64)
                order = np.argsort(np.abs(z), kind="mergesort")
            else:
                seed = (int(row["image_id"]) * 1009 + m * 9176 + a.seed) & 0xffffffff
                order = np.random.default_rng(seed).permutation(n)
            chosen = order[:take]
            hits += int(y[chosen].sum()); writes += take; errors += int(y.sum())
        return {"precision": hits / max(writes, 1), "error_coverage": hits / max(errors, 1),
                "hits": hits, "writes": writes, "errors": errors}

    point = {str(m): {arm: one(rows, m, arm) for arm in ("score", "uncertainty", "random")} for m in counts}
    rng = np.random.default_rng(a.seed); deltas: Dict[str, List[float]] = defaultdict(list)
    for _ in range(a.resamples):
        sample = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        for m in counts:
            score, unc, ran = (one(sample, m, arm) for arm in ("score", "uncertainty", "random"))
            deltas[f"m{m}_score_minus_uncertainty_precision"].append(score["precision"] - unc["precision"])
            deltas[f"m{m}_score_minus_random_precision"].append(score["precision"] - ran["precision"])
    ci = {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)),
              "p_gt_zero": float(np.mean(np.asarray(v) > 0))} for k, v in deltas.items()}
    # Preregistered route gate: M=8 must select a clear majority of errors and
    # improve precision over both non-learned controls on held-out images.
    m = 8 if 8 in counts else counts[0]
    pass_score = point[str(m)]["score"]["precision"] >= .65
    pass_controls = all(ci[f"m{m}_score_minus_{arm}_precision"]["low"] > 0 for arm in ("uncertainty", "random"))
    return {"point": point, "paired_image_bootstrap": ci, "resamples": a.resamples, "seed": a.seed,
            "pre_registered_gate": {"verdict": "pass" if pass_score and pass_controls else "fail",
                "rule": f"M={m}: score precision >=0.65 and both paired-image precision CI lower bounds >0",
                "score_precision_majority": pass_score, "beats_both_controls": pass_controls}}


def _action_summary(rows: Sequence[Mapping[str, Any]], a: argparse.Namespace) -> Dict[str, Any]:
    """Decompose selected-error correction into direction, gate, and magnitude."""
    keys = ("all_error", "selected_error", "direction_agree", "current_reach", "sign_reach", "gate_reach", "both_reach", "cap_reach")
    def ratios(sample: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        sums = {k: sum(float(r["action"][k]) for r in sample) for k in keys}
        e = max(sums["selected_error"], 1.)
        return {"selected_error_coverage": sums["selected_error"] / max(sums["all_error"], 1.),
                "raw_direction_agreement": sums["direction_agree"] / e,
                "current_reach": sums["current_reach"] / e,
                "sign_oracle_reach": sums["sign_reach"] / e,
                "gate_oracle_reach": sums["gate_reach"] / e,
                "both_oracle_reach": sums["both_reach"] / e,
                "cap_oracle_reach": sums["cap_reach"] / e}
    for row in rows:
        q = row["action"]
        if q["all_error"] < q["selected_error"] or any(q[k] > q["selected_error"] for k in keys[2:]) or q["cap_reach"] < q["both_reach"]:
            raise RuntimeError(f"invalid action decomposition counts for image {row['image_id']}")
    point = ratios(rows); rng = np.random.default_rng(a.seed); vals: Dict[str, List[float]] = defaultdict(list)
    for _ in range(a.resamples):
        m = ratios([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        vals["sign_minus_current"].append(m["sign_oracle_reach"] - m["current_reach"])
        vals["gate_minus_current"].append(m["gate_oracle_reach"] - m["current_reach"])
        vals["both_minus_gate"].append(m["both_oracle_reach"] - m["gate_oracle_reach"])
        vals["both_minus_sign"].append(m["both_oracle_reach"] - m["sign_oracle_reach"])
        vals["cap_minus_both"].append(m["cap_oracle_reach"] - m["both_oracle_reach"])
    ci = {k: {"low": float(np.quantile(v, .025)), "high": float(np.quantile(v, .975)), "p_gt_zero": float(np.mean(np.asarray(v) > 0))} for k, v in vals.items()}
    return {"point": point, "paired_image_bootstrap": ci, "resamples": a.resamples, "definition": "2x2 factor: current=q*d; sign=q*s*abs(d); gate=d; both=s*abs(d). All reaches are selected pre-tail errors; cap is feasibility m=-s*z < delta_logit_max."}


def main() -> int:
    a = parse_args()
    if not (0 < a.match_iou <= 1 and a.max_train_per_class > 0 and a.resamples > 0):
        raise SystemExit("invalid audit arguments")
    ckpt_path = Path(a.checkpoint).resolve(); ckpt = _load_checkpoint(ckpt_path); snap = _snapshot(ckpt)
    ns = SimpleNamespace(checkpoint=str(ckpt_path), config=None, split="validation", data_root=None,
                         ann_file=None, image_subdir=None, image_size=None, batch_size=1,
                         sam2_repo=None, sam2_ckpt=None)
    cfg, config_source = _resolve_model_config(ns, snap)
    if a.attach_zero_tail:
        cfg = copy.deepcopy(cfg)
        cfg["roi_head"]["mask_head"]["decoder_tail_refiner_cfg"] = dict(
            enabled=True, num_points=64, hidden_dim=128, point_loss_weight=1.0,
            delta_logit_max=2.0, mode="confidence_gated", gate_init_prob=.1,
            gate_loss_weight=1.0, keep_loss_weight=.05,
        )
    model = _register_and_build(cfg)
    strict = _load_model_state(model, _select_state_dict(ckpt, "model"), allow_nonstrict=a.attach_zero_tail)
    expected_missing = (sorted(k for k in model.state_dict() if ".decoder_tail_refiner." in k)
                        if a.attach_zero_tail else [])
    if sorted(strict["missing_keys"]) != expected_missing or strict["unexpected_keys"]:
        raise RuntimeError(f"strict checkpoint load failed: {strict}")
    device = torch.device(a.device); model.to(device).eval(); head = model.roi_head.mask_head
    if not head.decoder_tail_enabled:
        raise RuntimeError("audit requires a checkpoint with decoder tail enabled")
    out = Path(a.output_dir).resolve()
    if out.exists(): raise RuntimeError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    val_contract = _resolve_dataset_contract(ns, snap)
    train_contract = dict(val_contract); train_contract.update({"ann_file": a.train_ann_file, "image_subdir": a.train_image_subdir})
    torch.manual_seed(a.seed)

    def collect(contract: Mapping[str, Any], train: bool, readouts: Mapping[str, Any] | None = None) -> Any:
        packed: List[torch.Tensor] = []; labels: List[torch.Tensor] = []; rows: Dict[int, Dict[str, Any]] = {}
        state: Dict[str, Any] = {"queue": None, "cursor": 0, "boxes": None, "ids": None, "labels": None, "tail": None, "gts": None, "metas": None}
        original_predict, original_tail = model.roi_head.predict_mask, head.decoder_tail_refiner.forward

        def predict(*fa: Any, **kw: Any) -> Any:
            results = kw.get("results_list", fa[2] if len(fa) > 2 else None)
            state["queue"] = torch.cat([r.labels.detach() for r in results]) if results else torch.zeros(0, dtype=torch.long, device=device)
            state["cursor"] = 0
            return original_predict(*fa, **kw)

        def pre(_m: Any, _fa: Tuple[Any, ...], kw: Dict[str, Any]) -> None:
            boxes, ids = kw.get("boxes"), kw.get("roi_img_ids")
            state["boxes"], state["ids"] = boxes, ids
            if boxes is not None:
                s, e = state["cursor"], state["cursor"] + boxes.shape[0]
                state["labels"], state["cursor"] = state["queue"][s:e], e

        def tail(logits: torch.Tensor, upscaled: torch.Tensor, tokens: torch.Tensor) -> Any:
            refined, outputs = original_tail(logits, upscaled, tokens)
            base, patches = _point_inputs(logits, upscaled, tokens, outputs["indices"])
            state["tail"] = (outputs, base.cpu(), patches.cpu())
            return refined, outputs

        hook = head.register_forward_pre_hook(pre, with_kwargs=True)
        model.roi_head.predict_mask = predict; head.decoder_tail_refiner.forward = tail
        try:
            with torch.inference_mode():
                limit = a.max_train_batches if train else a.max_val_batches
                loader = _build_loader(contract, 0)
                for batch_id, batch in enumerate(loader):
                    if limit and batch_id >= limit: break
                    inputs, samples = _build_data_samples(batch, device)
                    processed = model.data_preprocessor({"inputs": inputs, "data_samples": samples}, training=False)
                    state["gts"], state["metas"], state["tail"] = _gt_cache(processed["data_samples"]), batch["img_metas"], None
                    model.predict(processed["inputs"], processed["data_samples"], rescale=False)
                    if state["tail"] is None or state["boxes"] is None: continue
                    if state["cursor"] != int(state["queue"].numel()): raise RuntimeError("proposal label queue mismatch")
                    outputs, base, patches = state["tail"]
                    idx = outputs["indices"].detach().cpu(); raw = outputs["base_selected_logits"].detach().cpu()
                    for i in range(idx.shape[0]):
                        image_index = int(state["ids"][i]); entry = state["gts"][image_index]
                        gt_index = _match_gt_index(state["boxes"][i].detach().cpu(), state["labels"][i].detach().cpu(), entry, a.match_iou)
                        if gt_index is None: continue
                        target = _target_for_final_mask_contract(head, entry[2][gt_index], state["boxes"][i].detach().cpu(), (256, 256))
                        target = target.reshape(-1).gather(0, idx[i]).bool()
                        error = raw[i].ge(0).ne(target)
                        if train:
                            packed.append(torch.cat((base[i], patches[i, :, :9]), 1)); labels.append(error.float())
                        else:
                            image_id = int(state["metas"][image_index].get("image_id", image_index))
                            row = rows.setdefault(image_id, {"image_id": image_id, "base": [], "patch": [], "sham": [], "label": [], "action": {k: 0. for k in ("all_error", "selected_error", "direction_agree", "current_reach", "sign_reach", "gate_reach", "both_reach", "cap_reach")}})
                            row["base"].append(base[i]); row["patch"].append(patches[i, :, :9]); row["sham"].append(patches[i, :, 9:]); row["label"].append(error)
                            target_f = target.float(); s = target_f.mul(2.).sub(1.)
                            raw_delta = outputs["raw_delta"][i].detach().cpu(); gate = outputs["gate"][i].detach().cpu()
                            margin = (-s * raw[i]).clamp_min(0.)
                            full_target = _target_for_final_mask_contract(head, entry[2][gt_index], state["boxes"][i].detach().cpu(), (256, 256)).bool()
                            full_base = outputs["base_logits_detached"][i].detach().cpu()
                            all_error = full_base.ge(0).ne(full_target)
                            action = row["action"]
                            action["all_error"] += float(all_error.sum())
                            action["selected_error"] += float(error.sum())
                            action["direction_agree"] += float((error & (raw_delta * s > 0)).sum())
                            action["current_reach"] += float((error & ((raw[i] + gate * raw_delta) * s >= 0)).sum())
                            action["sign_reach"] += float((error & ((raw[i] + gate * s * raw_delta.abs()) * s >= 0)).sum())
                            action["gate_reach"] += float((error & ((raw[i] + raw_delta) * s >= 0)).sum())
                            action["both_reach"] += float((error & ((raw[i] + s * raw_delta.abs()) * s >= 0)).sum())
                            action["cap_reach"] += float((error & (margin <= head.decoder_tail_refiner.delta_logit_max)).sum())
        finally:
            hook.remove(); model.roi_head.predict_mask = original_predict; head.decoder_tail_refiner.forward = original_tail
        if train:
            return torch.cat(packed), torch.cat(labels)
        if readouts is None: raise RuntimeError("validation requires fitted readouts")
        finalized: List[Dict[str, Any]] = []
        for row in rows.values():
            base, patch, sham, y = (torch.cat(row[k]) for k in ("base", "patch", "sham", "label"))
            i0 = _score(base.to(device), readouts["i0"]).cpu().numpy()
            i1 = _score(torch.cat((base, patch), 1).to(device), readouts["i1"]).cpu().numpy()
            # Same aligned-patch readout, but only its patch channels are spatially displaced.
            s = _score(torch.cat((base, sham), 1).to(device), readouts["i1"]).cpu().numpy()
            # z is retained only for the fixed uncertainty control, not as an
            # extra learned feature or label.
            finalized.append({"image_id": row["image_id"], "i0": i0.tolist(), "i1": i1.tolist(), "sham": s.tolist(), "base_z": base[:, -4].numpy().tolist(), "label": y.numpy().astype(np.uint8).tolist(), "action": row["action"]})
        return finalized

    x, y = collect(train_contract, True)
    ids0, ids1 = (y == 0).nonzero().flatten(), (y == 1).nonzero().flatten()
    if not len(ids0) or not len(ids1): raise RuntimeError("both error classes are required")
    g = torch.Generator().manual_seed(a.seed)
    keep = torch.cat((ids0[torch.randperm(len(ids0), generator=g)[:a.max_train_per_class]], ids1[torch.randperm(len(ids1), generator=g)[:a.max_train_per_class]]))
    keep = keep[torch.randperm(len(keep), generator=g)]; x, y = x[keep].to(device), y[keep].to(device)
    base_dim = x.shape[1] - 9
    readouts = {"i0": _fit_linear(x[:, :base_dim], y, a), "i1": _fit_linear(x, y, a)}
    rows = collect(val_contract, False, readouts)
    if not rows: raise RuntimeError("no held-out matched proposal records")
    boot = _bootstrap(rows, a)
    action = _action_summary(rows, a)
    selectivity = _write_selectivity(rows, a)
    summary = {"checkpoint": str(ckpt_path), "strict_load": strict, "config_source": config_source,
               "train_contract": train_contract, "validation_contract": val_contract,
               "feature_contract": {"i0": "exact tail point input [upscaled32, mask_token256, z,x,y,abs(z)]", "i1": "i0 + detached aligned 3x3 native-logit patch", "sham": "same i1 readout with per-ROI half-grid shifted patch"},
               "label": "pre-tail native hard-label error on matched positive proposal; GT is never supplied to frozen forward", "fit_samples": int(len(y)), "fit_error_fraction_balanced": float(y.mean()), "validation_images": len(rows), "bootstrap": boot, "gate": _verdict(boot), "write_selectivity": selectivity, "action_decomposition": action,
               "tail_capture": "zero-init attached to tail-off checkpoint; output identity verified by the companion equivalence guard" if a.attach_zero_tail else "checkpoint-native tail"}
    _write_json(out / "summary.json", summary); _write_json(out / "image_records.json", rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
