#!/usr/bin/env python3
"""Offline train-520 -> val-130 pre-PromptEncoder gate readability verdict.

Inputs are produced by ``canvas_accuracy_probe.py --export-pre-prompt-features``
for learned/current and learned/forced-alpha=1.  This program never constructs
a model: GT is read only to create train Oracle labels and report held-out AUC.
The deployed selector uses the readout score alone for *all* val proposals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from inference.probes.dense_gate_readability import binary_auc, fit_balanced_linear, row_permute
from inference.probes.dense_gate_selector_oracle import _mixed_records, make_selectors


def _load(directory: Path):
    dt = json.loads((directory / "dt_records.json").read_text())
    gt = json.loads((directory / "gt_records.json").read_text())
    images = json.loads((directory / "images.json").read_text())
    manifest = json.loads((directory / "run_manifest.json").read_text())
    path = directory / "pre_prompt_features.npz"
    if not path.is_file():
        raise SystemExit(f"missing {path}; rerun collector with --export-pre-prompt-features")
    values = np.load(path)
    ids, scalar, full = values["image_ids"], values["scalar"], values["full"]
    if not (len(dt) == len(ids) == len(scalar) == len(full)):
        raise SystemExit(f"feature/DT row count mismatch in {directory}")
    if not np.array_equal(ids, np.asarray([int(x["image_id"]) for x in dt])):
        raise SystemExit(f"feature/DT image-id order mismatch in {directory}")
    if not (manifest.get("dataset") or {}).get("num_classes"):
        raise SystemExit(f"manifest lacks explicit dataset class contract: {directory}")
    return dt, gt, images, manifest, torch.from_numpy(scalar).float(), torch.from_numpy(full).float()


def _labels(current, forced, gt, match_iou, random_trials=1):
    oracle, _random, audit = make_selectors(current, forced, gt, match_iou, random_trials, 44)
    matched = torch.tensor([bool(row["matched"]) for row in audit])
    labels = torch.tensor([bool(row.get("choose_forced", False)) for row in audit], dtype=torch.float32)
    return matched, labels, oracle


def _write(out, name, current_dir, dt, forced, selected, extra):
    arm = out / name; arm.mkdir(parents=True, exist_ok=False)
    for filename in ("gt_records.json", "images.json"):
        (arm / filename).write_text((current_dir / filename).read_text())
    (arm / "dt_records.json").write_text(json.dumps(_mixed_records(dt, forced, selected)) + "\n")
    manifest = json.loads((current_dir / "run_manifest.json").read_text())
    manifest.update(dict(selector=name, selected_detections=len(selected), **extra))
    (arm / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-current", required=True); p.add_argument("--train-forced", required=True)
    p.add_argument("--val-current", required=True); p.add_argument("--val-forced", required=True)
    p.add_argument("--output-dir", required=True); p.add_argument("--match-iou", type=float, default=.5)
    p.add_argument("--steps", type=int, default=400); p.add_argument("--lr", type=float, default=.03)
    p.add_argument("--threshold", type=float, default=.5); p.add_argument("--seed", type=int, default=44)
    a = p.parse_args()
    if not 0 < a.threshold < 1 or not 0 < a.match_iou <= 1: raise SystemExit("invalid threshold/match-iou")
    tc, tf, vc, vf, out = (Path(x).resolve() for x in (a.train_current, a.train_forced, a.val_current, a.val_forced, a.output_dir))
    if out.exists(): raise SystemExit(f"output exists: {out}")
    tdt, tgt, _ti, tm, ts, tx = _load(tc); tfdt, tfgt, _tfi, tfm, _tfs, _tfx = _load(tf)
    vdt, vgt, _vi, vm, vs, vx = _load(vc); vfdt, vfgt, _vfi, vfm, _vfs, _vfx = _load(vf)
    for name, left, right in (("train", tm, tfm), ("val", vm, vfm)):
        if left.get("checkpoint") != right.get("checkpoint") or left.get("processed_image_ids") != right.get("processed_image_ids"):
            raise SystemExit(f"{name} current/forced checkpoint or image identities differ")
    tm_match, ty, _to = _labels(tdt, tfdt, tgt, a.match_iou)
    vm_match, vy, _vo = _labels(vdt, vfdt, vgt, a.match_iou)
    if not bool(tm_match.any()) or not bool(vm_match.any()): raise SystemExit("no matched Oracle labels")
    scalar_dim = ts.shape[1]
    variants = {
        "raw": (ts, vs),
        "full": (tx, vx),
        "permuted": (torch.cat((ts, row_permute(tx[:, scalar_dim:], seed=a.seed)), 1),
                      torch.cat((vs, row_permute(vx[:, scalar_dim:], seed=a.seed + 1)), 1)),
    }
    out.mkdir(parents=True)
    summary = dict(protocol="fit train-only Oracle labels; val threshold fixed before labels; all val proposals selectable", threshold=a.threshold, match_iou=a.match_iou, train=dict(matched=int(tm_match.sum()), total=len(tm_match)), val=dict(matched=int(vm_match.sum()), total=len(vm_match)), arms={})
    for name, (train_x, val_x) in variants.items():
        model = fit_balanced_linear(train_x[tm_match], ty[tm_match], steps=a.steps, lr=a.lr, seed=a.seed)
        score = model.score(val_x); selected = set(torch.where(score >= a.threshold)[0].tolist())
        _write(out, name, vc, vdt, vfdt, selected, dict(readout_feature_set=name, train_only=True, threshold=a.threshold))
        summary["arms"][name] = dict(auc_matched=float(binary_auc(score[vm_match], vy[vm_match])), selected=len(selected), selected_fraction=len(selected) / max(len(score), 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
