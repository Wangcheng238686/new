#!/usr/bin/env python
"""Step-0 bitwise preflight for the densecap64 stage-1 adapter (design doc
dense_capacity_stage1_design.md, launch gate #4).

Builds TWO models from the SAME real A0 E300 checkpoint:
  A: plain A0, from the checkpoint's self-describing config;
  B: densecap64 arm env (DENSE_CAPACITY_HEAD_ENABLED=1), warm-started from the
     same checkpoint under the heat-start whitelist (only dense_capacity_head.*
     keys may be missing; zero unexpected; zero shape mismatch).

Runs identical real validation batches through both in eval mode and asserts
bitwise identity of: detector fields (bboxes/scores/labels), PE point slots
(coords/labels), the delivered dense canvas (PE masks input), and the final
predicted masks.  Additionally asserts the adapter residual is exactly zero
at step 0.

Exit 0 = gate PASS; nonzero with a field diff = FAIL.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import os
import random
import subprocess
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
    _resolve_sam2_repo,
    _select_state_dict,
    _snapshot,
)

# This script already lives inside the mainline repository.
REPO = MAINLINE_ROOT


def _set_determinism() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _tensor_hash(t) -> str:
    a = t.detach().cpu().contiguous()
    return hashlib.sha256(np.ascontiguousarray(a.numpy()).tobytes()).hexdigest()[:16]


def _arm_env(arm: str) -> dict:
    out = subprocess.run(
        ["bash", str(REPO / "scripts/ablations/vhr10_p2v2_dev.sh"), arm],
        env={**os.environ, "DEV_DUMP_ENV": "1", "P2V2_PROTOCOL": "dev100"},
        capture_output=True, text=True, check=True,
    ).stdout
    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(
        "/data/wangcheng/checkpoint/portable_sam2_explicit_coarse/ablations/"
        "vhr10_p2v2_matrix300_a0_pbm_d5b_tr1.0_va1.0/last_model_epoch300.pth"))
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    _set_determinism()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = _load_checkpoint(ckpt_path)
    snapshot = _snapshot(checkpoint)
    ns = SimpleNamespace(
        checkpoint=str(ckpt_path), config=None, split="validation",
        data_root=None, ann_file=None, image_subdir=None, image_size=None,
        batch_size=None, sam2_repo=None, sam2_ckpt=None,
    )
    ns.sam2_repo = _resolve_sam2_repo(ns, snapshot)

    # ---- Model A: plain A0 from the self-describing config ----
    model_a_cfg, _ = _resolve_model_config(ns, snapshot)
    model_a = _register_and_build(model_a_cfg)
    rep = _load_model_state(
        model_a, _select_state_dict(checkpoint, "model"), allow_nonstrict=False
    )
    if rep["missing_keys"] or rep["unexpected_keys"]:
        print(f"[FAIL] plain A0 strict load: {rep}")
        return 1
    model_a.to(device).eval()

    # ---- Model B: densecap64 arm env, heat-start under whitelist ----
    arm_env = _arm_env("densecap64")
    saved_env = {k: os.environ.get(k) for k in arm_env}
    os.environ.update(arm_env)
    from mmengine.config import Config
    model_b_cfg = Config.fromfile(
        str(REPO / "configs/vhr10_baseplus_explicit_coarse.py")
    ).model
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    model_b = _register_and_build(dict(model_b_cfg))
    rep_b = _load_model_state(
        model_b, _select_state_dict(checkpoint, "model"), allow_nonstrict=True
    )
    missing = rep_b["missing_keys"]
    unexpected = rep_b["unexpected_keys"]
    adapter_prefix = "roi_head.mask_head.dense_capacity_head."
    expected_missing = sorted(
        key for key in model_b.state_dict() if key.startswith(adapter_prefix)
    )
    bad_missing = [k for k in missing if not k.startswith(adapter_prefix)]
    print(f"[heat-start] missing={len(missing)} (all dense_capacity_head.*: "
          f"{not bad_missing}) unexpected={len(unexpected)}")
    if bad_missing or unexpected or sorted(missing) != expected_missing:
        print(f"[FAIL] whitelist violated: bad_missing={bad_missing[:5]} "
              f"unexpected={unexpected[:5]} expected_missing={len(expected_missing)}")
        return 1
    model_b.to(device).eval()

    adapter = model_b.roi_head.mask_head.dense_capacity_head
    if adapter is None:
        print("[FAIL] dense_capacity_head is None in arm build")
        return 1

    # ---- capture hooks on both models ----
    def make_capture(model):
        cap = {"points": [], "masks": []}
        head = model.roi_head.mask_head
        orig_transform = head._shape_prior_to_prompt_mask

        def wrap_transform(roi_mask_logits, boxes):
            out = original = orig_transform(roi_mask_logits, boxes)
            cap["masks"].append(out[0].detach().cpu())
            return out

        def pe_pre(module, fn_args, kwargs):
            pts = kwargs.get("points")
            if pts is not None and pts[0] is not None:
                cap["points"].append(
                    (pts[0].detach().cpu().clone(), pts[1].detach().cpu().clone())
                )
            return fn_args, kwargs

        head._shape_prior_to_prompt_mask = wrap_transform
        handle = head.prompt_encoder.register_forward_pre_hook(
            pe_pre, with_kwargs=True
        )
        return cap, head, orig_transform, handle

    cap_a, head_a, orig_a, h_a = make_capture(model_a)
    cap_b, head_b, orig_b, h_b = make_capture(model_b)

    delta_vals = []
    def adapter_hook(_module, _inputs, output):
        delta_vals.append(output.detach().cpu().clone())

    h_delta = adapter.register_forward_hook(adapter_hook)

    contract = _resolve_dataset_contract(ns, snapshot)
    loader = _build_loader(contract, 0)

    fields = {"bboxes": [], "scores": [], "labels": [], "masks": []}
    fields_b = {"bboxes": [], "scores": [], "labels": [], "masks": []}
    try:
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                if index >= args.batches:
                    break
                imgs, data_samples = _build_data_samples(batch, device)
                pre = model_a.data_preprocessor(
                    {"inputs": imgs, "data_samples": data_samples}, training=False
                )
                imgs1 = pre["inputs"].clone()
                ds1 = copy.deepcopy(pre["data_samples"])
                out_a = model_a.predict(imgs1, ds1, rescale=False)
                out_b = model_b.predict(pre["inputs"], pre["data_samples"], rescale=False)
                for oa, ob in zip(out_a, out_b):
                    pa, pb = oa.pred_instances, ob.pred_instances
                    fields["bboxes"].append(pa.bboxes.detach().cpu())
                    fields_b["bboxes"].append(pb.bboxes.detach().cpu())
                    fields["scores"].append(pa.scores.detach().cpu())
                    fields_b["scores"].append(pb.scores.detach().cpu())
                    fields["labels"].append(pa.labels.detach().cpu())
                    fields_b["labels"].append(pb.labels.detach().cpu())
                    fields["masks"].append(pa.masks.detach().cpu())
                    fields_b["masks"].append(pb.masks.detach().cpu())
    finally:
        head_a._shape_prior_to_prompt_mask = orig_a
        head_b._shape_prior_to_prompt_mask = orig_b
        h_a.remove()
        h_b.remove()
        h_delta.remove()

    delta_zero = bool(delta_vals) and all(torch.count_nonzero(delta) == 0 for delta in delta_vals)
    print(f"[{'PASS' if delta_zero else 'FAIL'}] adapter residual on real RoIs: "
          f"{len(delta_vals)} captures")
    if not delta_zero:
        return 1

    ok = True
    for key in ("bboxes", "scores", "labels", "masks"):
        same = len(fields[key]) == len(fields_b[key]) and all(
            torch.equal(x, y) for x, y in zip(fields[key], fields_b[key])
        )
        n = len(fields[key])
        ha = _tensor_hash(torch.cat([
            f.reshape(-1) if f.dim() == 1 else f.flatten()
            for f in fields[key][:1]
        ])) if n else "-"
        print(f"[{'PASS' if same else 'FAIL'}] predictions.{key}: "
              f"{n} samples, first-hash={ha}")
        ok = ok and same

    n_pts = len(cap_a["points"])
    pts_same = n_pts > 0 and n_pts == len(cap_b["points"]) and all(
        torch.equal(cap_a["points"][i][0], cap_b["points"][i][0])
        and torch.equal(cap_a["points"][i][1], cap_b["points"][i][1])
        for i in range(n_pts)
    )
    print(f"[{'PASS' if pts_same else 'FAIL'}] PE point slots: {n_pts} captures")
    ok = ok and pts_same

    n_mask = len(cap_a["masks"])
    canvas_same = n_mask > 0 and n_mask == len(cap_b["masks"]) and all(
        torch.equal(cap_a["masks"][i], cap_b["masks"][i])
        for i in range(n_mask)
    )
    if n_mask:
        print(f"[{'PASS' if canvas_same else 'FAIL'}] dense canvas: {n_mask} "
              f"captures, hash_a={_tensor_hash(cap_a['masks'][0])}")
    else:
        print("[FAIL] dense canvas: no captures")
        canvas_same = False
    ok = ok and canvas_same

    print("=" * 60)
    print("STEP-0 PREFLIGHT:", "PASS (bitwise)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
