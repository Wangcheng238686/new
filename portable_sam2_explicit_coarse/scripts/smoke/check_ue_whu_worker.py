#!/usr/bin/env python3
"""Isolated source/candidate numerical gate using the same temporary weights.

Run in separate processes with --package-root pointing at the source or the
migration package. Outputs are exact tensor digests, not performance metrics.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


def seed():
    random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    torch.cuda.manual_seed_all(44)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def digest(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    if isinstance(value, (list, tuple)):
        return [digest(item) for item in value]
    if isinstance(value, dict):
        return {key: digest(item) for key, item in value.items()}
    if value is None:
        return None
    raise TypeError(type(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--sam2-ckpt", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--audit-old", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root))
    sys.path.insert(0, args.sam2_repo)
    os.environ["SAM2_REPO"] = args.sam2_repo
    os.environ["SAM2_CKPT"] = args.sam2_ckpt
    from mmengine.config import Config
    from inference.model_rebuild import (
        _register_and_build, _load_model_state, _resolve_model_config,
        _targeted_sam2_checkpoint_override,
    )
    from rsprompter.architecture_contract import architecture_contract, file_sha256
    from types import SimpleNamespace

    seed()
    config = Config.fromfile(str(args.package_root / "configs/a3sgm_whu1024.py")).model.to_dict()
    _targeted_sam2_checkpoint_override(config, args.sam2_ckpt)
    expected_contract = architecture_contract(config)
    if args.initialize:
        from ue.contract import baseline_metadata, raw_data_contract
        model = _register_and_build(copy.deepcopy(config))
        snapshot = {
            "architecture_contract": expected_contract, "model_config": config,
            "ue_baseline": baseline_metadata(config, validation_only=True),
            "data_config": raw_data_contract(str(args.package_root / "data/ue_raw_hero_uav_pairset_20261011")),
            "runtime_config": {"sam2_repo": args.sam2_repo,
                               "sam2_checkpoint_sha256": file_sha256(args.sam2_ckpt)},
            "training_args": {"seed": 44, "val_batch_size": 1, "run_tag": "MIGRATION_SMOKE_ONLY"},
        }
        torch.save({"epoch": -1, "model": model.state_dict(), "config_snapshot": snapshot}, args.fixture)
        report = {"status": "temporary_fixture_only", "architecture_contract": expected_contract}
    else:
        checkpoint = torch.load(args.fixture, map_location="cpu", weights_only=False)
        rebuilt, _ = _resolve_model_config(SimpleNamespace(config=None, sam2_ckpt=args.sam2_ckpt), checkpoint["config_snapshot"])
        model = _register_and_build(rebuilt)
        report = {"architecture_contract": expected_contract,
                  "strict_load": _load_model_state(model, checkpoint["model"], False)}
        if args.audit_old:
            old = torch.load(args.audit_old, map_location="cpu", weights_only=False)
            state, wanted = old["model"], model.state_dict()
            report["old_architecture_contract"] = old["config_snapshot"]["architecture_contract"]
            report["missing_keys"] = sorted(set(wanted) - set(state))
            report["unexpected_keys"] = sorted(set(state) - set(wanted))
            report["shape_mismatch"] = [key for key in wanted.keys() & state.keys() if wanted[key].shape != state[key].shape]
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                report["old_strict_load"] = "REJECTED"
                report["reason"] = str(error)
            else:
                raise RuntimeError("old B0 unexpectedly loaded; audit needs numerical follow-up")
        else:
            from inference.eval_records import _build_data_samples
            model = model.to(args.device).eval()
            batch = torch.load(args.batch, map_location="cpu", weights_only=False)

            def inputs(training):
                images, samples = _build_data_samples(batch, torch.device(args.device))
                return model.data_preprocessor({"inputs": images, "data_samples": samples}, training=training)

            seed()
            with torch.inference_mode():
                data = inputs(False)
                report["preprocessed_inputs"] = digest(data["inputs"])
                features = model.extract_feat(data["inputs"])
                assert tuple(features[0][1].shape[1:]) == (512, 128, 128)
                report["features"] = digest(features)
                outputs = model.predict(data["inputs"], data["data_samples"], rescale=False)
                report["predictions"] = {key: digest(getattr(outputs[0].pred_instances, key))
                                         for key in ("bboxes", "scores", "labels", "masks")}
            model.train()
            seed()
            data = inputs(True)
            losses = model.loss(data["inputs"], data["data_samples"])
            report["losses"] = digest(losses)
            parts = []
            for value in losses.values():
                if torch.is_tensor(value):
                    parts.append(value)
                elif isinstance(value, (tuple, list)):
                    parts.extend(item for item in value if torch.is_tensor(item))
            total = sum(parts)
            if not torch.isfinite(total):
                raise RuntimeError("nonfinite validation loss")
            total.backward()
            gradients = {name: digest(parameter.grad) for name, parameter in model.named_parameters()
                         if parameter.requires_grad and parameter.grad is not None}
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("nonfinite gradient: " + name)
            for group in ("backbone", "neck", "rpn_head", "roi_head.bbox_head",
                          "roi_head.mask_head.shape_injector", "roi_head.mask_head.decoder_tail_refiner"):
                if not any(name.startswith(group) for name in gradients):
                    raise RuntimeError("missing expected gradients: " + group)
            report["gradients"] = gradients
            report["total_loss"] = float(total.detach())
            model.zero_grad(set_to_none=True)
            # Serialize and strict-reload after backward without an optimizer update.
            reload_path = args.output.with_suffix(".reload.pth")
            torch.save(model.state_dict(), reload_path)
            model.load_state_dict(torch.load(reload_path, map_location=args.device, weights_only=True), strict=True)
            reload_path.unlink()
            report["save_reload"] = "PASS"
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
