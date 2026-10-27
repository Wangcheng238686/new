#!/usr/bin/env python3
"""Fail-closed pre-launch harness for UBR-RESEAL-20261027 (prereg checks).

Per docs/UBR-RESEAL-20261027_preregistration.md, run BEFORE the full 100ep
launch (and before/after smoke). Every check below is a launch gate; the
harness writes a PASS/FAIL json and exits non-zero on any FAIL.

Checks:
  1. config_diff_exact   resolved diff of configs/ablations/ue_ubr_single_512.py
                         vs the sealed configs/ue_whu_a3sgm_512.py is EXACTLY
                         {decoder_tail_refiner_cfg emptied, ubr_integration_cfg
                         added} — nothing else changed.
  2. old_seal_untouched  SHA256 of the frozen_refs copies (SHA256SUMS of
                         FREEZE-REF-20261026), of EXP-0025 best_model
                         (ep72, seal fingerprint 2fcae2ff...) and of the
                         old-last anchor last_model_epoch100.pth.
  3. dataset_sha         the 7 dataset pins of the EXP-0025 wrapper.
  4. model_build_ctor    the level-3 model builds with the UBR class patch;
                         every configured checkpoint is the SAM2 ctor
                         checkpoint (no task weights); the frozen no-mask
                         embedding contract holds (ctor weights loaded);
                         UBR is a registered submodule, 163,537 params, all
                         trainable, zero-init output layer; NO
                         decoder_tail_refiner keys in state_dict.
  5. fwd_bwd_finite      one real train batch through model loss: loss dict
                         contains loss_ubr and NOT loss_decoder_tail; every
                         loss finite; backward finite; UBR+decoder grads
                         non-None and finite (gradient path 1-2 live).
  6. predict_refine      predict on a real val batch: paste parity holds;
                         zero-init UBR changes 0 pixels (neutral start);
                         with the UBR output biased, refinement changes a
                         nonzero number of pixels, all inside the band
                         (asserted fail-closed inside the integration), and
                         changed_px <= band_px.
  7. launch_preconds     the full-run output dir does NOT exist; protocol
                         constants in the runner match the EXP-0025 seal.

Run with CUDA_VISIBLE_DEVICES=3 (small memory; ~1 batch).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # portable_sam2_explicit_coarse (repo root)
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

CONFIG_NEW = ROOT / "configs/ablations/ue_ubr_single_512.py"
CONFIG_SEALED = ROOT / "configs/ue_whu_a3sgm_512.py"
SAM2_CKPT = "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"
FROZEN_REFS = Path("/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_probe_20261026/frozen_refs")
EXP25_DIR = Path("/data2/wangcheng/checkpoint/citysample_v1/"
                 "citysample_v1_512single_defaug_ddp4_s44_EXP-0025")
OUT_ROOT = Path("/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_20261027")
REPORT_PATH = OUT_ROOT.parent / "ue_ubr_reseal_20261027_harness_report.json"

OLD_BEST_SHA = "2fcae2ff082fdb2e4eb5397930fe8b240b13ffb7e586b44af646ce9f245724eb"
OLD_LAST_SHA = "736c1532755718ed678d61eb6ef3c900a3db2377873f2a7bb2c987bbdda373c9"
OLD_LAST_NAME = "last_model_epoch100.pth"

# expected exact config diff (flattened leaf paths)
EXPECTED_ADDED = sorted(
    [f"/roi_head/mask_head/ubr_integration_cfg/{k}" for k in (
        "enabled", "band_k", "loss_weight", "window_margin",
        "focal_gamma", "uav_port", "neighbors_train",
        "neighbors_predict")]
    + ["/roi_head/mask_head/point_warmup_cfg/_delete_",
       "/roi_head/mask_head/box_prompt_cfg/_delete_"]
    + ["/roi_head/mask_head/ubr_integration_cfg/aug_gaussian",
       "/roi_head/mask_head/ubr_integration_cfg/aug_jitter"])
EXPECTED_REMOVED = sorted(
    f"/roi_head/mask_head/decoder_tail_refiner_cfg/{k}" for k in (
        "enabled", "num_points", "hidden_dim", "point_loss_weight",
        "delta_logit_max", "mode", "crossing_margin", "boundary_logit"))


def _spec(relpath):
    spec = importlib.util.spec_from_file_location(
        relpath.stem, relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


class Harness:
    def __init__(self):
        self.results = []
        self.timings = {}

    def check(self, name):
        def deco(fn):
            t0 = time.time()
            try:
                detail = fn() or {}
                self.results.append(dict(name=name, status="PASS",
                                         detail=detail,
                                         secs=round(time.time() - t0, 1)))
                print(f"[PASS] {name} ({time.time() - t0:.1f}s) {detail}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 — record and continue
                self.results.append(
                    dict(name=name, status="FAIL",
                         error=f"{type(exc).__name__}: {exc}",
                         trace=traceback.format_exc()[-4000:],
                         secs=round(time.time() - t0, 1)))
                print(f"[FAIL] {name}: {exc}", flush=True)
        return deco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(REPORT_PATH))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    args = ap.parse_args()
    out_root = Path(args.out_root)

    h = Harness()
    os.environ.setdefault("SAM2_CKPT", SAM2_CKPT)  # required by both configs
    # replicate the sealed trainer's registry bootstrap (its main() does
    # the same DetDataPreprocessor registration)
    import mmdet.models  # noqa: F401
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor)
    runner = _spec(HERE / "ue_ubr_reseal_20261027.py")

    @h.check("config_diff_exact")
    def _():  # noqa: E306
        from mmengine.config import Config
        sealed = _flatten(Config.fromfile(str(CONFIG_SEALED)).model)
        new = _flatten(Config.fromfile(str(CONFIG_NEW)).model)
        added = sorted(set(new) - set(sealed))
        removed = sorted(set(sealed) - set(new))
        changed = sorted(k for k in set(sealed) & set(new)
                         if sealed[k] != new[k])
        assert added == EXPECTED_ADDED, f"added {added}"
        assert removed == EXPECTED_REMOVED, f"removed {removed}"
        assert not changed, f"changed {changed}"
        return dict(added=len(added), removed=len(removed), changed=0)

    @h.check("old_seal_untouched")
    def _():  # noqa: E306
        sums = (FROZEN_REFS / "SHA256SUMS").read_text().strip().splitlines()
        checked = []
        for line in sums:
            sha, path = line.split()
            got = runner.sha256_of(Path(path))
            assert got == sha, f"frozen ref drift {path}"
            checked.append(Path(path).name)
        best = EXP25_DIR / "best_model_epoch72.pth"
        assert runner.sha256_of(best) == OLD_BEST_SHA, "old best (seal fingerprint) drifted"
        last = EXP25_DIR / OLD_LAST_NAME
        assert runner.sha256_of(last) == OLD_LAST_SHA, "old last anchor drifted"
        return dict(frozen_refs=checked, exp25_best=OLD_BEST_SHA[:8],
                    exp25_last=OLD_LAST_SHA[:8])

    @h.check("dataset_sha")
    def _():  # noqa: E306
        info = runner.assert_dataset(Path(runner.DS_ROOT_DEFAULT))
        return dict(root=runner.DS_ROOT_DEFAULT,
                    label_revision=info["label_revision"],
                    pins=len(runner.DATASET_PINS))

    @h.check("launch_preconds")
    def _():  # noqa: E306
        assert not out_root.exists(), f"output dir exists: {out_root}"
        p = runner.PROTOCOL
        sealed = dict(seed=44, batch=1, accum=2, epochs=100, lr=5e-4,
                      wd=0.05, warmup=100, flip=0.5, vflip=0.0, ms=0.0)
        assert p["seed"] == sealed["seed"] and p["batch_size"] == sealed["batch"]
        assert p["grad_accum_steps"] == sealed["accum"]
        assert p["epochs"] == sealed["epochs"] and p["lr"] == sealed["lr"]
        assert p["weight_decay"] == sealed["wd"] and p["warmup_iters"] == sealed["warmup"]
        assert p["flip_prob"] == sealed["flip"] and p["vflip_prob"] == sealed["vflip"]
        assert p["ms_resize_prob"] == sealed["ms"]
        return dict(out_root_absent=True, protocol=sealed)

    # ---------- GPU checks (build + one train batch + one val batch) ------
    build = {}

    @h.check("model_build_ctor")
    def _():  # noqa: E306
        t0 = time.time()
        from mmdet.registry import MODELS
        import rsprompter  # noqa: F401  (registers model types)
        from mmengine.config import Config
        from rsprompter.uav import ubr_train_integration as integ

        cfg = Config.fromfile(str(CONFIG_NEW))
        integ.install_class_patch(cfg.model.roi_head.mask_head.ubr_integration_cfg)
        model = MODELS.build(cfg.model)
        model = model.to("cuda")
        build["model"] = model
        build["integration"] = model.ubr_integration
        h.timings["build_secs"] = round(time.time() - t0, 1)

        # ctor-only loading: every configured checkpoint is the SAM2 ctor ckpt
        cfg_ckpts = set()

        def _collect(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ("checkpoint", "checkpoint_path", "ckpt_path") and isinstance(v, str):
                        cfg_ckpts.add(v)
                    else:
                        _collect(v)
        _collect(cfg.model.to_dict())
        assert cfg_ckpts == {SAM2_CKPT}, f"ctor ckpts {cfg_ckpts}"

        # frozen no-mask embedding contract proves SAM2 ctor weights loaded
        model.roi_head.mask_head.assert_no_mask_embedding_contract("harness")

        mh = model.roi_head.mask_head
        assert not mh.decoder_tail_enabled and mh.decoder_tail_refiner is None

        sd = model.state_dict()
        tail_keys = [k for k in sd if "decoder_tail_refiner" in k]
        assert not tail_keys, f"UDPR keys present: {tail_keys[:5]}"
        ubr_keys = [k for k in sd if k.startswith("ubr_refiner.")]
        assert len(ubr_keys) > 0 and all(
            torch.isfinite(sd[k]).all() for k in ubr_keys)

        n_par = sum(p.numel() for p in model.ubr_refiner.parameters())
        assert n_par == 163537, n_par
        assert all(p.requires_grad for p in model.ubr_refiner.parameters())
        out_w = model.ubr_refiner.out.weight
        out_b = model.ubr_refiner.out.bias
        assert float(out_w.abs().max()) == 0.0 and float(out_b.abs().max()) == 0.0, (
            "UBR output layer must be zero-init (neutral start)")
        return dict(params=n_par, ubr_state_keys=len(ubr_keys),
                    udpr_keys=0, build_secs=h.timings["build_secs"])

    @h.check("fwd_bwd_finite")
    def _():  # noqa: E306
        from data.loader import create_train_loader
        from train.gt_samples import _build_gt_data_samples
        model = build["model"]
        os.environ.update(WHU_TRAIN_ANN_FILE="annotations/train.json",
                          WHU_VAL_ANN_FILE="annotations/val.json",
                          WHU_TRAIN_IMG_SUBDIR="", WHU_VAL_IMG_SUBDIR="")

        def _loader(train_ratio, val_ratio):
            return create_train_loader(
                data_root=runner.DS_ROOT_DEFAULT, batch_size=1,
                dataset_format="whu_coco", image_size=(512, 512),
                num_workers=0, seed=44, val_batch_size=1,
                whu_train_ann_file=os.environ["WHU_TRAIN_ANN_FILE"],
                whu_val_ann_file=os.environ["WHU_VAL_ANN_FILE"],
                whu_train_img_subdir=os.environ["WHU_TRAIN_IMG_SUBDIR"],
                whu_val_img_subdir=os.environ["WHU_VAL_IMG_SUBDIR"],
                train_subset_ratio=train_ratio, val_subset_ratio=val_ratio)
        train_loader, _, _ = _loader(runner.subset_ratio(2142, 6),
                                     runner.subset_ratio(459, 3))
        batch = next(iter(train_loader))
        t0 = time.time()
        device = "cuda"
        data_samples = _build_gt_data_samples(
            batch["imgs"], batch["img_metas"], batch["gt_bboxes"],
            batch["gt_labels"], batch["gt_masks"], device)
        payload = model.data_preprocessor(
            {"inputs": batch["imgs"].to(device), "data_samples": data_samples},
            training=True)
        with torch.autocast("cuda"):
            losses = model(payload["inputs"], payload["data_samples"],
                           mode="loss")
        assert "loss_ubr" in losses, f"loss_ubr missing: {sorted(losses)}"
        assert "loss_decoder_tail" not in losses, "loss_decoder_tail must be absent"

        def _tensors(d):
            for v in d.values():
                vals = v if isinstance(v, (list, tuple)) else [v]
                for item in vals:
                    if torch.is_tensor(item):
                        yield item
        for t in _tensors(losses):
            assert torch.isfinite(t).all(), "non-finite loss component"
        total = sum(_tensors(losses))
        scaler = torch.amp.GradScaler("cuda")
        scaler.scale(total).backward()
        # Zero-init output layer: at step 0 only ubr_refiner.out.* carries
        # gradient (upstream blocks see a zero-weight readout — the same
        # neutral-start property the level-2 trainer had).
        grads = dict(zip(
            ("out.weight", "out.bias"),
            (model.ubr_refiner.out.weight.grad, model.ubr_refiner.out.bias.grad)))
        for name, g in grads.items():
            assert g is not None and torch.isfinite(g).all(), name
            assert float(g.abs().sum()) > 0, f"{name} got no gradient"
        # Full-graph grad finiteness is asserted on an fp32 pass: a fresh
        # GradScaler's first steps can overflow fp16 backbone grads (the
        # production trainer's scaler skip-steps and lowers the scale —
        # normal AMP behavior, not a launch-blocker).
        model.zero_grad(set_to_none=True)
        losses32 = model(payload["inputs"], payload["data_samples"],
                         mode="loss")
        sum(_tensors(losses32)).backward()
        h.timings["fwd_bwd_secs"] = round(time.time() - t0, 1)
        for name, p in list(model.named_parameters()):
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad {name}"
        # Prereg gradient path 1-2 proof: with a nonzero readout, loss_ubr
        # alone must reach the decoder through the tapped decoder feat and
        # pasted z' (both are supposed to be differentiable inputs).
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.ubr_refiner.out.weight.normal_(0.0, 1e-3)
        losses3 = model(payload["inputs"], payload["data_samples"],
                        mode="loss")
        losses3["loss_ubr"].backward()
        dec_grads = [p.grad for n, p in model.named_parameters()
                     if "mask_decoder.decoder.output_upscaling" in n
                     and p.grad is not None and float(p.grad.abs().sum()) > 0]
        assert dec_grads, (
            "loss_ubr gradient does not reach the SAM2 decoder (taps "
            "detached?)")
        with torch.no_grad():
            model.ubr_refiner.out.weight.zero_()
        model.zero_grad(set_to_none=True)
        integ_stats = build["integration"].last_train_stats
        assert 0.0 <= integ_stats["UBR/band_nonempty_frac"] <= 1.0
        loss_detail = {}
        for k, v in losses.items():
            vals = v if isinstance(v, (list, tuple)) else [v]
            loss_detail[k] = round(sum(float(x) for x in vals
                                       if torch.is_tensor(x)), 5)
        return dict(losses=loss_detail,
                    ubr_instances=integ_stats["UBR/instances"],
                    band_nonempty_frac=round(
                        integ_stats["UBR/band_nonempty_frac"], 4),
                    dec_path_tensors=len(dec_grads),
                    fwd_bwd_secs=h.timings["fwd_bwd_secs"])

    @h.check("predict_refine")
    def _():  # noqa: E306
        from data.loader import create_train_loader
        from train.gt_samples import _build_gt_data_samples
        model = build["model"]
        integ = build["integration"]
        _, val_loader, _ = create_train_loader(
            data_root=runner.DS_ROOT_DEFAULT, batch_size=1,
            dataset_format="whu_coco", image_size=(512, 512),
            num_workers=0, seed=44, val_batch_size=1,
            whu_train_ann_file=os.environ["WHU_TRAIN_ANN_FILE"],
            whu_val_ann_file=os.environ["WHU_VAL_ANN_FILE"],
            whu_train_img_subdir=os.environ["WHU_TRAIN_IMG_SUBDIR"],
            whu_val_img_subdir=os.environ["WHU_VAL_IMG_SUBDIR"],
            train_subset_ratio=1.0, val_subset_ratio=runner.subset_ratio(459, 3))
        batch = next(iter(val_loader))
        model.eval()
        device = "cuda"
        with torch.no_grad():
            data_samples = _build_gt_data_samples(
                batch["imgs"], batch["img_metas"], batch["gt_bboxes"],
                batch["gt_labels"], batch["gt_masks"], device)
            payload = model.data_preprocessor(
                {"inputs": batch["imgs"].to(device),
                 "data_samples": data_samples}, training=False)
            t0 = time.time()
            out = model(payload["inputs"], payload["data_samples"],
                        mode="predict")
            h.timings["predict_secs"] = round(time.time() - t0, 1)
        s0 = dict(integ.last_refine_stats)
        assert s0["changed_px"] == 0.0, (
            f"zero-init UBR must be an exact copy, changed {s0}")
        assert s0["instances"] > 0, "no instances in the val batch"

        # bias the UBR decision positive -> in-band flips must appear
        with torch.no_grad():
            model.ubr_refiner.out.bias.fill_(5.0)
        with torch.no_grad():
            out = model(payload["inputs"], payload["data_samples"],
                        mode="predict")
        s1 = dict(integ.last_refine_stats)
        assert s1["changed_px"] > 0, "biased UBR changed no pixels"
        assert s1["changed_px"] <= s1["band_px"], (
            "changed pixels exceed the band (containment violated)")
        with torch.no_grad():
            model.ubr_refiner.out.bias.zero_()
        return dict(zero_init=s0, biased=s1,
                    predict_secs=h.timings["predict_secs"],
                    band_containment=True)

    passed = all(r["status"] == "PASS" for r in h.results)
    report = dict(
        tag="UBR-RESEAL-20261027-harness",
        overall="PASS" if passed else "FAIL",
        torch=torch.__version__, cuda=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        checks=h.results, timings=h.timings,
        out_root=str(out_root),
    )
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
