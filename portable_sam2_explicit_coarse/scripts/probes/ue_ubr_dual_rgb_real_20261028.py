#!/usr/bin/env python3
"""Runner for the UBR dual-stream +RGB arm (EXP-0030, 2026-10-28).

Spec: docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md. Launches the SAME sealed
production trainer entry (train/train_rsprompter_fusion.py) as EXP-0029,
with the +RGB config configs/ablations/ue_ubr_dual_rgb_512.py, after
installing the dual+RGB class patch (rsprompter/uav/
ubr_dual_rgb_integration.py - the EXP-0029 dual install extended by the
only test-positive RGB consumption shape: P0 source encoder -> per-cell
equal-weight mean -> block4 zero-init residual injection). Nothing in the
sealed trainer or model files is modified. This module REUSES the EXP-0029
runner lineage by importing it (ue_ubr_reseal_dual_20261027) - protocol,
dataset pins, trainer argv, env and smoke discipline are inherited, not
copied.

  --rgb-content real      UBR-DUAL-20261028-REAL (EXP-0030, the launch arm)
  --rgb-content constant  reserved fourth arm (ONLY after an RGB_WINS
                          verdict; nothing auto-launches it)

QUEUE DISCIPLINE (spec section 6): the full launch waits for EXP-0028-R2
or EXP-0029 to finish (the wrapper scripts/ue/run_ubr_dual_rgb_real.sh
enforces the max-2-training-processes-per-card guard; a third co-tenant
job would push val peaks toward OOM). Smoke/gates modes are co-tenant
safe (GPU3, <5 GB).

Protocol: verbatim EXP-0029 (seed 44, DDP4 b1 x accum2, lr 5e-4, wd 0.05,
warmup 100, cosine, 100 epochs, 3090-line photometric-only augmentation,
sharded val 459 every epoch, last-epoch selection, test never read).
Output roots: /data2/wangcheng/remote-lthpc/checkpoint/
  ue_ubr_dual_rgb_real_20261028/      (EXP-0030)
  ue_ubr_dual_rgb_constant_20261028/  (fourth arm, post-RGB_WINS only)

RESEAL_CONTRACT v4 (echoed fail-closed before cards are pulled): v3 dual
declarations verbatim + the rgb block (encoder ctor/lora config, P0 256 ->
project 32, anchor cache path + manifest SHA + scene count, injection site
block4, init file list = the generic SAM2 foundation only,
task_weights_loaded=False), the remote port-source SHAs, and the updated
dual - single enumeration {port 5, loss_geometry, rgb_injection}.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # portable_sam2_explicit_coarse (repo root)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import ue_ubr_reseal_dual_20261027 as base  # noqa: E402  (EXP-0029 lineage)

from rsprompter.uav import ubr_dual_rgb_integration as rgb  # noqa: E402

CONFIG_REL = "configs/ablations/ue_ubr_dual_rgb_512.py"
OUT_ROOT_DEFAULT = "/data2/wangcheng/remote-lthpc/checkpoint"
ARMS = rgb.RGB_CONTENT_MODES  # ('real', 'constant')
TAG_PREFIX = "UBR-DUAL-20261028"
# v4 source hashes: every v3 file + the +RGB wiring (spec section 3: full
# source_hashes INCLUDING the ported file and the remote source SHAs).
CONTRACT_SOURCE_FILES = tuple(base.CONTRACT_SOURCE_FILES) + (
    "rsprompter/uav/rgb_p0_encoder.py",
    "rsprompter/uav/ubr_dual_rgb_integration.py",
    "scripts/probes/ue_ubr_dual_rgb_real_20261028.py",
    "configs/ablations/ue_ubr_dual_rgb_512.py",
)
CONTRACT_V4_DECLARATION = (
    "v4 dual+RGB: v3 dual (EXP-0029) verbatim - port 5 real native512, "
    "geometry head + loss_geometry 1.0, r(s)/0.25*s/250 inference - plus "
    "the single test-positive RGB shape (remote candidate_detail +0.89pp "
    "CI): source LoRA SAM2 P0 encoder (r16/a32/dropout .05, generic "
    "foundation ctor only, optimizer backbone group), rgb_project 256->32, "
    "FP64 candidate-anchor sampling (<=8/cell, <=2/view, cache order), "
    "per-cell equal-weight mean, block4 zero-init residual injection; "
    "rgb_output zero-init => forward bit-identical to EXP-0029 at "
    "construction (gate G2); reserved rgb_content=constant control "
    "(train-only mean descriptor, same support mask)")


def anchor_manifest(cache_dir: Path) -> dict:
    """Per-file sha256 manifest of the mirrored FP64 anchor cache.

    Records (scene_count, manifest_sha256, bytes_total) into the contract
    (spec section 2/3); a name+size-matched sidecar is reused, otherwise
    the 8.5 GB cache is re-hashed (one-time, launch-cost only).
    """
    files = sorted(cache_dir.glob("*.npz"))
    assert files, f"empty anchor cache {cache_dir}"
    sidecar = cache_dir.parent / "ue_s2db_anchor_manifest_v2.jsonl"
    if sidecar.is_file():
        records = [json.loads(line) for line in
                   sidecar.read_text().splitlines() if line.strip()]
        by_name = {r["name"]: r for r in records}
        if (len(records) == len(files)
                and all(by_name.get(f.name, {}).get("bytes") == f.stat().st_size
                        for f in files)):
            return dict(cache_dir=str(cache_dir), scene_count=len(files),
                        anchor_version=rgb.ANCHOR_VERSION,
                        manifest_path=str(sidecar),
                        manifest_sha256=base.sha256_of(sidecar),
                        bytes_total=sum(f.stat().st_size for f in files))
    lines = []
    for f in files:
        rec = dict(name=f.name, bytes=f.stat().st_size,
                   sha256=base.sha256_of(f))
        lines.append(json.dumps(rec))
    sidecar.write_text("\n".join(lines) + "\n")
    return dict(cache_dir=str(cache_dir), scene_count=len(files),
                anchor_version=rgb.ANCHOR_VERSION,
                manifest_path=str(sidecar),
                manifest_sha256=base.sha256_of(sidecar),
                bytes_total=sum(f.stat().st_size for f in files))


def _source_hashes() -> dict:
    return {rel: base.sha256_of(ROOT / rel) for rel in CONTRACT_SOURCE_FILES}


def _build_contract(data_root, out_root, content, resume_from,
                    verdict) -> dict:
    """RESEAL_CONTRACT v4 (echoed before cards are pulled, fail-closed)."""
    from mmengine.config import Config
    from rsprompter.uav import ubr_dual_integration as dual

    cfg = Config.fromfile(str(ROOT / CONFIG_REL))
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    foundation = os.environ.get("SAM2_CKPT", base.SAM2_CKPT_DEFAULT)
    return dict(
        tag=f"{TAG_PREFIX}-{content.upper()}",
        contract_version=4,
        declaration=CONTRACT_V4_DECLARATION,
        config=CONFIG_REL,
        protocol=base.PROTOCOL,
        arm="real",               # EXP-0029 arm 2 (uav_port) - retained
        rgb_content=content,
        geometry=dict(            # v3 geometry block, verbatim
            head_params=dual.GEOMETRY_HEAD_PARAMS,
            loss_weight=dual.GEOMETRY_LOSS_WEIGHT,
            cache_dir=os.environ.get(
                "UBR_GEOM_CACHE", dual.GEOM_CACHE_DEFAULT),
            cache_version=dual.GEOM_VERSION,
            constants_path=os.environ.get(
                "UBR_GEOM_CONSTANTS", dual.GEOM_CONSTANTS_DEFAULT),
            constants_sha256=dual.CONSTANTS_SHA256,
            candidate_band=0.25,
            prefix_band="r(s)=0.5+0.5/(1-log s)",
            instance_cap=250,
        ),
        rgb=dict(
            source_encoder=dict(
                type="RGBP0Encoder over RSSAM2VisionEncoder(base_plus)",
                lora=rgb.RGB_LORA_CONFIG,
                foundation_file=foundation,
                foundation_sha256=base.sha256_of(Path(foundation)),
                init_files=[foundation],
                task_weights_loaded=False,
                seed_fork_rng=rgb.RGB_SOURCE_SEED,
                optimizer_group="backbone",
                optimizer_lr=base.PROTOCOL["lr"],
                optimizer_wd=base.PROTOCOL["weight_decay"]),
            p0_channels=rgb.P0_CHANNELS,
            project_dim=rgb.RGB_PROJECT_DIM,
            sampling="FP64 continuous anchors, <=8 candidates/cell, "
                     "<=2/view, cache rank order (sample_source_continuous)",
            reduce="per-cell equal-weight mean (reduce_rgb mean branch)",
            injection=dict(site="satellite Hiera trunk blocks[4] (64x64)",
                           channels=rgb.SAT_BLOCK4_DIM,
                           init="zero (weight and bias) - G2 identity"),
            anchor_cache=anchor_manifest(
                Path(os.environ.get("UBR_RGB_ANCHOR_CACHE",
                                    rgb.ANCHOR_CACHE_DEFAULT))),
            port_sources=dict(commit=rgb.PORT_COMMIT,
                              shas=rgb.PORT_SOURCE_SHAS),
            content=content,
        ),
        dual_minus_single=["port5", "loss_geometry", "rgb_injection"],
        ubr_capacity=dict(hidden=ubr_cfg["refiner_hidden"],
                          depth=ubr_cfg["refiner_depth"]),
        dataset=str(data_root),
        out_root=str(out_root),
        resume_from=resume_from,
        audit_verdict=str(verdict),
        commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip(),
        source_hashes=_source_hashes(),
        val_shard_across_ranks=1,
    )


def _install_epoch_telemetry(trainer, out_dir: str) -> None:
    """v3 epoch telemetry + the RGB sentinel (spec section 6).

    After each train phase the RGB means are drained from the integration
    (encoder secs - the 5-view P0 forward share, support fraction,
    descriptor RMS, residual RMS, injection_active); after each val phase
    the latest staged scene's RGB telemetry is recorded.
    """
    import torch

    orig_train = trainer.train_one_epoch
    orig_val = trainer.validate_one_epoch
    log_path = Path(out_dir) / "epoch_telemetry.jsonl"
    is_main = int(os.environ.get("RANK", "0")) == 0

    def _rgb_extra(integ, phase):
        if integ is None or not hasattr(integ, "drain_rgb_stats"):
            return {}
        if phase == "train":
            return {f"rgb_{k}": round(v, 6)
                    for k, v in integ.drain_rgb_stats().items()}
        return {f"rgb_{k}": round(v, 6)
                for k, v in integ.rgb_last_stats().items()}

    def _record(phase, epoch, t0, model_arg):
        if not (is_main and torch.cuda.is_available()):
            return
        integ = base._integration_of(model_arg)
        rec = dict(
            phase=phase, epoch=int(epoch),
            secs=round(time.perf_counter() - t0, 2),
            peak_mem_alloc_gb=round(
                torch.cuda.max_memory_allocated() / 2**30, 3),
            peak_mem_reserved_gb=round(
                torch.cuda.max_memory_reserved() / 2**30, 3),
        )
        if integ is not None:
            stats = (integ.drain_geo_stats() if phase == "train"
                     else integ.drain_val_stats())
            prefix = "geo_" if phase == "train" else "dual_"
            rec.update({f"{prefix}{k}": round(v, 6)
                        for k, v in stats.items()})
            rec.update(_rgb_extra(integ, phase))
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def train_wrapped(*a, **k):
        if is_main and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = orig_train(*a, **k)
        _record("train", a[13] if len(a) > 13 else k.get("epoch", -1),
                t0, a[1] if len(a) > 1 else k.get("model"))
        return out

    def val_wrapped(*a, **k):
        t0 = time.perf_counter()
        out = orig_val(*a, **k)
        _record("val", a[16] if len(a) > 16 else k.get("epoch", -1),
                t0, a[1] if len(a) > 1 else k.get("model"))
        return out

    trainer.train_one_epoch = train_wrapped
    trainer.validate_one_epoch = val_wrapped


def run_worker(argv: list) -> None:
    """Module substitution + the sealed trainer entry, in-process."""
    os.chdir(ROOT)
    content = os.environ["UBR_RGB_CONTENT"]
    assert content in ARMS, content
    base.set_protocol_env(os.environ.get("SAM2_CKPT", base.SAM2_CKPT_DEFAULT))
    from mmengine.config import Config

    cfg = Config.fromfile(str(ROOT / CONFIG_REL))
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    # fail-closed: tail deleted, dual+rgb keys exact, v1.1 flags off for DDP4
    assert cfg.model.roi_head.mask_head.decoder_tail_refiner_cfg == {}, (
        "decoder_tail_refiner_cfg must resolve empty (deleted, not toggled)")
    assert ubr_cfg["bucket_backward"] is False, (
        "bucket_backward must stay OFF for the DDP4 protocol")
    assert ubr_cfg["grad_checkpoint"] is False, (
        "grad_checkpoint is OFF at 512 (required only at 1024)")
    assert ubr_cfg["uav_port"] == "real", (
        f"config uav_port {ubr_cfg['uav_port']!r} != 'real' (+RGB arm)")
    assert ubr_cfg["rgb_content"] == content, (
        f"config rgb_content {ubr_cfg['rgb_content']!r} != launched "
        f"content {content!r}")
    rgb.install_dual_rgb_class_patch(ubr_cfg, content)
    if content == "constant":
        assert os.environ.get("UBR_RGB_DESCRIPTOR_MEAN"), (
            "the constant arm requires UBR_RGB_DESCRIPTOR_MEAN")

    sys.path.insert(0, str(ROOT / "train"))
    import train_rsprompter_fusion  # noqa: E402  (sealed trainer entry)

    out_dir = argv[argv.index("--checkpoint-dir") + 1]
    _install_epoch_telemetry(train_rsprompter_fusion, out_dir)
    sys.argv = argv
    base._install_3090_augmentation()
    train_rsprompter_fusion.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-content", default=os.environ.get(
        "UBR_RGB_CONTENT", "real"), choices=ARMS,
        help="RGB consumption content (default env UBR_RGB_CONTENT or "
             "'real'; 'constant' is the reserved fourth arm)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir", default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--nproc", type=int, default=base.PROTOCOL["nproc"])
    ap.add_argument("--out-root", default=os.environ.get(
        "UBR_RGB_OUT_ROOT",
        f"{OUT_ROOT_DEFAULT}/ue_ubr_dual_rgb_{{content}}_20261028"))
    ap.add_argument("--data-root", default=os.environ.get(
        "UBR_RGB_DATA_ROOT", base.DS_ROOT_DEFAULT))
    ap.add_argument("--sam2-ckpt", default=base.SAM2_CKPT_DEFAULT)
    ap.add_argument("--anchor-cache", default=os.environ.get(
        "UBR_RGB_ANCHOR_CACHE", rgb.ANCHOR_CACHE_DEFAULT))
    ap.add_argument("--audit-verdict", default=os.environ.get(
        "UBR_RGB_AUDIT_VERDICT", ""),
        help="path of the independent wiring-audit file containing "
             "'VERDICT: PASS'; REQUIRED for the full (non-smoke) launch")
    ap.add_argument("--_worker", action="store_true")
    ap.add_argument("--worker-argv", default=None)
    a = ap.parse_args()

    if a._worker:
        run_worker(json.loads(a.worker_argv))
        return

    content = a.rgb_content
    assert content in ARMS, content
    os.environ["UBR_RGB_CONTENT"] = content  # authoritative for workers
    os.environ["UBR_RGB_DATA_ROOT"] = str(a.data_root)
    os.environ["UBR_RGB_ANCHOR_CACHE"] = str(a.anchor_cache)
    out_root = Path(a.out_root.format(content=content)
                    if "{content}" in a.out_root else a.out_root)
    data_root = Path(a.data_root)
    base.assert_dataset(data_root)
    manifest = anchor_manifest(Path(a.anchor_cache))
    assert manifest["scene_count"] == 3060, manifest["scene_count"]

    if a.smoke:
        assert not a.resume_from, "smoke mode does not resume"
        out_dir = (Path(a.smoke_dir) if a.smoke_dir else
                   out_root.parent / (out_root.name + "_smoke"))
        assert not out_dir.exists(), f"smoke out dir exists: {out_dir}"
        out_dir.mkdir(parents=True)
        argv = base.trainer_argv(
            CONFIG_REL, f"ue_ubr_dual_rgb_{content}_20261028_smoke",
            str(data_root), str(out_dir), epochs=base.SMOKE["epochs"],
            train_ratio=base.subset_ratio(2142, base.SMOKE["train_scenes"]),
            val_ratio=base.subset_ratio(459, base.SMOKE["val_scenes"]),
            num_workers=2)
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", "3")
        cmd = [sys.executable, str(HERE / "ue_ubr_dual_rgb_real_20261028.py"),
               "--_worker", "--worker-argv", json.dumps(argv)]
        print("SMOKE_LAUNCH", content, " ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd, cwd=str(ROOT), env=env))

    # ---- full launch gates (prereg integrity contract, fail-closed) ----
    assert a.nproc == base.PROTOCOL["nproc"], (
        f"protocol requires DDP world size {base.PROTOCOL['nproc']}")
    if a.resume_from:
        assert out_root.is_dir(), f"resume requires existing run dir: {out_root}"
        assert Path(a.resume_from).is_file(), f"resume ckpt missing: {a.resume_from}"
    else:
        assert not out_root.exists(), f"output directory already exists: {out_root}"
    assert a.audit_verdict, (
        "--audit-verdict <docs/reviews/...> is required for the full launch")
    verdict = Path(a.audit_verdict)
    assert verdict.is_file(), f"audit verdict file missing: {verdict}"
    assert "VERDICT: PASS" in verdict.read_text(), (
        f"no 'VERDICT: PASS' in {verdict}")
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                           capture_output=True, text=True).stdout.strip()
    assert not dirty, "full training requires a clean git worktree"
    if content == "constant":
        assert os.environ.get("UBR_RGB_DESCRIPTOR_MEAN"), (
            "the constant arm requires UBR_RGB_DESCRIPTOR_MEAN "
            "(train-only descriptor mean json)")

    argv = base.trainer_argv(CONFIG_REL, f"ue_ubr_dual_rgb_{content}_20261028",
                             str(data_root), str(out_root),
                             epochs=base.PROTOCOL["epochs"],
                             resume_from=a.resume_from)
    base.set_protocol_env(a.sam2_ckpt)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nnodes=1", f"--nproc_per_node={a.nproc}",
           str(HERE / "ue_ubr_dual_rgb_real_20261028.py"),
           "--_worker", "--worker-argv", json.dumps(argv)]
    contract = _build_contract(data_root, out_root, content, a.resume_from,
                               verdict)
    print("RESEAL_CONTRACT_V4", json.dumps(contract), flush=True)
    sys.exit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
