#!/usr/bin/env python3
"""Runner for UBR-RESEAL-20261027-R2 (level 3 v1.1: 100ep relaunch).

Launches the SAME production trainer entry EXP-0025 used
(train/train_rsprompter_fusion.py, invoked by
scripts/ue/run_exp0025_ddp4.sh) with the level-3 config
configs/ablations/ue_ubr_single_512.py, after installing the UBR
integration class patch (rsprompter/uav/ubr_train_integration.py).
Nothing in the sealed trainer or model files is modified; the wiring is
pure module substitution, so the trainer's protocol handling (DDP, seed,
AMP, cosine schedule, per-epoch mmdet val, checkpointing) is bit-for-bit
the code path EXP-0025 ran.

UBR v1.1 (docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md): the dead first
launch (EXP-0028, output root ue_ubr_reseal_20261027/, launch2.log) is
archived untouched as the equivalence-replay reference. This R2 runner
differs from it ONLY in: new output root (ue_ubr_reseal_r2_20261027),
VAL_SHARD_ACROSS_RANKS=1 (A5 sharded validation: rank-strided batches,
CPU/Gloo gather_object to rank0, global-order reassembly, canonical
COCO metrics), the RESEAL_CONTRACT v2 echo (source hashes + capacity
declaration), per-epoch peak-memory/seconds telemetry
(epoch_telemetry.jsonl), and the v1.1 execution layer/capacity inside
ubr_train_integration.py + ubr.py (equivalence-gated by G1-G4,
scripts/probes/ue_ubr_v11_gates_20261027.py).

Protocol (preregistered, verbatim EXP-0025): seed 44, DDP4 batch1 x
accum2 (global 8). PI b2xa1 order 2026-10-27 REVERTED: rank activations
doubled to 45GB/card and OOMed at step ~50 (47.38GB limit); b1xa2 keeps
the effective batch 8 invariant within memory.
lr 5e-4, wd 0.05, warmup 100 optimizer steps, cosine,
100 epochs, default augmentation (3090 line: no flip; train-only
ColorJitter(.15,.15,.10,.02)@p=.5 + Gaussian noise p=.5/std=.02), val 459
every epoch, last-epoch selection (SAVE_LAST_MODEL=1), test never read.
Asserted fail-closed below, along with the dataset SHA pins of the
EXP-0025 wrapper.

Modes:
  default        full 100ep run. Requires: output dir absent, clean git
                 tree, dataset SHA pins, and --audit-verdict pointing at
                 the independent wiring-audit file containing
                 'VERDICT: PASS' (prereg ordering: audit BEFORE cards).
  --resume PATH  continue an interrupted run from its last_checkpoint.pth
                 (epoch boundary); output dir must already exist.
  --smoke        2 epochs on a 20-scene seeded train subset + val on 5
                 scenes, single process, output under {out}/smoke.
  --_worker      internal: the substituted trainer process itself.

Output root: /data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_r2_20261027/
(on the lthpc host the same name under /data/wangcheng/checkpoint/ —
override with --out-root / RESEAL_OUT_ROOT). The dead v1.0 arm's root
(ue_ubr_reseal_20261027) is archived as-is and never written to.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # portable_sam2_explicit_coarse (repo root)
sys.path.insert(0, str(ROOT))

CONFIG_REL = "configs/ablations/ue_ubr_single_512.py"
OUT_ROOT_DEFAULT = "/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_reseal_r2_20261027"
DS_ROOT_DEFAULT = "/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021"
SAM2_CKPT_DEFAULT = "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"
RUN_TAG = "ue_ubr_reseal_r2_20261027"

# RESEAL_CONTRACT v2 (echoed at launch): v1.1 execution layer is
# numerically neutral (G1 equivalence replay, rel err < 1e-3) and refiner
# capacity 32ch x 6 conv blocks is the NEW UNIFIED definition for every
# single-/dual-stream arm from EXP-0028-R2 onward (legacy 48x7 pinned for
# the G1 replay only; band_k unchanged).
CONTRACT_V2_DECLARATION = (
    "v1.1 execution-layer changes numerically neutral (G1 replay <1e-3); "
    "capacity 32ch x 6 conv blocks is the new unified definition "
    "(48x7 legacy pinned for G1 old-capacity equivalence replay only); "
    "band_k unchanged")
# source_hashes bump (G4): every file this arm's wiring depends on that
# v1.1 touched; recorded in the contract so the run is bound to its code.
CONTRACT_SOURCE_FILES = (
    "rsprompter/uav/ubr.py",
    "rsprompter/uav/ubr_train_integration.py",
    "scripts/probes/ue_ubr_reseal_20261027.py",
    "configs/ablations/ue_ubr_single_512.py",
)

# Sealed protocol (EXP-0025 wrapper run_exp0025_ddp4.sh). Any drift here
# is a launch refusal.
PROTOCOL = dict(
    seed=44,
    batch_size=1,
    grad_accum_steps=2,
    epochs=100,
    lr=5e-4,
    weight_decay=0.05,
    warmup_iters=100,
    flip_prob=0.5,
    vflip_prob=0.0,
    ms_resize_prob=0.0,
    val_every_n_epochs=1,
    val_batch_size=1,
    compute_val_loss=0,
    amp=1,
    ema_enabled=0,
    image_size=(512, 512),
    nproc=4,
)
SMOKE = dict(epochs=2, train_scenes=20, val_scenes=5)

# Dataset SHA pins, verbatim from scripts/ue/run_exp0025_ddp4.sh
# (CitySample_SatUAV_TrainMinimal_v1_20261021, label revision
# review99_v3_user_approved_20261021; splits 2142/459/459).
DATASET_PINS = {
    "dataset.json": "b42a11edae080f4a31b07a25128bbb8322db84f85769093d8327ba1a45164379",
    "splits/train.json": "965b17041b669ded9a9ddf6d32e5820c2333133543c81902f0611fd0f854fa5e",
    "splits/val.json": "2cb80f5206baefd8b2220a61a5dd75ebc3b3873349f272c5530771ed8cee3c37",
    "splits/test.json": "f388cf35703252c33d8a40d769b79452f6ca7c71c594c3cc8bfd217517d856da",
    "annotations/train.json": "ce262bf0303ed0ee02355142802f22301f782b2b61e66dd363f6bb74e58e656d",
    "annotations/val.json": "9f9ead36f8c35dda8ff054c510c9da3bb32ffe2e39aed64f70b0c5174402f7f1",
    "annotations/test.json": "6d8f29a148d1a023930a0052e47fa906939a72dc6f9b094ef424ab4c23e619b3",
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_dataset(ds_root: Path) -> dict:
    """Dataset SHA pins + split counts (fail-closed)."""
    for rel, pin in DATASET_PINS.items():
        path = ds_root / rel
        assert path.is_file(), f"missing dataset file {path}"
        got = sha256_of(path)
        assert got == pin, f"dataset pin mismatch {rel}: {got} != {pin}"
    ds = json.loads((ds_root / "dataset.json").read_text())
    assert ds["label_revision"] == "review99_v3_user_approved_20261021"
    assert [ds["splits"][s]["points"] for s in ("train", "val", "test")] == [2142, 459, 459]
    assert [ds["splits"][s]["instances"] for s in ("train", "val", "test")] == [
        126214, 14980, 44988]
    assert ds["image_size"] == [512, 512]
    return ds


def trainer_argv(cfg_rel: str, run_tag: str, data_root: str, out_dir: str,
                 epochs: int, resume_from: str = None,
                 train_ratio: float = 1.0, val_ratio: float = 1.0,
                 num_workers: int = 4) -> list:
    """EXACTLY the EXP-0025 trainer argv, with only the registered deltas:
    --config (level-3 arm), --run-tag/--checkpoint-dir (new out dir),
    --epochs/--train-subset-ratio/--val-subset-ratio/--num-workers for the
    smoke mode, and optional --resume-from."""
    p = PROTOCOL
    argv = [
        "train/train_rsprompter_fusion.py",
        "--config", cfg_rel,
        "--run-tag", run_tag,
        "--data-root", data_root,
        "--dataset", "whu",
        "--image-size", "512", "512",
        "--batch-size", str(p["batch_size"]),
        "--grad-accum-steps", str(p["grad_accum_steps"]),
        "--epochs", str(epochs),
        "--early-stopping-patience", "0",
        "--lr", str(p["lr"]),
        "--weight-decay", str(p["weight_decay"]),
        "--warmup-iters", str(p["warmup_iters"]),
        "--seed", str(p["seed"]),
        "--val-batch-size", str(p["val_batch_size"]),
        "--val-every-n-epochs", str(p["val_every_n_epochs"]),
        "--compute-val-loss", str(p["compute_val_loss"]),
        "--amp", str(p["amp"]),
        "--ema-enabled", str(p["ema_enabled"]),
        "--ema-eval", "0",
        "--checkpoint-dir", out_dir,
        "--num-workers", str(num_workers),
    ]
    if train_ratio < 1.0 or val_ratio < 1.0:
        argv += ["--train-subset-ratio", repr(train_ratio),
                 "--val-subset-ratio", repr(val_ratio)]
    if resume_from:
        argv += ["--resume-from", resume_from]
    return argv


def set_protocol_env(sam2_ckpt: str):
    p = PROTOCOL
    os.environ.setdefault("SAM2_CKPT", sam2_ckpt)
    os.environ["WHU_TRAIN_ANN_FILE"] = "annotations/train.json"
    os.environ["WHU_VAL_ANN_FILE"] = "annotations/val.json"
    os.environ["WHU_TRAIN_IMG_SUBDIR"] = ""
    os.environ["WHU_VAL_IMG_SUBDIR"] = ""
    os.environ["TRAIN_FLIP_PROB"] = str(p["flip_prob"])
    os.environ["TRAIN_VFLIP_PROB"] = str(p["vflip_prob"])
    os.environ["TRAIN_MULTI_SCALE_RESIZE_PROB"] = str(p["ms_resize_prob"])
    os.environ["SAVE_LAST_MODEL"] = "1"
    # A5 (UBR v1.1): sharded validation — every DDP rank owns loader
    # batches rank, rank+W, ...; predictions are gathered to rank0 over
    # the CPU/Gloo control group and reassembled in global batch order,
    # so COCO inputs match the rank0-only path exactly (the remote-proven
    # pattern; FP32 + no_grad unchanged, ~0.6 min/epoch instead of 2.1).
    os.environ["VAL_SHARD_ACROSS_RANKS"] = "1"
    # UBR v1.1 doc section 5 remedy, enabled up front: the v1.0 arm's
    # reserved-vs-allocated gap (fragmentation from bucketed train shapes
    # alternating with the val refine path) is allocator-level; expandable
    # segments keep reserved close to allocated. Numerics-neutral
    # (allocator config only); an explicit caller value wins (setdefault).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert float(os.environ["TRAIN_FLIP_PROB"]) == 0.5
    assert float(os.environ["TRAIN_VFLIP_PROB"]) == 0.0
    assert float(os.environ["TRAIN_MULTI_SCALE_RESIZE_PROB"]) == 0.0


def _source_hashes() -> dict:
    """G4 source_hashes: sha256 of every CONTRACT_SOURCE_FILES entry."""
    return {rel: sha256_of(ROOT / rel) for rel in CONTRACT_SOURCE_FILES}


def _build_contract(data_root, out_root, resume_from, verdict) -> dict:
    """RESEAL_CONTRACT v2 (echoed before cards are pulled, fail-closed)."""
    from mmengine.config import Config
    cfg = Config.fromfile(str(ROOT / CONFIG_REL))
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    return dict(
        tag="UBR-RESEAL-20261027-R2",
        contract_version=2,
        declaration=CONTRACT_V2_DECLARATION,
        config=CONFIG_REL,
        protocol=PROTOCOL,
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
    """Per-epoch wall seconds + CUDA peak memory -> epoch_telemetry.jsonl.

    UBR v1.1 monitoring sentinel (doc section 5): rank0 appends one record
    per train/val phase. Wraps the trainer's own loop functions in place
    (module substitution; sealed files untouched).
    """
    import torch

    orig_train = trainer.train_one_epoch
    orig_val = trainer.validate_one_epoch
    log_path = Path(out_dir) / "epoch_telemetry.jsonl"
    is_main = int(os.environ.get("RANK", "0")) == 0

    def _record(phase, epoch, t0):
        if not (is_main and torch.cuda.is_available()):
            return
        rec = dict(
            phase=phase, epoch=int(epoch),
            secs=round(time.perf_counter() - t0, 2),
            peak_mem_alloc_gb=round(
                torch.cuda.max_memory_allocated() / 2**30, 3),
            peak_mem_reserved_gb=round(
                torch.cuda.max_memory_reserved() / 2**30, 3),
        )
        with open(log_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def train_wrapped(*a, **k):
        if is_main and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = orig_train(*a, **k)
        _record("train", a[13] if len(a) > 13 else k.get("epoch", -1), t0)
        return out

    def val_wrapped(*a, **k):
        t0 = time.perf_counter()
        out = orig_val(*a, **k)
        _record("val", a[16] if len(a) > 16 else k.get("epoch", -1), t0)
        return out

    trainer.train_one_epoch = train_wrapped
    trainer.validate_one_epoch = val_wrapped


def run_worker(argv: list) -> None:
    """Module substitution + the sealed trainer entry, in-process."""
    import torch

    os.chdir(ROOT)
    set_protocol_env(os.environ.get("SAM2_CKPT", SAM2_CKPT_DEFAULT))
    from mmengine.config import Config
    from rsprompter.uav import ubr_train_integration

    cfg = Config.fromfile(str(ROOT / CONFIG_REL))
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    # fail-closed: tail deleted in the resolved config, UBR keys exact
    assert cfg.model.roi_head.mask_head.decoder_tail_refiner_cfg == {}, (
        "decoder_tail_refiner_cfg must resolve empty (deleted, not toggled)")
    # fail-closed v1.1 launch flags: A4 is single-process-only (DDP
    # reducer guard) and B2 is OFF for the 512 protocol.
    assert ubr_cfg["bucket_backward"] is False, (
        "bucket_backward must stay OFF for the DDP4 protocol")
    assert ubr_cfg["grad_checkpoint"] is False, (
        "grad_checkpoint is OFF at 512 (required only at 1024)")
    ubr_train_integration.install_class_patch(ubr_cfg)

    sys.path.insert(0, str(ROOT / "train"))
    import train_rsprompter_fusion  # noqa: E402  (sealed trainer entry)

    # A4 plumbing (dormant while the flag is OFF): register the trainer's
    # GradScaler with the integration so a future single-process
    # bucketed-backward run scales manual contributions consistently.
    _orig_scaler_init = torch.amp.GradScaler.__init__

    def _scaler_init(self, *a, **kw):
        _orig_scaler_init(self, *a, **kw)
        ubr_train_integration.register_grad_scaler(self)

    torch.amp.GradScaler.__init__ = _scaler_init

    out_dir = argv[argv.index("--checkpoint-dir") + 1]
    _install_epoch_telemetry(train_rsprompter_fusion, out_dir)
    sys.argv = argv
    # Protocol: no flip (3090 line). Set authoritatively here so worker
    # subprocesses inherit it regardless of caller env.
    os.environ["TRAIN_FLIP_PROB"] = "0"
    os.environ["TRAIN_VFLIP_PROB"] = "0"
    _install_3090_augmentation()
    train_rsprompter_fusion.main()


def _install_3090_augmentation():
    """Class patch: train-split ColorJitter + Gaussian noise, zero flip.

    Semantics copied verbatim from data/ue_minimal_dataset.py
    (MinimalSatelliteDataset, the 3090-line protocol): RandomApply(
    ColorJitter(.15,.15,.10,.02), p=.5) on the BGR->RGB uint8 image and
    gaussian_noise_prob=.5/std=.02, train split only; flip stays 0 via env.
    """
    import torch
    from torchvision.transforms import ColorJitter, RandomApply
    from data.whu_instance_dataset import WHUCocoInstanceDataset
    jitter = RandomApply([ColorJitter(brightness=.15, contrast=.15,
                                      saturation=.10, hue=.02)], p=.5)
    orig_init = WHUCocoInstanceDataset.__init__
    orig_get = WHUCocoInstanceDataset.__getitem__

    def init_(self, *a, **kw):
        orig_init(self, *a, **kw)
        ann = str(kw.get('ann_file') or (a[1] if len(a) > 1 else ''))
        if 'annotations/train' in ann:
            self.gaussian_noise_prob = 0.5
            self.gaussian_noise_std = 0.02
            self._ubr3090_jitter = jitter

    def get_(self, idx):
        item = orig_get(self, idx)
        j = getattr(self, '_ubr3090_jitter', None)
        if j is not None and isinstance(item, dict) and 'img' in item:
            rgb = item['img'][[2, 1, 0]].to(torch.uint8)
            item['img'] = j(rgb)[[2, 1, 0]].float()
        return item

    WHUCocoInstanceDataset.__init__ = init_
    WHUCocoInstanceDataset.__getitem__ = get_


def subset_ratio(n_full: int, n_want: int) -> float:
    ratio = n_want / float(n_full)
    assert int(n_full * ratio) == n_want, (n_full, n_want, ratio)
    return ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir", default=None)
    ap.add_argument("--resume-from", default=None,
                    help="checkpoint path (epoch boundary) to continue from")
    ap.add_argument("--nproc", type=int, default=PROTOCOL["nproc"],
                    help="DDP world size for the full run")
    ap.add_argument("--out-root", default=os.environ.get(
        "RESEAL_OUT_ROOT", OUT_ROOT_DEFAULT))
    ap.add_argument("--data-root", default=os.environ.get(
        "RESEAL_DATA_ROOT", DS_ROOT_DEFAULT))
    ap.add_argument("--sam2-ckpt", default=SAM2_CKPT_DEFAULT)
    ap.add_argument("--audit-verdict", default=os.environ.get(
        "RESEAL_AUDIT_VERDICT", ""),
        help="path of the independent wiring-audit file containing "
             "'VERDICT: PASS'; REQUIRED for the full (non-smoke) launch "
             "(prereg: audit lands in docs/reviews/ before cards are pulled)")
    ap.add_argument("--_worker", action="store_true",
                    help="internal: substituted trainer process")
    ap.add_argument("--worker-argv", default=None,
                    help="internal: JSON-encoded trainer argv")
    a = ap.parse_args()

    if a._worker:
        run_worker(json.loads(a.worker_argv))
        return

    out_root = Path(a.out_root)
    data_root = Path(a.data_root)
    assert_dataset(data_root)

    if a.smoke:
        assert not a.resume_from, "smoke mode does not resume"
        # smoke artifacts live OUTSIDE the preregistered full-run output
        # dir: the full launch requires that dir to be absent, so smoke
        # must never create it (sibling '<name>_smoke' instead).
        out_dir = (Path(a.smoke_dir) if a.smoke_dir else
                   out_root.parent / (out_root.name + "_smoke"))
        assert not out_dir.exists(), f"smoke out dir exists: {out_dir}"
        out_dir.mkdir(parents=True)
        train_ratio = subset_ratio(2142, SMOKE["train_scenes"])
        val_ratio = subset_ratio(459, SMOKE["val_scenes"])
        argv = trainer_argv(
            CONFIG_REL, f"{RUN_TAG}_smoke", str(data_root),
            str(out_dir), epochs=SMOKE["epochs"], train_ratio=train_ratio,
            val_ratio=val_ratio, num_workers=2)
        # smoke: single process on the allowed GPU (e.g. GPU3), no DDP
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", "3")
        cmd = [sys.executable, str(HERE / "ue_ubr_reseal_20261027.py"),
               "--_worker", "--worker-argv", json.dumps(argv)]
        print("SMOKE_LAUNCH", " ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd, cwd=str(ROOT), env=env))

    # ---- full launch gates (prereg integrity contract, fail-closed) ----
    assert a.nproc == PROTOCOL["nproc"], (
        f"protocol requires DDP world size {PROTOCOL['nproc']}, got {a.nproc}")
    if a.resume_from:
        # crash continuation at an epoch boundary: the run dir exists and
        # must contain the resume checkpoint (strict load incl. UBR keys)
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

    argv = trainer_argv(CONFIG_REL, RUN_TAG, str(data_root),
                        str(out_root), epochs=PROTOCOL["epochs"],
                        resume_from=a.resume_from)
    set_protocol_env(a.sam2_ckpt)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nnodes=1", f"--nproc_per_node={a.nproc}",
           str(HERE / "ue_ubr_reseal_20261027.py"),
           "--_worker", "--worker-argv", json.dumps(argv)]
    contract = _build_contract(data_root, out_root, a.resume_from, verdict)
    print("RESEAL_CONTRACT_V2", json.dumps(contract), flush=True)
    sys.exit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
