#!/usr/bin/env python3
"""Runner for the level-4 dual-stream UBR arms (EXP-0029, 2026-10-27).

Launches the SAME sealed production trainer entry EXP-0025/EXP-0028-R2
used (train/train_rsprompter_fusion.py) with the dual config
configs/ablations/ue_ubr_dual_512.py, after installing the DUAL
integration class patch (rsprompter/uav/ubr_dual_integration.py, which
extends - never forks - the v1.1 install ubr_train_integration.py).
Nothing in the sealed trainer or model files is modified.

Arms (prereg docs/UBR-RESEAL-20261027_preregistration.md, "级4 双流臂预注册"):
  --arm real      arm ② dual-REAL: port 5 = native512 cache fields.
  --arm constant  arm ③ dual-CONSTANT: port 5 = support-matched train-only
                  per-channel means (ubr.build_constant_geometry). Both
                  arms train the geometry head on the REAL map; the arm
                  diff is the port-5 content alone.
  Launch discipline (PI instruction 2026-10-27): arm ② launches
  co-tenant with the running EXP-0028-R2 (max 2 training processes per
  card); arm ③ is QUEUED for a later manual launch - nothing in this
  repo auto-launches it.

Protocol: verbatim EXP-0028-R2 (seed 44, DDP4 b1 x accum2, lr 5e-4,
wd 0.05, warmup 100, cosine, 100 epochs, 3090-line augmentation = zero
flip + train-only ColorJitter(.15,.15,.10,.02)@p.5 + Gauss(.5,.02),
sharded val 459 every epoch, last-epoch selection, test never read).
Output roots: /data2/wangcheng/remote-lthpc/checkpoint/
  ue_ubr_dual_real_20261027/     (arm ②)
  ue_ubr_dual_constant_20261027/ (arm ③)

RESEAL_CONTRACT v3 (echoed fail-closed before cards are pulled): adds to
the v2 contract the dual declarations - arm/port-5 mode, geometry head
(202,581 params, pilot network), lambda_geo 1.0, native512 cache version
pin + constants SHA, candidate band contract (r(s) prefix / 0.25*s
candidates / 250 cap), and the source hashes of the dual wiring files.

Modes:
  default        full 100ep run. Requires: output dir absent, clean git
                 tree, dataset SHA pins, --arm in {real, constant}, and
                 --audit-verdict pointing at the independent wiring-audit
                 file containing 'VERDICT: PASS' (audit BEFORE cards).
  --resume PATH  continue an interrupted run from its last_checkpoint.pth.
  --smoke        2 epochs on a 20-scene seeded train subset + val on 5
                 scenes, single process, output under {out}/smoke. NOTE:
                 this is the FULL-protocol smoke for when cards are free;
                 the co-tenant-safe (peak < 6 GB, GPU3) batch-level smoke
                 lives in scripts/probes/ue_ubr_dual_gates_20261027.py.
  --_worker      internal: the substituted trainer process itself.
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

CONFIG_REL = "configs/ablations/ue_ubr_dual_512.py"
OUT_ROOT_DEFAULT = "/data2/wangcheng/remote-lthpc/checkpoint"
DS_ROOT_DEFAULT = "/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021"
SAM2_CKPT_DEFAULT = "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"
ARMS = ("real", "constant")

# RESEAL_CONTRACT v3: v2 (numerically-neutral execution layer; 32ch x 6
# unified capacity) + the dual declarations.
CONTRACT_V3_DECLARATION = (
    "v3 dual: v1.1 execution layer + unified 32ch x 6 capacity inherited "
    "bit-exact from EXP-0028-R2; dual arms add port 5 (real native512 "
    "fields | support-matched train-only means) and loss_geometry "
    "(pilot geometry head, lambda_geo 1.0, direct GT center-box "
    "supervision, candidates never enter detector losses = FW1); "
    "inference = frozen interface r(s) prefix band + 0.25*s candidate "
    "band, dedup+NMS vs prefix, 250 cap, UBR refine on every instance")
CONTRACT_SOURCE_FILES = (
    "rsprompter/uav/ubr.py",
    "rsprompter/uav/ubr_train_integration.py",
    "rsprompter/uav/ubr_dual_integration.py",
    "scripts/probes/ue_ubr_reseal_dual_20261027.py",
    "configs/ablations/ue_ubr_dual_512.py",
)

# Sealed protocol (EXP-0028-R2 runner; EXP-0025 lineage). Any drift is a
# launch refusal. b1 x a2 (PI b2xa1 order was reverted after the 45GB/card
# OOM - see the R2 runner docstring).
PROTOCOL = dict(
    seed=44,
    batch_size=1,
    grad_accum_steps=2,
    epochs=100,
    lr=5e-4,
    weight_decay=0.05,
    warmup_iters=100,
    flip_prob=0.0,
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

# Dataset SHA pins, verbatim from the EXP-0025 wrapper (unchanged in R2).
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
    """Dataset SHA pins + split counts (fail-closed; verbatim R2)."""
    for rel, pin in DATASET_PINS.items():
        path = ds_root / rel
        assert path.is_file(), f"missing dataset file {path}"
        got = sha256_of(path)
        assert got == pin, f"dataset pin mismatch {rel}: {got} != {pin}"
    ds = json.loads((ds_root / "dataset.json").read_text())
    assert ds["label_revision"] == "review99_v3_user_approved_20261021"
    assert [ds["splits"][s]["points"] for s in ("train", "val", "test")] == [
        2142, 459, 459]
    assert [ds["splits"][s]["instances"] for s in ("train", "val", "test")] == [
        126214, 14980, 44988]
    assert ds["image_size"] == [512, 512]
    return ds


def trainer_argv(cfg_rel: str, run_tag: str, data_root: str, out_dir: str,
                 epochs: int, resume_from: str = None,
                 train_ratio: float = 1.0, val_ratio: float = 1.0,
                 num_workers: int = 4) -> list:
    """EXACTLY the EXP-0025/R2 trainer argv, with only the registered
    deltas: --config (dual arm), --run-tag/--checkpoint-dir, smoke-mode
    ratios, and optional --resume-from."""
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
    """3090-line protocol env (zero flip; sharded val; expandable segs)."""
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
    os.environ["VAL_SHARD_ACROSS_RANKS"] = "1"
    # canonical AP250: the dual val must evaluate the MERGED <=250 set -
    # prefix r(s) band + 0.25*s candidates (frozen interface, joint-eval
    # contract max_dets=[1,10,250]). The trainer applies this env both as
    # the model-side per-image cap and as the COCO maxDets list. Without
    # it the eval would run at maxDets=100 and rank every candidate below
    # the prefix band (r(s) > 0.5 > 0.25 >= s), hiding the arm's whole
    # candidate stream from per-epoch telemetry. Collection-time judged
    # evals of R1/R2 readouts use the same canonical AP250.
    os.environ.setdefault("TEST_MAX_PER_IMG", "250")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    assert float(os.environ["TRAIN_FLIP_PROB"]) == 0.0, "dual: no flip"
    assert float(os.environ["TRAIN_VFLIP_PROB"]) == 0.0, "dual: no vflip"
    assert float(os.environ["TRAIN_MULTI_SCALE_RESIZE_PROB"]) == 0.0


def _source_hashes() -> dict:
    return {rel: sha256_of(ROOT / rel) for rel in CONTRACT_SOURCE_FILES}


def _build_contract(data_root, out_root, arm, resume_from,
                    verdict) -> dict:
    """RESEAL_CONTRACT v3 (echoed before cards are pulled, fail-closed)."""
    from mmengine.config import Config
    from rsprompter.uav import ubr_dual_integration as dual

    cfg = Config.fromfile(str(ROOT / CONFIG_REL))
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    return dict(
        tag=f"UBR-DUAL-20261027-{arm.upper()}",
        contract_version=3,
        declaration=CONTRACT_V3_DECLARATION,
        config=CONFIG_REL,
        protocol=PROTOCOL,
        arm=arm,
        uav_port=arm,
        geometry=dict(
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


def _integration_of(trainer_or_model):
    model = trainer_or_model
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "roi_head"):  # a bare model
        return getattr(model, "ubr_integration", None)
    return None


def _install_epoch_telemetry(trainer, out_dir: str) -> None:
    """Per-epoch wall secs + CUDA peak memory + DUAL geometry telemetry.

    Extends the R2 telemetry: after each train phase the geometry epoch
    means (loss_geometry/focal/offset/size, instances per image) are
    drained from the integration; after each val phase the dual merge
    telemetry (candidates/img, accepted/img, prefix mean, prefix score
    mean) is drained. rank0 appends one jsonl record per phase.
    """
    import torch

    orig_train = trainer.train_one_epoch
    orig_val = trainer.validate_one_epoch
    log_path = Path(out_dir) / "epoch_telemetry.jsonl"
    is_main = int(os.environ.get("RANK", "0")) == 0

    def _extra(model_arg, phase):
        integ = _integration_of(model_arg)
        if integ is None:
            return {}
        if phase == "train":
            stats = integ.drain_geo_stats()
            return {f"geo_{k}": round(v, 6) for k, v in stats.items()}
        stats = integ.drain_val_stats()
        return {f"dual_{k}": round(v, 6) for k, v in stats.items()}

    def _record(phase, epoch, t0, model_arg):
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
        rec.update(_extra(model_arg, phase))
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
    arm = os.environ["UBR_DUAL_ARM"]
    assert arm in ARMS, arm
    set_protocol_env(os.environ.get("SAM2_CKPT", SAM2_CKPT_DEFAULT))
    from mmengine.config import Config
    from rsprompter.uav import ubr_dual_integration

    cfg = Config.fromfile(str(ROOT / CONFIG_REL))
    ubr_cfg = cfg.model.roi_head.mask_head.ubr_integration_cfg
    # fail-closed: tail deleted, dual keys exact, v1.1 flags off for DDP4
    assert cfg.model.roi_head.mask_head.decoder_tail_refiner_cfg == {}, (
        "decoder_tail_refiner_cfg must resolve empty (deleted, not toggled)")
    assert ubr_cfg["bucket_backward"] is False, (
        "bucket_backward must stay OFF for the DDP4 protocol")
    assert ubr_cfg["grad_checkpoint"] is False, (
        "grad_checkpoint is OFF at 512 (required only at 1024)")
    assert ubr_cfg["uav_port"] == arm, (
        f"config uav_port {ubr_cfg['uav_port']!r} != launched arm {arm!r}")
    ubr_dual_integration.install_dual_class_patch(ubr_cfg, arm)

    sys.path.insert(0, str(ROOT / "train"))
    import train_rsprompter_fusion  # noqa: E402  (sealed trainer entry)

    out_dir = argv[argv.index("--checkpoint-dir") + 1]
    _install_epoch_telemetry(train_rsprompter_fusion, out_dir)
    sys.argv = argv
    _install_3090_augmentation()
    train_rsprompter_fusion.main()


def _install_3090_augmentation():
    """Class patch: train-split ColorJitter + Gaussian noise, zero flip.

    Semantics copied verbatim from the R2 runner (data/
    ue_minimal_dataset.py MinimalSatelliteDataset, the 3090-line
    protocol). Photometric only - the geometry map is a separate,
    fixed per-scene artifact and is never augmented by construction.
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
            self._dual3090_jitter = jitter

    def get_(self, idx):
        item = orig_get(self, idx)
        j = getattr(self, '_dual3090_jitter', None)
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
    ap.add_argument("--arm", default=os.environ.get("UBR_DUAL_ARM", "real"),
                    choices=ARMS,
                    help="dual arm to run (port-5 content; default env "
                         "UBR_DUAL_ARM or 'real')")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-dir", default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--nproc", type=int, default=PROTOCOL["nproc"])
    ap.add_argument("--out-root", default=os.environ.get(
        "UBR_DUAL_OUT_ROOT",
        f"{OUT_ROOT_DEFAULT}/ue_ubr_dual_{{arm}}_20261027"))
    ap.add_argument("--data-root", default=os.environ.get(
        "UBR_DUAL_DATA_ROOT", DS_ROOT_DEFAULT))
    ap.add_argument("--sam2-ckpt", default=SAM2_CKPT_DEFAULT)
    ap.add_argument("--audit-verdict", default=os.environ.get(
        "UBR_DUAL_AUDIT_VERDICT", ""),
        help="path of the independent wiring-audit file containing "
             "'VERDICT: PASS'; REQUIRED for the full (non-smoke) launch")
    ap.add_argument("--_worker", action="store_true")
    ap.add_argument("--worker-argv", default=None)
    a = ap.parse_args()

    if a._worker:
        run_worker(json.loads(a.worker_argv))
        return

    arm = a.arm
    assert arm in ARMS, arm
    os.environ["UBR_DUAL_ARM"] = arm  # authoritative for the worker(s)
    out_root = Path(a.out_root.format(arm=arm) if "{arm}" in a.out_root
                    else a.out_root)
    data_root = Path(a.data_root)
    assert_dataset(data_root)

    if a.smoke:
        assert not a.resume_from, "smoke mode does not resume"
        out_dir = (Path(a.smoke_dir) if a.smoke_dir else
                   out_root.parent / (out_root.name + "_smoke"))
        assert not out_dir.exists(), f"smoke out dir exists: {out_dir}"
        out_dir.mkdir(parents=True)
        argv = trainer_argv(
            CONFIG_REL, f"ue_ubr_dual_{arm}_20261027_smoke", str(data_root),
            str(out_dir), epochs=SMOKE["epochs"],
            train_ratio=subset_ratio(2142, SMOKE["train_scenes"]),
            val_ratio=subset_ratio(459, SMOKE["val_scenes"]), num_workers=2)
        env = dict(os.environ)
        env.setdefault("CUDA_VISIBLE_DEVICES", "3")
        cmd = [sys.executable, str(HERE / "ue_ubr_reseal_dual_20261027.py"),
               "--_worker", "--worker-argv", json.dumps(argv)]
        print("SMOKE_LAUNCH", arm, " ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd, cwd=str(ROOT), env=env))

    # ---- full launch gates (prereg integrity contract, fail-closed) ----
    assert a.nproc == PROTOCOL["nproc"], (
        f"protocol requires DDP world size {PROTOCOL['nproc']}, got {a.nproc}")
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

    argv = trainer_argv(CONFIG_REL, f"ue_ubr_dual_{arm}_20261027",
                        str(data_root), str(out_root),
                        epochs=PROTOCOL["epochs"], resume_from=a.resume_from)
    set_protocol_env(a.sam2_ckpt)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
           "--nnodes=1", f"--nproc_per_node={a.nproc}",
           str(HERE / "ue_ubr_reseal_dual_20261027.py"),
           "--_worker", "--worker-argv", json.dumps(argv)]
    contract = _build_contract(data_root, out_root, arm, a.resume_from,
                               verdict)
    print("RESEAL_CONTRACT_V3", json.dumps(contract), flush=True)
    sys.exit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
