"""UBR dual-stream +RGB arm: EXP-0029 dual(D+pose) + the CI-positive RGB path.

Authoritative spec: docs/UBR-DUAL-RGB-LAUNCH-SPEC-20261028.md (EXP-0030,
UBR-DUAL-20261028-REAL). This module EXTENDS - never forks - the EXP-0029
dual install (rsprompter/uav/ubr_dual_integration.py, commit af71c49):
every EXP-0029 element is retained verbatim (geometry head + loss_geometry,
port-5 wiring, candidate band inference) and exactly one new consumption
path is added - the only test-positive RGB shape of the remote
candidate_detail matrix (+0.89pp [+0.64, +1.20]):

  UAV RGB x5 views -> source LoRA SAM2 encoder (RGBP0Encoder, ctor weights
  from the SAME generic foundation file as the satellite trunk) -> P0
  pyramid level (256ch) -> rgb_project -> 32d descriptors -> candidate
  anchor sampling (FP64 continuous read, <=8 candidates/cell, <=2/view,
  original cache order) -> per-cell equal-weight mean (reduce_rgb mean
  branch) -> satellite trunk block4 ZERO-initialized residual injection
  (independent residual path - compact's interaction CNN is NOT ported).

Ports (spec section 2, SHA-locked in PORT_SOURCE_SHAS / contract v4):
  rsprompter/uav/rgb_p0_encoder.py        verbatim remote copy (file sha
                                           af32556a... at remote commit
                                           1618908b..., blob 017e8339...)
  sample_source_continuous/source_uv_grid rsprompter/uav/attention_ladder.py
                                           (file sha 4ceb15c3..., blob
                                           a5e3cf46...) - verbatim
  mean-reduce + LN-on-descriptors + block4 hook site + zero-init output:
                                           remote compact lineage
                                           (rsprompter/uav/compact_joint.py
                                           file sha be9c8c7c..., blob
                                           b4c049fe...; the mean branch is
                                           InstanceInteraction.fuse_rgb /
                                           candidate_detail reduce_rgb
                                           'detail' variant)
  candidate index + FP64 anchor cache     remote
  /data/wangcheng/cache/ue_v1_s2db_geometry_v2 (version
  s2db_anchor_metadata_v2_fp64, <=2 tokens per view/cell by observed
  satellite depth, <=8 per cell, cache rank order) mirrored to
  /data2/wangcheng/remote-lthpc/cache/ue_v1_s2db_geometry_v2 (8.5 GB,
  3060 scenes - under the 50 GB mirror cap).

NOT ported (spec section 7): the compact geometry CNN (geometry is already
consumed at three sites - double consumption would confound attribution),
Q-style per-candidate attention reading (attention_ladder.masked_weights;
test CI negative -2.92pp), RGB direct injection into port 5 or the
geometry-head input (frozen-era refuted).

Identity invariant (gate G2): rgb_output is zero-initialized (weight AND
bias), so at construction the injected residual is exactly +0.0 and the
arm's forward is bit-identical to the EXP-0029 build under equal weights.
The addition base + 0.0 is IEEE-exact for every value except -0.0 (which
flips to +0.0); G2 therefore reports value equality, bitwise equality and
zero-sign-only equality separately (fail-closed on the first, evidence on
the other two).

Optimizer treatment: the whole RGB branch is registered as
model.backbone.rgb_source_branch, so train/optimizer.py's existing rule
match_name.startswith("backbone.") routes every RGB parameter into the
BACKBONE group - the same treatment as the satellite trunk LoRA
(lr * backbone_lr_mult = 5e-4, weight_decay 0.05, spec section 1).

Init discipline (spec section 2): the source encoder ctor loads ONLY the
generic SAM2 foundation file (model.backbone.checkpoint_path - the same
file the satellite trunk ctor read); task_weights_loaded stays False; the
gates audit actual torch.load paths. Construction runs under fork_rng
(seed RGB_SOURCE_SEED) so the shared initialization stream is unperturbed
(the dual build's tensors stay bit-identical - gate G1).

Reserved fourth arm: rgb_content='constant' replaces the per-cell pooled
descriptor with a train-only per-channel mean vector under the SAME
candidate-support mask (zero-cost switch; the source encoder is skipped
entirely - content control with no RGB compute). Unit-drilled by gate G5,
never auto-launched.

Asserts (fail-closed): dual+rgb config keys exact; anchor cache version
and per-scene cameras.json provenance; index/anchor shapes and finiteness;
block4 output shape; staging present whenever the trunk runs in train
mode; injection residual exactly zero at ctor.
"""
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import ubr_dual_integration as dual
from . import ubr_train_integration as v11
from .local_adapter_probe import YAWS

# ------------------------------------------------------------- RGB constants
RGB_CONTENT_MODES = ("real", "constant")  # rgb_content values (spec section 3)
RGB_CFG_EXTRA_KEYS = {"rgb_content"}      # the only added config key
RGB_BRANCH_ATTR = "rgb_source_branch"     # child of model.backbone -> its
# optimizer group (train/optimizer.py backbone rule; spec section 1)
P0_CHANNELS = 256        # RGBP0Encoder lateral output (neck.convs[3])
RGB_PROJECT_DIM = 32     # rgb_project output dim (compact add_rgb)
SAT_BLOCK4_DIM = 224     # Hiera base+ stage-2 dim at the injection site
RGB_SOURCE_SEED = 2045   # compact_joint.CompactJoint source_seed (fork_rng)
# compact InstanceInteraction.add_rgb lora_config, verbatim.
RGB_LORA_CONFIG = dict(r=16, lora_alpha=32, target_modules=["qkv", "proj"],
                       lora_dropout=0.05, bias="none", modules_to_save=[])
# compact add_rgb registered buffers, verbatim (ImageNet stats; RGB order).
RGB_MEAN = (123.675, 116.28, 103.53)
RGB_STD = (58.395, 57.12, 57.375)
# Source anchor/candidate-index cache (spec section 2); env override exists.
ANCHOR_CACHE_DEFAULT = "/data2/wangcheng/remote-lthpc/cache/ue_v1_s2db_geometry_v2"
ANCHOR_VERSION = "s2db_anchor_metadata_v2_fp64"
ANCHOR_INDEX_SHAPE = (128 * 128, 8)          # (16384, 8) stride-4 cells
ANCHOR_FIELDS_SHAPE = (128 * 128, 8, 6)      # fp64 u,v,z,sat_u,sat_v,sat_z
DATASET_ROOT_DEFAULT = (
    "/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021")
# Remote port provenance (read-only remote lthpc:/home/wangcheng2021/
# project/new-uav-dual, commit pinned at port time; file sha256 / git blob).
PORT_COMMIT = "1618908b55f96b6fcc324a0a019ddd9938972c01"
PORT_SOURCE_SHAS = {
    "rsprompter/uav/rgb_p0_encoder.py": (
        "af32556a8d27f3495559e5b0cfc1fc9d4bbe34173003f1607744c04a7d273897",
        "017e83395f99cffdac0eb92bc4d34f587efab09e"),
    "rsprompter/uav/compact_joint.py": (
        "be9c8c7cb868a33819be2344b957751246ce0cf575d4eff64daf619c4314d972",
        "b4c049fe0a425044a32fa61a23aa6bfbceccba6d"),
    "rsprompter/uav/attention_ladder.py": (
        "4ceb15c3b6bd7221659ec7c5cae0ed3f7e23dc0157b3e69b9c066b6f1451e6b1",
        "a5e3cf462bb13b4af037523077d6eb247a6adbea"),
}


# ------------------------------------------------- attention_ladder verbatim
def source_uv_grid(uv: torch.Tensor) -> torch.Tensor:
    """Integer-center image indices -> normalized stride-4 feature coords.

    Verbatim port of rsprompter/uav/attention_ladder.py::source_uv_grid
    (remote file sha 4ceb15c3...; the L1 feature coordinate contract).
    """
    return (uv + 2.) / 256. - 1.


def sample_source_continuous(features: torch.Tensor, index: torch.Tensor,
                             anchor: torch.Tensor) -> torch.Tensor:
    """Sample normalized 5x128x128 token maps at retained anchors.

    Verbatim port of rsprompter/uav/attention_ladder.py::
    sample_source_continuous (remote file sha 4ceb15c3...): features
    [5*16384, 32] (view-major cell order), index [16384, 8] with entries
    view*16384 + token (-1 = no candidate), anchor [16384, 8, 6] fp64
    whose first two fields are the retained continuous source u/v pixels.
    Returns per-candidate descriptors [16384, 8, 32].
    """
    valid = index >= 0
    result = features.new_zeros((*index.shape, RGB_PROJECT_DIM))
    maps = features.reshape(5, 128, 128, RGB_PROJECT_DIM).permute(0, 3, 1, 2)
    # Each selected source token identifies the view; its original
    # continuous pixel anchor is retained by s2db_geometry. No resampling
    # of the candidates.
    for view in range(5):
        take = valid & (index // 16384 == view)
        if not bool(take.any()):
            continue
        uv = anchor[take][:, :2]
        grid = source_uv_grid(uv).reshape(1, -1, 1, 2).to(maps.dtype)
        result[take] = F.grid_sample(
            maps[view:view + 1], grid,
            mode="bilinear", padding_mode="border",
            align_corners=False)[0, :, :, 0].T
    return result


def dataset_root() -> str:
    """Dataset tree holding points/*/uav views (env, else the sealed default)."""
    return os.environ.get("UBR_RGB_DATA_ROOT", DATASET_ROOT_DEFAULT)


# ------------------------------------------------------------- scene cache
class SceneRGBAnchorCache:
    """Per-process LRU over (5-view RGB, candidate index, FP64 anchor).

    One npz + five rgb.png reads per scene; the per-scene report's
    cameras.json sha is checked against THIS dataset tree so the remote
    cache is provably bound to the local scenes (fail-closed).
    """

    def __init__(self, cache_dir: str, data_root: str, capacity: int = 4):
        self.dir = Path(cache_dir)
        self.data_root = Path(data_root)
        self.capacity = int(capacity)
        self._pool: "OrderedDict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = (
            OrderedDict())

    def _load_rgb(self, scene_id: int) -> np.ndarray:
        import cv2
        folder = self.data_root / "points" / f"{int(scene_id):06d}"
        views = []
        for yaw in YAWS:
            image = cv2.imread(str(folder / "uav" / f"yaw_{yaw:03d}" / "rgb.png"))
            assert image is not None and image.shape == (512, 512, 3), (
                f"missing uav rgb view yaw_{yaw:03d} scene {scene_id}")
            # BGR -> RGB, CHW (ue_joint_inputs.py loader, verbatim).
            views.append(image[:, :, ::-1].transpose(2, 0, 1).copy())
        return np.stack(views)

    def load(self, scene_id: int
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (rgb uint8 [5,3,512,512], index int64, anchor float64)."""
        key = int(scene_id)
        if key not in self._pool:
            from .local_adapter_probe import file_sha
            with np.load(self.dir / f"{key:06d}.npz") as z:
                assert str(z["version"]) == ANCHOR_VERSION, (
                    f"anchor cache version drift scene {key}")
                index = z["index"].astype(np.int64)
                anchor = z["anchor"].copy()
                report = json.loads(str(z["report"]))
            assert index.shape == ANCHOR_INDEX_SHAPE, (key, index.shape)
            assert anchor.shape == ANCHOR_FIELDS_SHAPE, (key, anchor.shape)
            assert anchor.dtype == np.float64 and bool(np.isfinite(anchor).all()), key
            cameras = self.data_root / "points" / f"{key:06d}" / "cameras.json"
            assert report["source_hashes"]["cameras.json"] == file_sha(cameras), (
                f"anchor cache scene {key} does not belong to this dataset")
            rgb = self._load_rgb(key)
            self._pool[key] = (rgb, index, anchor)
            self._pool.move_to_end(key)
            while len(self._pool) > self.capacity:
                self._pool.popitem(last=False)
        return self._pool[key]


# ------------------------------------------------------------- RGB branch
class RGBSourceBranch(nn.Module):
    """P0 source encoder + projection + zero-init block4 output (compact port).

    Constructed under fork_rng by the integration (shared-stream
    discipline); the ctor loads only the generic SAM2 foundation file.
    """

    def __init__(self, foundation_path: str):
        super().__init__()
        from ..backbone.sam2_vision import RSSAM2VisionEncoder
        from .rgb_p0_encoder import RGBP0Encoder
        assert Path(foundation_path).is_file(), foundation_path
        # compact InstanceInteraction.add_rgb ctor, verbatim (the P0Source
        # wrap folded in: RGBP0Encoder IS the retained P0 path of the source).
        source = RSSAM2VisionEncoder(
            checkpoint_path=foundation_path, model_size="base_plus",
            use_gradient_checkpointing=True, use_lora=True,
            freeze_backbone=False, lora_config=dict(RGB_LORA_CONFIG))
        self.p0_encoder = RGBP0Encoder(source)
        self.p0_encoder.audit_source_state(source)  # bit-equal ctor state
        self.rgb_project = nn.Conv2d(P0_CHANNELS, RGB_PROJECT_DIM, 1)
        # Zero-initialized residual output (compact InstanceInteraction
        # self.output discipline): the G2 identity invariant.
        self.rgb_output = nn.Linear(RGB_PROJECT_DIM, SAT_BLOCK4_DIM)
        nn.init.zeros_(self.rgb_output.weight)
        nn.init.zeros_(self.rgb_output.bias)
        self.register_buffer(
            "rgb_mean", torch.tensor(RGB_MEAN).view(1, 3, 1, 1))
        self.register_buffer(
            "rgb_std", torch.tensor(RGB_STD).view(1, 3, 1, 1))

    @property
    def injection_is_zero(self) -> bool:
        """True while the zero-init output has never been updated (G2)."""
        w, b = self.rgb_output.weight, self.rgb_output.bias
        return bool((w == 0).all()) and bool((b == 0).all())


# ------------------------------------------------------------- integration
class UBRDualRGBIntegration(dual.UBRDualIntegration):
    """UBRDualIntegration upgraded in place with the RGB consumption path.

    Installed by class-swap (integration.__class__ = UBRDualRGBIntegration)
    after the dual integration is wired - the remote candidate_detail
    install pattern - so every EXP-0029 wrap is inherited, never duplicated.
    """

    def setup_rgb(self, rgb_content: str) -> None:
        """Register the RGB branch, the block4 hook and the loss/predict wraps."""
        assert rgb_content in RGB_CONTENT_MODES, rgb_content
        model = self.model
        assert self.arm == "real", (
            "the +RGB arm builds on the EXP-0029 real arm (uav_port='real')")
        assert RGB_BRANCH_ATTR not in dict(model.backbone.named_children()), (
            "RGB branch already installed")
        assert hasattr(model.backbone, "checkpoint_path"), (
            "satellite backbone must expose its foundation checkpoint_path")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(RGB_SOURCE_SEED)
            branch = RGBSourceBranch(model.backbone.checkpoint_path)
        setattr(model.backbone, RGB_BRANCH_ATTR, branch)  # registers module
        self.rgb_content = rgb_content
        self.rgb_branch = branch  # plain handle (the module lives on backbone)
        self.anchor_cache_dir = os.environ.get(
            "UBR_RGB_ANCHOR_CACHE", ANCHOR_CACHE_DEFAULT)
        self._anchor_cache = SceneRGBAnchorCache(
            self.anchor_cache_dir, dataset_root())
        if rgb_content == "constant":
            self._load_descriptor_mean()
        # block4 injection site (compact InstanceInteraction.attach, verbatim
        # trunk discovery + site asserts).
        trunks = [m for m in model.backbone.modules()
                  if hasattr(m, "stage_ends") and hasattr(m, "blocks")]
        assert len(trunks) == 1, "expected exactly one Hiera trunk"
        trunk = trunks[0]
        assert list(trunk.stage_ends) == [1, 4, 20, 23]
        assert trunk.blocks[4].dim_out == SAT_BLOCK4_DIM
        trunk.blocks[4].register_forward_hook(self._inject_rgb)
        # staging state + telemetry
        self._rgb_pooled: Optional[torch.Tensor] = None
        self._rgb_support64: Optional[torch.Tensor] = None
        self._rgb_last_stats: Dict[str, float] = {}
        self._rgb_totals = dict(batches=0.0, encoder_secs=0.0, stage_secs=0.0,
                                support_fraction=0.0, candidates_per_cell=0.0,
                                descriptor_rms=0.0, residual_rms=0.0,
                                base_rms=0.0)
        self._wire_rgb()

    # ------------------------------------------------ constant-arm control
    def _load_descriptor_mean(self) -> None:
        """rgb_content='constant': SHA-recorded train-only mean descriptor."""
        import hashlib
        path = os.environ.get("UBR_RGB_DESCRIPTOR_MEAN", "")
        assert path, ("rgb_content='constant' requires UBR_RGB_DESCRIPTOR_MEAN"
                      " pointing at the train-only descriptor-mean json")
        raw = Path(path).read_bytes()
        values = json.loads(raw)["descriptor_mean"]
        assert len(values) == RGB_PROJECT_DIM, len(values)
        self._descriptor_mean = torch.tensor(values, dtype=torch.float32)
        self._descriptor_mean_sha256 = hashlib.sha256(raw).hexdigest()

    # ------------------------------------------------------------- staging
    def _stage_rgb(self, data_samples) -> None:
        """Load + consume this batch's RGB views through the P0 -> 32d path.

        Per scene (b1 protocol): the 5 views pass the source encoder as ONE
        batch (spec section 6 sentinel: 5-view merged forward), descriptors
        are layer-normed per token (compact fuse_rgb), sampled at the FP64
        candidate anchors and mean-reduced per cell (reduce_rgb mean branch).
        The constant control skips the encoder entirely and writes the
        train-only mean descriptor under the same candidate-support mask.
        """
        assert len(data_samples) == 1, (
            f"the +RGB arm stages per scene; got batch {len(data_samples)}")
        ident = data_samples[0].metainfo.get("image_id", None)
        assert ident is not None, "data samples must carry image_id"
        t_stage = time.perf_counter()
        rgb_u8, index_np, anchor_np = self._anchor_cache.load(int(ident))
        model = self.model
        dev = next(model.parameters()).device
        index = torch.from_numpy(index_np).to(dev)
        valid = index >= 0
        support = valid.any(1)
        with torch.set_grad_enabled(model.training):
            if self.rgb_content == "real":
                branch = self.rgb_branch
                rgb = torch.from_numpy(rgb_u8).to(device=dev, dtype=torch.float32)
                t0 = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=dev.type == "cuda"):
                    maps = branch.rgb_project(branch.p0_encoder(
                        (rgb - branch.rgb_mean) / branch.rgb_std))
                encoder_secs = time.perf_counter() - t0
                features = maps.permute(0, 2, 3, 1).reshape(
                    -1, RGB_PROJECT_DIM).float()
                features = F.layer_norm(features, (RGB_PROJECT_DIM,))
                selected = sample_source_continuous(
                    features, index, torch.from_numpy(anchor_np).to(dev))
                # reduce_rgb mean branch (compact fuse_rgb / candidate_detail
                # 'detail' variant), verbatim: equal weight over the retained
                # candidates of the cell, zero where none.
                pooled = ((selected * valid[..., None]).sum(1)
                          / valid.sum(1, keepdim=True).clamp_min(1))
            else:
                encoder_secs = 0.0
                pooled = (self._descriptor_mean[None].to(dev)
                          * support[:, None].float())
        self._rgb_pooled = pooled.T.reshape(1, RGB_PROJECT_DIM, 128, 128)
        # nominal-lattice support map 128 -> 64 (compact interact's
        # max_pool2d support mapping; any candidate in the 2x2 block).
        self._rgb_support64 = F.max_pool2d(
            support.reshape(1, 1, 128, 128).float(), 2, 2).reshape(-1) > 0
        self._record_rgb_stage(int(ident), encoder_secs,
                               time.perf_counter() - t_stage,
                               pooled, support, valid)

    def _record_rgb_stage(self, scene_id: int, encoder_secs: float,
                          stage_secs: float, pooled: torch.Tensor,
                          support: torch.Tensor, valid: torch.Tensor) -> None:
        with torch.no_grad():
            sel = pooled[support]
            stats = dict(
                rgb_scene=float(scene_id),
                rgb_encoder_secs=round(encoder_secs, 4),
                rgb_stage_secs=round(stage_secs, 4),
                rgb_support_fraction=float(support.float().mean()),
                rgb_candidates_per_cell=float(valid.sum(1).float().mean()),
                rgb_descriptor_mean=float(sel.mean()) if sel.numel() else 0.0,
                rgb_descriptor_std=float(sel.std()) if sel.numel() > 1 else 0.0,
                rgb_descriptor_rms=float(sel.square().mean().sqrt())
                if sel.numel() else 0.0,
            )
        self._rgb_last_stats = stats
        t = self._rgb_totals
        t["batches"] += 1
        t["encoder_secs"] += encoder_secs
        t["stage_secs"] += stage_secs
        for key in ("support_fraction", "candidates_per_cell", "descriptor_rms"):
            t[key] += stats[f"rgb_{key}"]

    # ------------------------------------------------------------ injection
    def _inject_rgb(self, _module, _inputs, base: torch.Tensor) -> torch.Tensor:
        """block4 residual hook: base + rgb_output(mean-reduced 32d map).

        Zero-init rgb_output makes the residual exactly +0.0 at construction
        (G2 identity); the support mask keeps candidate-free cells inert.
        Returns base unchanged when nothing is staged (eval-only paths that
        bypass the standard loss/predict wraps; training must always stage).
        """
        if self._rgb_pooled is None:
            assert not self.model.training, (
                "RGB branch: trunk forward without staged context in train "
                "mode - the arm would silently degrade to no-RGB")
            return base
        assert tuple(base.shape) == (1, 64, 64, SAT_BLOCK4_DIM), tuple(base.shape)
        # nominal CNN lattice mapping 128 -> 64 (compact interact's
        # avg_pool2d(u128, 2, 2); Hiera's first 2x2 Q-pool shares it).
        pooled64 = F.avg_pool2d(self._rgb_pooled, 2, 2)
        cells = pooled64.permute(0, 2, 3, 1).reshape(-1, RGB_PROJECT_DIM)
        delta = self.rgb_branch.rgb_output(cells)
        delta = delta * self._rgb_support64.reshape(-1, 1).to(delta.dtype)
        out = base + delta.reshape(1, 64, 64, SAT_BLOCK4_DIM).to(base.dtype)
        with torch.no_grad():
            self._rgb_last_stats.update(
                rgb_residual_rms=float(
                    delta.detach().float().square().mean().sqrt()),
                rgb_base_rms=float(
                    base.detach().float().square().mean().sqrt()))
            t = self._rgb_totals
            t["residual_rms"] += self._rgb_last_stats["rgb_residual_rms"]
            t["base_rms"] += self._rgb_last_stats["rgb_base_rms"]
        return out

    # ------------------------------------------------------------ telemetry
    def drain_rgb_stats(self) -> Dict[str, float]:
        """Per-train-phase RGB means (runner epoch telemetry; spec section 6)."""
        t = self._rgb_totals
        n = max(1.0, t["batches"])
        out = dict(
            batches=t["batches"],
            encoder_secs_mean=round(t["encoder_secs"] / n, 4),
            stage_secs_mean=round(t["stage_secs"] / n, 4),
            support_fraction_mean=t["support_fraction"] / n,
            candidates_per_cell_mean=t["candidates_per_cell"] / n,
            descriptor_rms_mean=t["descriptor_rms"] / n,
            residual_rms_mean=t["residual_rms"] / n,
            base_rms_mean=t["base_rms"] / n,
            injection_active=float(not self.rgb_branch.injection_is_zero),
        )
        self._rgb_totals = dict(batches=0.0, encoder_secs=0.0, stage_secs=0.0,
                                support_fraction=0.0, candidates_per_cell=0.0,
                                descriptor_rms=0.0, residual_rms=0.0,
                                base_rms=0.0)
        return out

    def rgb_last_stats(self) -> Dict[str, float]:
        """Latest staged scene's RGB telemetry (val-phase record)."""
        return dict(self._rgb_last_stats)

    # -------------------------------------------------------------- wire-up
    def _wire_rgb(self) -> None:
        """Wrap model.loss/model.predict to stage RGB before extract_feat.

        The parent wraps live deeper (roi_head.mask_loss, model.predict via
        v1.1); these outer wraps only guarantee the RGB context exists when
        the trunk's block4 hook fires inside extract_feat.
        """
        model = self.model
        integration = self
        orig_loss = model.loss
        orig_predict = model.predict

        def loss(*args, **kwargs):
            integration._stage_rgb(
                args[1] if len(args) > 1 else kwargs["batch_data_samples"])
            return orig_loss(*args, **kwargs)

        def predict(*args, **kwargs):
            integration._stage_rgb(
                args[1] if len(args) > 1 else kwargs["batch_data_samples"])
            return orig_predict(*args, **kwargs)

        model.loss = loss
        model.predict = predict


# ---------------------------------------------------------------- install
def validate_dual_rgb_config(ubr_cfg: dict) -> dict:
    """Fail-closed validation of the +RGB config's ubr_integration_cfg.

    The block must be the dual prereg block plus exactly {rgb_content},
    with uav_port pinned to 'real' (this line is EXP-0029(real) + RGB; the
    geometry-port constant arm is EXP-0029 arm 3, not this config).
    """
    cfg = dict(ubr_cfg or {})
    expected = (set(v11.EXPECTED_UBR_CFG_KEYS) | dual.DUAL_CFG_EXTRA_KEYS
                | RGB_CFG_EXTRA_KEYS)
    assert set(cfg) == expected, (
        f"dual+rgb ubr_integration_cfg keys {sorted(cfg)} != expected "
        f"{sorted(expected)}")
    assert cfg["uav_port"] == "real", (
        f"the +RGB arm requires uav_port='real', got {cfg['uav_port']!r}")
    assert cfg["rgb_content"] in RGB_CONTENT_MODES, cfg["rgb_content"]
    assert cfg["geometry_loss_weight"] == dual.GEOMETRY_LOSS_WEIGHT, (
        f"geometry_loss_weight must stay the prereg {dual.GEOMETRY_LOSS_WEIGHT}")
    base = {k: v for k, v in cfg.items()
            if k != "geometry_loss_weight" and k not in RGB_CFG_EXTRA_KEYS}
    base["uav_port"] = "zero"
    v11.validate_ubr_config(base)  # every v1.1 pin, unchanged
    return cfg


def apply_dual_rgb_upgrade(model, rgb_content: str) -> UBRDualRGBIntegration:
    """Upgrade a wired UBRDualIntegration model in place; returns it."""
    integ = getattr(model, "ubr_integration", None)
    assert isinstance(integ, dual.UBRDualIntegration), (
        "the +RGB arm requires the EXP-0029 dual integration to be wired "
        "(install_dual_class_patch) before the RGB upgrade")
    assert not isinstance(integ, UBRDualRGBIntegration), "already upgraded"
    integ.__class__ = UBRDualRGBIntegration
    integ.setup_rgb(rgb_content)
    return integ


def install_dual_rgb_class_patch(ubr_cfg: dict,
                                 rgb_content: Optional[str] = None) -> None:
    """Patch RSPrompterAnchor so MODELS.build wires the dual+RGB arm.

    Idempotent; chains the dual class patch (which the gates may already
    have installed) and upgrades every built model in place - the sealed
    trainer's MODELS.build needs no change.
    """
    cfg = validate_dual_rgb_config(ubr_cfg)
    content = rgb_content or os.environ.get(
        "UBR_RGB_CONTENT") or cfg["rgb_content"]
    assert content in RGB_CONTENT_MODES, content
    dual.install_dual_class_patch(
        {k: v for k, v in cfg.items() if k not in RGB_CFG_EXTRA_KEYS})
    from ..detection.anchor_detector import RSPrompterAnchor
    if getattr(RSPrompterAnchor, "_ubr_dual_rgb_patched", False):
        return
    current_init = RSPrompterAnchor.__init__  # sealed or dual-patched

    def patched_init(self, *args, **kwargs):
        current_init(self, *args, **kwargs)
        apply_dual_rgb_upgrade(self, content)

    RSPrompterAnchor.__init__ = patched_init
    RSPrompterAnchor._ubr_dual_rgb_patched = True
