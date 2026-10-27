"""Level-4 dual-stream UBR arms: geometry head + live UAV port-5 wiring.

Extends the v1.1 single-stream install (rsprompter/uav/
ubr_train_integration.py, EXP-0028-R2) for the preregistered level-4 arms
(docs/UBR-RESEAL-20261027_preregistration.md, section "级4 双流臂预注册";
architecture docs/UBR-UNIFIED-ARCHITECTURE-DESIGN-20261026.md §2/§3):

  arm REAL     - port 5 of every UnifiedBoundaryRefiner call receives the
                 real native-512 geometry map (4 fp16 fields + support) of
                 the batch scene, cropped per window.
  arm CONSTANT - identical, except port 5 receives
                 ubr.build_constant_geometry(real, train-only mean_fields):
                 per-channel means where support == 1, zeros in holes; the
                 support channel distribution matches the real arm exactly
                 (the content-gate control).

  Both arms additionally own a GeometryProposalHead — the proven pilot
  network (scripts/probes/ue_geometry_proposal_pilot_20261024.py; network,
  center-box loss and decode ported bit-exactly) — trained by loss_geometry
  (weight 1.0, design §3.3 lambda_geo) with DIRECT instance supervision
  from the satellite GT boxes of the sealed batch. The head input is the
  REAL geometry map in BOTH arms; only the UBR port-5 content differs
  (arm diff collapses to port 5 alone, per the prereg wording).

Firewall FW1: geometry candidates NEVER enter any detector loss. The head's
training-time forward is consumed only by loss_geometry (GT-assembled
targets); its decode output exists only inside the eval-path predict wrap
under torch.no_grad. Verified structurally (decode is unreachable from the
loss path) and by the dual gates (zero decode calls during a train batch).

Training-time data flow (nothing in the sealed trainer changes):
  - roi_head.mask_loss receives batch_img_metas (the sealed roi head passes
    sample.metainfo); the dual wrap reads metainfo['image_id'], loads the
    scene's native-512 geometry ONCE per batch in-process (LRU cache; the
    npz cache is a fixed per-scene artifact, so it is structurally exempt
    from image augmentation - and the 3090 protocol has zero geometric
    augmentation anyway), and stages it for both consumers:
      a) loss_geometry: full 512^2 map -> GeometryProposalHead.
      b) loss_ubr: the v1.1 bucketed window crops get 5 extra input
         channels (port 5), real or constant per arm.

Inference (val) - the frozen-interface dual pipeline
(scripts/probes/ue_joint_eval_20261024.py merge contract, bit-exact):
  1. satellite prefix: the model's standard predict path (RPN -> RoI ->
     prompts -> SAM2 mask decode), scores remapped to the trusted band
     r(s) = 0.5 + 0.5/(1 - log s) (> 0.5, strictly increasing);
  2. geometry-head candidates decoded on the real map (top-250 local heat
     peaks under known support), dedup vs prefix at IoU <= nms.iou_threshold
     + NMS among candidates, capped to 250 - len(prefix), decoded through
     the model's own mask path (roi_head.predict_mask), scored 0.25 * s
     (<= 0.25; the two bands never cross);
  3. UBR band reclassification on EVERY decoded instance (prefix + new),
     with port-5 crops (the v1.1 refine path extended by port 5; chunked
     forward is numerically identical - see UBRDualIntegration.refine_...).

Asserts (fail-closed): dual config keys exact; geometry cache version and
constants SHA pinned; candidate total <= 250; band bounds (0, 0.25] /
(0.5, 1]; paste parity; refined pixels only inside the |SDT| <= k band;
losses finite.
"""
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from . import ubr
from . import ubr_train_integration as v11

# ---------------------------------------------------------------- constants
# Level-4 prereg keys (fail-closed; the config must carry exactly these).
DUAL_PORT_MODES = ("real", "constant")  # uav_port values of the dual arms
GEOMETRY_LOSS_WEIGHT = 1.0  # lambda_geo, design doc 20261026 §3.3 (remote-
# verified pilot value; the local 0.1 line is NOT introduced)
GEOMETRY_HEAD_PARAMS = 202581  # pilot Net param count (torch-measured)
HEAT_BIAS_INIT = -2.19  # pilot common init: prior odds of a dull head
GEOM_GRID = 128  # center heatmap grid (512 / 4)
CANDIDATE_CAP = 250  # frozen interface: max instances per image
CANDIDATE_BAND = 0.25  # candidate scores = 0.25 * s (never cross r(s) > .5)
NMS_DEDUP_KEY = "iou_threshold"
REFINE_CHUNK = 32  # eval refine forward chunk (numerically neutral; the
# chunk groups are window-area sorted, A1-style, so padding stays local)

# Native geometry cache (the level-2 probe's cache, host-pinned defaults;
# env overrides exist for the lthpc host where roots differ).
GEOM_CACHE_DEFAULT = "/data2/wangcheng/remote-lthpc/cache/ue_v1_native_geometry"
GEOM_CONSTANTS_DEFAULT = (
    "/data2/wangcheng/remote-lthpc/checkpoint/"
    "ue_geometry_mechanism_20261022/common_constants.json")
GEOM_VERSION = "native512_height_dx_dy_viewcount_v1"
CONSTANTS_SHA256 = (
    "761d9795282c01ec5af1077b6d03cf49401b6e59021a7b9a20c4215628a5a289")

DUAL_CFG_EXTRA_KEYS = {"geometry_loss_weight"}
# Pilot loss constants (ue_geometry_proposal_pilot_20261024.py protocol.json):
# "smoothL1 offset + .1 log-wh; focal center normalized by unique cells".
SIZE_LOSS_WEIGHT = 0.1
FOCAL_GAMMA_P = 2.0  # pilot focal powers: (1-p)^2 for positives
PROB_CLAMP = 1e-4


def geometry_head_param_count() -> int:
    """Pure arithmetic source for the GEOMETRY_HEAD_PARAMS pin."""
    def block_params(cin: int, cout: int) -> int:
        conv1 = cin * cout * 9 + cout
        conv2 = cout * cout * 9 + cout
        return conv1 + conv2 + 2 * (2 * cout)  # two GroupNorm(4) pairs

    total = (block_params(5, 16) + block_params(16, 32)
             + block_params(32, 64) + block_params(64, 64)
             + block_params(160, 32))
    return total + (32 * 1 + 1) + (32 * 4 + 4)  # heat + reg 1x1 convs


# ------------------------------------------------------------ geometry head
class GeometryProposalHead(nn.Module):
    """Pilot proposal network (20261024), production port - bit-exact.

    Layer-for-layer identical to scripts/probes/ue_geometry_proposal_
    20261024.py::Net (4 stride-2 blocks + FPN-ish fuse + heat/reg heads,
    GroupNorm(4)+GELU, heat bias -2.19), so the pilot's common-init
    fingerprint semantics and its proven learnability carry over. Input
    [B,5,512,512] (fields + support); output (heat [B,1,128,128],
    reg [B,4,128,128]).
    """

    def __init__(self):
        super().__init__()
        self.stem = self._block(5, 16, 2)
        self.stage2 = self._block(16, 32, 2)
        self.stage3 = self._block(32, 64, 2)
        self.stage4 = self._block(64, 64, 2)
        self.fuse = self._block(160, 32, 1)
        self.heat = nn.Conv2d(32, 1, 1)
        self.reg = nn.Conv2d(32, 4, 1)
        nn.init.constant_(self.heat.bias, HEAT_BIAS_INIT)

    @staticmethod
    def _block(cin: int, cout: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, padding=1),
            nn.GroupNorm(4, cout), nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(4, cout), nn.GELU())

    def forward(self, x: torch.Tensor):
        b = self.stage2(self.stem(x))
        c = self.stage3(b)
        d = self.stage4(c)
        up = [b]
        for feat in (c, d):
            up.append(nn.functional.interpolate(
                feat, size=b.shape[-2:], mode="bilinear", align_corners=False))
        z = self.fuse(torch.cat(up, 1))
        return self.heat(z), self.reg(z)


# --------------------------------------------------------- data-side helpers
class SceneGeometryCache:
    """Per-process LRU over the native-512 geometry npz cache."""

    def __init__(self, cache_dir: str, capacity: int = 8):
        self.dir = Path(cache_dir)
        self.capacity = int(capacity)
        self._pool: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def load(self, scene_id: int) -> np.ndarray:
        """Returns float32 [5,512,512] (fields + support). Fail-closed."""
        key = int(scene_id)
        if key not in self._pool:
            with np.load(self.dir / f"{key:06d}.npz") as z:
                assert str(z["version"]) == GEOM_VERSION, f"geom ver {key}"
                fields = z["fields"].astype(np.float32)
                support = z["support"].astype(np.float32)
            assert fields.shape == (4, 512, 512) and support.shape == (512, 512)
            self._pool[key] = np.concatenate([fields, support[None]])
            self._pool.move_to_end(key)
            while len(self._pool) > self.capacity:
                self._pool.popitem(last=False)
        return self._pool[key]


def build_center_targets(boxes: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT -> (heat, reg, valid) on the 128 grid, pilot dataset logic.

    Verbatim semantics of the pilot's Data.__getitem__: Gaussian instance
    centers with sigma = clip(min(wh_grid)/6, 1, 4) written as running max;
    a shared center cell keeps the LAST (smallest-area) instance's
    regression; valid marks occupied center cells (the focal/offset
    denominator). boxes: [N,4] float (x1, y1, x2, y2) on the 512 grid.
    """
    heat = np.zeros((1, GEOM_GRID, GEOM_GRID), np.float32)
    reg = np.zeros((4, GEOM_GRID, GEOM_GRID), np.float32)
    valid = np.zeros((1, GEOM_GRID, GEOM_GRID), np.float32)
    if boxes.shape[0] == 0:
        return heat, reg, valid
    wh_all = boxes[:, 2:4] - boxes[:, 0:2]
    # descending area, stable: the smallest box wins a shared center cell
    order = np.argsort(-np.prod(wh_all, axis=1), kind="stable")
    for box in boxes[order]:
        wh = (box[2:4] - box[0:2]) / 4
        center = (box[0:2] + box[2:4]) / 8
        ix, iy = np.clip(np.floor(center).astype(int), 0, GEOM_GRID - 1)
        sigma = float(np.clip(min(wh) / 6, 1, 4))
        radius = int(math.ceil(3 * sigma))
        x0, x1 = max(0, ix - radius), min(GEOM_GRID, ix + radius + 1)
        y0, y1 = max(0, iy - radius), min(GEOM_GRID, iy + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gauss = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma ** 2))
        heat[0, y0:y1, x0:x1] = np.maximum(heat[0, y0:y1, x0:x1], gauss)
        reg[:2, iy, ix] = center - [ix, iy]
        reg[2:, iy, ix] = np.log(np.maximum(wh, 1e-3))
        valid[0, iy, ix] = 1
    return heat, reg, valid


def center_box_loss(heat: torch.Tensor, reg: torch.Tensor,
                    heat_t: torch.Tensor, reg_t: torch.Tensor,
                    valid: torch.Tensor
                    ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Pilot loss_fn: focal heat + smoothL1 offset + 0.1 * log-wh.

    fp32 loss from the (possibly fp16 autocast) head outputs, exactly as
    the pilot computed it; normalized by unique center cells. heat/reg are
    the head outputs [1,1,128,128] / [1,4,128,128]; *_t are the matching
    target tensors (same 4-D shapes).
    """
    assert heat.shape == heat_t.shape and reg.shape == reg_t.shape, (
        heat.shape, heat_t.shape, reg.shape, reg_t.shape)
    assert valid.shape == heat_t.shape, (valid.shape, heat_t.shape)
    prob = heat.float().sigmoid().clamp(PROB_CLAMP, 1 - PROB_CLAMP)
    positive = (heat_t == 1).float()
    negative = (heat_t < 1).float()
    denom = valid.sum().clamp_min(1)
    focal = -(prob.log() * (1 - prob) ** FOCAL_GAMMA_P * positive
              + (1 - prob).log() * prob ** FOCAL_GAMMA_P
              * (1 - heat_t) ** 4 * negative).sum() / denom
    error = nn.functional.smooth_l1_loss(
        reg.float(), reg_t, reduction="none") * valid
    offset = error[:, :2].sum() / denom
    size = error[:, 2:].sum() / denom
    loss = focal + offset + SIZE_LOSS_WEIGHT * size
    return loss, dict(focal=float(focal.detach()), offset=float(offset.detach()),
                      size=float(size.detach()))


@torch.no_grad()
def decode_candidates(heat: torch.Tensor, reg: torch.Tensor,
                      support: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pilot decode: top-250 local heat peaks under known support.

    heat [1,1,128,128], reg [1,4,128,128], support [1,1,512,512] (REAL
    support in both arms, as in the pilot). Returns (boxes [N,4] on the
    512 grid, scores [N]); N <= 250, score order descending.
    """
    score = heat.float().sigmoid()
    local = score == torch.nn.functional.max_pool2d(score, 3, 1, 1)
    # any native support within the corresponding local 12x12 neighborhood
    known = torch.nn.functional.max_pool2d(
        torch.nn.functional.max_pool2d(support.float(), 4, 4), 3, 1, 1) > 0
    score = score.masked_fill(~local | ~known, -1)
    vals, ids = score.flatten().topk(CANDIDATE_CAP)
    keep = vals >= 0
    vals, ids = vals[keep], ids[keep]
    r = reg[0].reshape(4, -1)[:, ids].T.float()
    center = torch.stack((ids % GEOM_GRID, ids // GEOM_GRID), 1).float()
    center = center + r[:, :2].clamp(0, 1)
    wh = r[:, 2:].clamp(math.log(.25), math.log(GEOM_GRID)).exp()
    boxes = torch.cat((center - wh / 2, center + wh / 2), 1).mul(4)
    return boxes.clamp(0, 512), vals


def prefix_band_score(scores: torch.Tensor) -> torch.Tensor:
    """Trusted band r(s) = 0.5 + 0.5/(1 - log s); s <= 0 maps to 0.5.

    Bit-exact port of ue_geometry_conditional_complement_20261024::
    trusted_score (the frozen scoring interface): strictly increasing for
    every positive float32 score, every mapped value > 0.5.
    """
    x = scores.detach().double().cpu().numpy()
    out = np.full_like(x, 0.5)
    pos = x > 0
    out[pos] = 0.5 + 0.5 / (1 - np.log(x[pos]))
    return torch.from_numpy(out.astype(np.float32)).to(scores.device)


def select_candidates(prefix_boxes: torch.Tensor, cand_boxes: torch.Tensor,
                      cand_scores: torch.Tensor, test_cfg: dict
                      ) -> torch.Tensor:
    """Frozen-interface candidate admission (ue_joint_eval hypotheses).

    Validity filter -> dedup vs prefix (IoU <= nms.iou_threshold) -> NMS
    among candidates (cfg.nms) -> cap 250 - len(prefix). Returns the kept
    candidate indices in descending score order.
    """
    from mmdet.structures.bbox import bbox_overlaps
    from mmcv.ops import batched_nms

    nms_iou = float(test_cfg["nms"][NMS_DEDUP_KEY])
    n_pref = int(prefix_boxes.shape[0])
    if cand_boxes.shape[0] == 0 or n_pref >= CANDIDATE_CAP:
        return torch.zeros(0, dtype=torch.long, device=cand_boxes.device)
    valid = (torch.isfinite(cand_boxes).all(1)
             & (cand_boxes[:, 2:] > cand_boxes[:, :2]).all(1)
             & torch.isfinite(cand_scores) & (cand_scores > 0))
    if n_pref:
        valid &= bbox_overlaps(cand_boxes, prefix_boxes).max(1).values <= nms_iou
    cand_boxes, cand_scores = cand_boxes[valid], cand_scores[valid]
    if cand_boxes.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=cand_boxes.device)
    _, keep = batched_nms(
        cand_boxes.float(), cand_scores.float(),
        torch.zeros(len(cand_boxes), device=cand_boxes.device,
                    dtype=torch.long), test_cfg["nms"])
    keep = keep[:max(0, CANDIDATE_CAP - n_pref)]  # score-descending order
    src = torch.nonzero(valid, as_tuple=False).flatten()  # pre-filter ids
    return src[keep]


# --------------------------------------------------------------- integration
class UBRDualIntegration(v11.UBRIntegration):
    """v1.1 single-stream install extended by port 5 + geometry head."""

    def __init__(self, model, arm: str, geometry_loss_weight: float, **kwargs):
        super().__init__(model, **kwargs)
        assert arm in DUAL_PORT_MODES, f"unknown dual arm {arm!r}"
        assert float(geometry_loss_weight) == GEOMETRY_LOSS_WEIGHT, (
            f"geometry_loss_weight {geometry_loss_weight} != prereg "
            f"{GEOMETRY_LOSS_WEIGHT}")
        self.arm = arm
        self.geometry_loss_weight = float(geometry_loss_weight)
        self.geom_cache_dir = os.environ.get(
            "UBR_GEOM_CACHE", GEOM_CACHE_DEFAULT)
        self.constants_path = os.environ.get(
            "UBR_GEOM_CONSTANTS", GEOM_CONSTANTS_DEFAULT)
        self._cache = SceneGeometryCache(self.geom_cache_dir)
        self._channel_means = self._load_channel_means()
        # Submodule registration (same contract as ubr_refiner): device
        # moves, optimizer groups, the DDP wrap and checkpoints include it.
        assert "geometry_head" not in dict(model.named_children())
        model.geometry_head = GeometryProposalHead()
        n_par = sum(p.numel() for p in model.geometry_head.parameters())
        assert n_par == GEOMETRY_HEAD_PARAMS == geometry_head_param_count(), (
            f"geometry head params {n_par} != pin {GEOMETRY_HEAD_PARAMS}")
        for p in model.geometry_head.parameters():
            p.requires_grad_(True)
        # per-batch staged state: scene ids, port maps (numpy for the train
        # bucket crops, device tensors for the eval refine path) and the
        # per-val-pass telemetry accumulators.
        self._scene_ids: List[int] = []
        self._port_np: List[np.ndarray] = []
        self._val_geom: List[torch.Tensor] = []
        self._train_stats_geo: Dict[str, float] = {}
        self._geo_totals = dict(batches=0.0, focal=0.0, offset=0.0,
                                size=0.0, images=0.0, instances=0.0)
        self._val_totals = dict(images=0.0, prefix=0.0, candidates=0.0,
                                accepted=0.0, prefix_score_sum=0.0)

    def _load_channel_means(self) -> torch.Tensor:
        """Train-only mean_fields[:4], SHA-pinned (constant arm control)."""
        import hashlib
        import json
        raw = Path(self.constants_path).read_bytes()
        got = hashlib.sha256(raw).hexdigest()
        assert got == CONSTANTS_SHA256, f"constants sha drift {got}"
        vals = json.loads(raw)["mean_fields"]
        assert len(vals) == 4, vals
        return torch.tensor(vals[:4], dtype=torch.float32)

    # ------------------------------------------------------------- staging
    def scene_real(self, scene_id: int, device) -> torch.Tensor:
        """Real [5,512,512] map on device (fp32, no grad: cached artifact)."""
        return torch.from_numpy(self._cache.load(scene_id)).to(device)

    def _stage(self, img_metas: List[dict]) -> None:
        """Load + stage this batch's port maps (ids from metainfo)."""
        self._scene_ids = []
        for meta in img_metas:
            ident = meta.get("image_id", None)
            assert ident is not None, "img metas must carry image_id"
            self._scene_ids.append(int(ident))
        self._port_np = []
        self._val_geom = []
        for i in self._scene_ids:
            real = self._cache.load(i)  # fp32 [5,512,512], cached artifact
            port = real if self.arm == "real" else ubr.build_constant_geometry(
                torch.from_numpy(real)[None], self._channel_means)[0].numpy()
            self._port_np.append(np.ascontiguousarray(port))

    # -------------------------------------------------- train: loss path
    def compute_loss_geometry(self, batch_gt_instances) -> torch.Tensor:
        """loss_geometry: direct center-box supervision from satellite GT.

        Per image: full-512 real map -> head -> pilot center loss; the
        batch value is the per-image mean (at the b1 protocol it is the
        single image's loss, matching the pilot's per-batch normalization).
        Empty-GT images contribute a graph-connected zero so every DDP rank
        and every head parameter stays in the backward graph.
        """
        model = self.model
        dev = next(model.parameters()).device
        zero = sum(p.sum() for p in model.geometry_head.parameters()) * 0.0
        assert self._scene_ids and len(self._scene_ids) == len(batch_gt_instances), (
            "compute_loss_geometry called without staged geometry")
        losses, parts = [], dict(focal=0.0, offset=0.0, size=0.0)
        n_inst = 0
        for i, gt_instances in enumerate(batch_gt_instances):
            boxes = gt_instances.bboxes.detach().cpu().numpy().reshape(-1, 4)
            n_inst += boxes.shape[0]
            if boxes.shape[0] == 0:
                losses.append(zero)
                continue
            heat_t, reg_t, valid = build_center_targets(boxes)
            real = torch.from_numpy(self._cache.load(self._scene_ids[i])).to(dev)
            heat, reg = model.geometry_head(real[None])  # head sees REAL map
            loss_i, part_i = center_box_loss(
                heat, reg,
                torch.from_numpy(heat_t)[None].to(dev),
                torch.from_numpy(reg_t)[None].to(dev),
                torch.from_numpy(valid)[None].to(dev))
            losses.append(loss_i)
            for k in parts:
                parts[k] += part_i[k]
        n_img = len(losses)
        loss = self.geometry_loss_weight * torch.stack(losses).mean()
        assert bool(torch.isfinite(loss)), "non-finite loss_geometry"
        self._train_stats_geo = {
            "GEO/focal": parts["focal"] / n_img,
            "GEO/offset": parts["offset"] / n_img,
            "GEO/size": parts["size"] / n_img,
            "GEO/images": float(n_img),
            "GEO/instances": float(n_inst),
        }
        t = self._geo_totals  # per-epoch accumulation (runner telemetry)
        t["batches"] += 1
        t["focal"] += self._train_stats_geo["GEO/focal"]
        t["offset"] += self._train_stats_geo["GEO/offset"]
        t["size"] += self._train_stats_geo["GEO/size"]
        t["images"] += n_img
        t["instances"] += n_inst
        return loss

    def drain_geo_stats(self) -> Dict[str, float]:
        """Per-train-phase geometry means (called by the epoch telemetry)."""
        t = self._geo_totals
        n = max(1.0, t["batches"])
        out = dict(
            batches=t["batches"],
            loss_geometry_mean=(
                t["focal"] + t["offset"] + SIZE_LOSS_WEIGHT * t["size"]) / n,
            focal_mean=t["focal"] / n,
            offset_mean=t["offset"] / n,
            size_mean=t["size"] / n,
            instances_per_img=t["instances"] / max(1.0, t["images"]),
        )
        self._geo_totals = dict(batches=0.0, focal=0.0, offset=0.0,
                                size=0.0, images=0.0, instances=0.0)
        return out

    # ------------------------------------------- train: port-5 bucket crops
    def _bucket_step(self, idx, wins, fwins, gt_np, nbr_full, img_of, gt_of,
                     pred_bin, up, probs512, n_pos, dev
                     ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
                                int, torch.Tensor]:
        """v1.1 bucket step + port-5 geometry crops (the only delta).

        Identical to ubr_train_integration.UBRIntegration._bucket_step
        except: per-window crops of the staged port map are padded into
        the bucket and passed as the refiner's uav_geom argument (real or
        constant arm content). Keep in sync with the v1.1 parent.
        """
        n = int(idx.size)
        h_max = int(wins[idx, 2].max())
        w_max = int(wins[idx, 3].max())
        half = torch.float16 if dev.type == "cuda" else torch.float32
        crops = torch.zeros((n, 1, h_max, w_max), dtype=torch.bool,
                            device=pred_bin.device)
        gt_pad = np.zeros((n, 1, h_max, w_max), dtype=np.float32)
        nbr_pad = np.zeros((n, 1, h_max, w_max), dtype=np.float32)
        geom_pad = np.zeros((n, ubr.UAV_GEOMETRY_CHANNELS, h_max, w_max),
                            dtype=np.float32)
        inside = np.zeros((n, h_max, w_max), dtype=bool)
        for slot, j in enumerate(idx):
            h0, w0, h, w = wins[j]
            crops[slot, 0, :h, :w] = pred_bin[j, h0:h0 + h, w0:w0 + w]
            sl = (slice(int(h0), int(h0) + int(h)),
                  slice(int(w0), int(w0) + int(w)))
            own = gt_np[img_of[j]][gt_of[j]][sl]
            gt_pad[slot, 0, :h, :w] = own
            nbr_pad[slot, 0, :h, :w] = nbr_full[img_of[j]][sl] - own
            geom_pad[slot, :, :h, :w] = self._port_np[
                int(img_of[j])][:, sl[0], sl[1]]
            inside[slot, :h, :w] = True
        sdt32 = ubr.signed_distance_transform_numpy(crops.cpu().numpy()[:, 0])
        band_np = (np.abs(sdt32) <= float(self.band_k)) & inside
        n_empty = int((~band_np.any(axis=(1, 2))).sum())
        sdt_t = v11._to_dev(sdt32, dev, half).unsqueeze(1)
        band_t = torch.from_numpy(band_np).unsqueeze(1)
        gt_t = v11._to_dev(gt_pad, dev, torch.float32)
        nbr_t = v11._to_dev(nbr_pad, dev, half)
        geom_t = v11._to_dev(geom_pad, dev, half)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=dev.type == "cuda"):
            feats = []
            logits = []
            for j in idx:  # per-window bilinear keeps exact v1.0 sampling
                fh0, fw0, fh, fw = fwins[j]
                h0, w0, h, w = wins[j]
                feats.append(nn.functional.interpolate(
                    up[j:j + 1, :, fh0:fh0 + fh, fw0:fw0 + fw].to(half),
                    size=(int(h), int(w)), mode="bilinear", align_corners=False))
                logits.append(probs512[j:j + 1, h0:h0 + h, w0:w0 + w].to(half))
            out = self.model.ubr_refiner(
                v11._pad_batch(feats, 32), v11._pad_batch(logits, 1),
                sdt_t, nbr_t, geom_t)
        assert tuple(out.shape) == (n, 1, h_max, w_max), tuple(out.shape)
        assert bool(torch.isfinite(out.float()).all()), "non-finite UBR logits"
        focals, dices = v11.band_loss_per_instance(
            out, gt_t, band_t, self.focal_gamma)
        loss_b = self.loss_weight * (sum(focals) + sum(dices)) / float(n_pos)
        return focals, dices, n_empty, loss_b

    # ------------------------------------------------- eval: dual predict
    def _merge_geometry_candidates(self, roi_head, x, batch_data_samples,
                                   results_list, rescale, image_embeddings,
                                   image_positional_embeddings,
                                   high_res_features):
        """Append geometry-candidate instances to each image's prefix."""
        from mmengine.structures import InstanceData
        model = self.model
        dev = next(model.parameters()).device
        x_pe = roi_head._inject_extra_pe(x)  # once, exactly as predict does
        test_cfg = dict(roi_head.test_cfg)
        merged = []
        for i, res in enumerate(results_list):
            meta = batch_data_samples[i].metainfo
            real = self.scene_real(self._scene_ids[i], dev)
            self._val_geom.append(
                torch.from_numpy(self._port_np[i]).to(dev))
            with torch.no_grad():
                heat, reg = model.geometry_head(real[None])
                cand_boxes, cand_scores = decode_candidates(
                    heat, reg, real[4:5][None])
            keep = select_candidates(
                res.bboxes, cand_boxes, cand_scores, test_cfg)
            cand_boxes, cand_scores = cand_boxes[keep], cand_scores[keep]
            n_pref = int(res.bboxes.shape[0])
            assert n_pref + cand_boxes.shape[0] <= CANDIDATE_CAP
            if cand_boxes.shape[0]:
                inst = InstanceData(
                    bboxes=cand_boxes,
                    scores=cand_scores.float(),
                    labels=torch.zeros(len(cand_boxes), device=dev,
                                       dtype=torch.long))
                with torch.no_grad():
                    cand = roi_head.predict_mask(
                        x_pe, [meta], [inst], rescale=rescale,
                        image_embeddings=image_embeddings,
                        image_positional_embeddings=image_positional_embeddings,
                        high_res_features=high_res_features)[0]
                res = self._concat_instances(res, cand, cand_scores)
            self._tally_val(n_pref, res.scores[:n_pref], int(keep.numel()))
            merged.append(res)
        return merged

    def _concat_instances(self, prefix, cand, cand_scores):
        """Prefix + candidates with frozen-band scores r(s) / 0.25*s."""
        from mmengine.structures import InstanceData
        allowed = {"bboxes", "scores", "labels", "masks"}
        assert set(prefix.keys()) <= allowed, sorted(set(prefix.keys()))
        n_pref = int(prefix.bboxes.shape[0])
        n_cand = int(cand.bboxes.shape[0])
        out = InstanceData()
        out.bboxes = torch.cat([prefix.bboxes, cand.bboxes])
        out.labels = torch.cat([prefix.labels, cand.labels])
        cand_band = CANDIDATE_BAND * cand_scores.float().to(
            prefix.scores.dtype)
        out.scores = torch.cat(
            [prefix_band_score(prefix.scores), cand_band])
        assert bool((out.scores[n_pref:] > 0).all())
        assert bool((out.scores[n_pref:] <= CANDIDATE_BAND + 1e-6).all()), (
            "candidate scores left the 0.25 band")
        if n_pref:
            assert bool((out.scores[:n_pref] > 0.5).all()), (
                "prefix scores left the r(s) band")
        out.masks = torch.cat(
            [prefix.masks, cand.masks]) if n_pref else cand.masks
        assert out.bboxes.shape[0] == n_pref + n_cand
        return out

    def _tally_val(self, n_pref: int, prefix_scores: torch.Tensor,
                   n_accepted: int) -> None:
        t = self._val_totals
        t["images"] += 1
        t["prefix"] += n_pref
        t["candidates"] += float(CANDIDATE_CAP)  # decode top-k before dedup
        t["accepted"] += n_accepted
        t["prefix_score_sum"] += float(prefix_scores.double().sum())

    def drain_val_stats(self) -> Dict[str, float]:
        """Per-val-pass means (called by the runner's epoch telemetry)."""
        t = self._val_totals
        n = max(1.0, t["images"])
        out = dict(
            images=t["images"],
            prefix_mean=t["prefix"] / n,
            candidates_mean=t["candidates"] / n,
            accepted_mean=t["accepted"] / n,
            prefix_score_mean=t["prefix_score_sum"] / max(1.0, t["prefix"]),
        )
        self._val_totals = dict(images=0.0, prefix=0.0, candidates=0.0,
                                accepted=0.0, prefix_score_sum=0.0)
        return out

    def geo_train_stats(self) -> Dict[str, object]:
        return {k: torch.tensor(v) for k, v in self._train_stats_geo.items()}

    # ------------------------------------------------ eval: refine + port5
    @torch.no_grad()
    def refine_predictions(self, batch_data_samples) -> None:
        """UBR band reclassification on every instance, port 5 live.

        Same contract as the v1.1 parent refine (paste parity asserted per
        instance; refined pixels may differ only inside the |SDT| <= k band)
        with two dual deltas: (1) the port-5 crop of this image's staged
        map is concatenated per instance; (2) the refiner forward is
        chunked (REFINE_CHUNK) - a numerically neutral execution choice:
        per-instance windows are independent, padding beyond a window edge
        cannot reach in-band outputs (receptive-field radius depth+1 = 7px
        < window_margin - band_k = 24px), the same argument the v1.1 A1
        bucketing rests on.
        """
        from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
        from mmdet.structures.mask import BitmapMasks

        model = self.model
        thr = float(model.test_cfg["rcnn"]["mask_thr_binary"])
        z, up = self._collect()
        self._release()  # free the per-chunk originals; keep the cat'd views
        counts = [int(ds.pred_instances.bboxes.shape[0])
                  if hasattr(ds.pred_instances, "bboxes") else 0
                  for ds in batch_data_samples]
        assert z.shape[0] == sum(counts), (z.shape[0], sum(counts))
        # Pass 1 (assembly): paste-parity check + per-instance crops per
        # image; the fp32 taps (z', upscaled embedding) and the fp32 paste
        # are freed as soon as their crops are copied (co-tenant memory).
        assembled = []
        offset = 0
        for img_i, (ds, n) in enumerate(zip(batch_data_samples, counts)):
            if n == 0:
                continue
            pi = ds.pred_instances
            masks_obj = pi.masks
            if hasattr(masks_obj, "masks"):  # BitmapMasks
                masks = torch.from_numpy(
                    np.asarray(masks_obj.masks).astype(bool)).to(z.device)
            else:
                masks = masks_obj.detach().to(z.device).bool()
            assert tuple(masks.shape[1:]) == (v11.IMAGE_SIZE, v11.IMAGE_SIZE)
            boxes = pi.bboxes.detach()
            probs = _do_paste_mask(
                torch.sigmoid(z[offset:offset + n].float()), boxes.float(),
                v11.IMAGE_SIZE, v11.IMAGE_SIZE, skip_empty=False)[0]
            assert bool(((probs >= thr) == masks).all()), "paste parity broken"
            port = self._val_geom[img_i].float()
            inputs = self._refine_inputs(
                masks, boxes, probs, up[offset:offset + n], port)
            del probs  # crops copied; free the fp32 paste before forwards
            assembled.append((ds, masks_obj, masks, boxes, inputs))
            offset += n
        del up, z  # every crop is copied; free the fp32 taps
        # Pass 2 (forward + write-back): chunked UBR + band reclass.
        total_changed = total_band_px = total_inst = 0
        for ds, masks_obj, masks, boxes, inputs in assembled:
            n = masks.shape[0]
            nbr_full = masks.sum(0).float()  # predicted neighbors
            out = self._refine_forward(inputs, masks, nbr_full, n)
            pi = ds.pred_instances
            refined_np = masks.cpu().numpy().copy()
            for i, s in enumerate(inputs):
                h0, w0, h, w = s["win"]
                mask_crop = masks[i, h0:h0 + h, w0:w0 + w].float()[None, None]
                rec = ubr.apply_band_reclassification(
                    mask_crop, out[i:i + 1, :, :h, :w], s["sdt"][None],
                    self.band_k)
                changed = (rec[0, 0] > 0.5) != (mask_crop[0, 0] > 0.5)
                assert not bool((changed & ~s["band"][0]).any()), (
                    "refined mask changed pixels outside the band")
                total_changed += int(changed.sum())
                total_band_px += int(s["band"][0].sum())
                total_inst += 1
                refined_np[i, h0:h0 + h, w0:w0 + w] = rec[0, 0].cpu().numpy() > 0.5
            if hasattr(masks_obj, "masks"):
                pi.masks = BitmapMasks(refined_np.astype(np.uint8),
                                       v11.IMAGE_SIZE, v11.IMAGE_SIZE)
            else:
                pi.masks = torch.from_numpy(refined_np).to(masks.device)
        self.last_refine_stats = {
            "instances": float(total_inst),
            "changed_px": float(total_changed),
            "band_px": float(total_band_px),
        }

    def _refine_inputs(self, masks, boxes, probs, up_img, port):
        """Per-instance window crops for the eval refine (v1.1 + port 5).

        Storage discipline (A3, co-tenant memory): every math op (paste,
        SDT, bilinear) runs fp32 exactly like the v1.1 eval path; the
        per-channel crops are then cast to fp16 - the identical values
        the autocast conv consumes - halving the stored crop volume.
        The fp32 sdt is kept alongside for the band/reclassification
        semantics (bit-exact vs the single-stream refine).
        """
        import torch.nn.functional as F

        inputs = []
        for i in range(masks.shape[0]):
            win = v11.window_around(boxes[i].cpu().numpy(),
                                    self.window_margin, v11.IMAGE_SIZE)
            h0, w0, h, w = win
            outside = masks[i].clone()
            outside[h0:h0 + h, w0:w0 + w] = False
            assert not bool(outside.any()), f"pred mask {i} outside window"
            sdt = ubr.signed_distance_transform(
                masks[i, h0:h0 + h, w0:w0 + w].float()[None, None])[0]
            fh0, fw0, fh, fw = v11.feat_window(win)
            inputs.append(dict(
                win=win,
                sdt=sdt,
                band=ubr.band_mask(sdt, self.band_k),
                feat=F.interpolate(
                    up_img[i:i + 1, :, fh0:fh0 + fh, fw0:fw0 + fw].float(),
                    size=(h, w), mode="bilinear", align_corners=False).half(),
                logit=probs[i:i + 1, h0:h0 + h, w0:w0 + w].half(),
                geom=port[:, h0:h0 + h, w0:w0 + w].half()[None],
                sdt_h=sdt.half(),
            ))
        return inputs

    def _refine_forward(self, inputs, masks, nbr_full, n) -> torch.Tensor:
        """Area-sorted chunked refiner forward (A1 discipline at eval).

        Instances are processed in window-area order so each chunk pads
        to its own local max instead of the image max; outputs scatter
        back to the original capture order. Numerically neutral per
        instance: padding beyond a window edge cannot reach in-band
        outputs (receptive-field radius depth+1 = 7px < window_margin -
        band_k = 24px) - the same argument the v1.1 train bucketing
        rests on.
        """
        h_all = max(s["win"][2] for s in inputs)
        w_all = max(s["win"][3] for s in inputs)
        areas = [s["win"][2] * s["win"][3] for s in inputs]
        order = sorted(range(n), key=lambda i: areas[i])
        out_all = None
        for start in range(0, n, REFINE_CHUNK):
            sel = order[start:start + REFINE_CHUNK]
            chunk = [inputs[i] for i in sel]
            geom = v11._pad_batch([s["geom"] for s in chunk],
                                  ubr.UAV_GEOMETRY_CHANNELS)
            batch_in = dict(
                feat=v11._pad_batch([s["feat"] for s in chunk], 32),
                logit=v11._pad_batch([s["logit"] for s in chunk], 1),
                sdt=v11._pad_batch([s["sdt_h"] for s in chunk], 1))
            nbrs = v11._pad_batch([
                (nbr_full[s["win"][0]:s["win"][0] + s["win"][2],
                          s["win"][1]:s["win"][1] + s["win"][3]]
                 - masks[j, s["win"][0]:s["win"][0] + s["win"][2],
                         s["win"][1]:s["win"][1] + s["win"][3]].float())[None]
                for j, s in zip(sel, chunk)], 1).half()
            with torch.autocast("cuda", enabled=masks.is_cuda):
                out = self.model.ubr_refiner(
                    batch_in["feat"], batch_in["logit"], batch_in["sdt"],
                    nbrs, geom).float()
            if out_all is None:
                out_all = out.new_zeros(n, 1, h_all, w_all)
            for slot, i in enumerate(sel):
                h, w = inputs[i]["win"][2], inputs[i]["win"][3]
                out_all[i, :, :h, :w] = out[slot, :, :h, :w]
        return out_all

    # ---------------------------------------------------------- wire-up
    def wire(self) -> None:
        """Compose the v1.1 wraps with the dual loss/predict extensions."""
        super().wire()  # installs the parent mask_loss + predict wraps
        model = self.model
        integration = self

        parent_mask_loss = model.roi_head.mask_loss

        def mask_loss(*args, **kwargs):
            img_metas = kwargs.get("batch_img_metas",
                                   args[6] if len(args) > 6 else None)
            assert img_metas is not None, "dual mask_loss needs img metas"
            integration._stage(img_metas)
            losses, telemetry = parent_mask_loss(*args, **kwargs)
            batch_gt = kwargs.get(
                "batch_gt_instances", args[2] if len(args) > 2 else None)
            assert batch_gt is not None
            losses["loss_geometry"] = integration.compute_loss_geometry(batch_gt)
            telemetry.update(integration.geo_train_stats())
            return losses, telemetry

        orig_roi_predict = model.roi_head.predict

        def roi_predict(x, rpn_results_list, batch_data_samples,
                        rescale=False, **kw):
            results_list = orig_roi_predict(
                x, rpn_results_list, batch_data_samples, rescale=rescale, **kw)
            if not model.training:
                integration._stage([m.metainfo for m in batch_data_samples])
                assert len(results_list) == 1, (
                    "dual candidate merge is defined for batch 1 (the "
                    "val/test protocol); got %d" % len(results_list))
                results_list = integration._merge_geometry_candidates(
                    model.roi_head, x, batch_data_samples, results_list,
                    rescale, kw.get("image_embeddings"),
                    kw.get("image_positional_embeddings"),
                    kw.get("high_res_features"))
            return results_list

        model.roi_head.mask_loss = mask_loss
        model.roi_head.predict = roi_predict


# ---------------------------------------------------------------- install
def validate_dual_config(ubr_cfg: dict) -> dict:
    """Fail-closed validation of the dual config's ubr_integration_cfg.

    The dual block must be the SINGLE-stream prereg block plus exactly
    {geometry_loss_weight}, with uav_port in {real, constant}; every other
    key keeps the sealed v1.1 value (checked by delegating to the parent
    validator with uav_port normalized to the single-stream 'zero').
    """
    cfg = dict(ubr_cfg or {})
    expected = set(v11.EXPECTED_UBR_CFG_KEYS) | DUAL_CFG_EXTRA_KEYS
    assert set(cfg) == expected, (
        f"dual ubr_integration_cfg keys {sorted(cfg)} != expected "
        f"{sorted(expected)}")
    assert cfg["uav_port"] in DUAL_PORT_MODES, cfg["uav_port"]
    assert cfg["geometry_loss_weight"] == GEOMETRY_LOSS_WEIGHT, (
        f"geometry_loss_weight must stay the prereg {GEOMETRY_LOSS_WEIGHT}")
    base = {k: v for k, v in cfg.items() if k != "geometry_loss_weight"}
    base["uav_port"] = "zero"
    v11.validate_ubr_config(base)  # every other v1.1 pin, unchanged
    return cfg


def apply_dual_integration(model, ubr_cfg: dict, arm: Optional[str] = None
                           ) -> UBRDualIntegration:
    """Wire one built model for a dual arm; returns the integration."""
    cfg = validate_dual_config(ubr_cfg)
    arm = arm or os.environ.get("UBR_DUAL_ARM") or cfg["uav_port"]
    assert arm in DUAL_PORT_MODES, f"UBR_DUAL_ARM must be one of {DUAL_PORT_MODES}"
    integration = UBRDualIntegration(
        model,
        arm=arm,
        geometry_loss_weight=cfg["geometry_loss_weight"],
        band_k=cfg["band_k"],
        loss_weight=cfg["loss_weight"],
        window_margin=cfg["window_margin"],
        focal_gamma=cfg["focal_gamma"],
        refiner_hidden=cfg["refiner_hidden"],
        refiner_depth=cfg["refiner_depth"],
        grad_checkpoint=cfg["grad_checkpoint"],
        bucket_backward=cfg["bucket_backward"],
        debug_containment_assert=cfg["debug_containment_assert"],
    )
    integration.wire()
    return integration


def install_dual_class_patch(ubr_cfg: dict, arm: Optional[str] = None) -> None:
    """Patch RSPrompterAnchor so MODELS.build wires the dual integration.

    Idempotent; the runner calls this BEFORE the sealed trainer entry
    (same pattern as ubr_train_integration.install_class_patch). The
    trainer's MODELS.build(cfg.model) then returns the dual-integrated
    model without any change to sealed files.
    """
    from ..detection.anchor_detector import RSPrompterAnchor

    cfg = validate_dual_config(ubr_cfg)
    if getattr(RSPrompterAnchor, "_ubr_dual_patched", False):
        return
    original_init = RSPrompterAnchor.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        apply_dual_integration(self, cfg, arm)

    RSPrompterAnchor.__init__ = patched_init
    RSPrompterAnchor._ubr_dual_patched = True
