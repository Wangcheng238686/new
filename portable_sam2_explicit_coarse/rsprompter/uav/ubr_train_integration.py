"""Runtime UBR wiring for the UBR-RESEAL-20261027 level-3 single-stream arm.

This module integrates rsprompter/uav/ubr.py::UnifiedBoundaryRefiner into
the sealed a3sgm trainer WITHOUT touching any existing file: the runner
(scripts/probes/ue_ubr_reseal_20261027.py) installs a class-level patch on
RSPrompterAnchor so every MODELS.build of the level-3 config
(configs/ablations/ue_ubr_single_512.py) returns an integrated model.

UBR v1.1 (docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md, EXP-0028-R2):
  Execution layer (A1-A4, numerically neutral, equivalence-gated by G1):
    A1 window-area buckets (<=96^2 / <=192^2 / <=384^2 / rest): per-bucket
       padding and ONE ubr_refiner forward per bucket replaces the v1.0
       pad-everything-to-the-batch-max forward; band losses keep capture
       order, so per-instance equal weight is unchanged.
    A2 vectorized assembly: batched window math, one CPU round-trip per
       bucket for the SDT, one pinned H2D per bucket key; the v1.0
       per-instance 512^2 containment assert becomes a one-sync debug
       check (debug_containment_assert, default off).
    A3 UBR path runs fp16 (autocast + fp16 storage); band focal/dice
       stay fp32 (computed from the fp16 logits, as in v1.0).
    A4 optional per-bucket backward (bucket_backward flag, default off;
       single-process only — see _bucketed_backward).
  Capacity (B1): refiner_hidden x refiner_depth = 32 x 6 is the NEW
    UNIFIED definition for every single-/dual-stream arm from EXP-0028-R2
    onward; (48, 7) stays pinned ONLY for the G1 old-capacity equivalence
    replay against the dead EXP-0028 arm. band_k and every other prereg
    key are unchanged.
  B2: grad_checkpoint flag recompute UBR block activations in backward;
    default OFF at the 512 protocol, required at 1024.

Contract (docs/UBR-RESEAL-20261027_preregistration.md):

  Train  - a forward hook on roi_head.mask_head captures the per-RoI final
           mask logits z' [N,1,128,128]; a forward hook on
           mask_head.mask_decoder.decoder.output_upscaling[-1] captures the
           32-ch upscaled embedding [N,32,128,128] (full-image stride-4
           grid; the exact tap the level-2 frozen-ref cache builder
           verified). With the UDPR tail deleted the decoder wrapper no
           longer returns it, so the tap is registered here directly
           (graph-attached in train mode -> gradients reach the decoder).
         - roi_head.mask_loss is wrapped: after the standard loss pipeline
           runs unmodified, loss_ubr is computed for every sampled POS RoI
           on its box+32px window:
               input 1  decoder feat  upscaled_embedding crop (box/4
                        registered window) bilinearly upsampled x4
               input 2  mask logits   _do_paste_mask(sigmoid(z'), box,
                        512, 512) crop - the exact op the roi_local paste
                        performs; differentiable, so gradients flow back
                        into the SAM2 decoder through z' as well
               input 3  sdt           SDT of the binarized predicted mask
               input 4  neighbors     sum of OTHER GT masks in the same
                        image (documented train/inference choice: GT
                        neighbors during training for stability, predicted
                        neighbors at inference)
               input 5  uav geometry  all zeros (UBR-off single stream,
                        port 5 dead)
           band |sdt| <= k (k=8); band focal BCE (gamma 2) + soft Dice vs
           the matched GT crop, per-instance equal weight, lambda 0.1;
           appended to the loss dict as 'loss_ubr'.
         - telemetry: loss_ubr rides the loss dict (epoch train log);
           band statistics ride the roi_head telemetry channel into the
           same log.

  Predict- model.predict is wrapped: after the standard predict produced
           binarized masks, per instance UBR band reclassification
           (ubr.apply_band_reclassification) writes the in-band decision
           back, so val/test metrics see the refined masks. Paste parity
           of the tapped z' against the produced masks is asserted per
           instance (the invariant the level-2 cache builder proved).

Asserts (fail-closed): UDPR tail absent; shapes; finite losses; pasted
mask contained in its box+margin window (debug flag); refined masks may
differ from the input only inside the band.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mmdet.models.roi_heads.mask_heads.fcn_mask_head import _do_paste_mask
from mmdet.structures.mask import BitmapMasks

from . import ubr

# Preregistration constants (UBR-RESEAL-20261027). Fail-closed: the
# installer rejects any config that disagrees with these.
PREREG = dict(
    band_k=8,
    loss_weight=0.1,
    window_margin=32,
    focal_gamma=2.0,
    uav_port="zero",
    neighbors_train="gt",
    neighbors_predict="pred",
    # Augmentation protocol keys (PI instruction 2026-10-27, 3090 line):
    # declared here, consumed by the runner's dataset patch; values fixed.
    aug_gaussian="0.5,0.02",
    aug_jitter="0.15,0.15,0.10,0.02,0.5",
)
# v1.1 capacity contract (docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md B1):
# (32, 6) = 32ch x 6 conv blocks is the NEW UNIFIED definition for every
# single-/dual-stream arm from EXP-0028-R2 onward; (48, 7) is the dead
# v1.0 arm's capacity, accepted ONLY for the G1 old-capacity equivalence
# replay. Pin both param counts so any arithmetic drift fails closed.
CAPACITY_UNIFIED = (32, 6)
CAPACITY_LEGACY = (48, 7)
UBR_PARAM_PINS = {CAPACITY_UNIFIED: 67457, CAPACITY_LEGACY: 163537}
CAPACITY_KEYS = ("refiner_hidden", "refiner_depth")
# v1.1 execution flags (A2 debug assert, A4 bucketed backward, B2 grad
# checkpointing). All default OFF; the 512 protocol runs with them off.
FLAG_KEYS = ("grad_checkpoint", "bucket_backward", "debug_containment_assert")
# A1 bucket edges on window area (px^2): <=96^2, <=192^2, <=384^2, >384^2.
BUCKET_AREA_BOUNDS = (96 * 96, 192 * 192, 384 * 384)
EXPECTED_UBR_CFG_KEYS = (set(PREREG) | {"enabled"} | set(CAPACITY_KEYS)
                         | set(FLAG_KEYS))
IMAGE_SIZE = 512  # fixed 512-native protocol
FEAT_GRID = 128   # upscaled_embedding grid = 512/4


def window_around(box: np.ndarray, margin: int, limit: int) -> Tuple[int, int, int, int]:
    """Clamped int window (y0, x0, h, w) around box on a limit x limit grid.

    Verbatim from the level-2 cache builder (ue_ubr_addprobe_cache_20261026).
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    w0 = max(0, int(np.floor(x1)) - margin)
    w1 = min(limit, int(np.ceil(x2)) + margin)
    h0 = max(0, int(np.floor(y1)) - margin)
    h1 = min(limit, int(np.ceil(y2)) + margin)
    assert 0 <= h0 < h1 <= limit and 0 <= w0 < w1 <= limit, (box, h0, w0, h1, w1)
    return h0, w0, h1 - h0, w1 - w0


def feat_window(win: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Stride-4 window registered to the 512-grid window (//4, ceil end).

    Level-2 review finding: flooring box/4 independently left a 0-3px
    shift; the feature window must be derived from the logit window.
    """
    h0, w0, h, w = win
    fh0, fw0 = h0 // 4, w0 // 4
    fh = -(-(h0 + h) // 4) - fh0
    fw = -(-(w0 + w) // 4) - fw0
    return fh0, fw0, fh, fw


def bucket_windows(boxes: np.ndarray, margin: int,
                   limit: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized window_around + feat_window over an [N,4] box batch.

    Bit-identical to looping the scalar helpers: float64 floor/ceil of the
    same box values, the same clamp order, the same //4 registration.
    Returns (wins [N,4] as y0,x0,h,w; fwins [N,4] as fy0,fx0,fh,fw).
    """
    assert boxes.ndim == 2 and boxes.shape[1] == 4
    x1, y1, x2, y2 = boxes.T.astype(np.float64)
    w0 = np.maximum(0, np.floor(x1).astype(np.int64) - margin)
    w1 = np.minimum(limit, np.ceil(x2).astype(np.int64) + margin)
    h0 = np.maximum(0, np.floor(y1).astype(np.int64) - margin)
    h1 = np.minimum(limit, np.ceil(y2).astype(np.int64) + margin)
    assert bool(((0 <= h0) & (h0 < h1) & (h1 <= limit)).all()), "y window clamp"
    assert bool(((0 <= w0) & (w0 < w1) & (w1 <= limit)).all()), "x window clamp"
    h, w = h1 - h0, w1 - w0
    fh0, fw0 = h0 // 4, w0 // 4
    wins = np.stack([h0, w0, h, w], 1)
    fwins = np.stack([fh0, fw0, -(-(h0 + h) // 4) - fh0,
                      -(-(w0 + w) // 4) - fw0], 1)
    return wins, fwins


def band_loss(pred: torch.Tensor, gt: torch.Tensor, band: torch.Tensor,
              gamma: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Focal BCE + soft Dice inside the band; per-instance equal weight.

    Verbatim semantics of ue_ubr_addprobe_20261026.py::band_loss (the
    proven level-2 trainer); reimplemented here because the probe script
    lives outside the package.
    """
    focals, dices = [], []
    for b in range(pred.shape[0]):
        m = band[b]
        if not m.any():
            focals.append(pred.new_zeros(()))
            dices.append(pred.new_zeros(()))
            continue
        lg, y = pred[b][m].float(), gt[b][m].float()
        bce = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
        p = torch.sigmoid(lg)
        p_t = p * y + (1 - p) * (1 - y)
        focals.append(((1 - p_t) ** gamma * bce).mean())
        inter = (p * y).sum()
        dices.append(1 - 2 * inter / (p.sum() + y.sum() + 1e-6))
    f, dd = torch.stack(focals).mean(), torch.stack(dices).mean()
    return f + dd, f, dd


def band_loss_per_instance(pred: torch.Tensor, gt: torch.Tensor,
                           band: torch.Tensor, gamma: float
                           ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-instance focal/dice lists — band_loss semantics without the mean.

    The A1 bucketed path collects per-instance values and stacks them in
    original capture order afterwards, so the reduced loss keeps band_loss's
    exact per-instance-equal-weight semantics.
    """
    focals, dices = [], []
    for b in range(pred.shape[0]):
        m = band[b]
        if not m.any():
            focals.append(pred.new_zeros(()))
            dices.append(pred.new_zeros(()))
            continue
        lg, y = pred[b][m].float(), gt[b][m].float()
        bce = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
        p = torch.sigmoid(lg)
        p_t = p * y + (1 - p) * (1 - y)
        focals.append(((1 - p_t) ** gamma * bce).mean())
        dices.append(1 - 2 * (p * y).sum() / (p.sum() + y.sum() + 1e-6))
    return focals, dices


def _pad_batch(tensors: List[torch.Tensor], channels: int,
               bool_: bool = False) -> torch.Tensor:
    """Top-left zero-pad [B,C,h,w] crops to the batch max (band pads 0)."""
    max_h = max(t.shape[-2] for t in tensors)
    max_w = max(t.shape[-1] for t in tensors)
    out = tensors[0].new_zeros(len(tensors), channels, max_h, max_w)
    for b, t in enumerate(tensors):
        out[b, ..., : t.shape[-2], : t.shape[-1]] = t
    return out.bool() if bool_ else out


def _pasted_inputs(model, z: torch.Tensor, boxes: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable 512-paste probs + binarized predictions (unchanged).

    Same calls as the v1.0 path: fp32 sigmoid(z') paste (gradients reach
    the SAM2 decoder through z') and the thr-binarized no-grad predictions
    that feed the SDT.
    """
    assert bool(torch.isfinite(boxes).all()
                and (boxes[:, 2:] > boxes[:, :2]).all()), (
        "degenerate RoI box in positive samples")
    thr = float(model.test_cfg["rcnn"]["mask_thr_binary"])
    probs512 = _do_paste_mask(
        torch.sigmoid(z.float()), boxes.float(), IMAGE_SIZE, IMAGE_SIZE,
        skip_empty=False)[0]  # [N,512,512]; differentiable w.r.t. z'
    with torch.no_grad():
        pred_bin = probs512.detach() >= thr
    return probs512, pred_bin


def _gt_numpy(batch_gt_instances, sampling_results, pos_counts
              ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    """Per-image GT stacks, neighbor sums and per-instance gt indices.

    Neighbor semantics unchanged: nbr = sum of ALL GT masks of the image
    minus the instance's own GT crop (GT neighbors during training).
    """
    gt_np: List[np.ndarray] = []
    for gt_instances in batch_gt_instances:
        masks = gt_instances.masks
        arr = masks.masks if hasattr(masks, "masks") else np.asarray(masks)
        gt_np.append(np.ascontiguousarray(
            arr, dtype=np.uint8).reshape(-1, IMAGE_SIZE, IMAGE_SIZE))
    nbr_full = [
        g.sum(0) if g.shape[0] else np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        for g in gt_np
    ]
    img_of = np.repeat(np.arange(len(pos_counts), dtype=np.int64), pos_counts)
    gt_of = np.concatenate([
        res.pos_assigned_gt_inds.detach().cpu().numpy().astype(
            np.int64).reshape(-1)
        for res in sampling_results])
    return gt_np, nbr_full, img_of, gt_of


def _assert_containment(pred_bin: torch.Tensor, wins: np.ndarray) -> None:
    """A2 debug check (default off): no pasted pixel outside its window.

    Vectorized one-shot replacement of the v1.0 per-instance 512^2 clone
    assert: one cover tensor, one .any() sync for the whole batch.
    """
    cover = torch.zeros_like(pred_bin)
    for j in range(pred_bin.shape[0]):
        h0, w0, h, w = wins[j]
        cover[j, h0:h0 + h, w0:w0 + w] = True
    assert not bool((pred_bin & ~cover).any()), "pasted mask outside window"


def _to_dev(arr: np.ndarray, dev, dtype) -> torch.Tensor:
    """One pinned H2D per bucket key (A2); fp32 host copy stays valid."""
    return torch.from_numpy(np.ascontiguousarray(arr)).pin_memory().to(
        dtype=dtype, device=dev, non_blocking=True)


# A4 hook: the runner registers the trainer's GradScaler (module
# substitution; the sealed trainer is never touched) so the optional
# per-bucket backward can scale manual contributions consistently.
_GRAD_SCALER = None


def register_grad_scaler(scaler) -> None:
    """Register the trainer's torch.amp.GradScaler for A4 (bucketed backward)."""
    global _GRAD_SCALER
    _GRAD_SCALER = scaler


class UBRIntegration:
    """Owns the UBR module, the capture hooks and the two method wraps."""

    def __init__(self, model, band_k: int, loss_weight: float,
                 window_margin: int, focal_gamma: float,
                 refiner_hidden: int, refiner_depth: int,
                 grad_checkpoint: bool = False, bucket_backward: bool = False,
                 debug_containment_assert: bool = False):
        self.model = model
        self.band_k = int(band_k)
        self.loss_weight = float(loss_weight)
        self.window_margin = int(window_margin)
        self.focal_gamma = float(focal_gamma)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.bucket_backward = bool(bucket_backward)
        self.debug_containment_assert = bool(debug_containment_assert)

        mask_head = model.roi_head.mask_head
        # Fail-closed: UDPR must be deleted, not disabled-but-present.
        assert not mask_head.decoder_tail_enabled, "UDPR tail must be absent"
        assert mask_head.decoder_tail_refiner is None, "UDPR refiner instantiated"
        assert mask_head.mask_decoder.capture_upscaled_embedding is False, (
            "wrapper upscaled capture belongs to the UDPR arm; the UBR tap "
            "registers its own hook"
        )
        assert mask_head.final_mask_coordinate_mode == "roi_local", (
            "the paste-parity contract is proven for roi_local mode"
        )
        assert "ubr_refiner" not in dict(model.named_children()), (
            "UBR already installed"
        )

        # Submodule registration: the attribute assignment makes UBR part
        # of the model -> device moves, optimizer param groups, the DDP
        # wrap and checkpoint state_dict all include it automatically.
        capacity = (int(refiner_hidden), int(refiner_depth))
        assert capacity in UBR_PARAM_PINS, f"unpinned UBR capacity {capacity}"
        model.ubr_refiner = ubr.UnifiedBoundaryRefiner(
            hidden=capacity[0], depth=capacity[1],
            grad_checkpoint=self.grad_checkpoint)
        n_par = sum(p.numel() for p in model.ubr_refiner.parameters())
        assert n_par == UBR_PARAM_PINS[capacity] == ubr.refiner_param_count(
            *capacity), f"UBR param count {n_par} != pin {UBR_PARAM_PINS[capacity]}"
        for p in model.ubr_refiner.parameters():
            p.requires_grad_(True)

        self._stash_z: List[torch.Tensor] = []
        self._stash_up: List[torch.Tensor] = []
        self.last_train_stats: Dict[str, float] = {}
        self.last_refine_stats: Dict[str, float] = {}
        mask_head.register_forward_hook(self._hook_mask_head)
        mask_head.mask_decoder.decoder.output_upscaling[-1].register_forward_hook(
            self._hook_upscale
        )

    # -------------------------------------------------------------- taps
    def _hook_mask_head(self, _module, _inputs, output):
        # Keep the graph in train mode: loss_ubr backprops through z'
        # into the SAM2 decoder. Detached on the predict path (eval).
        z = output.mask_logits
        self._stash_z.append(z if self.model.training else z.detach())

    def _hook_upscale(self, _module, _inputs, output):
        up = output
        self._stash_up.append(up if self.model.training else up.detach())

    def _collect(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._stash_z and self._stash_up, (
            "mask head ran without decoder captures (tail-less forward?)"
        )
        z = torch.cat(self._stash_z, 0)
        up = torch.cat(self._stash_up, 0)
        assert z.shape[0] == up.shape[0]
        assert tuple(z.shape[1:]) == (1, FEAT_GRID, FEAT_GRID), tuple(z.shape)
        assert tuple(up.shape[1:]) == (32, FEAT_GRID, FEAT_GRID), tuple(up.shape)
        return z, up

    def _release(self):
        self._stash_z.clear()
        self._stash_up.clear()

    # -------------------------------------------------------- train loss
    def compute_loss_ubr(self, sampling_results, batch_gt_instances) -> torch.Tensor:
        """loss_ubr over all sampled positive RoIs of the batch (v1.1).

        v1.1 execution layer (docs/UBR-V1.1-PERF-AND-RELAUNCH-20261027.md,
        items A1-A4; numerically neutral vs the v1.0 full-batch path):
        instances are grouped into <=4 window-area buckets, each bucket is
        padded to its OWN max and gets ONE ubr_refiner forward; per-instance
        band losses keep capture order, so the reduction semantics are
        unchanged. Zero padding beyond a window is identical zeros in both
        paths, and band pixels sit >= window_margin - band_k = 24 px from
        the window edge while the refiner's receptive-field radius is
        depth+1 <= 8 px, so in-band conv outputs are unchanged. The fp16
        UBR path (A3) feeds the conv the same fp16-rounded values the v1.0
        outer autocast already produced at the conv cast; focal/dice stay
        fp32 (computed from the fp16 logits via .float(), as before).
        """
        model = self.model
        dev = next(model.parameters()).device
        pos_counts = [int(res.pos_priors.shape[0]) for res in sampling_results]
        n_pos = int(sum(pos_counts))
        zero = sum(p.sum() for p in model.ubr_refiner.parameters()) * 0.0
        if n_pos == 0:
            self.last_train_stats = {"UBR/instances": 0.0,
                                     "UBR/band_nonempty_frac": 0.0,
                                     "UBR/focal": 0.0, "UBR/dice": 0.0}
            return zero
        z, up = self._collect()
        assert z.shape[0] == n_pos, (z.shape[0], n_pos)
        boxes = torch.cat([res.pos_priors for res in sampling_results], 0)
        probs512, pred_bin = _pasted_inputs(model, z, boxes)
        wins, fwins = bucket_windows(
            boxes.detach().cpu().numpy(), self.window_margin, IMAGE_SIZE)
        gt_np, nbr_full, img_of, gt_of = _gt_numpy(
            batch_gt_instances, sampling_results, pos_counts)
        if self.debug_containment_assert:
            _assert_containment(pred_bin, wins)  # A2: one-sync check
        bucket_of = np.digitize(wins[:, 2] * wins[:, 3], BUCKET_AREA_BOUNDS)
        focal_all: List[torch.Tensor] = [None] * n_pos  # type: ignore[assignment]
        dice_all: List[torch.Tensor] = [None] * n_pos  # type: ignore[assignment]
        bucket_losses, n_empty_band = [], 0
        for b_id in range(len(BUCKET_AREA_BOUNDS) + 1):
            idx = np.nonzero(bucket_of == b_id)[0]
            if idx.size == 0:
                continue
            focals, dices, n_empty, loss_b = self._bucket_step(
                idx, wins, fwins, gt_np, nbr_full, img_of, gt_of,
                pred_bin, up, probs512, n_pos, dev)
            for slot, j in enumerate(idx):
                focal_all[int(j)] = focals[slot]
                dice_all[int(j)] = dices[slot]
            n_empty_band += n_empty
            bucket_losses.append(loss_b)
        assert all(v is not None for v in focal_all + dice_all)
        focal = torch.stack(focal_all).mean()
        dice = torch.stack(dice_all).mean()
        loss = self.loss_weight * (focal + dice)
        if self.bucket_backward and len(bucket_losses) > 1:
            loss = self._bucketed_backward(bucket_losses)  # A4
        assert bool(torch.isfinite(loss)), "non-finite loss_ubr"
        self.last_train_stats = {
            "UBR/instances": float(n_pos),
            "UBR/band_nonempty_frac": float(n_pos - n_empty_band) / n_pos,
            "UBR/focal": float(focal.detach()),
            "UBR/dice": float(dice.detach()),
        }
        return loss

    def _bucket_step(self, idx, wins, fwins, gt_np, nbr_full, img_of, gt_of,
                     pred_bin, up, probs512, n_pos, dev
                     ) -> Tuple[List[torch.Tensor], List[torch.Tensor],
                                int, torch.Tensor]:
        """Assemble one A1 bucket (padded to the bucket max) + one forward.

        A2: pred crops move to CPU once per bucket; SDT/band are computed
        in fp32 numpy (bit-identical to the v1.0 per-instance scipy calls);
        sdt/gt/nbr move back in one pinned H2D per key. A3: every conv
        input is stored fp16 (the values the v1.0 conv cast produced).
        """
        n = int(idx.size)
        h_max = int(wins[idx, 2].max())
        w_max = int(wins[idx, 3].max())
        half = torch.float16 if dev.type == "cuda" else torch.float32
        crops = torch.zeros((n, 1, h_max, w_max), dtype=torch.bool,
                            device=pred_bin.device)
        gt_pad = np.zeros((n, 1, h_max, w_max), dtype=np.float32)
        nbr_pad = np.zeros((n, 1, h_max, w_max), dtype=np.float32)
        inside = np.zeros((n, h_max, w_max), dtype=bool)
        for slot, j in enumerate(idx):
            h0, w0, h, w = wins[j]
            crops[slot, 0, :h, :w] = pred_bin[j, h0:h0 + h, w0:w0 + w]
            sl = (slice(int(h0), int(h0) + int(h)), slice(int(w0), int(w0) + int(w)))
            own = gt_np[img_of[j]][gt_of[j]][sl]
            gt_pad[slot, 0, :h, :w] = own
            nbr_pad[slot, 0, :h, :w] = nbr_full[img_of[j]][sl] - own
            inside[slot, :h, :w] = True
        sdt32 = ubr.signed_distance_transform_numpy(crops.cpu().numpy()[:, 0])
        band_np = (np.abs(sdt32) <= float(self.band_k)) & inside  # pad -> False
        n_empty = int((~band_np.any(axis=(1, 2))).sum())
        sdt_t = _to_dev(sdt32, dev, torch.float16 if dev.type == "cuda"
                        else torch.float32).unsqueeze(1)
        band_t = torch.from_numpy(band_np).unsqueeze(1)
        gt_t = _to_dev(gt_pad, dev, torch.float32)
        nbr_t = _to_dev(nbr_pad, dev, half)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=dev.type == "cuda"):
            feats = []
            logits = []
            for j in idx:  # per-window bilinear keeps exact v1.0 sampling
                fh0, fw0, fh, fw = fwins[j]
                h0, w0, h, w = wins[j]
                feats.append(F.interpolate(
                    up[j:j + 1, :, fh0:fh0 + fh, fw0:fw0 + fw].to(half),
                    size=(int(h), int(w)), mode="bilinear", align_corners=False))
                logits.append(probs512[j:j + 1, h0:h0 + h, w0:w0 + w].to(half))
            out = self.model.ubr_refiner(
                _pad_batch(feats, 32), _pad_batch(logits, 1), sdt_t, nbr_t)
        assert tuple(out.shape) == (n, 1, h_max, w_max), tuple(out.shape)
        assert bool(torch.isfinite(out.float()).all()), "non-finite UBR logits"
        focals, dices = band_loss_per_instance(out, gt_t, band_t, self.focal_gamma)
        # A4 budget: the per-bucket graph loss, weight x (sum_i)/n_pos
        loss_b = self.loss_weight * (sum(focals) + sum(dices)) / float(n_pos)
        return focals, dices, n_empty, loss_b

    def _bucketed_backward(self, bucket_losses) -> torch.Tensor:
        """A4: backward every bucket but the last; the tail rides the
        trainer's scaled backward through the returned loss.

        Gradient accumulation is mathematically the single-backward sum;
        each bucket's UBR activations free at its own backward while the
        small shared upstream (z', probs512, upscaled embedding) is
        retained. Manual contributions are scaled by the registered
        GradScaler's current scale so they match the trainer's scaled
        tail backward. Constraint: single-process only — under the DDP
        reducer a second backward over the same parameters trips its
        mark-ready-once guard, so the flag must stay OFF for DDP runs.
        """
        scale = float(_GRAD_SCALER.get_scale()) if _GRAD_SCALER is not None else 1.0
        head = sum(bucket_losses[:-1])
        torch.autograd.backward(scale * head, retain_graph=True)
        return bucket_losses[-1] + head.detach()

    def train_stats(self) -> Dict[str, object]:
        """Telemetry tensors for the roi_head channel (train log)."""
        return {k: torch.tensor(v) for k, v in self.last_train_stats.items()}

    # --------------------------------------------------- predict refine
    @torch.no_grad()
    def refine_predictions(self, batch_data_samples) -> None:
        """In-place UBR band reclassification on every predicted mask.

        Runs on the eval path only (the trainer's validation loop and the
        harness); neighbors are PREDICTED masks (documented choice).
        """
        model = self.model
        thr = float(model.test_cfg["rcnn"]["mask_thr_binary"])
        z, up = self._collect()
        counts = []
        for ds in batch_data_samples:
            pi = ds.pred_instances
            counts.append(int(pi.bboxes.shape[0]) if hasattr(pi, "bboxes") else 0)
        assert z.shape[0] == sum(counts), (z.shape[0], sum(counts))
        offset = 0
        total_changed = total_band_px = total_inst = 0
        for ds, n in zip(batch_data_samples, counts):
            if n == 0:
                continue
            pi = ds.pred_instances
            masks_obj = pi.masks
            if hasattr(masks_obj, "masks"):  # BitmapMasks
                masks = torch.from_numpy(
                    np.asarray(masks_obj.masks).astype(bool)).to(z.device)
            else:
                masks = masks_obj.detach().to(z.device).bool()
            assert tuple(masks.shape[1:]) == (IMAGE_SIZE, IMAGE_SIZE)
            boxes = pi.bboxes.detach()
            probs = _do_paste_mask(
                torch.sigmoid(z[offset:offset + n].float()), boxes.float(),
                IMAGE_SIZE, IMAGE_SIZE, skip_empty=False)[0]
            # paste parity: the tapped z' must reproduce the produced masks
            assert bool(((probs >= thr) == masks).all()), "paste parity broken"
            wins = [window_around(b, self.window_margin, IMAGE_SIZE)
                    for b in boxes.cpu().numpy()]
            fwins = [feat_window(w) for w in wins]
            nbr_full = masks.sum(0).float()  # predicted neighbors

            inputs = []
            for i in range(n):
                h0, w0, h, w = wins[i]
                outside = masks[i].clone()
                outside[h0:h0 + h, w0:w0 + w] = False
                assert not bool(outside.any()), f"pred mask {i} outside window"
                sdt = ubr.signed_distance_transform(
                    masks[i, h0:h0 + h, w0:w0 + w].float()[None, None])[0]
                fh0, fw0, fh, fw = fwins[i]
                inputs.append(dict(
                    win=wins[i],
                    sdt=sdt,
                    band=ubr.band_mask(sdt, self.band_k),
                    feat=F.interpolate(
                        up[offset + i: offset + i + 1,
                           :, fh0:fh0 + fh, fw0:fw0 + fw].float(),
                        size=(h, w), mode="bilinear", align_corners=False),
                    logit=probs[i:i + 1, h0:h0 + h, w0:w0 + w],
                ))
            batch_in = dict(
                feat=_pad_batch([s["feat"] for s in inputs], 32),
                logit=_pad_batch([s["logit"] for s in inputs], 1),
                sdt=_pad_batch([s["sdt"] for s in inputs], 1),
            )
            nbrs = _pad_batch([
                (nbr_full[s["win"][0]:s["win"][0] + s["win"][2],
                          s["win"][1]:s["win"][1] + s["win"][3]]
                 - masks[i, s["win"][0]:s["win"][0] + s["win"][2],
                         s["win"][1]:s["win"][1] + s["win"][3]].float())[None]
                for i, s in enumerate(inputs)], 1)
            with torch.autocast("cuda", enabled=z.is_cuda):
                out = model.ubr_refiner(
                    batch_in["feat"], batch_in["logit"], batch_in["sdt"],
                    nbrs).float()
            refined_np = masks.cpu().numpy().copy()
            for i, s in enumerate(inputs):
                h0, w0, h, w = s["win"]
                mask_crop = masks[i, h0:h0 + h, w0:w0 + w].float()[None, None]
                rec = ubr.apply_band_reclassification(
                    mask_crop, out[i:i + 1, :, :h, :w], s["sdt"][None],
                    self.band_k)
                changed = (rec[0, 0] > 0.5) != (mask_crop[0, 0] > 0.5)
                # fail-closed containment: changed pixels only in band
                assert not bool((changed & ~s["band"][0]).any()), (
                    "refined mask changed pixels outside the band")
                total_changed += int(changed.sum())
                total_band_px += int(s["band"][0].sum())
                total_inst += 1
                refined_np[i, h0:h0 + h, w0:w0 + w] = rec[0, 0].cpu().numpy() > 0.5
            if hasattr(masks_obj, "masks"):
                pi.masks = BitmapMasks(refined_np.astype(np.uint8),
                                       IMAGE_SIZE, IMAGE_SIZE)
            else:
                pi.masks = torch.from_numpy(refined_np).to(masks.device)
            offset += n
        self.last_refine_stats = {
            "instances": float(total_inst),
            "changed_px": float(total_changed),
            "band_px": float(total_band_px),
        }

    # ---------------------------------------------------------- wire-up
    def wire(self) -> None:
        """Wrap roi_head.mask_loss and model.predict on this instance."""
        model = self.model
        integration = self
        model.ubr_integration = self  # non-Module attr: inspection handle
        orig_mask_loss = model.roi_head.mask_loss
        orig_predict = model.predict

        def mask_loss(*args, **kwargs):
            integration._release()
            losses, telemetry = orig_mask_loss(*args, **kwargs)
            sampling_results = kwargs.get(
                "sampling_results", args[1] if len(args) > 1 else None)
            batch_gt_instances = kwargs.get(
                "batch_gt_instances", args[2] if len(args) > 2 else None)
            assert sampling_results is not None and batch_gt_instances is not None
            losses["loss_ubr"] = integration.compute_loss_ubr(
                sampling_results, batch_gt_instances)
            telemetry.update(integration.train_stats())
            integration._release()
            return losses, telemetry

        def predict(*args, **kwargs):
            integration._release()
            out = orig_predict(*args, **kwargs)
            if integration._stash_z and integration._stash_up:
                integration.refine_predictions(out)
            integration._release()
            return out

        model.roi_head.mask_loss = mask_loss
        model.predict = predict


def validate_ubr_config(ubr_cfg: dict) -> dict:
    """Fail-closed validation of the config's ubr_integration_cfg block."""
    cfg = dict(ubr_cfg or {})
    keys = set(cfg)
    assert keys == EXPECTED_UBR_CFG_KEYS, (
        f"ubr_integration_cfg keys {sorted(keys)} != expected "
        f"{sorted(EXPECTED_UBR_CFG_KEYS)}")
    assert cfg["enabled"] is True, "UBR integration must be enabled"
    for key, expected in PREREG.items():
        assert cfg[key] == expected, (
            f"ubr_integration_cfg[{key}]={cfg[key]!r} drifts from prereg "
            f"{expected!r}")
    capacity = (int(cfg["refiner_hidden"]), int(cfg["refiner_depth"]))
    assert capacity in UBR_PARAM_PINS, (
        f"UBR capacity {capacity} is not pinned; pinned capacities "
        f"(hidden, depth): {sorted(UBR_PARAM_PINS)}")
    for flag in FLAG_KEYS:
        assert isinstance(cfg[flag], bool), f"{flag} must be a bool"
    # NOTE bucket_backward=True is single-process-only (the DDP reducer's
    # mark-ready-once guard); launchers of DDP runs must reject it — the
    # reseal runner asserts it OFF for the EXP-0028-R2 protocol.
    return cfg


def apply_ubr_integration(model, ubr_cfg: dict) -> UBRIntegration:
    """Wire one built model; returns the integration for inspection."""
    cfg = validate_ubr_config(ubr_cfg)
    integration = UBRIntegration(
        model,
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


def install_class_patch(ubr_cfg: dict) -> None:
    """Patch RSPrompterAnchor so MODELS.build wires UBR automatically.

    Idempotent; the runner calls this BEFORE invoking the sealed trainer
    entry, so the model the trainer builds (MODELS.build(cfg.model) in
    train/train_rsprompter_fusion.py) is integrated without touching the
    trainer or any model file. The config's ubr_integration_cfg block is
    swallowed harmlessly by the mask head's **kwargs; the wiring constants
    come from this validated dict (single source: the config file).
    """
    from ..detection.anchor_detector import RSPrompterAnchor

    validate_ubr_config(ubr_cfg)
    if getattr(RSPrompterAnchor, "_ubr_integration_patched", False):
        return
    original_init = RSPrompterAnchor.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        apply_ubr_integration(self, ubr_cfg)

    RSPrompterAnchor.__init__ = patched_init
    RSPrompterAnchor._ubr_integration_patched = True
