"""Opt-in P1 bridge. Original native mask/coarse/detection losses stay native.

The runner supplies the FULL CURRENT, ALREADY SELECTED pre-UBR prediction
pool in resized-image coordinates. No new candidate generation, NMS, source
quota, score transform, or GT proposal insertion occurs in this module.
Single-image API (existing per-rank batch=1); average image losses externally
for larger batches, INCLUDING images with no active matches.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .ubr_coordinate_adapter import adapt_native_roi, residual_probabilities
from .ubr_prediction_context import PredictionContext
from .unified_boundary_refiner import (UnifiedBoundaryRefiner, signed_distance_transform,
                                      band_mask, boundary_loss)


def disable_udpr_config(model_cfg):
    """Call BEFORE model construction: no UDPR parameters/module instantiated."""
    cfg = deepcopy(model_cfg)
    cfg['roi_head']['mask_head']['decoder_tail_refiner_cfg'] = {'enabled': False}
    return cfg


@dataclass(frozen=True)
class PredictionPool:
    boxes: Tensor
    scores: Tensor
    candidate_ids: Tensor
    source_ids: Tuple[str, ...]

    def validate(self):
        n = len(self.boxes)
        if self.boxes.shape != (n, 4) or self.scores.shape != (n,) or self.candidate_ids.shape != (n,):
            raise ValueError('pool boxes/scores/IDs have inconsistent shapes')
        if len(self.source_ids) != n or self.candidate_ids.dtype != torch.int64:
            raise ValueError('source IDs and int64 candidate IDs required')
        if len(torch.unique(self.candidate_ids)) != n:
            raise ValueError('pool candidate IDs must be unique and stable')
        if self.boxes.requires_grad or self.scores.requires_grad:
            raise ValueError('prediction boxes/scores must be detached')
        if not torch.isfinite(self.boxes).all() or not torch.isfinite(self.scores).all():
            raise ValueError('nonfinite prediction pool')
        if (self.boxes[:, 2:] <= self.boxes[:, :2]).any():
            raise ValueError('degenerate prediction box')
        if self.scores.device != self.boxes.device or self.candidate_ids.device != self.boxes.device:
            raise ValueError('pool tensors must share a device')


@dataclass(frozen=True)
class PreparedImage:
    roi_features: Tuple[Tensor, ...]  # already extraPE-adjusted ONCE
    image_embeddings: Tensor
    image_positional_embeddings: Optional[Tensor]
    high_res_features: Optional[Tuple[Tensor, ...]]
    meta: dict


@torch.no_grad()
def active_prediction_indices(pool, gt_boxes, max_active=64):
    """IoU>=.5, GT-order tie break; score-desc/ID-asc cap, no GT insertion."""
    pool.validate()
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4:
        raise ValueError('gt_boxes must be [G,4] on resized-image canvas')
    if not len(pool.boxes) or not len(gt_boxes):
        empty = pool.candidate_ids.new_empty(0)
        return empty, empty
    a, b = pool.boxes, gt_boxes.to(pool.boxes)
    lo = torch.maximum(a[:, None, :2], b[None, :, :2])
    hi = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (hi-lo).clamp_min(0).prod(-1)
    union = (a[:, 2:]-a[:, :2]).prod(-1)[:, None] + (b[:, 2:]-b[:, :2]).prod(-1)[None] - inter
    quality, assigned = (inter / union.clamp_min(1e-12)).max(1)
    keep = torch.where(quality >= .5)[0].tolist()
    keep.sort(key=lambda i: (-float(pool.scores[i]), int(pool.candidate_ids[i])))
    indices = pool.candidate_ids.new_tensor(keep[:max_active])
    return indices, assigned[indices]


class UBRDetectorIntegration:
    """Non-module controller; parameters register once as satellite.ubr_refiner.

    Install BEFORE optimizer/DDP. Decode uses existing mask head unchanged;
    the capture hook observes final act2 only during this controller's calls.
    Not reentrant; no mutable global cache survives a decoder call.
    """
    def __init__(self, satellite, refiner=None):
        self.satellite = satellite
        self.head = satellite.roi_head
        mask_head = self.head.mask_head
        if mask_head.decoder_tail_enabled or mask_head.decoder_tail_refiner is not None:
            raise ValueError('construct with disable_udpr_config first; do not silently unload UDPR')
        if mask_head.final_mask_coordinate_mode != 'roi_local' or not mask_head.class_agnostic:
            raise ValueError('P1 requires historical class-agnostic roi_local mask head')
        if mask_head.mask_decoder.capture_upscaled_embedding:
            raise ValueError('mask decoder must retain the original three-tuple return')
        if hasattr(satellite, 'ubr_refiner'):
            raise ValueError('UBR already installed')
        if float(self.head.test_cfg.mask_thr_binary) != .4:
            raise ValueError('P1 deployment threshold is fixed at .4')
        satellite.add_module('ubr_refiner', refiner if refiner is not None else
                             UnifiedBoundaryRefiner(checkpoint_blocks=True))
        # Match pre-existing model device; caller still owns model dtype/transfer.
        satellite.ubr_refiner.to(device=next(satellite.parameters()).device)
        self._capturing = False
        self._capture = None
        self.last_stats = {}
        self._handle = mask_head.mask_decoder.decoder.output_upscaling[-1].register_forward_hook(self._hook)

    @property
    def refiner(self):
        return self.satellite.ubr_refiner

    def _hook(self, module, args, output):
        if self._capturing:
            if self._capture is not None:
                raise RuntimeError('multiple decoder captures in one local decode')
            self._capture = output

    def prepare(self, features, meta):
        """One extraPE application; reuse result for bbox and all mask calls."""
        x, embedding, pe, high = features
        if embedding.shape[0] != 1 or any(item.shape[0] != 1 for item in x):
            raise ValueError('P1 API expects exactly one image')
        return PreparedImage(tuple(self.head._inject_extra_pe(x)), embedding, pe,
                             tuple(high) if high is not None else None, dict(meta))

    def decode_native(self, prepared, boxes):
        """Public engineering seam: native masks/features before any paste."""
        if self._capturing or self._capture is not None:
            raise RuntimeError('capture must be local and nonreentrant')
        rois = torch.cat((boxes.new_zeros((len(boxes), 1)), boxes), dim=1)
        self._capturing = True
        try:
            result = self.head._mask_forward(prepared.roi_features, rois,
                image_embeddings=prepared.image_embeddings,
                image_positional_embeddings=prepared.image_positional_embeddings,
                high_res_features=prepared.high_res_features)
            feature = self._capture
            logits = result['mask_preds']
            if feature is None:
                raise RuntimeError('real SAM decoder failed to emit final act2')
            if feature.shape != (len(boxes), 32, *logits.shape[-2:]):
                raise RuntimeError('capture must share native logits grid and 32 channels')
            return logits, feature
        finally:
            self._capture = None
            self._capturing = False

    def _decode(self, prepared, boxes):
        logits, feature = self.decode_native(prepared, boxes)
        return adapt_native_roi(logits, feature, boxes, prepared.meta, rescale=False)

    def _prediction_chunks(self, pool):
        """Preserve legacy independent consumer calls and mask decoder batches.

        Current pool is ordered sat / sat_supplement / uav, each corresponding
        to ONE old consumer.masks invocation. Numerical conv kernels differ
        between batch1 and batch32: preserving boxes alone is insufficient for
        same-weight zero-UBR probability parity. source_ids MUST encode these
        consumer call boundaries, not merely an arbitrary regroupable category.
        """
        pool.validate()
        chunk_size = int(getattr(self.head, 'mask_roi_chunk_size', 32) or 0)
        start = 0
        while start < len(pool.boxes):
            end = start + 1
            while end < len(pool.boxes) and pool.source_ids[end] == pool.source_ids[start]:
                end += 1
            step = chunk_size if chunk_size > 0 else end - start
            for offset in range(start, end, step):
                yield offset, min(offset + step, end)
            start = end

    @contextmanager
    def _prediction_context_mode(self):
        """Eval/no_grad prediction without changing active train modes or RNG."""
        modules = list(self.head.modules())
        modes = [module.training for module in modules]
        device = next(self.head.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        try:
            self.head.eval()
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                yield
        finally:
            for module, mode in zip(modules, modes):
                module.training = mode

    def prediction_context(self, prepared, pool):
        """Full current pool, no additional NMS or GT; synchronous before UBR."""
        pool.validate()
        masks = []
        with self._prediction_context_mode():
            for begin, end in self._prediction_chunks(pool):
                native, feature = self.decode_native(prepared, pool.boxes[begin:end])
                for offset, i in enumerate(range(begin, end)):
                    state = adapt_native_roi(native[offset:offset+1], feature[offset:offset+1],
                                             pool.boxes[i:i+1], prepared.meta, rescale=False)
                    masks.append(state.probabilities[0, 0] >= .4)
        if masks:
            masks = torch.stack(masks)
        else:
            from .ubr_coordinate_adapter import fcn_canvas
            _, size = fcn_canvas(pool.boxes, prepared.meta)
            masks = pool.boxes.new_empty((0, *size), dtype=torch.bool)
        ids = pool.candidate_ids.detach().clone()
        images = torch.zeros_like(ids)
        # Input pool already selected by original prefix/hypothesis policy.
        # Every retained prediction represents itself. No fresh NMS is legal.
        return PredictionContext(ids, images, tuple(pool.source_ids), ids.clone(),
                                 ids.clone(), images.clone(), tuple(pool.source_ids),
                                 masks.detach().clone(), .4)

    def _refine(self, state, context, candidate_id, geometry):
        if geometry is not None and geometry.shape != (1, 5, *state.image_size):
            raise ValueError('geometry must already share exact resized-image pixel canvas')
        # Use SAVED probability, not inverse-logit roundtrip at .4 ties.
        sdt = signed_distance_transform(state.probabilities.detach() >= .4)
        neighbor = context.neighbors(candidate_id)[None, None]
        refined = self.refiner(state.decoder_features, state.logits, sdt, neighbor, geometry)
        return refined, band_mask(sdt, self.refiner.band_width)

    def training_loss(self, prepared, pool, gt_boxes, gt_masks, geometry=None):
        """Only additive UBR .1 loss; original native losses stay untouched.

        gt_masks: [G,H,W] assigned original instance masks in resized-image
        canvas, NOT bbox crops. One image is denominator even with zero matches.
        """
        if not torch.is_grad_enabled():
            raise ValueError('active UBR training must be gradient enabled')
        from .ubr_coordinate_adapter import fcn_canvas
        _, size = fcn_canvas(pool.boxes, prepared.meta)
        if gt_masks.shape != (len(gt_boxes), *size):
            raise ValueError('full-image GT masks must match canvas and GT order')
        context = self.prediction_context(prepared, pool)
        active, assigned = active_prediction_indices(pool, gt_boxes)
        # Always connect every UBR parameter on empty/sparse ranks.
        zero = sum(p.sum() * 0 for p in self.refiner.parameters())
        raw = zero
        nonempty = 0
        for i, g in zip(active.tolist(), assigned.tolist()):
            # Whole decoder->paste->UBR->loss checkpoint, not only UBR blocks:
            # accumulating 64 ordinary decoder graphs defeats per-instance chunks.
            # Bind indices NOW: recomputation occurs after this loop has finished.
            def active_loss(box, target=gt_masks[g:g+1, None],
                            candidate_id=int(pool.candidate_ids[i])):
                state = self._decode(prepared, box)
                refined, band = self._refine(state, context, candidate_id, geometry)
                value = boundary_loss(refined, target.to(refined), band,
                                      torch.zeros(1, device=refined.device, dtype=torch.long), 1)
                return value, band.any()
            # Legacy decoder catches Exception around its forward. Disable
            # checkpoint early-stop exceptions so it cannot wrap that control flow.
            with set_checkpoint_early_stop(False):
                value, has_band = checkpoint(active_loss, pool.boxes[i:i+1],
                                             use_reentrant=False, preserve_rng_state=True)
            raw = raw + value
            nonempty += int(has_band)
        self.last_stats = dict(context_count=len(pool.boxes), active_count=len(active),
                               active_redecode_count=len(active), nonempty_band_count=nonempty)
        return {'loss_ubr': .1 * raw / max(nonempty, 1)}

    @torch.no_grad()
    def predict(self, prepared, pool, geometry=None):
        """Image-grid output; NEVER send these masks through bbox paste again.

        Scores and boxes unchanged. Requires eval() on model as usual. Outputs
        remain resized-image coordinates; runner owns any final rescale.
        """
        if self.satellite.training:
            raise ValueError('predict requires satellite.eval()')
        context = self.prediction_context(prepared, pool)
        probabilities = []
        for begin, end in self._prediction_chunks(pool):
            native, feature = self.decode_native(prepared, pool.boxes[begin:end])
            for offset, i in enumerate(range(begin, end)):
                state = adapt_native_roi(native[offset:offset+1], feature[offset:offset+1],
                                         pool.boxes[i:i+1], prepared.meta, rescale=False)
                refined, _ = self._refine(state, context, int(pool.candidate_ids[i]), geometry)
                probabilities.append(residual_probabilities(state, refined)[0, 0])
        if probabilities:
            probabilities = torch.stack(probabilities)
        else:
            probabilities = context.retained_masks.float()
        return dict(probabilities=probabilities, masks=probabilities >= .4,
                    boxes=pool.boxes.clone(), scores=pool.scores.clone(),
                    candidate_ids=pool.candidate_ids.clone())


def install_ubr(satellite, refiner=None):
    return UBRDetectorIntegration(satellite, refiner)
