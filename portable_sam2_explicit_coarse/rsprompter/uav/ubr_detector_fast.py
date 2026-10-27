"""Independent efficiency implementation of the fixed P1 UBR contract.

No windows, resize, approximate EDT, candidate truncation or loss changes.
Training retains singleton native decoding and singleton image-grid UBR,
after measured train-microbatch4 offered no speed gain but doubled memory.
Eval keeps original source/chunk32 native batches and uses UBR microbatch4.
These are independent engineering batch sizes, not model hyperparameters.
The original integration and its running jobs are never modified here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .ubr_coordinate_adapter import (adapt_native_roi, fcn_canvas, paste_native_roi,
                                     residual_probabilities)
from .ubr_detector_integration import UBRDetectorIntegration, active_prediction_indices
from .ubr_prediction_context import PredictionContext
from .unified_boundary_refiner import signed_distance_transform, band_mask


@dataclass(frozen=True)
class FastPredictionContext(PredictionContext):
    mask_sum: Tensor
    id_to_index: Dict[int, int]

    @torch.no_grad()
    def neighbors(self, candidate_id):
        if candidate_id not in self.id_to_index:
            raise KeyError('candidate_id was not in the predicted context: %s' % candidate_id)
        index = self.id_to_index[candidate_id]
        # P1 pool is single-image, all retained IDs are their own representative.
        return self.mask_sum - self.retained_masks[index].to(torch.float32)


def _band_instance_sums(logits, target, band):
    """Sum (not mean) nonempty instance losses; caller divides ONCE per image."""
    z, y = logits.float(), target.float()
    valid = band.to(z.dtype)
    counts = valid.flatten(1).sum(1)
    nonempty = counts > 0
    p = z.sigmoid()
    pt = p * y + (1-p) * (1-y)
    focal = ((1-pt).square() * F.binary_cross_entropy_with_logits(z, y, reduction='none') * valid)
    focal = focal.flatten(1).sum(1) / counts.clamp_min(1)
    intersection = (p*y*valid).flatten(1).sum(1)
    denominator = ((p+y)*valid).flatten(1).sum(1)
    dice = 1 - 2*intersection / (denominator + 1e-6)
    return ((focal+dice) * nonempty).sum()


class FastUBRDetectorIntegration(UBRDetectorIntegration):
    def __init__(self, satellite, refiner=None, active_microbatch=1, inference_microbatch=4):
        self._native_controller = None
        self._set_microbatch(active_microbatch, inference_microbatch)
        super().__init__(satellite, refiner)

    def _set_microbatch(self, size, inference_size):
        if size not in (1, 2, 4) or inference_size not in (1, 2, 4):
            raise ValueError('engineering train/inference microbatch must be 1, 2 or 4')
        self.active_microbatch = int(size)
        self.inference_microbatch = int(inference_size)

    @classmethod
    def from_controller(cls, original, active_microbatch=1, inference_microbatch=4):
        """Engineering comparison on exactly the same model/parameters/hook.

        No second module owner or extra decoder hook. Do not run both
        controllers concurrently; the registered native capture is nonreentrant.
        """
        result = cls.__new__(cls)
        result.satellite, result.head = original.satellite, original.head
        result._native_controller = original
        result.last_stats = {}
        result._set_microbatch(active_microbatch, inference_microbatch)
        return result

    def decode_native(self, prepared, boxes):
        if self._native_controller is not None:
            return self._native_controller.decode_native(prepared, boxes)
        return super().decode_native(prepared, boxes)

    @contextmanager
    def _without_inner_checkpoint(self):
        # Outer checkpoint covers decoder + paste + entire UBR + loss already.
        previous = self.refiner.checkpoint_blocks
        self.refiner.checkpoint_blocks = False
        try:
            yield
        finally:
            self.refiner.checkpoint_blocks = previous

    def _context(self, prepared, pool, retain_native=False):
        pool.validate()
        masks, native_cache = [], []
        boxes, size = fcn_canvas(pool.boxes, prepared.meta, rescale=False)
        with self._prediction_context_mode():
            for begin, end in self._prediction_chunks(pool):
                native, feature = self.decode_native(prepared, pool.boxes[begin:end])
                # No 32-channel image-canvas paste in the context pass.
                probability = paste_native_roi(native.sigmoid(), boxes[begin:end], size,
                                               cpu_bbox_clip=boxes.device.type == 'cpu')
                masks.append(probability[:, 0] >= .4)
                if retain_native:
                    # Native grid/dtype, NOT 32x512x512 image features. Evaluation
                    # gate measures actual dimensions/memory; never cast to save RAM.
                    native_cache.append((begin, end, native.detach(), feature.detach()))
        mask = torch.cat(masks) if masks else pool.boxes.new_empty((0, *size), dtype=torch.bool)
        ids = pool.candidate_ids.detach().clone()
        images = torch.zeros_like(ids)
        context = FastPredictionContext(ids, images, tuple(pool.source_ids), ids.clone(),
            ids.clone(), images.clone(), tuple(pool.source_ids), mask.detach(), .4,
            mask.to(torch.float32).sum(0), dict(zip(ids.cpu().tolist(), range(len(ids)))))
        return context, native_cache

    def prediction_context(self, prepared, pool):
        return self._context(prepared, pool)[0]

    def training_loss(self, prepared, pool, gt_boxes, gt_masks, geometry=None):
        if not torch.is_grad_enabled():
            raise ValueError('active UBR training must be gradient enabled')
        _, size = fcn_canvas(pool.boxes, prepared.meta)
        if gt_masks.shape != (len(gt_boxes), *size):
            raise ValueError('full-image GT masks must match canvas and GT order')
        if geometry is not None and geometry.shape != (1, 5, *size):
            raise ValueError('geometry must already share exact resized-image pixel canvas')
        context = self.prediction_context(prepared, pool)
        active, assigned = active_prediction_indices(pool, gt_boxes)
        raw = sum(parameter.sum()*0 for parameter in self.refiner.parameters())
        nonempty = 0
        active_list, assigned_list = active.tolist(), assigned.tolist()
        for start in range(0, len(active_list), self.active_microbatch):
            indices = active_list[start:start+self.active_microbatch]
            gt_indices = assigned_list[start:start+self.active_microbatch]
            target = gt_masks[gt_indices, None].to(device=pool.boxes.device, dtype=torch.float32)
            neighbors = torch.stack([context.neighbors(int(pool.candidate_ids[i])) for i in indices])[:, None]
            local_geometry = geometry.expand(len(indices), -1, -1, -1) if geometry is not None else None
            # Local to THIS graph/microbatch. The first forward populates exact
            # detached state; checkpoint replay reuses it without EDT/CPU sync.
            state_cache = {}

            def active_loss(box, target=target, neighbors=neighbors,
                            local_geometry=local_geometry, cache=state_cache):
                # Preserve old decoder batch1 kernels and RNG call order.
                # Only the expensive image-grid paste/refiner/loss is batched.
                # Native128 feature chunks are small; do not conflate native
                # SAM batching with UBR's 512-pixel engineering microbatch.
                native_parts = [self.decode_native(prepared, box[i:i+1])
                                for i in range(len(box))]
                logits = torch.cat([part[0] for part in native_parts])
                feature = torch.cat([part[1] for part in native_parts])
                state = adapt_native_roi(logits, feature, box, prepared.meta)
                if not cache:
                    sdt = signed_distance_transform(state.probabilities.detach() >= .4)
                    band = band_mask(sdt, self.refiner.band_width)
                    cache.update(sdt=sdt, band=band,
                                 nonempty=int(band.flatten(1).any(1).sum()))
                with self._without_inner_checkpoint():
                    refined = self.refiner(state.decoder_features, state.logits,
                                            cache['sdt'], neighbors, local_geometry)
                return _band_instance_sums(refined, target, cache['band'])

            with set_checkpoint_early_stop(False):
                value = checkpoint(active_loss, pool.boxes[indices],
                                   use_reentrant=False, preserve_rng_state=True)
            raw = raw + value
            nonempty += state_cache['nonempty']
        self.last_stats = dict(context_count=len(pool.boxes), active_count=len(active),
            active_redecode_count=len(active), nonempty_band_count=nonempty,
            active_microbatch=self.active_microbatch,
            active_decode_calls=len(active),
            active_microbatch_calls=(len(active)+self.active_microbatch-1)//self.active_microbatch)
        return {'loss_ubr': .1 * raw / max(nonempty, 1)}

    @torch.no_grad()
    def predict(self, prepared, pool, geometry=None):
        if self.satellite.training:
            raise ValueError('predict requires satellite.eval()')
        context, cache = self._context(prepared, pool, retain_native=True)
        probabilities = []
        cache_bytes = sum(t.numel()*t.element_size() for row in cache for t in row[2:])
        for begin, end, native, feature in cache:
            for offset in range(0, end-begin, self.inference_microbatch):
                count = min(self.inference_microbatch, end-begin-offset)
                first = begin + offset
                state = adapt_native_roi(native[offset:offset+count], feature[offset:offset+count],
                                         pool.boxes[first:first+count], prepared.meta)
                if geometry is not None and geometry.shape != (1, 5, *state.image_size):
                    raise ValueError('geometry must already share exact resized-image pixel canvas')
                sdt = signed_distance_transform(state.probabilities.detach() >= .4)
                neighbor = torch.stack([context.neighbors(int(pool.candidate_ids[i]))
                                        for i in range(first, first+count)])[:, None]
                local_geometry = geometry.expand(count, -1, -1, -1) if geometry is not None else None
                with self._without_inner_checkpoint():
                    refined = self.refiner(state.decoder_features, state.logits, sdt, neighbor, local_geometry)
                probabilities.append(residual_probabilities(state, refined)[:, 0])
        probabilities = torch.cat(probabilities) if probabilities else context.retained_masks.float()
        self.last_stats = dict(context_count=len(pool.boxes), native_cache_bytes=cache_bytes,
            native_shape=list(cache[0][2].shape[-2:]) if cache else None,
            inference_microbatch=self.inference_microbatch)
        return dict(probabilities=probabilities, masks=probabilities >= .4,
                    boxes=pool.boxes.clone(), scores=pool.scores.clone(),
                    candidate_ids=pool.candidate_ids.clone())


def install_fast_ubr(satellite, refiner=None, active_microbatch=1, inference_microbatch=4):
    return FastUBRDetectorIntegration(satellite, refiner, active_microbatch, inference_microbatch)
