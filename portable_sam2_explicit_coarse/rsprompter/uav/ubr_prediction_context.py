"""Prediction-only, synchronous neighbour context for a future shared UBR.

This module is deliberately not wired into a detector. Both training and inference
must call it with the same *predicted*, pre-refinement candidate population. Positive
training RoIs and ground-truth instances are not valid substitutes for that population.
Boxes are continuous xyxy in the common image coordinate system; masks are already
aligned probabilities on the common image grid (never logits or resized RoI masks).
Candidate IDs must be globally unique in this call and stable across input ordering.
Use a source-namespaced ID allocator upstream; source IDs alone are not identities.
Production use is two-stage: ``select_prediction_candidates`` first performs NMS
using metadata only; decode masks only for ``selection.retained_indices``; finally
``selection.with_masks(probabilities, retained_ids)`` freezes the neighbour context.
The all-masks ``build_prediction_context`` wrapper is a reference convenience API.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class PredictionContext:
    """Detached snapshot, including lineage for suppressed and capped candidates.

    ``group_ids`` maps each original candidate to its NMS representative, including
    groups dropped by max_per_image. A suppressed target excludes its retained group
    representative. A target in a capped group has no retained self contribution to
    subtract and sees all retained neighbours in its image. Unknown IDs raise KeyError.
    The tensors are detached copies; callers must treat this snapshot as read-only.
    """

    candidate_ids: Tensor
    image_ids: Tensor
    source_ids: Tuple[str, ...]
    group_ids: Tensor
    retained_ids: Tensor
    retained_image_ids: Tensor
    retained_source_ids: Tuple[str, ...]
    retained_masks: Tensor
    probability_threshold: float

    @torch.no_grad()
    def neighbors(self, candidate_id: int) -> Tensor:
        """Sum other retained binary masks as float32, without iterative updates."""
        matches = torch.nonzero(self.candidate_ids == candidate_id, as_tuple=False).flatten()
        if matches.numel() != 1:
            raise KeyError("candidate_id was not in the predicted context: %s" % candidate_id)
        index = int(matches.item())
        keep = ((self.retained_image_ids == self.image_ids[index]) &
                (self.retained_ids != self.group_ids[index]))
        return self.retained_masks[keep].to(torch.float32).sum(dim=0)

    def lineage(self) -> Dict[int, int]:
        """Original candidate ID -> pre-cap NMS representative ID."""
        return dict(zip(self.candidate_ids.cpu().tolist(), self.group_ids.cpu().tolist()))


def _iou_one(box: Tensor, others: Tensor) -> Tensor:
    lo = torch.maximum(box[:2], others[:, :2])
    hi = torch.minimum(box[2:], others[:, 2:])
    intersection = (hi - lo).clamp(min=0).prod(dim=1)
    area = (box[2:] - box[:2]).prod()
    other_area = (others[:, 2:] - others[:, :2]).prod(dim=1)
    return intersection / (area + other_area - intersection)


@dataclass(frozen=True)
class PredictionSelection:
    """Metadata-only NMS result; indices refer to the original candidate tensors."""

    candidate_ids: Tensor
    image_ids: Tensor
    source_ids: Tuple[str, ...]
    group_ids: Tensor
    retained_indices: Tensor
    retained_ids: Tensor
    retained_image_ids: Tensor
    retained_source_ids: Tuple[str, ...]

    @torch.no_grad()
    def with_masks(
        self,
        mask_probabilities: Tensor,
        candidate_ids: Tensor,
        *,
        probability_threshold: float = 0.4,
    ) -> PredictionContext:
        """Attach decoded, image-aligned probabilities in retained-ID order only.

        IDs are required to reject accidental attachment in a different order. This
        operation does not decode, resize, sigmoid, or refine the supplied masks.
        """
        if candidate_ids.device != self.retained_ids.device or candidate_ids.dtype != torch.int64:
            raise ValueError("attached candidate IDs must be int64 on the selection device")
        if not torch.equal(candidate_ids, self.retained_ids):
            raise ValueError("attached candidate IDs must exactly match retained order")
        _validate_probabilities(mask_probabilities, len(self.retained_ids),
                                self.retained_ids.device, probability_threshold)
        return PredictionContext(
            candidate_ids=self.candidate_ids.clone(), image_ids=self.image_ids.clone(),
            source_ids=self.source_ids, group_ids=self.group_ids.clone(),
            retained_ids=self.retained_ids.clone(),
            retained_image_ids=self.retained_image_ids.clone(),
            retained_source_ids=self.retained_source_ids,
            retained_masks=(mask_probabilities >= probability_threshold).detach().clone(),
            probability_threshold=float(probability_threshold),
        )


def _validate_probabilities(probabilities: Tensor, n: int, device: torch.device,
                            threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError("probability threshold must be in [0, 1]")
    if probabilities.ndim != 3 or probabilities.shape[0] != n:
        raise ValueError("mask probabilities must be [N,H,W] on the image grid")
    if min(probabilities.shape[1:]) < 1:
        raise ValueError("mask grid must be nonempty")
    if probabilities.device != device or not probabilities.is_floating_point():
        raise ValueError("mask probabilities must be floating point on the selection device")
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("mask probabilities must be finite")
    if bool(((probabilities < 0) | (probabilities > 1)).any()):
        raise ValueError("mask probabilities must be in [0,1]; logits are not accepted")


@torch.no_grad()
def select_prediction_candidates(
    boxes: Tensor,
    scores: Tensor,
    image_ids: Tensor,
    candidate_ids: Tensor,
    source_ids: Tuple[str, ...],
    *,
    nms_iou_threshold: float,
    max_per_image: int = 250,
) -> PredictionSelection:
    """Select metadata before mask decoding, without an invented score filter.

    NMS is class-agnostic for this single-class task, within each image only.
    Order is image ID ascending, score descending, stable candidate ID ascending.
    Suppression uses IoU > threshold (matching conventional box NMS). Scores and
    boxes use float64 internally so half precision does not overflow box areas.
    Max 250 is an explicit cap, not a source quota; training sampling is upstream.
    """
    if not isinstance(max_per_image, int) or isinstance(max_per_image, bool) or not 1 <= max_per_image <= 250:
        raise ValueError("max_per_image must be an integer in [1, 250]")
    if not 0 <= nms_iou_threshold <= 1:
        raise ValueError("NMS threshold must be in [0, 1]")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must be [N,4] xyxy")
    n = boxes.shape[0]
    if scores.shape != (n,) or image_ids.shape != (n,) or candidate_ids.shape != (n,):
        raise ValueError("scores/image_ids/candidate_ids must be [N]")
    tensors = (boxes, scores, image_ids, candidate_ids)
    if any(t.device != boxes.device for t in tensors):
        raise ValueError("all tensors must share a device")
    if image_ids.dtype != torch.int64 or candidate_ids.dtype != torch.int64:
        raise ValueError("image_ids and candidate_ids must be int64")
    if torch.unique(candidate_ids).numel() != n:
        raise ValueError("candidate IDs must be globally unique")
    if len(source_ids) != n or any(not isinstance(s, str) or not s.strip() or s != s.strip() for s in source_ids):
        raise ValueError("one nonempty canonical source string per candidate is required")
    if not all(t.is_floating_point() for t in (boxes, scores)):
        raise ValueError("boxes and scores must be floating point")
    if not all(bool(torch.isfinite(t).all()) for t in (boxes, scores)):
        raise ValueError("prediction inputs must be finite")
    if bool((boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("boxes must have positive width and height")

    b = boxes.detach().to(torch.float64)
    # Sorting integer/score metadata in Python gives a deterministic tie rule across
    # torch versions and devices. This reference helper is not a fused NMS kernel.
    ids = candidate_ids.cpu().tolist()
    images = image_ids.cpu().tolist()
    score_values = scores.detach().cpu().tolist()
    order = sorted(range(n), key=lambda i: (images[i], -score_values[i], ids[i]))
    groups = candidate_ids.detach().clone()
    retained = []
    for image in sorted(set(images)):
        remaining = [i for i in order if images[i] == image]
        representatives = []
        while remaining:
            current = remaining[0]
            representatives.append(current)
            if len(remaining) == 1:
                break
            rest = torch.tensor(remaining[1:], dtype=torch.int64, device=boxes.device)
            suppressed = _iou_one(b[current], b[rest]) > nms_iou_threshold
            groups[rest[suppressed]] = candidate_ids[current]
            remaining = rest[~suppressed].cpu().tolist()
        retained.extend(representatives[:max_per_image])
    retained_indices = torch.tensor(retained, dtype=torch.int64, device=boxes.device)
    return PredictionSelection(
        candidate_ids=candidate_ids.detach().clone(),
        image_ids=image_ids.detach().clone(),
        source_ids=tuple(source_ids),
        group_ids=groups,
        retained_indices=retained_indices,
        retained_ids=candidate_ids[retained_indices].detach().clone(),
        retained_image_ids=image_ids[retained_indices].detach().clone(),
        retained_source_ids=tuple(source_ids[i] for i in retained),
    )


@torch.no_grad()
def build_prediction_context(
    boxes: Tensor,
    scores: Tensor,
    mask_probabilities: Tensor,
    image_ids: Tensor,
    candidate_ids: Tensor,
    source_ids: Tuple[str, ...],
    *,
    nms_iou_threshold: float,
    probability_threshold: float = 0.4,
    max_per_image: int = 250,
) -> PredictionContext:
    """Reference wrapper when probabilities for all candidates already exist.

    Production should select metadata first, decode retained masks only, then call
    selection.with_masks. Probability threshold uses >= explicitly in both APIs.
    """
    selection = select_prediction_candidates(
        boxes, scores, image_ids, candidate_ids, source_ids,
        nms_iou_threshold=nms_iou_threshold, max_per_image=max_per_image)
    _validate_probabilities(mask_probabilities, boxes.shape[0], boxes.device,
                            probability_threshold)
    return selection.with_masks(mask_probabilities[selection.retained_indices],
                                selection.retained_ids,
                                probability_threshold=probability_threshold)
