"""Compatibility facade: implementations live in sibling modules.

sam2_vision.py            helpers + RSSAM2PositionalEmbedding + RSSAM2VisionEncoder
sam2_decoder.py           RSSAM2MaskDecoderWrapper
sam2_neck.py              RSFeatureAggregatorSAM2 + RSSAM2PAFPN
sam2_mask_head.py         RSPrompterAnchorMaskHeadSAM2 (init/forward/predict)
sam2_mask_head_helpers.py _MaskHeadPromptHelpers mixin
sam2_mask_head_targets.py _MaskHeadTargetsMixin mixin
Registry names and every external import path are unchanged.
"""
from .sam2_vision import (
    RSSAM2PositionalEmbedding,
    RSSAM2VisionEncoder,
    _DiscardedIoUPredictionHead,
    _UnusedPromptEmbedding,
    _load_pretrained_no_mask_embedding,
    _load_sam2_checkpoint,
    _resolve_sam2_checkpoint_path,
)
from .sam2_decoder import RSSAM2MaskDecoderWrapper
from .sam2_neck import RSFeatureAggregatorSAM2, RSSAM2PAFPN
from .sam2_mask_head import RSPrompterAnchorMaskHeadSAM2
from .sam2_mask_head_helpers import _MaskHeadPromptHelpers
from .sam2_mask_head_targets import _MaskHeadTargetsMixin

__all__ = [
    "RSSAM2PositionalEmbedding",
    "RSSAM2VisionEncoder",
    "RSSAM2MaskDecoderWrapper",
    "RSFeatureAggregatorSAM2",
    "RSSAM2PAFPN",
    "RSPrompterAnchorMaskHeadSAM2",
]
