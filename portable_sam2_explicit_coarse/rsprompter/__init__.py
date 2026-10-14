"""Registry facade; light mode is reserved for isolated tensor tests.

Implementations live in domain subpackages:

detection/   anchor_detector.py, anchor_roi_head.py, sam2_neck.py
backbone/    sam2_vision.py (LoRA encoder + strict loaders)
cspm/        mask_head{,_helpers,_targets}.py, shape_prior_injector.py,
             shape_point_miner.py, dense_prompt_utils.py, coarse_mask_loss.py
decoder/     sam2_decoder.py, decoder_tail_refiner.py

Registry names and every externally imported symbol are unchanged.
"""
import os

if os.environ.get("RSPROMPTER_LIGHT_IMPORT", "0") != "1":
    from .detection.anchor_detector import RSPrompterAnchor
    from .detection.anchor_roi_head import RSPrompterAnchorRoIPromptHead
    from .backbone.sam2_vision import (
        RSSAM2PositionalEmbedding,
        RSSAM2VisionEncoder,
        _load_pretrained_no_mask_embedding,
        _load_sam2_checkpoint,
        _resolve_sam2_checkpoint_path,
    )
    from .decoder.sam2_decoder import RSSAM2MaskDecoderWrapper
    from .detection.sam2_neck import RSSAM2PAFPN
    from .cspm.mask_head import RSPrompterAnchorMaskHeadSAM2
    from .cspm.mask_head_helpers import _MaskHeadPromptHelpers
    from .cspm.mask_head_targets import _MaskHeadTargetsMixin
    from .cspm.shape_prior_injector import ShapePriorInjector, SmallMaskDecoder
    from .cspm.shape_point_miner import ShapePointMiner
    from .detection.dual_detector import RSPrompterUAVDual
