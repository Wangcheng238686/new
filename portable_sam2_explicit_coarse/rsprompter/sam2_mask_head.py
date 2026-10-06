"""SAM2 mask head: config wiring (__init__), the a3sgm forward chain, and
per-RoI mask post-processing.

The forward chain is the paper's PromptMiner-SAM2 core (architecture guide
docs/a3sgm_architecture_guide.md, Step 4-11):

    RoI feature ─► ShapePriorInjector (coarse 64x64 logits, ROI-local)
                        ├─► dense canvas  (raw logits, sg-detached, Step 5)
                        └─► ShapePointMiner (2P2N points, Step 6)
    points + box + dense mask ─► frozen SAM2 PromptEncoder (Step 7)
    PE output ─► dense embedding = no_mask base + gated residual (Step 8)
    ─► SAM2 MaskDecoder (Step 9) ─► UDPR margin tail (Step 10)
    ─► roi_local paste / full_image resize post-processing (Step 11)

Prompt-modality ablations (paper Table: P / B / P+B / P+B+M) are driven by
``explicit_prompt_mode``; the two entry contracts differ in
``final_mask_coordinate_mode`` (WHU fast150 = roi_local, NWPU matrix300 =
full_image) and both predict paths below are load-bearing for the two paper
checkpoints.
"""

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.registry import MODELS
from mmdet.utils import ConfigType

from .ckpt_utils import load_module_state_dict_strict
from .decoder_tail_refiner import DecoderTailPointRefiner
from .dense_prompt_utils import replace_invalid_dense_with_base
from .sam2_decoder import RSSAM2MaskDecoderWrapper
from .sam2_vision import (
    SAM2_REPO,
    _load_pretrained_no_mask_embedding,
    _load_sam2_checkpoint,
    _resolve_sam2_checkpoint_path,
)

from .sam2_mask_head_helpers import _MaskHeadPromptHelpers
from .sam2_mask_head_targets import _MaskHeadTargetsMixin

# Sparse prompt route is fixed to the explicit shape-point path; the legacy
# RSPrompter MLP point embeddings ("point"/"point_box" modes) were baseline
# arms and are gone. The final-mask supervision coordinate contract, however,
# has two live values (see class docstring).
_VALID_EXPLICIT_PROMPT_MODES = {"points", "box", "points_box", "points_box_dense"}


@MODELS.register_module()
class RSPrompterAnchorMaskHeadSAM2(_MaskHeadPromptHelpers, _MaskHeadTargetsMixin, FCNMaskHead, BaseModule):
    """SAM2-compatible Mask Head for RSPrompter (a3sgm form)."""

    def __init__(
        self,
        sam2_mask_decoder,
        in_channels,
        roi_feat_size=14,
        per_pointset_point=5,
        with_sincos=True,
        multimask_output=False,
        attention_similarity=None,
        target_embedding=None,
        output_attentions=None,
        class_agnostic=False,
        prompt_sparse_mode: str = "point",
        explicit_prompt_mode: Optional[str] = None,
        prompt_encoder_enabled: bool = False,
        prompt_encoder_image_size: int = 512,
        prompt_encoder_embed_size: int = 16,
        load_pe_pretrained: bool = True,
        sam2_ckpt_for_pe: Optional[str] = None,
        prompt_encoder_cfg: Optional[Dict] = None,
        dense_prompt_mode: str = "none",
        prune_unused_prompt_components: bool = False,
        shape_base_dense_mode: str = "no_mask",
        coarse_mask_coordinate_mode: str = "roi_local",
        final_mask_coordinate_mode: str = "roi_local",
        final_mask_loss_cfg: Optional[Dict] = None,
        roi_sam_cfg: Optional[Dict] = None,
        mask_coordinate_mode: Optional[str] = None,
        roi_mask_size: int = 28,
        checkpoint_load_cfg: Optional[Dict] = None,
        coarse_mask_loss_cfg: Optional[Dict] = None,
        dense_prompt_cfg: Optional[Dict] = None,
        restrict_dense_prompt_to_box: bool = True,
        shape_point_miner_cfg: Optional[Dict] = None,
        point_warmup_cfg: Optional[Dict] = None,
        mask_prompt_box_cfg: Optional[Dict] = None,
        box_prompt_cfg: Optional[Dict] = None,
        freeze_mask_decoder: bool = False,
        freeze_no_mask_embed: bool = True,
        load_no_mask_pretrained: bool = True,
        shape_prior_cfg: Optional[Dict] = None,
        shape_prior_loss_weight: float = 0.5,
        # Legacy arms removed by the a3sgm-only cleanup. The parameters stay
        # in the signature because checkpoints embed these config keys and
        # MODELS.build passes every key as a kwarg; enabling any of them is a
        # hard error instead of a silent no-op.
        dense_capacity_cfg: Optional[Dict] = None,
        canvas_renderer_cfg: Optional[Dict] = None,
        p2_boundary_refiner_cfg: Optional[Dict] = None,
        decoder_tail_refiner_cfg: Optional[Dict] = None,
        quality_head_cfg: Optional[Dict] = None,
        segm_score_mode: str = "detector",
        loss_mask: ConfigType = dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0),
        init_cfg=None,
        *args,
        **kwargs,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self._reject_removed_arms(
            dense_capacity_cfg, canvas_renderer_cfg, p2_boundary_refiner_cfg,
            quality_head_cfg, roi_sam_cfg, prune_unused_prompt_components,
        )

        # --- decoder-tail (UDPR) config ------------------------------------
        self._decoder_tail_cfg = dict(decoder_tail_refiner_cfg or {})
        self._decoder_tail_cfg.setdefault("enabled", False)
        self.decoder_tail_enabled = bool(self._decoder_tail_cfg.pop("enabled"))
        self.decoder_tail_refiner = None
        self._last_tail_outputs = None

        # --- prompt modality (paper ablation axis) -------------------------
        self.prompt_sparse_mode = str(prompt_sparse_mode)
        if self.prompt_sparse_mode != "shape_point":
            raise ValueError(
                "prompt_sparse_mode must be 'shape_point' (the legacy MLP "
                "point/point_box routes were removed); got "
                f"{self.prompt_sparse_mode!r}"
            )
        if explicit_prompt_mode is None:
            # Legacy shape-point configs always included box tokens; infer
            # only whether the dense mask was enabled.
            explicit_prompt_mode = (
                "points_box_dense"
                if bool((shape_prior_cfg or {}).get("use_shape_dense", True))
                else "points_box"
            )
        self.explicit_prompt_mode = str(explicit_prompt_mode).strip().lower()
        if self.explicit_prompt_mode not in _VALID_EXPLICIT_PROMPT_MODES:
            raise ValueError(
                "explicit_prompt_mode must be one of "
                f"{sorted(_VALID_EXPLICIT_PROMPT_MODES)}, got "
                f"{self.explicit_prompt_mode!r}"
            )
        self.explicit_use_point_prompt = self.explicit_prompt_mode in {
            "points", "points_box", "points_box_dense"
        }
        self.explicit_use_box_prompt = self.explicit_prompt_mode in {
            "box", "points_box", "points_box_dense"
        }
        self.explicit_use_dense_prompt = (
            self.explicit_prompt_mode == "points_box_dense"
        )

        self.prompt_encoder_enabled = bool(prompt_encoder_enabled)
        self.dense_prompt_mode = str(dense_prompt_mode)
        if self.dense_prompt_mode not in ("none", ""):
            raise ValueError(
                f"Unsupported dense_prompt_mode={self.dense_prompt_mode!r}; "
                "the legacy 'residual' branch was removed, use dense_prompt_cfg."
            )
        self.point_warmup_cfg = dict(point_warmup_cfg or {})
        self.mask_prompt_box_cfg = dict(mask_prompt_box_cfg or {})
        self.prompt_encoder_image_size = int(prompt_encoder_image_size)
        self.prompt_encoder_embed_size = int(prompt_encoder_embed_size)
        self.prompt_encoder_mask_size = self.prompt_encoder_embed_size * 4
        # Historical Mask R-CNN knob; every target builder passes its own
        # mask_size explicitly, so this is only kept as an accepted attribute.
        self.roi_mask_size = int(roi_mask_size)

        # --- final-mask coordinate contract --------------------------------
        # Coarse logits are always ROI-local; the final SAM2 logits use either
        # ROI-local (WHU fast150 entry) or the native full-image grid (NWPU
        # matrix300 entry). Both predict/loss paths are live for the paper
        # checkpoints, so neither may be removed.
        self.coarse_mask_coordinate_mode = str(coarse_mask_coordinate_mode).lower()
        self.final_mask_coordinate_mode = str(final_mask_coordinate_mode).lower()
        if mask_coordinate_mode is not None:
            # Pre-split configs passed one flag for both semantics.
            self.coarse_mask_coordinate_mode = str(mask_coordinate_mode).lower()
            self.final_mask_coordinate_mode = str(mask_coordinate_mode).lower()
        for mode_name, mode_value in (
            ("coarse_mask_coordinate_mode", self.coarse_mask_coordinate_mode),
            ("final_mask_coordinate_mode", self.final_mask_coordinate_mode),
        ):
            if mode_value not in ("full_image", "roi_local"):
                raise ValueError(
                    f"{mode_name} must be 'full_image' or 'roi_local', got {mode_value!r}"
                )
        if self.coarse_mask_coordinate_mode != "roi_local":
            raise ValueError(
                "coarse_mask_coordinate_mode must be 'roi_local' (coarse mask "
                "is generated and supervised per-ROI)"
            )

        self._init_final_mask_loss_cfg(final_mask_loss_cfg)

        # --- PromptEncoder strict-loading / freezing contract ---------------
        self.checkpoint_load_cfg = dict(checkpoint_load_cfg or {})
        pe_cfg = dict(prompt_encoder_cfg or {})
        unknown_pe_keys = set(pe_cfg) - {
            "require_pretrained", "ckpt_path", "freeze_all", "train_mask_downscaling",
        }
        if unknown_pe_keys:
            raise ValueError(
                f"Unknown prompt_encoder_cfg keys: {sorted(unknown_pe_keys)}"
            )
        pe_cfg.setdefault("require_pretrained", bool(load_pe_pretrained))
        pe_cfg.setdefault("ckpt_path", sam2_ckpt_for_pe)
        pe_cfg.setdefault("freeze_all", True)
        pe_cfg.setdefault("train_mask_downscaling", False)
        self.prompt_encoder_cfg = pe_cfg
        self._pe_require_pretrained = bool(pe_cfg.get("require_pretrained", True))
        self._pe_cfg_path = pe_cfg.get("ckpt_path")  # None: reuse decoder ckpt

        # Segm scores come from the detector head; the G1/G2 quality-rerank
        # contract was removed together with its arm.
        self.segm_score_mode = str(segm_score_mode).strip().lower()
        if self.segm_score_mode != "detector":
            raise ValueError(
                "segm_score_mode must be 'detector' (the mask_quality rerank "
                f"arm was removed), got {self.segm_score_mode!r}"
            )
        self.segm_score_key = "scores"

        self.shape_base_dense_mode = str(shape_base_dense_mode)
        self.load_no_mask_pretrained = bool(load_no_mask_pretrained)
        self.freeze_no_mask_embed = bool(freeze_no_mask_embed)
        if self.shape_base_dense_mode not in {"no_mask", "zero"}:
            raise ValueError(
                "shape_base_dense_mode must be 'no_mask' or 'zero', got "
                f"{self.shape_base_dense_mode!r}"
            )

        # --- SAM2 mask decoder ----------------------------------------------
        sam2_mask_decoder = dict(sam2_mask_decoder)
        # The tail consumes the decoder's upscaled 32-ch feature map, so the
        # wrapper must capture it whenever the tail is on.
        sam2_mask_decoder["capture_upscaled_embedding"] = self.decoder_tail_enabled
        self.mask_decoder = RSSAM2MaskDecoderWrapper(**sam2_mask_decoder)
        if freeze_mask_decoder:
            for parameter in self.mask_decoder.parameters():
                parameter.requires_grad_(False)
            self.mask_decoder.eval()

        self._init_no_mask_embedding(sam2_mask_decoder)
        self._init_prompt_encoder(sam2_mask_decoder, bool(load_pe_pretrained))
        self._init_shape_prior(
            shape_prior_cfg, coarse_mask_loss_cfg, dense_prompt_cfg,
            box_prompt_cfg, in_channels, shape_prior_loss_weight,
            restrict_dense_prompt_to_box, shape_point_miner_cfg,
        )
        self._init_point_miner(shape_point_miner_cfg)
        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic
        self._init_decoder_tail()

        self.assert_no_mask_embedding_contract("model construction")

    # ------------------------------------------------------------------
    # __init__ helpers (split so each states one contract)
    # ------------------------------------------------------------------

    def _reject_removed_arms(
        self, dense_capacity_cfg, canvas_renderer_cfg,
        p2_boundary_refiner_cfg, quality_head_cfg, roi_sam_cfg,
        prune_unused_prompt_components,
    ) -> None:
        """Hard-error on any removed-arm config instead of silently no-op'ing.

        Checkpoints embed these keys (all ``enabled=False`` for a3sgm), so the
        keys must stay acceptable while the arms themselves are gone.
        """
        removed = {
            "dense_capacity_cfg": dense_capacity_cfg,
            "canvas_renderer_cfg": canvas_renderer_cfg,
            "p2_boundary_refiner_cfg": p2_boundary_refiner_cfg,
            "quality_head_cfg": quality_head_cfg,
            "roi_sam_cfg": roi_sam_cfg,
        }
        for name, cfg in removed.items():
            if cfg and cfg.get("enabled", False):
                raise ValueError(
                    f"{name} belongs to a removed development arm and cannot "
                    "be enabled in the a3sgm-only codebase"
                )
        if prune_unused_prompt_components:
            raise ValueError(
                "prune_unused_prompt_components belonged to removed baseline "
                "arms and cannot be enabled"
            )

    def _init_final_mask_loss_cfg(self, final_mask_loss_cfg) -> None:
        cfg = dict(final_mask_loss_cfg or {})
        allowed = {
            "mode", "roi_expand_ratio", "roi_bce_weight",
            "roi_dice_weight", "outside_bce_weight", "eps",
        }
        unknown = set(cfg) - allowed
        if unknown:
            raise ValueError(f"Unknown final_mask_loss_cfg keys: {sorted(unknown)}")
        cfg.setdefault("mode", "standard")
        cfg.setdefault("roi_expand_ratio", 1.20)
        cfg.setdefault("roi_bce_weight", 1.0)
        cfg.setdefault("roi_dice_weight", 1.0)
        cfg.setdefault("outside_bce_weight", 0.05)
        cfg.setdefault("eps", 1e-6)
        cfg["mode"] = str(cfg["mode"]).strip().lower()
        if cfg["mode"] not in {"standard", "roi_balanced_dice"}:
            raise ValueError(
                "final_mask_loss_cfg.mode must be standard or "
                f"roi_balanced_dice, got {cfg['mode']!r}"
            )
        if self.final_mask_coordinate_mode != "full_image" and cfg["mode"] != "standard":
            raise ValueError(
                "ROI-focused final-mask loss requires final_mask_coordinate_mode='full_image'"
            )
        if float(cfg["roi_expand_ratio"]) < 1.0:
            raise ValueError("final-mask ROI expand ratio must be >= 1.0")
        for weight_key in ("roi_bce_weight", "roi_dice_weight", "outside_bce_weight"):
            if float(cfg[weight_key]) < 0:
                raise ValueError(f"final-mask {weight_key} must be non-negative")
        if float(cfg["eps"]) <= 0:
            raise ValueError("final-mask loss eps must be positive")
        self.final_mask_loss_cfg = ConfigDict(cfg)

    def _init_no_mask_embedding(self, sam2_mask_decoder) -> None:
        """Frozen SAM2 no-mask dense base; must come from the pretrained ckpt."""
        if self.shape_base_dense_mode != "no_mask":
            self.no_mask_pretrained_loaded = False
            self.register_parameter("no_mask_embed", None)
            self.register_buffer(
                "_pretrained_no_mask_reference", torch.empty(0), persistent=False
            )
            if is_main_process():
                print("[MaskHead] base dense disabled")
            return
        # C1-C5 require the exact SAM2 embedding; a missing path/key is a hard
        # error rather than a silent zero fallback.
        no_mask_ckpt = self._pe_cfg_path or sam2_mask_decoder.get("checkpoint_path")
        if self.load_no_mask_pretrained:
            no_mask = _load_pretrained_no_mask_embedding(no_mask_ckpt)
            self.no_mask_pretrained_loaded = True
        else:
            no_mask = torch.zeros(1, 256, 1, 1)
            self.no_mask_pretrained_loaded = False
        self.no_mask_embed = nn.Parameter(no_mask)
        self.no_mask_embed.requires_grad_(not self.freeze_no_mask_embed)
        self.register_buffer(
            "_pretrained_no_mask_reference",
            no_mask.detach().clone() if self.no_mask_pretrained_loaded else torch.empty(0),
            persistent=False,
        )
        if is_main_process():
            source = "SAM2 ckpt" if self.no_mask_pretrained_loaded else "intentional zeros"
            state = "frozen" if self.freeze_no_mask_embed else "trainable"
            abs_mean = float(self.no_mask_embed.detach().abs().mean().item())
            print(
                f"[MaskHead] Initialized no_mask_embed from {source}, "
                f"{state}, abs_mean={abs_mean:.8f}"
            )

    def _init_prompt_encoder(self, sam2_mask_decoder, load_pe_pretrained: bool) -> None:
        """Frozen SAM2 PromptEncoder; only the dense-mask stack may train.

        train_mask_downscaling differs between the two entries (NWPU trains
        the 4,684-parameter stack at 0.1x LR, WHU freezes it), so it stays a
        resolved config field rather than a launcher env.
        """
        if not self.prompt_encoder_enabled:
            self.prompt_encoder = None
            return
        if SAM2_REPO and SAM2_REPO not in sys.path:
            sys.path.insert(0, SAM2_REPO)
        from sam2.modeling.sam.prompt_encoder import PromptEncoder

        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(
                int(self.prompt_encoder_embed_size),
                int(self.prompt_encoder_embed_size),
            ),
            input_image_size=(
                int(self.prompt_encoder_image_size),
                int(self.prompt_encoder_image_size),
            ),
            mask_in_chans=16,
        )
        if not bool(self.prompt_encoder_cfg.get("freeze_all", True)):
            raise ValueError(
                "Sparse-PE fine-tuning (freeze_all=False) belonged to removed "
                "arms; the canonical contract freezes all sparse embeddings"
            )
        for embedding in self.prompt_encoder.point_embeddings:
            embedding.weight.requires_grad_(False)
        self.prompt_encoder.not_a_point_embed.weight.requires_grad_(False)
        self.prompt_encoder.no_mask_embed.weight.requires_grad_(False)
        train_mask_downscaling = bool(
            self.prompt_encoder_cfg.get("train_mask_downscaling", False)
        )
        for parameter in self.prompt_encoder.mask_downscaling.parameters():
            parameter.requires_grad_(train_mask_downscaling)

        # Pretrained PE weights are mandatory by default: a randomly
        # initialized PE silently breaks geometric encoding while training
        # continues to "work". The raw flag still allows an explicit
        # opt-out when the stricter cfg key is unset.
        if not (load_pe_pretrained or self._pe_require_pretrained):
            return
        ckpt_path = _resolve_sam2_checkpoint_path(
            self._pe_cfg_path or sam2_mask_decoder.get("checkpoint_path")
        )
        if not (ckpt_path and os.path.exists(ckpt_path)):
            message = (
                f"PromptEncoder pretrained weights not found at {ckpt_path}. "
                f"require_pretrained={self._pe_require_pretrained}."
            )
            if self._pe_require_pretrained:
                raise RuntimeError(
                    f"[MaskHead] {message} Set prompt_encoder_cfg.require_pretrained=False "
                    "to allow random init, or provide a valid SAM2 checkpoint."
                )
            if is_main_process():
                print(f"[MaskHead] WARNING: {message} PE will be randomly initialized.")
            return
        ckpt = _load_sam2_checkpoint(ckpt_path)
        pe_state = {
            key[len("sam_prompt_encoder."):]: value
            for key, value in ckpt.items()
            if key.startswith("sam_prompt_encoder.")
        }
        if not pe_state:
            if self._pe_require_pretrained:
                raise RuntimeError(
                    f"[MaskHead] sam_prompt_encoder.* keys not found in {ckpt_path}, "
                    "but require_pretrained=True. Check checkpoint integrity / model_size match."
                )
            if is_main_process():
                print(f"[MaskHead] WARNING: no sam_prompt_encoder.* keys in {ckpt_path}")
            return
        load_module_state_dict_strict(
            self.prompt_encoder,
            pe_state,
            module_name="PromptEncoder",
            strict_reproduction=self.checkpoint_load_cfg.get("strict_reproduction", False),
            allowed_missing_patterns=tuple(
                self.checkpoint_load_cfg.get("allowed_missing_patterns", [])
            ),
            allowed_unexpected_patterns=tuple(
                self.checkpoint_load_cfg.get("allowed_unexpected_patterns", [])
            ),
        )
        if self.load_no_mask_pretrained:
            pe_no_mask = (
                self.prompt_encoder.no_mask_embed.weight.detach()
                .reshape_as(self.no_mask_embed)
            )
            if not torch.equal(pe_no_mask.cpu(), self.no_mask_embed.detach().cpu()):
                raise RuntimeError(
                    "PromptEncoder and MaskHead loaded different pretrained "
                    "no-mask embeddings"
                )

    def _init_shape_prior(
        self, shape_prior_cfg, coarse_mask_loss_cfg, dense_prompt_cfg,
        box_prompt_cfg, in_channels, shape_prior_loss_weight,
        restrict_dense_prompt_to_box, shape_point_miner_cfg,
    ) -> None:
        """Coarse head, its aux loss, the dense-canvas contract, and the gate.

        The dense gate is the bounded global-sigmoid form only
        (shape_dense_alpha_raw Parameter + shape_dense_alpha buffer); the
        legacy_unbounded/fixed ablation modes are gone, but BOTH state-dict
        keys must keep being registered (checkpoints carry both).
        """
        self.shape_prior_loss_weight = float(shape_prior_loss_weight)
        shape_prior_enabled = bool(
            shape_prior_cfg is not None and shape_prior_cfg.get("enabled", False)
        )
        if shape_prior_enabled:
            from .coarse_mask_loss import CoarseMaskLoss

            loss_cfg = dict(coarse_mask_loss_cfg or {})
            loss_cfg.setdefault("weight", self.shape_prior_loss_weight)
            loss_cfg.setdefault("bce_weight", 1.0)
            loss_cfg.setdefault("dice_weight", 1.0)
            loss_cfg.setdefault("boundary_weight", 0.0)
            loss_cfg.setdefault("distance_weight", 0.0)
            self.coarse_mask_loss_module = CoarseMaskLoss(**loss_cfg)
        else:
            self.coarse_mask_loss_module = None

        self.dense_prompt_cfg = dict(dense_prompt_cfg or {})
        self.restrict_dense_prompt_to_box = bool(restrict_dense_prompt_to_box)
        self.dense_prompt_cfg.setdefault("transform", "raw_logits")
        self.dense_prompt_cfg.setdefault("outside_fill_logit", 0.0)
        self.dense_prompt_cfg.setdefault("clamp_range", None)
        self.dense_prompt_cfg.setdefault("detach_input", False)
        if str(self.dense_prompt_cfg["transform"]) != "raw_logits":
            raise ValueError(
                "Unsupported dense prompt transform="
                f"{self.dense_prompt_cfg['transform']!r} (only raw_logits "
                "survives the a3sgm-only cleanup)"
            )

        self.shape_point_miner_cfg = dict(shape_point_miner_cfg or {})
        self.box_prompt_cfg = dict(box_prompt_cfg or {})
        self.box_prompt_cfg.setdefault("jitter_prob", 0.0)
        self.box_prompt_cfg.setdefault("jitter_scale", 0.05)
        self.box_prompt_cfg.setdefault("min_box_size", 2.0)
        self.box_prompt_cfg.setdefault("min_iou_with_original", 0.5)
        self.mask_prompt_box_cfg.setdefault("train_box_source", "proposal")
        self.mask_prompt_box_cfg.setdefault("test_box_source", "sabl_refined")
        self.mask_prompt_box_cfg.setdefault("fallback_to_proposal", True)
        self.mask_prompt_box_cfg.setdefault("min_refined_proposal_iou", 0.30)
        self.mask_prompt_box_cfg.setdefault("min_box_size", 2.0)
        self.mask_prompt_box_cfg.setdefault("clamp_to_image", True)
        self._last_shape_point_local_yx = None
        self._last_shape_point_labels = None

        if shape_prior_enabled:
            import math
            from .shape_prior import ShapePriorInjector

            # These keys are handled here, not by the injector.
            head_keys = {
                "enabled", "use_shape_dense", "shape_scale_mode",
                "prompt_scale_init", "shape_scale_init",
            }
            injector_cfg = {
                key: value for key, value in shape_prior_cfg.items()
                if key not in head_keys
            }
            injector_cfg.setdefault("roi_feat_channels", in_channels)
            self.shape_injector = ShapePriorInjector(**injector_cfg)

            self.shape_scale_mode = str(
                shape_prior_cfg.get("shape_scale_mode", "global_sigmoid")
            )
            if self.shape_scale_mode != "global_sigmoid":
                raise ValueError(
                    "shape_scale_mode must be 'global_sigmoid' (legacy/fixed "
                    f"gate ablations were removed), got {self.shape_scale_mode!r}"
                )
            self.use_shape_dense = bool(shape_prior_cfg.get("use_shape_dense", True))
            if self.use_shape_dense != self.explicit_use_dense_prompt:
                raise ValueError(
                    "shape_prior_cfg.use_shape_dense must match explicit_prompt_mode: "
                    f"mode={self.explicit_prompt_mode!r}, "
                    f"use_shape_dense={self.use_shape_dense}"
                )
            scale_init = float(shape_prior_cfg.get(
                "prompt_scale_init", shape_prior_cfg.get("shape_scale_init", 0.0)
            ))
            if self.use_shape_dense:
                # Bounded gate sigmoid(raw) in (0,1); buffer mirrors the init
                # value for logging and both keys live in the state dict.
                eps = 1e-4
                init = min(max(scale_init, eps), 1.0 - eps)
                raw_init = math.log(init / (1.0 - init))
                self.shape_dense_alpha_raw = nn.Parameter(torch.tensor(raw_init))
                self.register_buffer("shape_dense_alpha", torch.tensor(scale_init))
                self.register_parameter("shape_prompt_scale", None)
            else:
                # Sparse-only: no gate parameters, but the injector stays —
                # point mining still needs the coarse mask.
                self.register_parameter("shape_prompt_scale", None)
                self.register_parameter("shape_dense_alpha_raw", None)
                self.register_buffer("shape_dense_alpha", torch.tensor(0.0))
        else:
            self.shape_injector = None
            self.register_parameter("shape_prompt_scale", None)
            self.register_parameter("shape_dense_alpha_raw", None)
            self.register_buffer("shape_dense_alpha", torch.tensor(0.0))
            self.shape_scale_mode = "global_sigmoid"
            self.use_shape_dense = False

    def _init_point_miner(self, shape_point_miner_cfg) -> None:
        """2P2N miner over detached ROI-local coarse logits (no parameters)."""
        from .shape_prior import ShapePointMiner

        miner_cfg = dict(shape_point_miner_cfg or {})
        miner_cfg.setdefault("point_warmup_cfg", self.point_warmup_cfg)
        self.shape_point_miner = ShapePointMiner(**miner_cfg)

    def _init_decoder_tail(self) -> None:
        if self.decoder_tail_enabled:
            if not self.class_agnostic:
                raise ValueError("UDPR v1 requires class_agnostic=True")
            self.decoder_tail_refiner = DecoderTailPointRefiner(**self._decoder_tail_cfg)
        elif self._decoder_tail_cfg:
            raise ValueError(
                "decoder_tail_refiner_cfg may only contain enabled=False when disabled"
            )

    def init_weights(self) -> None:
        # Deliberately skip FCNMaskHead.init_weights: this head has none of
        # the parent's conv_logits layers, and the parent's initializer would
        # crash on them. BaseModule's no-op keeps module semantics intact.
        BaseModule.init_weights(self)

    def assert_no_mask_embedding_contract(self, context: str = "runtime") -> None:
        """Fail if a required frozen SAM2 no-mask embedding drifted or fell back."""
        if not self.load_no_mask_pretrained:
            return
        if not self.no_mask_pretrained_loaded:
            raise RuntimeError(
                f"{context}: pretrained no-mask embedding was requested but not loaded"
            )
        if self.no_mask_embed is None:
            raise RuntimeError(f"{context}: required MaskHead no_mask_embed is absent")
        expected = self._pretrained_no_mask_reference
        actual = self.no_mask_embed.detach()
        if tuple(actual.shape) != (1, 256, 1, 1):
            raise RuntimeError(
                f"{context}: invalid MaskHead no_mask_embed shape {tuple(actual.shape)}"
            )
        if not torch.isfinite(actual).all():
            raise RuntimeError(f"{context}: MaskHead no_mask_embed contains non-finite values")
        if expected.numel() != actual.numel():
            raise RuntimeError(f"{context}: pretrained no-mask reference is unavailable")
        max_diff = float(
            (actual.float().cpu() - expected.float().cpu()).abs().max().item()
        )
        if max_diff != 0.0:
            raise RuntimeError(
                f"{context}: MaskHead no_mask_embed no longer matches the SAM2 "
                f"pretrained value (max_abs_diff={max_diff:.6g}); reject silent "
                "zero/random/checkpoint overwrite"
            )
        if self.no_mask_embed.requires_grad != (not self.freeze_no_mask_embed):
            raise RuntimeError(
                f"{context}: no_mask_embed requires_grad="
                f"{self.no_mask_embed.requires_grad} "
                f"does not match freeze_no_mask_embed={self.freeze_no_mask_embed}"
            )

    # ------------------------------------------------------------------
    # forward chain
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        image_embeddings,
        image_positional_embeddings,
        roi_img_ids=None,
        high_res_features=None,
        boxes=None,
        p2_feature=None,
        prompt_rois=None,
        renderer_features=None,
    ):
        """One decoder pass per RoI. Returns the 6-tuple
        ``(mask_logits, iou_predictions, coarse_outputs, {}, quality, tokens)``.

        ``p2_feature``/``prompt_rois``/``renderer_features`` are legacy call
        slots kept so the RoI head's calling convention stays stable; the
        modules that consumed them were removed.
        """
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]
        debug_stats: Dict[str, float] = {}

        (
            shape_prior_mask_logits,
            coarse_outputs,
            shape_prior_prompt_logits,
            shape_dense_valid,
        ) = self._forward_coarse(
            x, image_embeddings, boxes, roi_img_ids, debug_stats,
        )

        sparse_pe, dense_pe, _, prompt_scale = self._forward_prompt_encode(
            image_embeddings, boxes,
            shape_prior_mask_logits, shape_prior_prompt_logits,
            roi_bs, debug_stats,
        )
        if sparse_pe is None:
            raise ValueError(
                "prompt_sparse_mode='shape_point' requires prompt_encoder_enabled=True, "
                "shape_prior enabled, and boxes to be provided"
            )
        sparse_embeddings = sparse_pe  # [N, 4 or 6, 256]

        num_roi_per_image = torch.bincount(
            roi_img_ids.long(), minlength=img_bs,
        )[:img_bs]

        dense_embeddings = self._forward_dense_embeddings(
            image_embeddings, dense_pe, shape_dense_valid, boxes, prompt_scale,
            image_embedding_size, roi_bs, debug_stats,
        )

        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        if self.prompt_encoder is not None and hasattr(self.prompt_encoder, "get_dense_pe"):
            image_positional_embeddings = self.prompt_encoder.get_dense_pe().to(
                device=image_embeddings.device,
                dtype=image_embeddings.dtype,
            )
            if image_positional_embeddings.shape[0] == 1 and img_bs > 1:
                image_positional_embeddings = image_positional_embeddings.expand(
                    img_bs, -1, -1, -1
                )
        if image_positional_embeddings.dim() == 3:
            image_positional_embeddings = image_positional_embeddings.unsqueeze(0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(
            num_roi_per_image, dim=0
        )

        if high_res_features is not None:
            high_res_features_expanded = [
                feat.repeat_interleave(num_roi_per_image, dim=0)
                for feat in high_res_features[:2]
            ]
        else:
            high_res_features_expanded = None

        decoder_outputs = self.mask_decoder(
            image_embeddings,
            image_positional_embeddings,
            sparse_embeddings,
            dense_embeddings,
            self.multimask_output,
            False,
            high_res_features=high_res_features_expanded,
        )
        if self.decoder_tail_enabled:
            low_res_masks, iou_predictions, mask_tokens_out, upscaled_embedding = decoder_outputs
            # The margin tail always deploys the refined write, in training
            # and inference alike (coupled co-adaptation channel; the
            # train-on-raw "tailstop" semantics belonged to a removed arm).
            low_res_masks, self._last_tail_outputs = self.decoder_tail_refiner(
                low_res_masks, upscaled_embedding, mask_tokens_out
            )
            debug_stats.update({
                "TAIL/selected_count": low_res_masks.new_tensor(
                    float(self._last_tail_outputs["indices"].numel())
                ),
                "TAIL/selected_abs_logit": (
                    self._last_tail_outputs["selected_abs"].detach().mean()
                    if self._last_tail_outputs["selected_abs"].numel()
                    else low_res_masks.new_zeros(())
                ),
                "TAIL/delta_abs": (
                    self._last_tail_outputs["delta"].detach().abs().mean()
                    if self._last_tail_outputs["delta"].numel()
                    else low_res_masks.new_zeros(())
                ),
            })
        else:
            # Tail-less prompt-path ablations (paper Table P / B / P+B rows).
            low_res_masks, iou_predictions, mask_tokens_out = decoder_outputs
            self._last_tail_outputs = None
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        mask_tokens_out = mask_tokens_out.reshape(
            roi_bs, -1, mask_tokens_out.shape[-1]
        )
        # Telemetry bus for the trainer (detached: never retains the graph).
        self._last_uav_debug_stats = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in debug_stats.items()
        }
        return (
            low_res_masks,
            iou_predictions,
            coarse_outputs,
            {},  # aux-losses slot kept for the 6-tuple contract; UAV removed
            iou_predictions,
            mask_tokens_out,
        )

    def _forward_coarse(
        self, x, image_embeddings, boxes, roi_img_ids, debug_stats,
    ):
        """Coarse 64x64 logits per RoI, plus the dense canvas when enabled.

        The coarse logits have exactly two downstream consumers — the dense
        canvas (detached under the sg contract) and point mining (no_grad by
        construction) — so the coarse head is trained solely by the coarse
        auxiliary loss.
        """
        shape_prior_mask_logits = None
        coarse_outputs = None
        shape_prior_prompt_logits = None
        shape_dense_valid = None
        if self.shape_injector is None or boxes is None:
            return (
                shape_prior_mask_logits,
                coarse_outputs,
                shape_prior_prompt_logits,
                shape_dense_valid,
            )

        context_tokens = self.shape_injector.forward_context_visual(image_embeddings)
        raw_shape_prior_logits, _, _, _ = self.shape_injector.forward_roi(
            context_tokens, x, boxes, roi_img_ids,
        )
        debug_stats.update(self.shape_injector.get_debug_stats())
        shape_prior_mask_logits = raw_shape_prior_logits
        coarse_outputs = {
            "raw_logits": raw_shape_prior_logits,
            "refined_logits": raw_shape_prior_logits,
        }
        if not self.use_shape_dense:
            # True sparse-only: keep the coarse mask for point mining and
            # auxiliary supervision, but skip transform/paste/PE mask work.
            return (
                shape_prior_mask_logits,
                coarse_outputs,
                shape_prior_prompt_logits,
                shape_dense_valid,
            )

        (
            shape_prior_prompt_logits,
            shape_dense_valid,
            dense_excavation_stats,
        ) = self._shape_prior_to_prompt_mask(shape_prior_mask_logits, boxes)
        for stat_name, stat_value in dense_excavation_stats.items():
            # Historical telemetry prefix; kept for log-series continuity.
            debug_stats[f"GAUSSIAN/{stat_name}"] = stat_value
        return (
            shape_prior_mask_logits,
            coarse_outputs,
            shape_prior_prompt_logits,
            shape_dense_valid,
        )

    def _forward_prompt_encode(
        self,
        image_embeddings,
        boxes,
        shape_prior_mask_logits,
        shape_prior_prompt_logits,
        roi_bs,
        debug_stats,
    ):
        """Single frozen PromptEncoder call: 2P2N points + box + dense mask."""
        box_embs = None
        dense_pe = None
        prompt_scale = None
        sparse_pe = None
        if self.prompt_encoder is None or boxes is None:
            return sparse_pe, dense_pe, box_embs, prompt_scale

        # Sparse-only variants never hand a mask to the PE.
        masks_for_pe = shape_prior_prompt_logits if self.use_shape_dense else None
        prompt_scale = self._effective_shape_dense_alpha()
        if prompt_scale is not None:
            gate = prompt_scale.detach().float()
            debug_stats["GATE/shape_dense_alpha_mean"] = gate.mean()
            debug_stats["GATE/shape_dense_alpha_min"] = gate.min()
            debug_stats["GATE/shape_dense_alpha_max"] = gate.max()
        if masks_for_pe is not None and masks_for_pe.numel() > 0:
            canvas = masks_for_pe.detach().float()
            debug_stats["DENSE/canvas_mean"] = canvas.mean()
            debug_stats["DENSE/canvas_std"] = canvas.std()
            debug_stats["DENSE/canvas_min"] = canvas.min()
            debug_stats["DENSE/canvas_max"] = canvas.max()
            debug_stats["DENSE/canvas_nonzero_ratio"] = (canvas != 0).float().mean()

        if shape_prior_mask_logits is None:
            raise ValueError(
                "prompt_sparse_mode='shape_point' requires shape_prior enabled "
                "with non-None mask_logits."
            )
        coords, labels, point_stats = self.shape_point_miner(
            shape_prior_mask_logits,
            boxes,
            self.prompt_encoder_image_size,
        )
        (self._last_shape_point_local_yx,
         self._last_shape_point_labels) = self.shape_point_miner.get_last_local_points()
        # point_stats are prediction-based (no GT); GT-based stats are
        # derived by the training loop.
        for stat_key, stat_value in point_stats.items():
            debug_stats[f"SP/{stat_key}"] = (
                stat_value.detach() if torch.is_tensor(stat_value) else stat_value
            )
        points_for_pe, boxes_for_pe, masks_for_pe = (
            self._select_explicit_prompt_inputs(coords, labels, boxes, masks_for_pe)
        )
        if (
            points_for_pe is None
            and boxes_for_pe is None
            and masks_for_pe is None
        ):
            # The PE cannot infer an ROI batch size when every modality is
            # absent; build the true empty sparse set directly (the canonical
            # base dense embedding is built for all roi_bs below).
            sparse_pe = image_embeddings.new_empty((roi_bs, 0, 256))
            dense_pe_from_pe = None
        else:
            sparse_pe, dense_pe_from_pe = self.prompt_encoder(
                points=points_for_pe,
                boxes=boxes_for_pe,
                masks=masks_for_pe,
            )
            if points_for_pe is not None:
                sparse_pe = self._neutralize_invalid_point_tokens(sparse_pe, labels)
        debug_stats["PROMPT/use_box_token"] = float(self.explicit_use_box_prompt)
        debug_stats["PROMPT/use_point_token"] = float(self.explicit_use_point_prompt)
        debug_stats["PROMPT/use_dense_mask"] = float(self.explicit_use_dense_prompt)
        if masks_for_pe is not None:
            dense_pe = dense_pe_from_pe  # [N,256,Hemb,Wemb]
        return sparse_pe, dense_pe, box_embs, prompt_scale

    def _forward_dense_embeddings(
        self,
        image_embeddings,
        dense_pe,
        shape_dense_valid,
        boxes,
        prompt_scale,
        image_embedding_size,
        roi_bs,
        debug_stats,
    ):
        """Dense embedding = no-mask base + gated shape-dense residual."""
        if self.no_mask_embed is None:
            base_dense = image_embeddings.new_zeros(
                roi_bs, 256, image_embedding_size[0], image_embedding_size[1]
            )
        else:
            base_dense = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
                roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
            )
        if dense_pe is None:
            return base_dense

        if shape_dense_valid is not None:
            dense_pe = replace_invalid_dense_with_base(
                dense_pe, base_dense, shape_dense_valid
            )
        if self.restrict_dense_prompt_to_box:
            support = self._dense_embedding_box_support(
                boxes, image_embedding_size, dense_pe.dtype
            )
            dense_pe = base_dense + support * (dense_pe - base_dense)
        dense_delta = dense_pe - base_dense
        dense_embeddings = base_dense + prompt_scale * dense_delta
        if dense_embeddings.numel() > 0:
            base_norm = base_dense.detach().float().flatten(1).norm(dim=1)
            source_delta_norm = dense_delta.detach().float().flatten(1).norm(dim=1)
            applied_delta_norm = (
                dense_embeddings.detach().float() - base_dense.detach().float()
            ).flatten(1).norm(dim=1)
            norm_denom = base_norm.clamp_min(1e-6)
            debug_stats["DENSE/residual_alpha"] = prompt_scale.detach().float().mean()
            debug_stats["DENSE/base_norm"] = base_norm.mean()
            debug_stats["DENSE/shape_norm"] = (
                dense_pe.detach().float().flatten(1).norm(dim=1).mean()
            )
            debug_stats["DENSE/final_norm"] = (
                dense_embeddings.detach().float().flatten(1).norm(dim=1).mean()
            )
            debug_stats["DENSE/source_delta_norm"] = source_delta_norm.mean()
            debug_stats["DENSE/applied_delta_norm"] = applied_delta_norm.mean()
            debug_stats["DENSE/source_delta_ratio"] = (source_delta_norm / norm_denom).mean()
            debug_stats["DENSE/applied_delta_ratio"] = (applied_delta_norm / norm_denom).mean()
        return dense_embeddings

    # ------------------------------------------------------------------
    # post-processing
    # ------------------------------------------------------------------

    def _check_tail_anchor_once(self, threshold: float) -> None:
        """One-time margin-anchor check on first inference.

        The margin loss pushes selected logits across the DEPLOYED
        binarisation logit; anchoring anywhere else (e.g. logit 0 for thr
        0.5) would train a different semantics than what paste-time
        binarisation evaluates, so drift is a hard error.
        """
        if getattr(self, "_tail_boundary_checked", False):
            return
        if self._decoder_tail_cfg.get("mode") != "residual_v1_margin":
            return
        expected = float(torch.log(torch.tensor(threshold / (1.0 - threshold))).item())
        if abs(self.decoder_tail_refiner.boundary_logit - expected) > 1e-4:
            raise RuntimeError(
                "stop-margin boundary_logit "
                f"{self.decoder_tail_refiner.boundary_logit:.6f} != deployed "
                f"binarisation logit {expected:.6f} (mask_thr_binary={threshold})"
            )
        self._tail_boundary_checked = True

    def _predict_by_feat_single(
        self,
        mask_preds: Tensor,
        bboxes: Tensor,
        labels: Tensor,
        img_meta: dict,
        rcnn_test_cfg: ConfigDict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> Tensor:
        if self.final_mask_coordinate_mode == "roi_local":
            # ROI-local logits: the FCNMaskHead parent handles sigmoid,
            # per-bbox paste, and paste-time binarisation.
            self._check_tail_anchor_once(float(rcnn_test_cfg.mask_thr_binary))
            return super()._predict_by_feat_single(
                mask_preds=mask_preds,
                bboxes=bboxes,
                labels=labels,
                img_meta=img_meta,
                rcnn_test_cfg=rcnn_test_cfg,
                rescale=rescale,
                activate_map=activate_map,
            )

        # Full-image logits: the decoder already produced the whole canvas;
        # resize once to the image grid and never paste through the bbox.
        del bboxes
        if not self.class_agnostic:
            row = torch.arange(mask_preds.shape[0], device=mask_preds.device)
            mask_preds = mask_preds[row, labels][:, None]
        elif mask_preds.shape[1] != 1:
            mask_preds = mask_preds[:, :1]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        if rescale:
            image_h, image_w = img_meta["ori_shape"][:2]
        else:
            image_h, image_w = img_meta.get(
                "img_shape", img_meta["ori_shape"]
            )[:2]
        mask_probs = F.interpolate(
            mask_preds,
            size=(int(image_h), int(image_w)),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        threshold = float(rcnn_test_cfg.mask_thr_binary)
        self._check_tail_anchor_once(threshold)
        if threshold >= 0:
            return mask_probs >= threshold
        return (mask_probs * 255).to(torch.uint8)
