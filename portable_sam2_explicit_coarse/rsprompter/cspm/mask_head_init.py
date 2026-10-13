"""Construction-time wiring of the SAM2 mask head, as a plain mixin.

Every method here is called by the concrete ``__init__`` in mask_head.py;
splitting them out keeps the forward chain readable while the __init__
signature (checkpoint-embedded config keys) stays byte-stable. The section
banner and method bodies are moved verbatim from the a3sgm cleanup state.
"""

import os
import sys

import torch
import torch.nn as nn
from mmengine import ConfigDict
from mmengine.dist import is_main_process

from ..backbone.sam2_vision import (
    SAM2_REPO,
    _load_pretrained_no_mask_embedding,
    _load_sam2_checkpoint,
    _resolve_sam2_checkpoint_path,
)
from ..ckpt_utils import load_module_state_dict_strict
from ..decoder.decoder_tail_refiner import DecoderTailPointRefiner


class _MaskHeadInitMixin:
    # ------------------------------------------------------------------
    # __init__ helpers (split so each states one contract)
    # ------------------------------------------------------------------

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
            from .shape_prior_injector import ShapePriorInjector

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
        from .shape_point_miner import ShapePointMiner

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
