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

import dataclasses
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.registry import MODELS
from mmdet.utils import ConfigType

from ..decoder.sam2_decoder import RSSAM2MaskDecoderWrapper
from .dense_prompt_utils import replace_invalid_dense_with_base
from .mask_head_helpers import _MaskHeadPromptHelpers
from .mask_head_init import _MaskHeadInitMixin
from .mask_head_targets import _MaskHeadTargetsMixin

# Sparse prompt route is fixed to the explicit shape-point path; the legacy
# RSPrompter MLP point embeddings ("point"/"point_box" modes) were baseline
# arms and are gone. The final-mask supervision coordinate contract, however,
# has two live values (see class docstring).
_VALID_EXPLICIT_PROMPT_MODES = {"points", "box", "points_box", "points_box_dense"}


@dataclasses.dataclass
class MaskHeadOutput:
    """What the mask head's forward hands to the RoI head.

    Only the three fields with real consumers survive the cleanup; the
    historical 6-tuple carried a permanently-empty aux-loss slot, a
    duplicated iou tensor, and tokens/feats that nothing read.
    """

    mask_logits: Tensor          # refined decoder logits [N,1,H,W] (z')
    coarse_outputs: Optional[Dict[str, Tensor]]  # coarse head dict (raw_logits)
    tail_outputs: Optional[Dict[str, Tensor]]    # UDPR write-back internals


@MODELS.register_module()
class RSPrompterAnchorMaskHeadSAM2(_MaskHeadInitMixin, _MaskHeadPromptHelpers, _MaskHeadTargetsMixin, FCNMaskHead, BaseModule):
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
        """One decoder pass per RoI; see MaskHeadOutput for the return contract.

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
        # Telemetry bus for the trainer (detached: never retains the graph).
        self._forward_telemetry = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in debug_stats.items()
        }
        return MaskHeadOutput(
            mask_logits=low_res_masks,
            coarse_outputs=coarse_outputs,
            tail_outputs=self._last_tail_outputs,
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
