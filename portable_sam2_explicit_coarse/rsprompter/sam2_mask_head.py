"""SAM2 mask head: config wiring (__init__), the causal-chain forward and predict. Helpers/targets live in the two sibling mixins (split from models_sam2)."""

"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

import os
import sys
from typing import Dict, Optional

import einops
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
from .sam2_decoder import RSSAM2MaskDecoderWrapper
from .sam2_vision import (
    SAM2_REPO,
    _DiscardedIoUPredictionHead,
    _UnusedPromptEmbedding,
    _load_pretrained_no_mask_embedding,
    _load_sam2_checkpoint,
    _resolve_sam2_checkpoint_path,
)



from .sam2_mask_head_helpers import _MaskHeadPromptHelpers
from .sam2_mask_head_targets import _MaskHeadTargetsMixin


@MODELS.register_module()
class RSPrompterAnchorMaskHeadSAM2(_MaskHeadPromptHelpers, _MaskHeadTargetsMixin, FCNMaskHead, BaseModule):
    """SAM2-compatible Mask Head for RSPrompter."""

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
        # ===== Prompt 编码迁移 (阶段1) =====
        # sparse prompt 三模式: "point"(point_emb) / "box"(SAM2 PE box) / "point_box"(concat)
        prompt_sparse_mode: str = "point",
        explicit_prompt_mode: Optional[str] = None,
        prompt_encoder_enabled: bool = False,      # 是否接回 SAM2 原生 PromptEncoder
        prompt_encoder_image_size: int = 512,      # input_image_size, 与图一致
        prompt_encoder_embed_size: int = 16,       # image_embedding_size, 实测 512→16×16
        load_pe_pretrained: bool = True,           # 加载 sam_prompt_encoder.* 预训练权重 (对比项目遗漏)
        sam2_ckpt_for_pe: Optional[str] = None,    # PE 权重来源, 默认复用 decoder ckpt
        prompt_encoder_cfg: Optional[Dict] = None, # 问题 3：PE 严格加载/冻结统一配置（见下）
        dense_prompt_mode: str = "none",           # "none"(no_mask 常量) / "residual"(+ROI 残差, 零初始化)
        prune_unused_prompt_components: bool = False,
        shape_base_dense_mode: str = "no_mask",
        # 问题 1（审查重构）：拆分为 coarse / final 两个独立坐标语义，禁止一个开关同时控制。
        # - coarse_mask_coordinate_mode：coarse mask（ShapePriorInjector 输出）的坐标系，
        #   固定 "roi_local"（coarse mask 在 ROI 局部生成与监督）。
        # - final_mask_coordinate_mode：最终实例 mask 的监督/后处理坐标系。
        #   B0/B1 用 "roi_local" 复现旧口径；显式 coarse 路线使用 SAM2 原生
        #   "full_image" 网格，训练对齐整图 target，推理只 resize 一次。
        # 旧的统一 mask_coordinate_mode 参数保留为兼容入口（见下方解析）。
        coarse_mask_coordinate_mode: str = "roi_local",
        final_mask_coordinate_mode: str = "roi_local",
        final_mask_loss_cfg: Optional[Dict] = None,
        roi_sam_cfg: Optional[Dict] = None,
        mask_coordinate_mode: Optional[str] = None,  # 兼容旧参数，None 时用上面两个默认
        roi_mask_size: int = 28,
        # 问题 13：mask_head 级别的 checkpoint 严格加载配置（透传给 PE；decoder/backbone 各自接收）
        checkpoint_load_cfg: Optional[Dict] = None,
        # 批次2 子任务1：coarse mask 独立强监督配置（替换旧 _shape_prior_aux_loss）
        coarse_mask_loss_cfg: Optional[Dict] = None,
        # 批次2 子任务3：dense prompt transform 配置
        dense_prompt_cfg: Optional[Dict] = None,
        restrict_dense_prompt_to_box: bool = True,
        # P2: explicit point miner / training-only box prompt jitter configs
        shape_point_miner_cfg: Optional[Dict] = None,
        point_warmup_cfg: Optional[Dict] = None,
        mask_prompt_box_cfg: Optional[Dict] = None,
        box_prompt_cfg: Optional[Dict] = None,
        # ===== 冻结策略 =====
        freeze_mask_decoder: bool = False,         # SAM2 MaskDecoder 可训练 (canonical 契约 §10/§14.1)
        freeze_no_mask_embed: bool = True,         # 冻结 no_mask_embed (True=加载ckpt并冻结 / False=零初始化可训练)
        load_no_mask_pretrained: bool = True,       # False 复现旧 MLP 基线的零初始化 no_mask_embed
        # ===== 阶段2: shape prior =====
        shape_prior_cfg: Optional[Dict] = None,    # dict(enabled=True, context_source="visual", ...) 或 None
        shape_prior_loss_weight: float = 0.5,       # 辅助 dice_bce loss 权重
        p2_boundary_refiner_cfg: Optional[Dict] = None,
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

        # Prompt 编码开关
        self.prompt_sparse_mode = str(prompt_sparse_mode)
        _explicit_mode_was_provided = explicit_prompt_mode is not None
        if self.prompt_sparse_mode == "shape_point":
            if explicit_prompt_mode is None:
                # Legacy shape-point configs always included box tokens; infer
                # only whether the dense mask was enabled.
                explicit_prompt_mode = (
                    "points_box_dense"
                    if bool((shape_prior_cfg or {}).get("use_shape_dense", True))
                    else "points_box"
                )
            self.explicit_prompt_mode = str(explicit_prompt_mode).strip().lower()
            _valid_explicit_modes = {
                "points", "box", "mask", "points_box", "points_box_dense"
            }
            if self.explicit_prompt_mode not in _valid_explicit_modes:
                raise ValueError(
                    "explicit_prompt_mode must be points, box, mask, "
                    "points_box or "
                    f"points_box_dense, got {self.explicit_prompt_mode!r}"
                )
            self.explicit_use_point_prompt = self.explicit_prompt_mode in {
                "points", "points_box", "points_box_dense"
            }
            self.explicit_use_box_prompt = self.explicit_prompt_mode in {
                "box", "points_box", "points_box_dense"
            }
            self.explicit_use_dense_prompt = (
                self.explicit_prompt_mode in {"mask", "points_box_dense"}
            )
            if _explicit_mode_was_provided:
                _cfg_use_dense = bool(
                    (shape_prior_cfg or {}).get("use_shape_dense", False)
                )
                if _cfg_use_dense != self.explicit_use_dense_prompt:
                    raise ValueError(
                        "explicit_prompt_mode and shape_prior_cfg.use_shape_dense "
                        "must agree: mode="
                        f"{self.explicit_prompt_mode!r}, use_shape_dense={_cfg_use_dense}"
                    )
        else:
            self.explicit_prompt_mode = None
            self.explicit_use_point_prompt = False
            self.explicit_use_box_prompt = False
            self.explicit_use_dense_prompt = False
        self.prompt_encoder_enabled = bool(prompt_encoder_enabled)
        self.dense_prompt_mode = str(dense_prompt_mode)
        self.point_warmup_cfg = dict(point_warmup_cfg or {})
        self.mask_prompt_box_cfg = dict(mask_prompt_box_cfg or {})
        self.prompt_encoder_image_size = int(prompt_encoder_image_size)
        self.prompt_encoder_embed_size = int(prompt_encoder_embed_size)
        self.prompt_encoder_mask_size = self.prompt_encoder_embed_size * 4
        self.prune_unused_prompt_components = bool(prune_unused_prompt_components)
        # 问题 1（审查重构）：coarse / final 双坐标语义，禁止混用。
        # 旧的统一 mask_coordinate_mode 仅作兼容入口：若显式传入则覆盖到对应字段并校验。
        self.coarse_mask_coordinate_mode = str(coarse_mask_coordinate_mode).lower()
        self.final_mask_coordinate_mode = str(final_mask_coordinate_mode).lower()
        for _mode_name, _mode_val in (
            ("coarse_mask_coordinate_mode", self.coarse_mask_coordinate_mode),
            ("final_mask_coordinate_mode", self.final_mask_coordinate_mode),
        ):
            if _mode_val not in ("full_image", "roi_local"):
                raise ValueError(
                    f"{_mode_name} must be 'full_image' or 'roi_local', got {_mode_val!r}"
                )
        # 兼容旧参数：若传了 mask_coordinate_mode，则同时作用于两者（向后兼容旧 config）
        if mask_coordinate_mode is not None:
            _legacy = str(mask_coordinate_mode).lower()
            if _legacy not in ("full_image", "roi_local"):
                raise ValueError(
                    f"mask_coordinate_mode must be 'full_image' or 'roi_local', got {mask_coordinate_mode!r}"
                )
            self.coarse_mask_coordinate_mode = _legacy
            self.final_mask_coordinate_mode = _legacy
        # Coarse logits are always ROI-local.  Final SAM2 logits may retain the
        # historical ROI-local interpretation for the frozen B0/B1 reproduction,
        # or use their native full-image grid in the explicit-coarse route.
        assert self.coarse_mask_coordinate_mode == "roi_local", (
            "coarse_mask_coordinate_mode must be 'roi_local' in current architecture "
            "(coarse mask is generated and supervised per-ROI)"
        )
        _final_loss_cfg = dict(final_mask_loss_cfg or {})
        _allowed_final_loss_keys = {
            "mode",
            "roi_expand_ratio",
            "roi_bce_weight",
            "roi_dice_weight",
            "outside_bce_weight",
            "eps",
        }
        _unknown_final_loss_keys = set(_final_loss_cfg) - _allowed_final_loss_keys
        if _unknown_final_loss_keys:
            raise ValueError(
                "Unknown final_mask_loss_cfg keys: "
                f"{sorted(_unknown_final_loss_keys)}"
            )
        _final_loss_cfg.setdefault("mode", "standard")
        _final_loss_cfg.setdefault("roi_expand_ratio", 1.20)
        _final_loss_cfg.setdefault("roi_bce_weight", 1.0)
        _final_loss_cfg.setdefault("roi_dice_weight", 1.0)
        _final_loss_cfg.setdefault("outside_bce_weight", 0.05)
        _final_loss_cfg.setdefault("eps", 1e-6)
        _final_loss_cfg["mode"] = str(_final_loss_cfg["mode"]).strip().lower()
        if _final_loss_cfg["mode"] not in {"standard", "roi_balanced_dice"}:
            raise ValueError(
                "final_mask_loss_cfg.mode must be standard or roi_balanced_dice, "
                f"got {_final_loss_cfg['mode']!r}"
            )
        if (
            self.final_mask_coordinate_mode != "full_image"
            and _final_loss_cfg["mode"] != "standard"
        ):
            raise ValueError(
                "ROI-focused final-mask loss requires final_mask_coordinate_mode='full_image'"
            )
        if float(_final_loss_cfg["roi_expand_ratio"]) < 1.0:
            raise ValueError("final-mask ROI expand ratio must be >= 1.0")
        for _weight_key in (
            "roi_bce_weight",
            "roi_dice_weight",
            "outside_bce_weight",
        ):
            if float(_final_loss_cfg[_weight_key]) < 0:
                raise ValueError(f"final-mask {_weight_key} must be non-negative")
        if float(_final_loss_cfg["eps"]) <= 0:
            raise ValueError("final-mask loss eps must be positive")
        self.final_mask_loss_cfg = ConfigDict(_final_loss_cfg)
        _roi_sam_cfg = dict(roi_sam_cfg or {})
        _allowed_roi_sam_keys = {"enabled", "sampling_ratio", "aligned"}
        _unknown_roi_sam_keys = set(_roi_sam_cfg) - _allowed_roi_sam_keys
        if _unknown_roi_sam_keys:
            raise ValueError(
                f"Unknown roi_sam_cfg keys: {sorted(_unknown_roi_sam_keys)}"
            )
        _roi_sam_cfg.setdefault("enabled", False)
        _roi_sam_cfg.setdefault("sampling_ratio", 2)
        _roi_sam_cfg.setdefault("aligned", True)
        self.roi_sam_enabled = bool(_roi_sam_cfg["enabled"])
        if int(_roi_sam_cfg["sampling_ratio"]) < 0:
            raise ValueError("roi_sam_cfg.sampling_ratio must be >= 0")
        if self.roi_sam_enabled:
            if self.prompt_sparse_mode != "shape_point":
                raise ValueError("ROI-SAM requires prompt_sparse_mode='shape_point'")
            if self.explicit_prompt_mode != "points":
                raise ValueError(
                    "Current ROI-SAM experiment is a strict C2 points-only variant"
                )
            if not self.prompt_encoder_enabled:
                raise ValueError("ROI-SAM requires the official PromptEncoder")
            if self.final_mask_coordinate_mode != "roi_local":
                raise ValueError(
                    "ROI-SAM decoder outputs require final_mask_coordinate_mode='roi_local'"
                )
            if self.final_mask_loss_cfg.mode != "standard":
                raise ValueError("ROI-SAM uses the standard ROI-local mask loss")
        self.roi_sam_cfg = ConfigDict(_roi_sam_cfg)
        self.roi_mask_size = int(roi_mask_size)
        # 问题 3：PromptEncoder 严格加载/冻结统一配置。
        # 向后兼容：旧的 load_pe_pretrained / sam2_ckpt_for_pe 裸参数仍可用，
        # 但若提供了 prompt_encoder_cfg，则以 cfg 为准（更细粒度）。
        self.checkpoint_load_cfg = dict(checkpoint_load_cfg or {})
        _pe_cfg = dict(prompt_encoder_cfg or {})
        _allowed_pe_cfg_keys = {
            "require_pretrained",
            "ckpt_path",
            "freeze_all",
            "train_mask_downscaling",
        }
        _unknown_pe_cfg_keys = set(_pe_cfg) - _allowed_pe_cfg_keys
        if _unknown_pe_cfg_keys:
            raise ValueError(
                "Unknown prompt_encoder_cfg keys: "
                f"{sorted(_unknown_pe_cfg_keys)}"
            )
        # 旧参数 → cfg 默认值（仅当 cfg 未显式指定时）
        _pe_cfg.setdefault("require_pretrained", bool(load_pe_pretrained))
        _pe_cfg.setdefault("ckpt_path", sam2_ckpt_for_pe)
        _pe_cfg.setdefault("freeze_all", True)
        _pe_cfg.setdefault("train_mask_downscaling", False)
        self.prompt_encoder_cfg = _pe_cfg
        # 暴露给后续加载逻辑使用的扁平字段（保持旧代码可读）
        self._pe_require_pretrained = bool(_pe_cfg.get("require_pretrained", True))
        self._pe_cfg_path = _pe_cfg.get("ckpt_path")  # None 表示复用 decoder ckpt
        self.quality_head_cfg = dict(quality_head_cfg or {})
        self.quality_head_enabled = bool(
            self.quality_head_cfg.get("enabled", False)
        )
        self.quality_head_type = str(
            self.quality_head_cfg.get("type", "native_iou")
        )
        if self.quality_head_type not in {"native_iou", "ap_ordinal"}:
            raise ValueError(
                "quality_head type must be 'native_iou' or 'ap_ordinal', got "
                f"{self.quality_head_type!r}"
            )
        self.quality_head_loss_weight = float(
            self.quality_head_cfg.get("loss_weight", 1.0)
        )
        self.quality_head_loss_beta = float(
            self.quality_head_cfg.get("loss_beta", 0.1)
        )
        self.quality_head_score_alpha = float(
            self.quality_head_cfg.get("score_alpha", 0.5)
        )
        if self.quality_head_loss_weight < 0:
            raise ValueError("quality_head loss_weight must be non-negative")
        if self.quality_head_loss_beta <= 0:
            raise ValueError("quality_head loss_beta must be positive")
        if self.quality_head_score_alpha < 0:
            raise ValueError("quality_head score_alpha must be non-negative")
        self.segm_score_mode = str(segm_score_mode).strip().lower()
        if self.segm_score_mode not in {"detector", "mask_quality"}:
            raise ValueError(
                "segm_score_mode must be 'detector' or 'mask_quality', got "
                f"{self.segm_score_mode!r}"
            )
        if self.segm_score_mode == "mask_quality" and not self.quality_head_enabled:
            raise ValueError(
                "segm_score_mode='mask_quality' requires quality_head_cfg.enabled=True; "
                "refusing to silently fall back to detector scores"
            )
        self.segm_score_key = (
            "mask_scores" if self.segm_score_mode == "mask_quality" else "scores"
        )
        self.shape_base_dense_mode = str(shape_base_dense_mode)
        self.load_no_mask_pretrained = bool(load_no_mask_pretrained)
        self.freeze_no_mask_embed = bool(freeze_no_mask_embed)
        if self.shape_base_dense_mode not in {"no_mask", "zero"}:
            raise ValueError(
                "shape_base_dense_mode must be 'no_mask' or 'zero', got "
                f"{self.shape_base_dense_mode!r}"
            )
        # Inference-only counterfactual switch. Normal training/inference keeps
        # this at "none"; the diagnostic runner sets one intervention at a time.
        self._diagnostic_forward_ablation = "none"
        self._diagnostic_prompt_frequency_mode = "none"
        self._diagnostic_prompt_spatial_perturbation = "none"
        self._diagnostic_directional_candidate_action = -1

        self.mask_decoder = RSSAM2MaskDecoderWrapper(**sam2_mask_decoder)
        if self.prune_unused_prompt_components and not self.quality_head_enabled:
            decoder = self.mask_decoder.decoder
            if decoder is None or not hasattr(decoder, "iou_prediction_head"):
                raise ValueError("Cannot prune missing SAM2 IoU prediction head")
            decoder.iou_prediction_head = _DiscardedIoUPredictionHead(
                decoder.num_mask_tokens
            )
        elif self.quality_head_enabled:
            decoder = self.mask_decoder.decoder
            if decoder is None or not hasattr(decoder, "iou_prediction_head"):
                raise ValueError("Quality calibration requires the SAM2 IoU prediction head")
            if isinstance(decoder.iou_prediction_head, _DiscardedIoUPredictionHead):
                raise ValueError("Quality calibration cannot use a discarded IoU head")
            if is_main_process() and self.quality_head_type == "native_iou":
                print(
                    "[MaskHead] G1 quality head enabled: "
                    f"loss_weight={self.quality_head_loss_weight:g}, "
                    f"score_alpha={self.quality_head_score_alpha:g}"
                )

        self.quality_head = None
        if self.quality_head_enabled and self.quality_head_type == "ap_ordinal":
            from .quality_head import APOrdinalQualityHead

            self.quality_head = APOrdinalQualityHead(
                token_dim=int(self.quality_head_cfg.get("token_dim", 256)),
                hidden_dim=int(self.quality_head_cfg.get("hidden_dim", 256)),
                thresholds=self.quality_head_cfg.get(
                    "thresholds",
                    tuple(0.50 + 0.05 * index for index in range(10)),
                ),
                image_size=int(
                    self.quality_head_cfg.get(
                        "image_size", self.prompt_encoder_image_size
                    )
                ),
                detach_inputs=bool(
                    self.quality_head_cfg.get("detach_inputs", True)
                ),
            )
            if is_main_process():
                print(
                    "[MaskHead] G2 AP-ordinal quality head enabled: "
                    f"thresholds={self.quality_head.thresholds}, "
                    f"detach_inputs={self.quality_head.detach_inputs}"
                )

        if self.shape_base_dense_mode == "no_mask":
            # C1-C5 require the exact SAM2 embedding.  Missing paths/keys used
            # to fall through to zeros here, even when the config explicitly
            # requested pretrained loading.  That is now a hard error.
            _no_mask_ckpt = (
                self._pe_cfg_path or sam2_mask_decoder.get("checkpoint_path")
            )
            if self.load_no_mask_pretrained:
                _no_mask = _load_pretrained_no_mask_embedding(_no_mask_ckpt)
                self.no_mask_pretrained_loaded = True
            else:
                _no_mask = torch.zeros(1, 256, 1, 1)
                self.no_mask_pretrained_loaded = False
            self.no_mask_embed = nn.Parameter(_no_mask)
            self.no_mask_embed.requires_grad_(not freeze_no_mask_embed)
            self.register_buffer(
                "_pretrained_no_mask_reference",
                _no_mask.detach().clone()
                if self.no_mask_pretrained_loaded
                else torch.empty(0),
                persistent=False,
            )
            if is_main_process():
                src = "SAM2 ckpt" if self.no_mask_pretrained_loaded else "intentional zeros"
                state = "frozen" if freeze_no_mask_embed else "trainable"
                abs_mean = float(self.no_mask_embed.detach().abs().mean().item())
                print(
                    f"[MaskHead] Initialized no_mask_embed from {src}, "
                    f"{state}, abs_mean={abs_mean:.8f}"
                )
        else:
            self.no_mask_pretrained_loaded = False
            self.register_parameter("no_mask_embed", None)
            self.register_buffer(
                "_pretrained_no_mask_reference", torch.empty(0), persistent=False
            )
            if is_main_process():
                print("[MaskHead] base dense disabled")

        # Freeze mask_decoder (默认为 True, 对齐原始行为)
        # freeze_mask_decoder=False 时: 解码器可训练 (对齐参考项目 B 的 decoder_freeze=False 行为)
        if freeze_mask_decoder:
            for _p in self.mask_decoder.parameters():
                _p.requires_grad_(False)
            self.mask_decoder.eval()
        # else: decoder 保持可训练, 由父类 decoder_freeze 开关二次控制

        # === SAM2 原生 PromptEncoder (可选) ===
        # box 通道: proposal box 坐标 → 2 个角点 token [N,2,256]
        if self.prompt_encoder_enabled:
            if SAM2_REPO and SAM2_REPO not in sys.path:
                sys.path.insert(0, SAM2_REPO)
            from sam2.modeling.sam.prompt_encoder import PromptEncoder

            self.prompt_encoder = PromptEncoder(
                embed_dim=256,
                image_embedding_size=(int(prompt_encoder_embed_size), int(prompt_encoder_embed_size)),
                input_image_size=(int(prompt_encoder_image_size), int(prompt_encoder_image_size)),
                mask_in_chans=16,
            )
            # 问题 3：冻结策略由 prompt_encoder_cfg 统一控制。
            # - 默认 freeze_all=True：PE 全部冻结（canonical 契约 §9.3/§14.2）。
            # - freeze_all=False：sparse_only 解冻 box 角点 point_embeddings[2]/[3]；
            #   shape_point 模式额外解冻 point_embeddings[0]/[1]（前景点/背景点）。
            # 其余冻结 → 避免 DDP unused params, 避免 dense 路径悄悄变成 PE no_mask
            _pe_freeze_all = bool(self.prompt_encoder_cfg.get("freeze_all", True))
            if _pe_freeze_all:
                _pe_trainable_indices = ()
            else:
                _pe_trainable_indices = (2, 3)
                if self.prompt_sparse_mode == "shape_point":
                    _pe_trainable_indices = (
                        (0, 1, 2, 3)
                        if self.explicit_use_box_prompt
                        else (0, 1)
                    )
            for _idx, _emb in enumerate(self.prompt_encoder.point_embeddings):
                _emb.weight.requires_grad_(_idx in _pe_trainable_indices)
            self.prompt_encoder.not_a_point_embed.weight.requires_grad_(False)
            self.prompt_encoder.no_mask_embed.weight.requires_grad_(False)
            # The dense-mask convolution stack may be trained independently of
            # sparse point/box embeddings.  This is a resolved model-config
            # field so checkpoints can reconstruct the exact trainability
            # contract without relying on a launcher-only environment value.
            _train_mask_downscaling = bool(
                self.prompt_encoder_cfg.get("train_mask_downscaling", False)
            )
            for _p in self.prompt_encoder.mask_downscaling.parameters():
                _p.requires_grad_(_train_mask_downscaling)

            # 加载预训练权重 (对比项目遗漏, 本计划增量)
            # 问题 3：require_pretrained 默认 True——ckpt 不存在或 missing key 时直接 raise，
            # 避免静默随机初始化导致 PE 几何编码错误却继续训练。
            if load_pe_pretrained or self._pe_require_pretrained:
                _ck_path = _resolve_sam2_checkpoint_path(
                    self._pe_cfg_path or sam2_mask_decoder.get("checkpoint_path")
                )
                if _ck_path and os.path.exists(_ck_path):
                    _ck = _load_sam2_checkpoint(_ck_path)
                    _pe_state = {
                        k[len("sam_prompt_encoder."):]: v
                        for k, v in _ck.items()
                        if k.startswith("sam_prompt_encoder.")
                    }
                    if not _pe_state:
                        if self._pe_require_pretrained:
                            raise RuntimeError(
                                f"[MaskHead] sam_prompt_encoder.* keys not found in {_ck_path}, "
                                "but require_pretrained=True. Check checkpoint integrity / model_size match."
                            )
                        elif is_main_process():
                            print(f"[MaskHead] WARNING: no sam_prompt_encoder.* keys in {_ck_path}")
                    else:
                        load_module_state_dict_strict(
                            self.prompt_encoder,
                            _pe_state,
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
                            _pe_no_mask = (
                                self.prompt_encoder.no_mask_embed.weight.detach()
                                .reshape_as(self.no_mask_embed)
                            )
                            if not torch.equal(
                                _pe_no_mask.cpu(), self.no_mask_embed.detach().cpu()
                            ):
                                raise RuntimeError(
                                    "PromptEncoder and MaskHead loaded different "
                                    "pretrained no-mask embeddings"
                                )
                else:
                    _msg = (f"PromptEncoder pretrained weights not found at {_ck_path}. "
                            f"require_pretrained={self._pe_require_pretrained}.")
                    if self._pe_require_pretrained:
                        raise RuntimeError(
                            f"[MaskHead] {_msg} Set prompt_encoder_cfg.require_pretrained=False "
                            "to allow random init, or provide a valid SAM2 checkpoint."
                        )
                    elif is_main_process():
                        print(f"[MaskHead] WARNING: {_msg} PE will be randomly initialized.")
            if self.prune_unused_prompt_components:
                if not (shape_prior_cfg and shape_prior_cfg.get("enabled", False)):
                    raise ValueError(
                        "prune_unused_prompt_components requires shape_prior_cfg.enabled=True"
                    )
                # shape_point 模式需要 point_embeddings[0]/[1]（前景点/背景点），不能 prune
                if self.prompt_sparse_mode != "shape_point":
                    # points=None and masks=shape_prompt are invariant in this path.
                    # Indices 2/3 remain because SAM2 uses them for box corners.
                    self.prompt_encoder.point_embeddings[0] = _UnusedPromptEmbedding()
                    self.prompt_encoder.point_embeddings[1] = _UnusedPromptEmbedding()
                    # torch.where 求值两个分支，not_a_point_embed 即使无 label=-1 也会被访问
                    self.prompt_encoder.not_a_point_embed = None
                # Sparse-only shape-point experiments still need SAM2's pretrained
                # no-mask dense embedding.  Prune it only when every forward is
                # guaranteed to provide a shape dense mask.
                if self.explicit_use_dense_prompt:
                    self.prompt_encoder.no_mask_embed = None
        else:
            self.prompt_encoder = None

        # === point_emb (RSPrompter 自造 5 点) ===
        # prompt_sparse_mode="box" / "shape_point" 时不实例化 → 省 DDP unused params
        # SAM2 decoder expects sparse_prompt_embeddings with dim=256 (matching image_embeddings)
        # Output: (batch, per_pointset_point, 256)
        num_sincos = 2 if with_sincos else 1
        embed_dim = 256  # Match SAM2 decoder embedding dimension
        if self.prompt_sparse_mode not in ("box", "shape_point"):
            self.point_emb = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(in_channels * roi_feat_size**2 // 4, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, in_channels),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels, embed_dim * num_sincos * per_pointset_point),
            )
        else:
            self.point_emb = None

        # === ShapePointMiner (shape_point mode) ===
        # Mine hard distance-based positive/negative coordinates from detached
        # ROI-local coarse logits, then encode them with the frozen SAM2 PE.
        if self.prompt_sparse_mode == "shape_point":
            from .shape_prior import ShapePointMiner
            _miner_cfg = dict(shape_point_miner_cfg or {})
            _miner_cfg.setdefault("point_warmup_cfg", self.point_warmup_cfg)
            self.shape_point_miner = ShapePointMiner(**_miner_cfg)
        else:
            self.shape_point_miner = None

        # === dense ROI residual ===
        # The legacy "residual" mode (per-ROI dense projection on top of the
        # no_mask base) was removed: PROMPT_DENSE_MODE was never exported by
        # any wrapper, so no checkpoint carries roi_dense_* keys. The dense
        # prompt route lives in dense_prompt_cfg (shape-derived canvas).
        if self.dense_prompt_mode not in ("none", ""):
            raise ValueError(
                f"Unsupported dense_prompt_mode={self.dense_prompt_mode!r}; "
                "the legacy 'residual' branch was removed, use dense_prompt_cfg."
            )

        # === ShapePriorInjector (阶段2, 可选) ===
        # visual 路径: image_embeddings → context tokens → cross-attn 调制 ROI feat → small mask decoder
        # 产出的 mask_logits 喂给 PE masks= 通道, 产生 shape-aware dense_pe
        # shape_prompt_scale 初始值可配置（默认 0.0 等价无 shape prior，安全热启动）
        self.shape_prior_loss_weight = float(shape_prior_loss_weight)
        # 批次2 子任务1：CoarseMaskLoss（替换旧 _shape_prior_aux_loss）。
        # Canonical weight is epoch-scheduled (0.20 then 0.10); the static
        # shape_prior_loss_weight remains only as enable/fallback behavior.
        _shape_prior_enabled = bool(
            shape_prior_cfg is not None and shape_prior_cfg.get("enabled", False)
        )
        if _shape_prior_enabled:
            from .coarse_mask_loss import CoarseMaskLoss
            _cml_cfg = dict(coarse_mask_loss_cfg or {})
            _cml_cfg.setdefault("weight", self.shape_prior_loss_weight)
            _cml_cfg.setdefault("bce_weight", 1.0)
            _cml_cfg.setdefault("dice_weight", 1.0)
            _cml_cfg.setdefault("boundary_weight", 0.0)
            _cml_cfg.setdefault("distance_weight", 0.0)
            self.coarse_mask_loss_module = CoarseMaskLoss(**_cml_cfg)
        else:
            # Pure RSPrompter MLP mode has no coarse mask and therefore no
            # shape-loss object or schedule in the model graph.
            self.coarse_mask_loss_module = None
        # 批次2 子任务3：dense prompt transform 配置。
        # 默认 raw_logits + outside_fill=0 + clamp_range=None（严格等价旧 _shape_prior_to_prompt_mask）。
        self.dense_prompt_cfg = dict(dense_prompt_cfg or {})
        self.restrict_dense_prompt_to_box = bool(restrict_dense_prompt_to_box)
        self.dense_prompt_cfg.setdefault("transform", "raw_logits")
        self.dense_prompt_cfg.setdefault("outside_fill_logit", 0.0)
        self.dense_prompt_cfg.setdefault("gamma", 1.0)
        self.dense_prompt_cfg.setdefault("strength", 1.0)
        self.dense_prompt_cfg.setdefault("temperature", 1.0)
        self.dense_prompt_cfg.setdefault("clamp_range", None)  # None=不 clamp（raw_logits 兼容）
        self.dense_prompt_cfg.setdefault("detach_input", False)
        self.dense_prompt_cfg.setdefault("foreground_threshold", 0.5)
        self.dense_prompt_cfg.setdefault("gaussian_omega", 15.0)
        self.dense_prompt_cfg.setdefault("gaussian_gamma", 4.0)
        _dense_transform = str(self.dense_prompt_cfg["transform"])
        if _dense_transform not in {
            "raw_logits", "confidence_signed", "gaussian_edt"
        }:
            raise ValueError(f"Unsupported dense prompt transform={_dense_transform!r}")
        if _dense_transform == "gaussian_edt":
            if not bool(self.dense_prompt_cfg["detach_input"]):
                raise ValueError("gaussian_edt requires detach_input=True")
            if self.dense_prompt_cfg["clamp_range"] is not None:
                raise ValueError("gaussian_edt must not be clamped")
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
        if _shape_prior_enabled:
            import math as _math
            from .shape_prior import ShapePriorInjector
            _mh_keys = {
                "enabled",
                "use_shape_dense",
                "shape_scale_mode",
                "prompt_scale_init",
                "shape_scale_init",
            }  # mask_head 自己处理的字段, 不传给 injector
            _sp_cfg = {k: v for k, v in shape_prior_cfg.items() if k not in _mh_keys}
            _sp_cfg.setdefault("roi_feat_channels", in_channels)
            self.shape_injector = ShapePriorInjector(**_sp_cfg)
            # 批次2 子任务4：shape dense gate 三模式（审查 2.1）。
            # legacy_unbounded=旧行为复现；fixed=固定 gate 消融；
            # global_sigmoid=canonical 有界 gate（默认 init=0.25）。
            self.shape_scale_mode = str(shape_prior_cfg.get("shape_scale_mode", "legacy_unbounded"))
            self.use_shape_dense = bool(shape_prior_cfg.get("use_shape_dense", True))
            if (
                self.prompt_sparse_mode == "shape_point"
                and self.use_shape_dense != self.explicit_use_dense_prompt
            ):
                raise ValueError(
                    "shape_prior_cfg.use_shape_dense must match explicit_prompt_mode: "
                    f"mode={self.explicit_prompt_mode!r}, "
                    f"use_shape_dense={self.use_shape_dense}"
                )
            _scale_init = float(shape_prior_cfg.get("prompt_scale_init",
                                                     shape_prior_cfg.get("shape_scale_init", 0.0)))
            if self.use_shape_dense:
                if self.shape_scale_mode == "legacy_unbounded":
                    # 旧实现原样：可训练无界 Parameter
                    self.shape_prompt_scale = nn.Parameter(torch.tensor(_scale_init))
                    self.register_buffer("shape_dense_alpha", torch.tensor(_scale_init))
                    self.register_parameter("shape_dense_alpha_raw", None)
                elif self.shape_scale_mode == "fixed":
                    # 固定 gate（buffer，不训练）
                    self.register_buffer("shape_dense_alpha", torch.tensor(_scale_init))
                    self.register_parameter("shape_prompt_scale", None)
                    self.register_parameter("shape_dense_alpha_raw", None)
                elif self.shape_scale_mode == "global_sigmoid":
                    # 有界 gate：sigmoid(raw) ∈ (0,1)
                    _eps = 1e-4
                    _init = min(max(_scale_init, _eps), 1.0 - _eps)
                    _raw_init = _math.log(_init / (1.0 - _init))
                    self.shape_dense_alpha_raw = nn.Parameter(torch.tensor(_raw_init))
                    self.register_buffer("shape_dense_alpha", torch.tensor(_scale_init))
                    self.register_parameter("shape_prompt_scale", None)
                else:
                    raise ValueError(
                        f"Unknown shape_scale_mode: {self.shape_scale_mode!r}. "
                        "Expected legacy_unbounded/fixed/global_sigmoid."
                    )
            else:
                # sparse-only：不创建 shape dense gate，gate 不进 optimizer。
                # 注意：shape_injector 保留（shape_point 模式仍需 coarse mask 挖点），
                # 只是 forward 里不把 coarse mask 转 dense_pe（masks_for_pe=None）。
                self.register_parameter("shape_prompt_scale", None)
                self.register_parameter("shape_dense_alpha_raw", None)
                self.register_buffer("shape_dense_alpha", torch.tensor(0.0))
        else:
            self.shape_injector = None
            self.register_parameter("shape_prompt_scale", None)
            self.register_parameter("shape_dense_alpha_raw", None)
            self.register_buffer("shape_dense_alpha", torch.tensor(0.0))
            self.shape_scale_mode = "legacy_unbounded"
            self.use_shape_dense = False

        # C5-v2 is a P2-driven local boundary modifier.  Construct it after all
        # shared random modules and restore the CPU RNG afterwards so enabling
        # the ablation cannot perturb R1-C4 shared initialization.
        self.p2_boundary_refiner_cfg = dict(p2_boundary_refiner_cfg or {})
        if self.p2_boundary_refiner_cfg.get("enabled", False):
            if self.shape_injector is None or self.prompt_encoder is None:
                raise ValueError(
                    "P2BoundaryRefiner requires ShapePrior and PromptEncoder"
                )
            from .p2_boundary_refiner import P2BoundaryRefiner

            _refiner_cfg = dict(self.p2_boundary_refiner_cfg)
            _refiner_cfg.pop("enabled", None)
            _refiner_cfg.setdefault("coarse_size", int(
                (shape_prior_cfg or {}).get("coarse_mask_output_size", 64)
            ))
            with torch.random.fork_rng(devices=[]):
                self.p2_boundary_refiner = P2BoundaryRefiner(**_refiner_cfg)
        else:
            self.p2_boundary_refiner = None

        # DecoderBR is an independent post-decoder residual branch. It is
        # initialized after the baseline modules so old checkpoint loading and
        # random initialization order remain unchanged when it is disabled.
        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

        self.assert_no_mask_embedding_contract("model construction")

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

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

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
    ):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]
        debug_stats: Dict[str, float] = {}
        if self.roi_sam_enabled:
            if boxes is None or roi_img_ids is None:
                raise ValueError("ROI-SAM requires proposal boxes and roi_img_ids")
            prompt_geometry_boxes = self._roi_sam_local_boxes(boxes)
            debug_stats["ROI_SAM/enabled"] = 1.0
        else:
            prompt_geometry_boxes = boxes

        # === sparse_embeddings 三模式 ===
        # point: RSPrompter 自造 point_emb [N,5,256]
        # box:   SAM2 原生 PE box 角点 [N,2,256]
        # point_box: concat 两者 [N,7,256]
        point_embs = None
        if self.point_emb is not None:
            pe = self.point_emb(x)
            pe = einops.rearrange(pe, "b (n c) -> b n c", n=self.per_pointset_point)
            if self.with_sincos:
                pe = torch.sin(pe[..., ::2]) + pe[..., 1::2]
            point_embs = pe  # [N, 5, 256]

        (
            shape_prior_mask_logits,
            coarse_outputs,
            shape_prior_prompt_logits,
            shape_dense_valid,
        ) = self._forward_coarse_p2(
            x, image_embeddings, boxes, roi_img_ids, p2_feature, prompt_rois,
            debug_stats,
        )

        sparse_pe, dense_pe, box_embs, prompt_scale = self._forward_prompt_encode(
            image_embeddings, boxes, prompt_geometry_boxes,
            shape_prior_mask_logits, shape_prior_prompt_logits,
            roi_bs, debug_stats,
        )

        if self.prompt_sparse_mode == "point":
            sparse_embeddings = point_embs                                    # [N,5,256]
        elif self.prompt_sparse_mode == "box":
            if box_embs is None:
                raise ValueError(
                    "prompt_sparse_mode='box' requires prompt_encoder_enabled=True "
                    "and boxes to be provided"
                )
            sparse_embeddings = box_embs                                      # [N,2,256]
        elif self.prompt_sparse_mode == "shape_point":
            # PE output is [N,4,256] for points, or [N,6,256] when box tokens are enabled.
            if sparse_pe is None:
                raise ValueError(
                    "prompt_sparse_mode='shape_point' requires prompt_encoder_enabled=True, "
                    "shape_prior enabled, and boxes to be provided"
                )
            sparse_embeddings = sparse_pe                                     # [N,4 or 6,256]
        elif self.prompt_sparse_mode == "point_box":
            if point_embs is None or box_embs is None:
                raise ValueError(
                    "prompt_sparse_mode='point_box' requires point embeddings, "
                    "prompt_encoder_enabled=True, and boxes to be provided"
                )
            sparse_embeddings = torch.cat([point_embs, box_embs], dim=1)      # [N,7,256]
        else:
            raise ValueError(f"Unknown prompt_sparse_mode='{self.prompt_sparse_mode}', "
                             f"expected 'point'/'box'/'point_box'/'shape_point'")
        if self._diagnostic_forward_ablation == "no_sparse":
            sparse_embeddings = torch.zeros_like(sparse_embeddings)

        num_roi_per_image = torch.bincount(
            roi_img_ids.long(), minlength=img_bs,
        )[:img_bs]

        dense_embeddings = self._forward_dense_embeddings(
            image_embeddings, dense_pe, shape_dense_valid, boxes, prompt_scale,
            image_embedding_size, roi_bs, debug_stats,
        )


        # P2BoundaryRefiner has already calibrated ROI-local coarse logits before the
        # frozen PromptEncoder.  No embedding-space residual is applied here.
        if self.roi_sam_enabled:
            image_embeddings = self._roi_sam_crop_feature(
                image_embeddings, boxes, roi_img_ids
            )
        else:
            image_embeddings = image_embeddings.repeat_interleave(
                num_roi_per_image, dim=0
            )

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
        if image_positional_embeddings.dim() == 4:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        else:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        if self._diagnostic_forward_ablation == "zero_image_pe":
            image_positional_embeddings = torch.zeros_like(image_positional_embeddings)
        
        if high_res_features is not None:
            if self.roi_sam_enabled:
                high_res_features_expanded = [
                    self._roi_sam_crop_feature(feat, boxes, roi_img_ids)
                    for feat in high_res_features[:2]
                ]
            else:
                high_res_features_expanded = [
                    feat.repeat_interleave(num_roi_per_image, dim=0)
                    for feat in high_res_features[:2]
                ]
            if self._diagnostic_forward_ablation == "zero_high_res":
                high_res_features_expanded = [
                    torch.zeros_like(feat) for feat in high_res_features_expanded
                ]
        else:
            high_res_features_expanded = None
        
        low_res_masks, iou_predictions, mask_tokens_out = self.mask_decoder(
            image_embeddings,
            image_positional_embeddings,
            sparse_embeddings,
            dense_embeddings,
            self.multimask_output,
            False,
            high_res_features=high_res_features_expanded,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        mask_tokens_out = mask_tokens_out.reshape(
            roi_bs, -1, mask_tokens_out.shape[-1]
        )
        quality_predictions = iou_predictions
        if self.quality_head is not None:
            if mask_tokens_out is None or boxes is None:
                raise ValueError("G2 quality head requires mask tokens and proposal boxes")
            quality_predictions = self.quality_head(
                mask_tokens=mask_tokens_out,
                mask_logits=low_res_masks,
                native_iou=iou_predictions,
                boxes=boxes,
            )
        # Persist the latest forward diagnostics for the trainer.  Previously
        # these values lived only in this local dictionary, so the logger could
        # not observe the effective dense residual even when debug logging was
        # enabled.  Detach tensors to avoid retaining the autograd graph.
        self._last_uav_debug_stats = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in debug_stats.items()
        }
        return (
            low_res_masks,
            iou_predictions,
            coarse_outputs,
            {},  # uav_aux_losses slot kept for the 6-tuple contract; UAV removed
            quality_predictions,
            mask_tokens_out,
        )
    def _forward_coarse_p2(
        self,
        x,
        image_embeddings,
        boxes,
        roi_img_ids,
        p2_feature,
        prompt_rois,
        debug_stats,
    ):
        """(forward phase) shape injector + P2BoundaryRefiner + canvas prep.

Verbatim-extracted from forward; operation order unchanged.
"""
        # === shape prior mask (阶段2, 必须在 PE 调用之前计算) ===
        shape_prior_mask_logits = None
        coarse_outputs = None
        shape_prior_prompt_logits = None
        shape_dense_valid = None
        if self.shape_injector is not None and boxes is not None:
            ctx_tokens = self.shape_injector.forward_context_visual(image_embeddings)
            raw_shape_prior_logits, _, _, _ = self.shape_injector.forward_roi(
                ctx_tokens, x, boxes, roi_img_ids,
            )
            debug_stats.update(self.shape_injector.get_debug_stats())
            refined_shape_prior_logits = raw_shape_prior_logits
            refiner_outputs = None
            if self.p2_boundary_refiner is not None:
                if p2_feature is None or prompt_rois is None:
                    raise ValueError(
                        "P2BoundaryRefiner requires full P2 and prompt_rois [N,5]"
                    )
                refiner_outputs = self.p2_boundary_refiner(
                    p2_feature=p2_feature,
                    prompt_rois=prompt_rois,
                    raw_logits=raw_shape_prior_logits,
                )
                refined_shape_prior_logits = refiner_outputs["refined_logits"]
                with torch.no_grad():
                    delta = refiner_outputs["delta_logits"].detach().float()
                    support_valid = refiner_outputs["support_valid"].detach()
                    # 饱和判据必须跟随本次训练自身上限 beta*delta_logit_max：
                    # 旧硬编码 0.396 按黄金跑 0.2*2.0=0.40 标定，对小上限配置
                    # （如 0.1*0.5=0.05）结构性恒 0。x0.99 保持黄金跑阈值不变。
                    saturation_threshold = (
                        float(self.p2_boundary_refiner.beta)
                        * float(self.p2_boundary_refiner.delta_logit_max)
                        * 0.99
                    )
                    support_pixels = (
                        refiner_outputs["search_support"].detach().float().flatten(1).sum(1)
                    )
                    saturated = delta.abs().ge(saturation_threshold).float()
                    debug_stats.update({
                        "P2BR/roi_count": delta.new_tensor(float(delta.shape[0])),
                        "P2BR/valid_support_count": support_valid.float().sum(),
                        "P2BR/rejected_support_count": (~support_valid).float().sum(),
                        "P2BR/search_coverage_sum": refiner_outputs["search_coverage_sum"],
                        "P2BR/delta_abs_sum": delta.abs().flatten(1).mean(1).sum(),
                        "P2BR/delta_nonzero_sum": delta.ne(0).float().flatten(1).mean(1).sum(),
                        "P2BR/delta_saturated_sum": saturated.flatten(1).mean(1).sum(),
                        "P2BR/delta_saturated_support_ratio_sum": (
                            saturated.flatten(1).sum(1) / support_pixels.clamp_min(1.0)
                        ).sum(),
                        "P2BR/saturation_threshold": delta.new_tensor(saturation_threshold),
                        "P2BR/projected_feature_norm_sum": refiner_outputs["projected_feature_norm_sum"],
                        "P2BR/highpass_feature_norm_sum": refiner_outputs["highpass_feature_norm_sum"],
                    })
            shape_prior_mask_logits = refined_shape_prior_logits
            coarse_outputs = {
                "raw_logits": raw_shape_prior_logits,
                "refined_logits": refined_shape_prior_logits,
                "delta_logits": None if refiner_outputs is None else refiner_outputs["delta_logits"],
                "raw_boundary_band": None if refiner_outputs is None else refiner_outputs["raw_boundary_band"],
                "search_support": None if refiner_outputs is None else refiner_outputs["search_support"],
                "support_valid": None if refiner_outputs is None else refiner_outputs["support_valid"],
            }
            # Refined coarse logits are the only downstream semantic source;
            # raw logits remain independently supervised by the C4 objective.
            if self.use_shape_dense:
                (
                    shape_prior_prompt_logits,
                    shape_dense_valid,
                    dense_excavation_stats,
                ) = self._shape_prior_to_prompt_mask(shape_prior_mask_logits, boxes)
                for stat_name, stat_value in dense_excavation_stats.items():
                    debug_stats[f"GAUSSIAN/{stat_name}"] = stat_value
            else:
                # True sparse-only: keep the coarse mask for point mining and
                # auxiliary supervision, but skip transform/paste/PE mask work.
                shape_prior_prompt_logits = None
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
        prompt_geometry_boxes,
        shape_prior_mask_logits,
        shape_prior_prompt_logits,
        roi_bs,
        debug_stats,
    ):
        """(forward phase) single frozen PromptEncoder call: diagnostics + mining.

Verbatim-extracted from forward; operation order unchanged.
"""
        # === PE 单次调用: box → sparse_pe, mask → dense_pe ===
        box_embs = None
        dense_pe = None
        if self.prompt_encoder is not None and boxes is not None:
            # 批次2 子任务4：sparse-only(use_shape_dense=False)时不传 mask 给 PE（审查 2.3）
            masks_for_pe = shape_prior_prompt_logits if self.use_shape_dense else None
            # 批次2 子任务4：shape dense gate 三模式有效值（审查 2.1）
            prompt_scale = self._effective_shape_dense_alpha()
            if prompt_scale is not None:
                _gate = prompt_scale.detach().float()
                debug_stats["GATE/shape_dense_alpha_mean"] = _gate.mean()
                debug_stats["GATE/shape_dense_alpha_min"] = _gate.min()
                debug_stats["GATE/shape_dense_alpha_max"] = _gate.max()
            if masks_for_pe is not None and masks_for_pe.numel() > 0:
                _dense_dbg = masks_for_pe.detach().float()
                debug_stats["DENSE/canvas_mean"] = _dense_dbg.mean()
                debug_stats["DENSE/canvas_std"] = _dense_dbg.std()
                debug_stats["DENSE/canvas_min"] = _dense_dbg.min()
                debug_stats["DENSE/canvas_max"] = _dense_dbg.max()
                debug_stats["DENSE/canvas_nonzero_ratio"] = (_dense_dbg != 0).float().mean()
            masks_for_pe = self._apply_diagnostic_prompt_frequency(
                masks_for_pe, debug_stats
            )
            masks_for_pe = self._apply_diagnostic_prompt_spatial(
                masks_for_pe, boxes, debug_stats
            )

            # === shape_point 模式: 从 shape_prior_mask 挖点，一次 PE 调用编码点+box+mask ===
            if self.prompt_sparse_mode == "shape_point":
                if shape_prior_mask_logits is None:
                    raise ValueError(
                        "prompt_sparse_mode='shape_point' requires shape_prior enabled "
                        "with non-None mask_logits. Got shape_prior_mask_logits=None."
                    )
                coords, labels, point_stats = self.shape_point_miner(
                    shape_prior_mask_logits,
                    prompt_geometry_boxes,
                    self.prompt_encoder_image_size,
                )
                (self._last_shape_point_local_yx,
                 self._last_shape_point_labels) = self.shape_point_miner.get_last_local_points()
                # point_stats 是 prediction-based（不含 GT）；GT-based stats 由上层训练循环算
                for _k, _v in point_stats.items():
                    debug_stats[f"SP/{_k}"] = (
                        _v.detach() if torch.is_tensor(_v) else _v
                    )
                points_for_pe, boxes_for_pe, masks_for_pe = (
                    self._select_explicit_prompt_inputs(
                        coords, labels, prompt_geometry_boxes, masks_for_pe
                    )
                )
                if (
                    points_for_pe is None
                    and boxes_for_pe is None
                    and masks_for_pe is None
                ):
                    # PromptEncoder cannot infer an ROI batch size when every
                    # modality is absent. Construct the true empty sparse set
                    # directly; the canonical base dense embedding is built
                    # below for all ``roi_bs`` instances.
                    sparse_pe = image_embeddings.new_empty((roi_bs, 0, 256))
                    dense_pe_from_pe = None
                else:
                    sparse_pe, dense_pe_from_pe = self.prompt_encoder(
                        points=points_for_pe,
                        boxes=boxes_for_pe,
                        masks=masks_for_pe,
                    )
                    if points_for_pe is not None:
                        sparse_pe = self._neutralize_invalid_point_tokens(
                            sparse_pe, labels
                        )
                debug_stats["PROMPT/use_box_token"] = float(
                    self.explicit_use_box_prompt
                )
                debug_stats["PROMPT/use_point_token"] = float(
                    self.explicit_use_point_prompt
                )
                debug_stats["PROMPT/use_dense_mask"] = float(
                    self.explicit_use_dense_prompt
                )
                # points: [N,4,256]; points_box(_dense): [N,6,256].
                box_embs = None  # explicit shape-point mode keeps PE output unified
            else:
                sparse_pe, dense_pe_from_pe = self.prompt_encoder(
                    points=None, boxes=boxes, masks=masks_for_pe,
                )
                box_embs = sparse_pe  # [N, 2, 256]
            if masks_for_pe is not None:
                dense_pe = dense_pe_from_pe  # [N,256,Hemb,Wemb], 1024输入通常为32x32
        else:
            prompt_scale = None
        return sparse_pe, dense_pe, box_embs, prompt_scale

    def _apply_diagnostic_prompt_frequency(self, masks_for_pe, debug_stats):
        """Diagnostic low-pass / high-boost filtering of the PE mask canvas."""
        frequency_mode = self._diagnostic_prompt_frequency_mode
        if masks_for_pe is not None and frequency_mode != "none":
            supported_frequency_modes = (
                "lowpass_16",
                "lowpass_8",
                "highboost_16_0p5",
                "highboost_16_1p0",
            )
            if frequency_mode not in supported_frequency_modes:
                raise ValueError(
                    "Unsupported prompt frequency diagnostic mode="
                    f"{frequency_mode!r}"
                )
            target_size = 8 if frequency_mode == "lowpass_8" else 16
            original_size = masks_for_pe.shape[-2:]
            original_masks = masks_for_pe
            low_frequency = F.interpolate(
                original_masks,
                size=(target_size, target_size),
                mode="area",
            )
            low_frequency = F.interpolate(
                low_frequency,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
            if frequency_mode.startswith("lowpass"):
                masks_for_pe = low_frequency
            else:
                boost = 0.5 if frequency_mode.endswith("0p5") else 1.0
                masks_for_pe = original_masks + boost * (
                    original_masks - low_frequency
                )
                debug_stats["DIAG/prompt_highboost"] = float(boost)
            debug_stats["DIAG/prompt_lowpass_target"] = float(target_size)
        return masks_for_pe

    def _apply_diagnostic_prompt_spatial(self, masks_for_pe, boxes, debug_stats):
        """Diagnostic edge side-perturbation of the PE mask canvas."""
        spatial_mode = self._diagnostic_prompt_spatial_perturbation
        if masks_for_pe is not None and spatial_mode != "none":
            try:
                side, direction, radius_text = spatial_mode.split("_")
                radius = int(radius_text)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "Prompt spatial perturbation must be SIDE_DIRECTION_RADIUS, "
                    f"got {spatial_mode!r}"
                ) from error
            if side not in ("top", "bottom", "left", "right"):
                raise ValueError(f"Unsupported prompt perturbation side={side!r}")
            if direction not in ("out", "in") or radius <= 0:
                raise ValueError(
                    f"Unsupported prompt perturbation={spatial_mode!r}"
                )
            kernel_size = 2 * radius + 1
            if direction == "out":
                perturbed = F.max_pool2d(
                    masks_for_pe,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                )
            else:
                perturbed = -F.max_pool2d(
                    -masks_for_pe,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=radius,
                )
            height, width = masks_for_pe.shape[-2:]
            scale_x = width / float(self.prompt_encoder_image_size)
            scale_y = height / float(self.prompt_encoder_image_size)
            center_x = 0.5 * (boxes[:, 0] + boxes[:, 2]) * scale_x
            center_y = 0.5 * (boxes[:, 1] + boxes[:, 3]) * scale_y
            yy = torch.arange(
                height, device=masks_for_pe.device, dtype=masks_for_pe.dtype
            ).view(1, 1, height, 1)
            xx = torch.arange(
                width, device=masks_for_pe.device, dtype=masks_for_pe.dtype
            ).view(1, 1, 1, width)
            if side == "top":
                side_mask = yy <= center_y[:, None, None, None]
            elif side == "bottom":
                side_mask = yy >= center_y[:, None, None, None]
            elif side == "left":
                side_mask = xx <= center_x[:, None, None, None]
            else:
                side_mask = xx >= center_x[:, None, None, None]
            masks_for_pe = torch.where(side_mask, perturbed, masks_for_pe)
            debug_stats["DIAG/prompt_spatial_radius"] = float(radius)
        return masks_for_pe

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
        """(forward phase) no-mask base + gated shape-dense residual + stats.

Verbatim-extracted from forward; operation order unchanged.
"""
        # === dense_embeddings ===
        if self.no_mask_embed is None:
            base_dense = image_embeddings.new_zeros(
                roi_bs, 256, image_embedding_size[0], image_embedding_size[1]
            )
        else:
            base_dense = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
                roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
            )
        if self._diagnostic_forward_ablation == "zero_base_dense":
            base_dense = torch.zeros_like(base_dense)

        if dense_pe is not None:
            if shape_dense_valid is not None:
                from .dense_prompt_utils import replace_invalid_dense_with_base
                dense_pe = replace_invalid_dense_with_base(
                    dense_pe, base_dense, shape_dense_valid
                )
            if self.restrict_dense_prompt_to_box:
                support = self._dense_embedding_box_support(
                    boxes, image_embedding_size, dense_pe.dtype
                )
                dense_pe = base_dense + support * (dense_pe - base_dense)
            # 所有模式统一走门控残差: base_dense + prompt_scale * (dense_pe - base_dense)
            # points_box_dense: bounded global-sigmoid interpolation between
            # pretrained no-mask and shape dense embedding.
            # Legacy/fixed modes remain only for controlled ablations.
            dense_delta = dense_pe - base_dense
            dense_embeddings = base_dense + prompt_scale * dense_delta
            if dense_embeddings.numel() > 0:
                _base_norm = base_dense.detach().float().flatten(1).norm(dim=1)
                _source_delta_norm = dense_delta.detach().float().flatten(1).norm(dim=1)
                _applied_delta_norm = (
                    dense_embeddings.detach().float() - base_dense.detach().float()
                ).flatten(1).norm(dim=1)
                _norm_denom = _base_norm.clamp_min(1e-6)
                debug_stats["DENSE/residual_alpha"] = prompt_scale.detach().float().mean()
                debug_stats["DENSE/base_norm"] = _base_norm.mean()
                debug_stats["DENSE/shape_norm"] = dense_pe.detach().float().flatten(1).norm(dim=1).mean()
                debug_stats["DENSE/final_norm"] = dense_embeddings.detach().float().flatten(1).norm(dim=1).mean()
                debug_stats["DENSE/source_delta_norm"] = _source_delta_norm.mean()
                debug_stats["DENSE/applied_delta_norm"] = _applied_delta_norm.mean()
                debug_stats["DENSE/source_delta_ratio"] = (_source_delta_norm / _norm_denom).mean()
                debug_stats["DENSE/applied_delta_ratio"] = (_applied_delta_norm / _norm_denom).mean()
        else:
            dense_embeddings = base_dense
        return dense_embeddings


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
            return super()._predict_by_feat_single(
                mask_preds=mask_preds,
                bboxes=bboxes,
                labels=labels,
                img_meta=img_meta,
                rcnn_test_cfg=rcnn_test_cfg,
                rescale=rescale,
                activate_map=activate_map,
            )

        # SAM2 decodes on the full image-embedding grid.  Resize that grid once
        # to the requested image canvas; never paste it through the bbox again.
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
        if threshold >= 0:
            return mask_probs >= threshold
        return (mask_probs * 255).to(torch.uint8)
