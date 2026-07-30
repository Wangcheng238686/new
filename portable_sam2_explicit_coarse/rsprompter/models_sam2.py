"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from torch import Tensor

from mmcv.ops import RoIAlign
from mmdet.models import MaskRCNN, StandardRoIHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask.mask_target import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType

from .ckpt_utils import load_module_state_dict_strict


class _DiscardedIoUPredictionHead(nn.Module):
    """Keep SAM2's decoder contract without computing an unused IoU MLP."""

    def __init__(self, num_outputs: int):
        super().__init__()
        self.num_outputs = int(num_outputs)

    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros((x.shape[0], self.num_outputs))


class _UnusedPromptEmbedding(nn.Module):
    """Parameter-free device anchor for PromptEncoder's unused point labels."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    @property
    def weight(self) -> Tensor:
        return self._device_anchor

    def forward(self, _: Tensor) -> Tensor:
        raise RuntimeError("Pruned point-label embedding cannot encode point prompts")


SAM2_AVAILABLE = False
PositionEmbeddingSine = None  # safe default; set when sam2 package imports
SAM2_REPO = os.environ.get("SAM2_REPO")
if SAM2_REPO and SAM2_REPO not in sys.path:
    sys.path.insert(0, SAM2_REPO)
try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    SAM2_AVAILABLE = True
except ImportError:
    pass


def _load_sam2_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict:
    """Load SAM2 checkpoint from file."""
    from mmengine.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)
    # SAM2 checkpoint format is {"model": {...}}
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _resolve_sam2_checkpoint_path(checkpoint_path: Optional[str]) -> Optional[str]:
    """Resolve a SAM2 checkpoint path with the same rule for every submodule."""
    if not checkpoint_path or os.path.isabs(checkpoint_path):
        return checkpoint_path
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    return os.path.join(project_root, checkpoint_path)


def _load_pretrained_no_mask_embedding(checkpoint_path: Optional[str]) -> Tensor:
    """Load SAM2's no-mask embedding or fail instead of returning zeros."""
    resolved_path = _resolve_sam2_checkpoint_path(checkpoint_path)
    if not resolved_path or not os.path.isfile(resolved_path):
        raise RuntimeError(
            "Pretrained no-mask embedding was requested, but the SAM2 "
            f"checkpoint does not exist: {resolved_path!r}"
        )
    checkpoint = _load_sam2_checkpoint(resolved_path)
    key = "sam_prompt_encoder.no_mask_embed.weight"
    if key not in checkpoint:
        raise RuntimeError(
            f"Pretrained no-mask embedding key {key!r} is missing from "
            f"{resolved_path}. Check checkpoint integrity and SAM2 model size."
        )
    value = checkpoint[key]
    if tuple(value.shape) != (1, 256):
        raise RuntimeError(
            f"Invalid pretrained no-mask embedding shape for {key!r}: "
            f"expected (1, 256), got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Pretrained no-mask embedding {key!r} contains non-finite values")
    return value.detach().reshape(1, 256, 1, 1).clone()


@MODELS.register_module()
class RSSAM2PositionalEmbedding(BaseModule):
    """Legacy learnable image PE used by the historical MLP baseline.

    B0/B1 intentionally reproduce the old bridge: a zero-initialized,
    trainable ``pos_embed`` is interpolated by ``RSPrompterAnchor`` to the
    selected image-embedding resolution.  This is not SAM2 PromptEncoder's
    random Fourier PE; coarse experiments use the real PromptEncoder instead.
    """

    def __init__(
        self,
        image_size: int = 1024,
        patch_size: int = 16,
        embedding_stride: int = 32,
        embed_dim: int = 256,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embedding_stride = int(embedding_stride)
        if self.embedding_stride not in (16, 32):
            raise ValueError(
                "embedding_stride must be 16 (official) or 32 (legacy), got "
                f"{self.embedding_stride}"
            )
        self.grid_size = image_size // patch_size
        self.embed_dim = embed_dim

        # Retained for checkpoint/DDP compatibility; get_dense_pe ignores it and
        # emits a fixed sinusoidal encoding instead (see docstring).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size, self.grid_size, embed_dim)
        )

        class PositionalEmbeddingWrapper:
            def __init__(self, pos_embed):
                self.positional_embedding = pos_embed

        self.shared_image_embedding = PositionalEmbeddingWrapper(self.pos_embed)

    def get_dense_pe(self) -> Tensor:
        """Return the trainable legacy PE in SAM2's [1,C,H,W] layout."""
        return self.pos_embed.permute(0, 3, 1, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.get_dense_pe()


@MODELS.register_module()
class RSSAM2VisionEncoder(BaseModule):
    """SAM2 Vision Encoder adapter using Hiera backbone.
    直接从checkpoint加载整个SAM2模型，然后提取encoder
    支持LoRA微调以降低小数据集上的过拟合风险
    支持Base+和Large模型
    """

    # SAM2 Hiera model configurations
    MODEL_CONFIGS = {
        "base_plus": {
            "embed_dim": 112,
            "num_heads": 2,
            "stages": (2, 3, 16, 3),
            "window_spec": (8, 4, 14, 7),
            "global_att_blocks": (12, 16, 20),
            "backbone_channel_list": [896, 448, 224, 112],
            "window_pos_embed_bkg_spatial_size": None,  # Base+ uses default
        },
        "large": {
            "embed_dim": 144,
            "num_heads": 2,
            "stages": (2, 6, 36, 4),
            "window_spec": (8, 4, 16, 8),
            "global_att_blocks": (23, 33, 43),
            "backbone_channel_list": [1152, 576, 288, 144],
            "window_pos_embed_bkg_spatial_size": (7, 7),  # Large needs this
        },
    }

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        init_cfg: Optional[Dict] = None,
        use_gradient_checkpointing: bool = False,
        freeze_backbone: bool = True,
        freeze_backbone_stages: int = -1,
        use_lora: bool = False,
        lora_config: Optional[Dict] = None,
        model_size: Optional[str] = None,  # "base_plus" or "large"
        checkpoint_load_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.checkpoint_path = checkpoint_path
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_stages = freeze_backbone_stages
        self.use_lora = use_lora
        self.lora_config = lora_config or {}
        self.model_size = model_size
        # 问题 13：checkpoint 严格加载配置。None 或空 dict 时保持历史行为（strict=False + 打印）。
        # config 可传 checkpoint_load_cfg=dict(strict_reproduction=True,
        #   allowed_missing_patterns=[...], allowed_unexpected_patterns=[...])
        self.checkpoint_load_cfg = dict(checkpoint_load_cfg or {})
        self.encoder = None
        self.encoder_out_channels = 256

        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not installed. Please install it via:\n"
                "pip install git+https://github.com/facebookresearch/sam2.git"
            )

        if init_cfg is not None and checkpoint_path is None:
            checkpoint_path = init_cfg.get("checkpoint")

        if checkpoint_path:
            if not os.path.isabs(checkpoint_path):
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                checkpoint_path = os.path.join(project_root, checkpoint_path)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Auto-detect model size from checkpoint path if not specified
            if self.model_size is None:
                self.model_size = self._detect_model_size(checkpoint_path)
            self._load_sam2_encoder(checkpoint_path)
        else:
            raise ValueError(
                f"SAM2 checkpoint not found at: {checkpoint_path}. "
                "Please download SAM2 checkpoint first."
            )

        # Apply LoRA if enabled (优先于freeze_backbone)
        # 问题 12 修复：LoRA 失败必须 raise，且失败后必须 fallback 到 _freeze_params，
        # 否则 backbone 原参数会保持 requires_grad=True → 意外全量微调。
        if self.use_lora and self.encoder is not None:
            try:
                self._apply_lora()
            except Exception:
                # LoRA 失败后，backbone 尚未被冻结，必须显式冻结以避免全量微调。
                # 然后 re-raise 让训练终止（用户应修复 LoRA 配置或关闭 use_lora）。
                self._freeze_params()
                raise
        elif freeze_backbone:
            self._freeze_params()

    def _detect_model_size(self, checkpoint_path: str) -> str:
        """Auto-detect model size from checkpoint path."""
        path_lower = checkpoint_path.lower()
        if "large" in path_lower or "_l" in path_lower:
            return "large"
        elif "base" in path_lower or "_b" in path_lower:
            return "base_plus"
        else:
            # Default to base_plus for backward compatibility
            return "base_plus"

    def _load_sam2_encoder(self, checkpoint_path: str):
        """Load SAM2 encoder from checkpoint by directly building the model."""
        # SAM2 import is resolved at module load via SAM2_REPO env var or the
        # installed package (see SAM2_AVAILABLE at the top of this file).
        if not SAM2_AVAILABLE:
            raise ImportError(
                "SAM2 is not properly installed. Install it (pip install -e .) "
                "or set the SAM2_REPO env var to its source checkout."
            )

        # Get model config based on detected or specified size
        config = self.MODEL_CONFIGS.get(self.model_size, self.MODEL_CONFIGS["base_plus"])
        
        if is_main_process():
            print(f"[RSSAM2VisionEncoder] Loading SAM2 {self.model_size} model")
            print(f"  - embed_dim: {config['embed_dim']}")
            print(f"  - stages: {config['stages']}")
            print(f"  - backbone_channels: {config['backbone_channel_list']}")

        try:
            # Build Hiera trunk with optional window_pos_embed_bkg_spatial_size
            hiera_kwargs = dict(
                embed_dim=config["embed_dim"],
                num_heads=config["num_heads"],
                drop_path_rate=0.1,
                q_pool=3,
                q_stride=(2, 2),
                stages=config["stages"],
                dim_mul=2.0,
                head_mul=2.0,
                window_spec=config["window_spec"],
                global_att_blocks=config["global_att_blocks"],
                return_interm_layers=True,
            )
            # Add window_pos_embed_bkg_spatial_size if specified
            if config.get("window_pos_embed_bkg_spatial_size") is not None:
                hiera_kwargs["window_pos_embed_bkg_spatial_size"] = config["window_pos_embed_bkg_spatial_size"]
            
            trunk = Hiera(**hiera_kwargs)

            position_encoding = PositionEmbeddingSine(
                num_pos_feats=256,
                normalize=True,
                temperature=10000,
            )

            neck = FpnNeck(
                position_encoding=position_encoding,
                d_model=256,
                backbone_channel_list=config["backbone_channel_list"],
                fpn_top_down_levels=[2, 3],
                fpn_interp_model="nearest",
            )

            image_encoder = ImageEncoder(
                trunk=trunk,
                neck=neck,
                scalp=0,
            )

            checkpoint = _load_sam2_checkpoint(checkpoint_path)
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("image_encoder."):
                    new_k = k[14:]
                    new_state_dict[new_k] = v

            # 审查 3.5：checkpoint 里没有 image_encoder.* 时直接报错，
            # 避免静默随机初始化 ImageEncoder 后继续训练。
            if not new_state_dict:
                raise RuntimeError(
                    f"No image_encoder.* keys found in checkpoint: {checkpoint_path}. "
                    "ImageEncoder would be randomly initialized. Check checkpoint integrity "
                    "or SAM2_MODEL_SIZE match."
                )

            # 问题 13：用严格加载工具，按白名单校验 missing/unexpected keys
            load_module_state_dict_strict(
                image_encoder,
                new_state_dict,
                module_name=f"ImageEncoder({self.model_size})",
                strict_reproduction=self.checkpoint_load_cfg.get("strict_reproduction", False),
                allowed_missing_patterns=tuple(self.checkpoint_load_cfg.get("allowed_missing_patterns", [])),
                allowed_unexpected_patterns=tuple(self.checkpoint_load_cfg.get("allowed_unexpected_patterns", [])),
            )

            self.encoder = image_encoder
            self.encoder_out_channels = 256

        except Exception as e:
            if is_main_process():
                print(f"Failed to load SAM2 encoder: {e}")
            raise

    def _apply_lora(self):
        """Apply LoRA to SAM2 encoder for efficient fine-tuning on small datasets.
        
        Improved LoRA configuration:
        - Higher rank (r=16) for better adaptation capacity
        - Include more target modules for comprehensive fine-tuning
        - Use lower dropout for better convergence
        - Save both neck and mask_head for task adaptation
        """
        if self.encoder is None:
            return
        
        try:
            from peft import LoraConfig, get_peft_model
            
            # Improved LoRA config for SAM2 Hiera backbone
            # Hiera architecture: Multi-scale block attention with qkv and proj layers
            default_lora_config = {
                "r": 16,  # Increased rank for better adaptation
                "lora_alpha": 32,  # Increased alpha for stronger adaptation
                "target_modules": ["qkv", "proj"],  # Hiera attention layers (mlp not supported by PEFT)
                "lora_dropout": 0.05,  # Reduced dropout for better convergence
                "bias": "none",
                "modules_to_save": ["neck"],  # Keep neck trainable
                "use_rslora": False,  # Use standard LoRA (can enable RSLora for large ranks)
            }
            
            # Update with user-provided config
            config_dict = default_lora_config.copy()
            config_dict.update(self.lora_config)
            
            # Filter out unsupported target modules
            # PEFT only supports nn.Linear, nn.Conv2d, nn.Embedding
            supported_modules = ["qkv", "proj", "fc1", "fc2", "conv"]
            target_modules = config_dict.get("target_modules", ["qkv", "proj"])
            filtered_targets = [m for m in target_modules if m in supported_modules]
            
            if len(filtered_targets) != len(target_modules):
                removed = set(target_modules) - set(filtered_targets)
                if is_main_process():
                    print(f"[RSSAM2VisionEncoder] Warning: Removed unsupported target_modules: {removed}")
            
            # Create LoRA config
            lora_config = LoraConfig(
                r=config_dict["r"],
                lora_alpha=config_dict["lora_alpha"],
                target_modules=filtered_targets,
                lora_dropout=config_dict["lora_dropout"],
                bias=config_dict["bias"],
                modules_to_save=config_dict.get("modules_to_save", ["neck"]),
            )
            
            # Apply LoRA to encoder
            self.encoder = get_peft_model(self.encoder, lora_config)

            if is_main_process():
                print(f"[RSSAM2VisionEncoder] Applied LoRA with config:")
                print(f"  - r={config_dict['r']}, alpha={config_dict['lora_alpha']}")
                print(f"  - target_modules={filtered_targets}")
                print(f"  - dropout={config_dict['lora_dropout']}")
                print(f"  - modules_to_save={config_dict.get('modules_to_save', ['neck'])}")
                # Print trainable parameters info
                self.encoder.print_trainable_parameters()

            # 问题 12：LoRA 参数审计——确保只有 lora_A/lora_B 和明确白名单（neck）参数可训练，
            # 否则意味着 LoRA 包装有遗漏，原 Hiera 参数仍可训练（变相全量微调）。
            self._audit_lora_trainable_params(config_dict.get("modules_to_save", ["neck"]))

        except ImportError as e:
            # 问题 12：peft 未安装时必须 raise（而非 print warning 继续）。
            # 继续会导致 backbone 未冻结 → 全量微调。
            raise ImportError(
                f"[RSSAM2VisionEncoder] peft is required when use_lora=True, but not installed: {e}. "
                "Install with `pip install peft`, or set use_lora=False / backbone.freeze_backbone=True."
            ) from e
        except Exception as e:
            # 问题 12：LoRA 应用失败必须 raise。调用方会 fallback 到 _freeze_params 后再 re-raise。
            import traceback
            msg = traceback.format_exc()
            raise RuntimeError(
                f"[RSSAM2VisionEncoder] Failed to apply LoRA: {e}.\n{msg}\n"
                "Fix the LoRA config or set use_lora=False."
            ) from e

    def _audit_lora_trainable_params(self, modules_to_save):
        """问题 12：审计 LoRA 包装后的可训练参数，发现意外可训练参数则报错。

        合法可训练：含 'lora_A'/'lora_B' 的参数，或在 modules_to_save 白名单中的模块。
        其它（原 Hiera 参数）必须 requires_grad=False，否则视为包装不完整。
        """
        if self.encoder is None:
            return
        total = 0
        trainable = 0
        lora_trainable = 0
        unexpected = []
        for name, p in self.encoder.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
                is_lora = ("lora_A." in name) or ("lora_B." in name)
                is_whitelisted = any(
                    f".{mod}." in f".{name}." or name.startswith(f"{mod}.")
                    for mod in modules_to_save
                )
                if is_lora:
                    lora_trainable += p.numel()
                elif not is_whitelisted:
                    unexpected.append(name)
        if is_main_process():
            print(f"[RSSAM2VisionEncoder] LoRA audit: backbone total={total}, "
                  f"trainable={trainable}, lora_trainable={lora_trainable}, "
                  f"unexpected_trainable={len(unexpected)}")
            if unexpected:
                sample = unexpected[:20]
                print(f"[RSSAM2VisionEncoder] Unexpected trainable params (first {len(sample)}): {sample}")
        if unexpected:
            raise RuntimeError(
                f"[RSSAM2VisionEncoder] LoRA audit failed: {len(unexpected)} unexpected trainable "
                f"parameters in backbone (expected only lora_A/lora_B + {modules_to_save}). "
                f"LoRA wrapping is incomplete; this would cause full fine-tuning. "
                f"First few: {unexpected[:10]}"
            )
        # 审查 3.4：LoRA 真正插入的参数数量必须 > 0，否则 LoRA 等于没生效
        # （可能出现 target_modules 全部被过滤掉，只剩 neck 可训练，审计却没发现异常）。
        if lora_trainable == 0:
            raise RuntimeError(
                "[RSSAM2VisionEncoder] LoRA audit failed: zero trainable LoRA parameters found. "
                "LoRA was not actually inserted (check target_modules matches the model's layer names)."
            )

    def _freeze_params(self):
        """Freeze encoder parameters with optional partial unfreezing."""
        if self.encoder is None:
            return
        
        if self.freeze_backbone_stages < 0:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            for name, param in self.encoder.named_parameters():
                stage_idx = -1
                for i in range(4):
                    if f"stages.{i}" in name or f"backbone_fpn.{i}" in name:
                        stage_idx = i
                        break
                
                if stage_idx >= 0 and stage_idx >= self.freeze_backbone_stages:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            self.encoder.train()

    def init_weights(self):
        if is_main_process():
            print("SAM2 Vision Encoder initialized")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """Forward pass returning intermediate features."""
        if self.encoder is None:
            raise RuntimeError("SAM2 encoder not loaded")

        if self.use_gradient_checkpointing and self.training:
            features = torch.utils.checkpoint.checkpoint(
                self.encoder, x, use_reentrant=False
            )
        else:
            features = self.encoder(x)

        if isinstance(features, dict):
            vision_features = features["vision_features"]
            backbone_fpn = features.get("backbone_fpn")
            if backbone_fpn is not None:
                return vision_features, tuple(backbone_fpn)
            else:
                return vision_features, (vision_features,)
        elif isinstance(features, (list, tuple)):
            return features[0], features
        else:
            return features, (features,)


class RSSAM2MaskDecoderWrapper(BaseModule):
    """Wrapper class to make RSSAM2MaskDecoder compatible with existing RSPrompterAnchorMaskHead."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_high_res_features: bool = False,
        init_cfg: Optional[Dict] = None,
        checkpoint_load_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.decoder = None
        self.use_high_res_features = use_high_res_features
        self.allow_decoder_fallback = os.environ.get("ALLOW_DECODER_FALLBACK", "0") == "1"
        # 问题 13：MaskDecoder 严格加载配置
        self.checkpoint_load_cfg = dict(checkpoint_load_cfg or {})

        if init_cfg is not None and checkpoint_path is None:
            checkpoint_path = init_cfg.get("checkpoint")

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_sam2_decoder(checkpoint_path)
        elif not self.allow_decoder_fallback:
            raise FileNotFoundError(f"SAM2 decoder checkpoint not found: {checkpoint_path}")

    def _load_sam2_decoder(self, checkpoint_path: str):
        """Load SAM2 decoder from checkpoint."""
        try:
            if SAM2_REPO and SAM2_REPO not in sys.path:
                sys.path.insert(0, SAM2_REPO)
            import sam2

            from sam2.modeling.sam.transformer import TwoWayTransformer
            if is_main_process():
                print(f"[SAM2] using package: {getattr(sam2, '__file__', 'unknown')}")

            transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                num_heads=8,
                mlp_dim=2048,
                activation=nn.GELU,
                attention_downsample_rate=2,
            )

            mask_decoder = MaskDecoder(
                transformer_dim=256,
                transformer=transformer,
                num_multimask_outputs=3,
                activation=nn.GELU,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                use_high_res_features=self.use_high_res_features,
                iou_prediction_use_sigmoid=True,
                pred_obj_scores=True,
                pred_obj_scores_mlp=True,
            )

            checkpoint = _load_sam2_checkpoint(checkpoint_path)
            decoder_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("sam_mask_decoder."):
                    new_k = k[17:]
                    decoder_state_dict[new_k] = v

            if not decoder_state_dict:
                raise RuntimeError(
                    f"No sam_mask_decoder.* keys found in checkpoint: {checkpoint_path}"
                )
            # 问题 13：用严格加载工具，按白名单校验
            load_module_state_dict_strict(
                mask_decoder,
                decoder_state_dict,
                module_name="MaskDecoder",
                strict_reproduction=self.checkpoint_load_cfg.get("strict_reproduction", False),
                allowed_missing_patterns=tuple(self.checkpoint_load_cfg.get("allowed_missing_patterns", [])),
                allowed_unexpected_patterns=tuple(self.checkpoint_load_cfg.get("allowed_unexpected_patterns", [])),
            )
            if is_main_process() and self.use_high_res_features:
                print(f"  use_high_res_features: True")
                print(f"  Has conv_s0: {hasattr(mask_decoder, 'conv_s0')}")
                print(f"  Has conv_s1: {hasattr(mask_decoder, 'conv_s1')}")

            self.decoder = mask_decoder
        except Exception as e:
            if is_main_process():
                print(f"Failed to load SAM2 decoder: {e}")
            if self.allow_decoder_fallback:
                self.decoder = None
            else:
                raise

    def forward(
        self,
        image_embeddings: Tensor,
        image_positional_embeddings: Optional[Tensor] = None,
        sparse_prompt_embeddings: Optional[Tensor] = None,
        dense_prompt_embeddings: Optional[Tensor] = None,
        multimask_output: bool = False,
        repeat_image: bool = False,
        attention_similarity: Optional[Tensor] = None,
        target_embedding: Optional[Tensor] = None,
        output_attentions: bool = False,
        high_res_features: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Forward pass compatible with original SAM mask decoder interface."""
        if self.decoder is None:
            if not self.allow_decoder_fallback:
                raise RuntimeError(
                    "SAM2 decoder is not loaded. Set ALLOW_DECODER_FALLBACK=1 "
                    "only for explicit zero-mask debugging."
                )
            batch_size = image_embeddings.shape[0]
            h = w = image_embeddings.shape[2]
            low_res_masks = torch.zeros(batch_size, 1, h, w, device=image_embeddings.device)
            iou_predictions = torch.ones(batch_size, 1, device=image_embeddings.device)
            return low_res_masks, iou_predictions, None

        batch_size = image_embeddings.shape[0]

        if sparse_prompt_embeddings is None:
            sparse_prompt_embeddings = torch.zeros(
                batch_size, 1, 1, 256, device=image_embeddings.device
            )

        if dense_prompt_embeddings is None:
            h = w = image_embeddings.shape[2]
            dense_prompt_embeddings = torch.zeros(
                batch_size, 1, h, w, device=image_embeddings.device
            )

        # Prepare positional embeddings for decoder
        if image_positional_embeddings is not None:
            pe = image_positional_embeddings
            if pe.dim() == 3:
                pe = pe.unsqueeze(0)
            if pe.dim() == 4 and pe.shape[0] != batch_size:
                pe = pe.expand(batch_size, -1, -1, -1)
            
            target_h, target_w = image_embeddings.shape[2], image_embeddings.shape[3]
            if pe.shape[2:] != (target_h, target_w):
                pe = torch.nn.functional.interpolate(
                    pe, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
            
            image_positional_embeddings_pe = pe[0:1, :, :, :].contiguous()
            image_positional_embeddings = pe
        else:
            # Create dummy positional embeddings if not provided
            h = w = image_embeddings.shape[2]
            image_positional_embeddings_pe = torch.zeros(
                1, 256, h, w, device=image_embeddings.device, dtype=image_embeddings.dtype
            )

        # Process high_res_features: apply conv_s0 and conv_s1 if available.
        processed_high_res = None
        if high_res_features is not None and self.use_high_res_features:
            if not (hasattr(self.decoder, 'conv_s0') and hasattr(self.decoder, 'conv_s1')):
                raise RuntimeError("SAM2 decoder is missing conv_s0/conv_s1 for high-res features")
            feat_s0, feat_s1 = high_res_features[:2]
            processed_high_res = [
                self.decoder.conv_s0(feat_s0),  # [B, 32, H, W]
                self.decoder.conv_s1(feat_s1),  # [B, 64, H, W]
            ]

        try:
            low_res_masks, iou_predictions, mask_tokens_out, _ = self.decoder(
                    image_embeddings,
                    image_positional_embeddings_pe,
                    sparse_prompt_embeddings,
                    dense_prompt_embeddings,
                    multimask_output,
                    repeat_image,
                    high_res_features=processed_high_res,
                )
        except Exception as e:
            if not self.allow_decoder_fallback:
                raise RuntimeError(
                    "SAM2 decoder forward failed with "
                    f"image_embeddings={tuple(image_embeddings.shape)}, "
                    f"image_pe={tuple(image_positional_embeddings_pe.shape)}, "
                    f"sparse={tuple(sparse_prompt_embeddings.shape)}, "
                    f"dense={tuple(dense_prompt_embeddings.shape)}, "
                    f"high_res={[tuple(f.shape) for f in high_res_features] if high_res_features is not None else None}, "
                    f"processed_high_res={[tuple(f.shape) for f in processed_high_res] if processed_high_res is not None else None}"
                ) from e
            if is_main_process():
                print(f"Warning: SAM2 decoder forward failed, using zero-mask fallback: {e}")
            h = w = image_embeddings.shape[2]
            low_res_masks = torch.zeros(batch_size, 1, h, w, device=image_embeddings.device)
            iou_predictions = torch.ones(batch_size, 1, device=image_embeddings.device)
            mask_tokens_out = None
        attn_weights = None

        return low_res_masks, iou_predictions, mask_tokens_out


@MODELS.register_module()
class RSFeatureAggregatorSAM2(BaseModule):
    """Feature aggregator adapted for SAM2's output dimensions."""

    in_channels_dict = {
        "tiny": [192, 96, 48, 24],
        "small": [384, 192, 96, 48],
        "base": [768, 384, 192, 96],
        "large": [1024, 512, 256, 128],
        "base_plus": [896, 448, 224, 112],
        # SAM2 Hiera 各 size 经 neck 聚合后各 stage 输出通道（FPN 输出统一 256）
        "sam2_hiera_tiny": [256, 256, 256, 256],
        "sam2_hiera_small": [256, 256, 256, 256],
        "sam2_hiera_base": [256, 256, 256, 256],
        "sam2_hiera_base_plus": [256, 256, 256, 256],
        "sam2_hiera_large": [256, 256, 256, 256],  # FPN输出都是256维
    }

    def __init__(
        self,
        in_channels: str = "sam2_hiera_base",
        hidden_channels: int = 64,
        out_channels: int = 256,
        select_layers: Optional[List[int]] = None,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        if "sam2" in in_channels.lower():
            model_size = in_channels
        else:
            model_size = "base_plus" if "base" in in_channels else "large" if "large" in in_channels else "small" if "small" in in_channels else "tiny"
        # 审查 3.2：禁止 neck key 静默 fallback。未知 key 直接报错，避免隐藏拼写/型号错误。
        if model_size not in self.in_channels_dict:
            raise KeyError(
                f"Unsupported SAM2 neck key: {model_size!r}. "
                f"Expected one of {sorted(self.in_channels_dict.keys())}."
            )
        self.in_channels = self.in_channels_dict[model_size]
        self.select_layers = select_layers or [0, 1, 2]

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            if i_layer < len(self.in_channels):
                in_ch = self.in_channels[i_layer]
            else:
                in_ch = self.in_channels[-1]

            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, hidden_channels, 1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.hidden_convs = nn.ModuleList()
        for _ in self.select_layers:
            self.hidden_convs.append(
                nn.Sequential(
                    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, inputs) -> Tuple[Tensor, ...]:
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], tuple):
            inputs = inputs[1]
        
        if not inputs:
            raise ValueError("Empty inputs to neck")

        inputs = tuple(x.float() for x in inputs)

        if is_main_process() and not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[DEBUG] RSFeatureAggregatorSAM2 inputs: {len(inputs)} tensors")
            for i, t in enumerate(inputs):
                print(f"  [{i}] shape={t.shape}, dtype={t.dtype}")

        if len(inputs) == 1:
            x = inputs[0]
            x = self.fusion_conv(x)
            feat_64 = x
            feat_32 = torch.nn.functional.max_pool2d(feat_64, kernel_size=2, stride=2)
            feat_16 = torch.nn.functional.max_pool2d(feat_32, kernel_size=2, stride=2)
            feat_8 = torch.nn.functional.max_pool2d(feat_16, kernel_size=2, stride=2)
            feat_4 = torch.nn.functional.max_pool2d(feat_8, kernel_size=2, stride=2)
            # return features in order of increasing stride (4, 8, 16, 32, 64)
            return (feat_64, feat_32, feat_16, feat_8, feat_4)

        while len(inputs) < len(self.in_channels):
            inputs = inputs + (inputs[-1],)
        
        inputs = [einops.rearrange(x, "b h w c -> b c h w") if x.dim() == 3 else x for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            if i_layer < len(inputs):
                x = inputs[i_layer]
                if x.dim() == 3:
                    x = x.permute(0, 2, 1)
                features.append(self.downconvs[idx](x))

        if not features:
            raise ValueError("No features to process")

        # Find largest feature map size
        target_h, target_w = 0, 0
        for feat in features:
            if feat.shape[2] > target_h:
                target_h = feat.shape[2]
                target_w = feat.shape[3]
        target_size = (target_h, target_w)

        aligned_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = torch.nn.functional.interpolate(
                    feat, size=target_size, mode='bilinear', align_corners=False
                )
            else:
                feat = feat
            aligned_features.append(feat)
        
        x = aligned_features[0]
        for idx, hidden_conv in enumerate(self.hidden_convs[1:], 1):
            if idx < len(aligned_features):
                x = x + aligned_features[idx]
            residual = hidden_conv(x)
            x = x + residual
        x = self.fusion_conv(x)
        
        feat_64 = x
        feat_32 = torch.nn.functional.max_pool2d(feat_64, kernel_size=2, stride=2)
        feat_16 = torch.nn.functional.max_pool2d(feat_32, kernel_size=2, stride=2)
        feat_8 = torch.nn.functional.max_pool2d(feat_16, kernel_size=2, stride=2)
        feat_4 = torch.nn.functional.max_pool2d(feat_8, kernel_size=2, stride=2)
        
        # return features in order of increasing stride (4, 8, 16, 32, 64)
        return (feat_64, feat_32, feat_16, feat_8, feat_4)


@MODELS.register_module()
class RSSAM2PAFPN(BaseModule):
    """Detection-oriented FPN/PAN neck for SAM2 backbone_fpn features.

    SAM2 provides four 256-channel maps at strides 4/8/16/32. This neck keeps
    those levels separate, sends semantics down with an FPN path, then sends
    localization back up with a PAN path. The output contract matches the
    existing detector: five 512-channel maps at strides 4/8/16/32/64.

    Channel design (mid/out decoupling):
      - ``mid_channels`` controls the internal FPN/PAN computation width. All
        lateral/top-down/downsample/bottom-up/p64 convs operate at this width.
        Default 256 matches the SAM2 backbone output, so no information is lost
        while the N×N conv cost drops from out² to mid² (4× at mid=256).
      - ``out_channels`` (512) is the detector interface contract: RPN/SABL/
        RoIExtractor all expect 512. A per-level 1×1 projection (``output_proj``)
        lifts mid→out at the very end, keeping downstream configs untouched.
      - To reproduce the legacy all-out behavior set ``mid_channels=out_channels``
        (the output_proj then becomes an identity-ish 1×1).
    """

    in_channels_dict = {
        "sam2_hiera_tiny": [256, 256, 256, 256],
        "sam2_hiera_small": [256, 256, 256, 256],
        "sam2_hiera_base": [256, 256, 256, 256],
        "sam2_hiera_base_plus": [256, 256, 256, 256],
        "sam2_hiera_large": [256, 256, 256, 256],
    }

    def __init__(
        self,
        in_channels: str = "sam2_hiera_large",
        out_channels: int = 512,
        mid_channels: int = 256,
        num_levels: int = 4,
        norm_groups: int = 32,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.out_channels = out_channels
        self.mid_channels = int(mid_channels)
        self.num_levels = num_levels
        # 审查 3.2：禁止 neck key 静默 fallback。未知 key 直接报错。
        if in_channels not in self.in_channels_dict:
            raise KeyError(
                f"Unsupported SAM2 neck key: {in_channels!r}. "
                f"Expected one of {sorted(self.in_channels_dict.keys())}."
            )
        self.in_channels = self.in_channels_dict[in_channels]
        if len(self.in_channels) < num_levels:
            self.in_channels = self.in_channels + [self.in_channels[-1]] * (num_levels - len(self.in_channels))

        # Internal FPN/PAN computation lives at mid_channels.
        self.lateral_convs = nn.ModuleList([
            self._conv_block(ch, self.mid_channels, kernel_size=1, padding=0, norm_groups=norm_groups)
            for ch in self.in_channels[:num_levels]
        ])
        self.top_down_refine = nn.ModuleList([
            self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, norm_groups=norm_groups)
            for _ in range(num_levels)
        ])
        self.downsample_convs = nn.ModuleList([
            self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, stride=2, padding=1, norm_groups=norm_groups)
            for _ in range(num_levels - 1)
        ])
        self.bottom_up_refine = nn.ModuleList([
            self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, norm_groups=norm_groups)
            for _ in range(num_levels)
        ])
        self.p64_conv = self._conv_block(self.mid_channels, self.mid_channels, kernel_size=3, stride=2, padding=1, norm_groups=norm_groups)
        # Per-level 1×1 projection lifts mid→out to satisfy the detector's
        # in_channels=512 contract without paying out² across all N×N convs.
        num_outputs = num_levels + 1  # + P64
        self.output_proj = nn.ModuleList([
            self._conv_block(self.mid_channels, out_channels, kernel_size=1, padding=0, norm_groups=norm_groups)
            for _ in range(num_outputs)
        ])
        self.level_scales = nn.Parameter(torch.ones(num_outputs))
        self._init_weights()

    @staticmethod
    def _conv_block(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        norm_groups: int,
        stride: int = 1,
    ) -> nn.Sequential:
        groups = min(norm_groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @staticmethod
    def _parse_inputs(inputs) -> Tuple[Tensor, ...]:
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], tuple):
            inputs = inputs[1]
        if not isinstance(inputs, (tuple, list)) or len(inputs) == 0:
            raise ValueError(f"Unexpected inputs to RSSAM2PAFPN: {type(inputs)}")
        return tuple(inputs)

    def forward(self, inputs) -> Tuple[Tensor, ...]:
        inputs = self._parse_inputs(inputs)
        while len(inputs) < self.num_levels:
            inputs = inputs + (inputs[-1],)
        inputs = inputs[:self.num_levels]

        if is_main_process() and not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            print(f"[DEBUG] RSSAM2PAFPN inputs: {len(inputs)} tensors")
            for i, t in enumerate(inputs):
                print(f"  [{i}] shape={t.shape}, dtype={t.dtype}")

        laterals = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        top_down = [None] * self.num_levels
        top_down[-1] = self.top_down_refine[-1](laterals[-1])
        for i in range(self.num_levels - 2, -1, -1):
            up = F.interpolate(top_down[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            top_down[i] = self.top_down_refine[i](laterals[i] + up)

        outputs = [None] * self.num_levels
        outputs[0] = self.bottom_up_refine[0](top_down[0])
        for i in range(1, self.num_levels):
            down = self.downsample_convs[i - 1](outputs[i - 1])
            if down.shape[-2:] != top_down[i].shape[-2:]:
                down = F.interpolate(down, size=top_down[i].shape[-2:], mode="nearest")
            outputs[i] = self.bottom_up_refine[i](top_down[i] + down)

        outputs.append(self.p64_conv(outputs[-1]))
        # Lift mid→out per level to satisfy the detector's in_channels contract.
        outputs = [proj(out) for proj, out in zip(self.output_proj, outputs)]
        outputs = [out * scale for out, scale in zip(outputs, self.level_scales)]
        return tuple(outputs)


@MODELS.register_module()
class RSPrompterAnchorMaskHeadSAM2(FCNMaskHead, BaseModule):
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
            _valid_explicit_modes = {"points", "points_box", "points_box_dense"}
            if self.explicit_prompt_mode not in _valid_explicit_modes:
                raise ValueError(
                    "explicit_prompt_mode must be points, points_box or "
                    f"points_box_dense, got {self.explicit_prompt_mode!r}"
                )
            self.explicit_use_box_prompt = self.explicit_prompt_mode in {
                "points_box", "points_box_dense"
            }
            self.explicit_use_dense_prompt = (
                self.explicit_prompt_mode == "points_box_dense"
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
        # 旧参数 → cfg 默认值（仅当 cfg 未显式指定时）
        _pe_cfg.setdefault("require_pretrained", bool(load_pe_pretrained))
        _pe_cfg.setdefault("ckpt_path", sam2_ckpt_for_pe)
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
            # Allow selective unfreeze of mask_downscaling (4684 params) so the
            # frozen PromptEncoder conv stack can adapt to weak coarse-mask
            # canvas signals.  Controlled via env-var so existing experiments
            # keep the committed freeze-all contract.
            _unfreeze_msk_ds = os.environ.get("UNFREEZE_MASK_DOWNSCALING", "0") == "1"
            for _p in self.prompt_encoder.mask_downscaling.parameters():
                _p.requires_grad_(_unfreeze_msk_ds)

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

        # === dense ROI residual (可选) ===
        # 在 no_mask 常量基底上叠加 per-ROI dense 残差, roi_dense_scale 零初始化 → 启动严格等价 B0
        if self.dense_prompt_mode == "residual":
            self.roi_dense_proj = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, 1),
                nn.GroupNorm(32, embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
                nn.GroupNorm(32, embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim, embed_dim, 1),
            )
            self.roi_dense_scale = nn.Parameter(torch.zeros(1))
        else:
            self.roi_dense_proj = None
            self.register_parameter("roi_dense_scale", None)

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
            if not self.use_shape_dense or not self.explicit_use_dense_prompt:
                raise ValueError(
                    "P2BoundaryRefiner requires points_box_dense prompt mode"
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

    def _shape_prior_to_prompt_mask(
        self,
        roi_mask_logits: Tensor,
        boxes: Tensor,
    ) -> Tensor:
        """批次2 子任务3：ROI-local coarse logits → full-image PE mask grid。

        处理顺序：ROI-local transform → resize 到 bbox 对应 prompt-grid → paste 到
        full-image canvas → bbox 外 outside_fill_logit → 最终 clamp（若非 None）。
        """
        from .dense_prompt_utils import transform_coarse_prompt, paste_roi_to_full_canvas
        # 1. ROI-local transform（在 resize/paste 之前完成）
        transformed = transform_coarse_prompt(
            roi_mask_logits,
            transform=self.dense_prompt_cfg.get("transform", "raw_logits"),
            gamma=self.dense_prompt_cfg.get("gamma", 1.0),
            strength=self.dense_prompt_cfg.get("strength", 1.0),
            clamp_range=self.dense_prompt_cfg.get("clamp_range", None),
            temperature=self.dense_prompt_cfg.get("temperature", 1.0),
        )
        # 2. paste 到 full-image canvas（PE mask 尺寸从 PE 属性读，不硬编码）
        mask_h, mask_w = self._pe_mask_input_size()
        return paste_roi_to_full_canvas(
            transformed, boxes, mask_h, mask_w,
            image_size=(self.prompt_encoder_image_size, self.prompt_encoder_image_size),
            outside_fill_logit=float(self.dense_prompt_cfg.get("outside_fill_logit", 0.0)),
        )

    def _select_explicit_prompt_inputs(
        self,
        coords: Tensor,
        labels: Tensor,
        boxes: Tensor,
        masks: Optional[Tensor],
    ):
        """Select SAM2 PromptEncoder inputs for the explicit-mask ablation.

        The detector boxes remain the internal geometry for ROI extraction and
        coordinate mapping in every mode.  This method controls only which
        prompt modalities are exposed to the frozen PromptEncoder.
        """
        if self.prompt_sparse_mode != "shape_point":
            raise RuntimeError(
                "_select_explicit_prompt_inputs is only valid for shape_point"
            )
        # A warm-up/all-invalid batch must be a true no-point prompt. Passing
        # four label=-1 entries would create four learned not-a-point tokens.
        points = None if bool((labels < 0).all()) else (coords, labels)
        boxes_for_pe = boxes if self.explicit_use_box_prompt else None
        masks_for_pe = masks if self.explicit_use_dense_prompt else None
        return points, boxes_for_pe, masks_for_pe

    @staticmethod
    def _neutralize_invalid_point_tokens(
        sparse_embeddings: Tensor, labels: Tensor
    ) -> Tensor:
        """Zero invalid point slots while preserving any following box tokens."""
        point_count = int(labels.shape[1])
        point_tokens = sparse_embeddings[:, :point_count]
        valid = (labels >= 0).unsqueeze(-1).to(dtype=point_tokens.dtype)
        point_tokens = point_tokens * valid
        return torch.cat(
            [point_tokens, sparse_embeddings[:, point_count:]], dim=1
        )

    def _effective_shape_dense_alpha(self):
        """批次2 子任务4：返回 shape dense gate 的有效值（三模式统一）。

        - legacy_unbounded：self.shape_prompt_scale（可训练无界，旧实现）
        - fixed：self.shape_dense_alpha（buffer，不训练）
        - global_sigmoid：sigmoid(self.shape_dense_alpha_raw) ∈ (0,1)
        - use_shape_dense=False：返回 None（forward 里据此跳过 dense 混合）
        """
        if not self.use_shape_dense:
            return None
        if self.shape_scale_mode == "legacy_unbounded":
            return self.shape_prompt_scale
        elif self.shape_scale_mode == "fixed":
            return self.shape_dense_alpha
        elif self.shape_scale_mode == "global_sigmoid":
            return torch.sigmoid(self.shape_dense_alpha_raw)
        return None

    def _dense_embedding_box_support(
        self, boxes: Tensor, spatial_size: Tuple[int, int], dtype: torch.dtype
    ) -> Tensor:
        """Restrict dense-prompt embedding changes to the proposal box."""
        height, width = int(spatial_size[0]), int(spatial_size[1])
        yy = (
            torch.arange(height, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) * (float(self.prompt_encoder_image_size) / float(height))
        xx = (
            torch.arange(width, device=boxes.device, dtype=boxes.dtype) + 0.5
        ) * (float(self.prompt_encoder_image_size) / float(width))
        x1, y1, x2, y2 = boxes.detach().unbind(dim=1)
        support = (
            (xx[None, None, None, :] >= x1[:, None, None, None])
            & (xx[None, None, None, :] < x2[:, None, None, None])
            & (yy[None, None, :, None] >= y1[:, None, None, None])
            & (yy[None, None, :, None] < y2[:, None, None, None])
        )
        return support.to(dtype=dtype)

    def _pe_mask_input_size(self) -> Tuple[int, int]:
        """批次2 子任务3：读取 PromptEncoder 的 mask 输入尺寸（不硬编码）。

        优先 self.prompt_encoder.mask_input_size（SAM2 PE 属性）；
        若 PE 不存在则 fallback 到 prompt_encoder_embed_size*4（旧逻辑）。
        不假设正方形（返回 (H,W)）。
        """
        pe = getattr(self, "prompt_encoder", None)
        if pe is not None and hasattr(pe, "mask_input_size"):
            mis = pe.mask_input_size
            return (int(mis[0]), int(mis[1]))
        es = self.prompt_encoder_embed_size
        return (es * 4, es * 4)

    @staticmethod
    def _debug_delta_ratio(after: Optional[Tensor], before: Optional[Tensor]) -> Optional[float]:
        if after is None or before is None:
            return None
        with torch.no_grad():
            denom = before.detach().float().norm().clamp_min(1e-6)
            ratio = (after.detach().float() - before.detach().float()).norm() / denom
            return float(ratio.item())

    def set_current_epoch(self, epoch_number: int) -> None:
        """Synchronize epoch-dependent point/loss schedules (1-based)."""
        self.current_epoch = max(1, int(epoch_number))
        if self.shape_point_miner is not None:
            self.shape_point_miner.set_current_epoch(self.current_epoch)
        if self.coarse_mask_loss_module is not None and hasattr(
            self.coarse_mask_loss_module, "set_current_epoch"
        ):
            self.coarse_mask_loss_module.set_current_epoch(self.current_epoch)

    def _shape_point_gt_stats(self, coarse_targets: Tensor) -> Dict[str, Tensor]:
        local_yx = self._last_shape_point_local_yx
        labels = self._last_shape_point_labels
        if local_yx is None or labels is None or coarse_targets.shape[0] == 0:
            zero = coarse_targets.new_zeros(())
            return {
                "SP/p1_in_gt_fg_count": zero, "SP/p1_gt_valid_count": zero,
                "SP/p2_in_gt_fg_count": zero, "SP/p2_gt_valid_count": zero,
                "SP/n1_in_gt_bg_count": zero, "SP/n1_gt_valid_count": zero,
                "SP/n2_in_gt_bg_count": zero, "SP/n2_gt_valid_count": zero,
            }
        if local_yx.shape[0] != coarse_targets.shape[0]:
            raise RuntimeError(
                "Shape-point/target ROI count mismatch: "
                f"{local_yx.shape[0]} vs {coarse_targets.shape[0]}"
            )
        h, w = coarse_targets.shape[-2:]
        y = local_yx[..., 0].clamp(0, h - 1)
        x = local_yx[..., 1].clamp(0, w - 1)
        batch = torch.arange(coarse_targets.shape[0], device=coarse_targets.device)[:, None]
        sampled = coarse_targets[batch, y, x] >= 0.5
        out: Dict[str, Tensor] = {}
        for slot, name, positive in (
            (0, "p1", True), (1, "p2", True),
            (2, "n1", False), (3, "n2", False),
        ):
            valid = labels[:, slot] >= 0
            correct = sampled[:, slot] if positive else ~sampled[:, slot]
            suffix = "in_gt_fg" if positive else "in_gt_bg"
            out[f"SP/{name}_{suffix}_count"] = (correct & valid).sum().detach()
            out[f"SP/{name}_gt_valid_count"] = valid.sum().detach()
        return out

    def get_uav_debug_stats(self) -> Dict[str, float]:
        # UAV branch removed (doc §1). This stub is kept for backward
        # compatibility with the legacy trainer logger call sites; it now
        # surfaces the GT-point diagnostics previously written here.
        return dict(getattr(self, "_last_uav_debug_stats", {}) or {})

    def _roi_sam_local_boxes(self, boxes: Tensor) -> Tensor:
        """Return canonical full-canvas boxes for ROI-local PromptEncoder input."""
        local_boxes = torch.zeros_like(boxes)
        local_boxes[:, 2:] = float(self.prompt_encoder_image_size)
        return local_boxes

    def _roi_sam_crop_feature(
        self,
        feature: Tensor,
        boxes: Tensor,
        roi_img_ids: Tensor,
    ) -> Tensor:
        """Normalize each proposal crop back to one native SAM2 feature canvas."""
        if feature.dim() != 4:
            raise ValueError(f"ROI-SAM feature must be BCHW, got {tuple(feature.shape)}")
        if boxes.shape[0] != roi_img_ids.shape[0]:
            raise ValueError("ROI-SAM boxes and roi_img_ids must have equal length")
        if boxes.numel() == 0:
            return feature.new_empty((0, feature.shape[1], *feature.shape[-2:]))
        if int(roi_img_ids.min()) < 0 or int(roi_img_ids.max()) >= feature.shape[0]:
            raise ValueError("ROI-SAM roi_img_ids contain an invalid image index")
        height, width = feature.shape[-2:]
        scale_x = float(width) / float(self.prompt_encoder_image_size)
        scale_y = float(height) / float(self.prompt_encoder_image_size)
        if not math.isclose(scale_x, scale_y, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(
                "ROI-SAM currently requires square images/features with one spatial scale"
            )
        rois = torch.cat(
            [roi_img_ids.to(device=boxes.device, dtype=boxes.dtype)[:, None], boxes],
            dim=1,
        )
        align = RoIAlign(
            output_size=(height, width),
            spatial_scale=scale_x,
            sampling_ratio=int(self.roi_sam_cfg.sampling_ratio),
            pool_mode="avg",
            aligned=bool(self.roi_sam_cfg.aligned),
        )
        return align(feature, rois)

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

        # === shape prior mask (阶段2, 必须在 PE 调用之前计算) ===
        shape_prior_mask_logits = None
        coarse_outputs = None
        shape_prior_prompt_logits = None
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
                    debug_stats.update({
                        "P2BR/roi_count": delta.new_tensor(float(delta.shape[0])),
                        "P2BR/valid_support_count": support_valid.float().sum(),
                        "P2BR/rejected_support_count": (~support_valid).float().sum(),
                        "P2BR/search_coverage_sum": refiner_outputs["search_coverage_sum"],
                        "P2BR/delta_abs_sum": delta.abs().flatten(1).mean(1).sum(),
                        "P2BR/delta_nonzero_sum": delta.ne(0).float().flatten(1).mean(1).sum(),
                        "P2BR/delta_saturated_sum": delta.abs().ge(0.396).float().flatten(1).mean(1).sum(),
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
                shape_prior_prompt_logits = self._shape_prior_to_prompt_mask(
                    shape_prior_mask_logits, boxes
                )
            else:
                # True sparse-only: keep the coarse mask for point mining and
                # auxiliary supervision, but skip transform/paste/PE mask work.
                shape_prior_prompt_logits = None

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

        # === dense_embeddings ===
        if self.no_mask_embed is None:
            base_dense = image_embeddings.new_zeros(
                roi_bs, 256, image_embedding_size[0], image_embedding_size[1]
            )
        else:
            base_dense = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
                roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
            )
        if self.roi_dense_proj is not None:
            roi_dense = self.roi_dense_proj(x)  # [N,256,14,14]
            roi_dense = F.interpolate(
                roi_dense, size=image_embedding_size,
                mode='bilinear', align_corners=False,
            )
            base_dense = base_dense + self.roi_dense_scale * roi_dense
        if self._diagnostic_forward_ablation == "zero_base_dense":
            base_dense = torch.zeros_like(base_dense)

        if dense_pe is not None:
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

    def get_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
        pos_priors_override: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Build ROI-local final-mask targets for historical B0/B1 reproduction.

        This legacy adapter deliberately interprets each decoder output as an
        ROI-local map, crops the assigned GT by the proposal and uses standard
        MMDetection bbox paste at prediction time.
        """
        pos_proposals = (
            pos_priors_override
            if pos_priors_override is not None
            else [res.pos_priors for res in sampling_results]
        )
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        return mask_target(
            pos_proposals,
            pos_assigned_gt_inds,
            gt_masks,
            rcnn_train_cfg,
        )

    def get_full_image_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        target_size: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        """Build one native full-image target for every positive proposal."""
        targets = []
        target_h, target_w = int(target_size[0]), int(target_size[1])
        for result, gt_instances in zip(sampling_results, batch_gt_instances):
            assigned = result.pos_assigned_gt_inds
            if assigned.numel() == 0:
                continue
            gt_masks = gt_instances.masks
            unique_ids, inverse = torch.unique(
                assigned.detach().cpu(), return_inverse=True
            )
            unique_masks = gt_masks[unique_ids].to_tensor(
                dtype=torch.float32, device=device
            )
            selected = unique_masks[inverse.to(device=unique_masks.device)]
            selected = F.interpolate(
                selected[:, None],
                size=(target_h, target_w),
                mode="nearest",
            )[:, 0]
            targets.append(selected)
        if not targets:
            return torch.zeros(
                (0, target_h, target_w), device=device, dtype=torch.float32
            )
        return torch.cat(targets, dim=0)

    def get_final_mask_roi_support(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        target_size: Tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        """Rasterize expanded positive proposal boxes on the final-mask grid.

        C2-L retains SAM2's full-image coordinate contract.  These proposal
        supports affect only loss weighting; they never crop or paste decoder
        predictions and are not needed by validation/inference.
        """
        target_h, target_w = int(target_size[0]), int(target_size[1])
        expand_ratio = float(self.final_mask_loss_cfg.roi_expand_ratio)
        supports = []
        for result, gt_instances in zip(sampling_results, batch_gt_instances):
            priors = getattr(result.pos_priors, "tensor", result.pos_priors)
            if priors.numel() == 0:
                continue
            full_h = int(gt_instances.masks.height)
            full_w = int(gt_instances.masks.width)
            scale_x = float(target_w) / float(full_w)
            scale_y = float(target_h) / float(full_h)
            for prior in priors.detach():
                x1, y1, x2, y2 = [float(value) for value in prior[:4].cpu()]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                half_w = max(0.5, 0.5 * (x2 - x1) * expand_ratio)
                half_h = max(0.5, 0.5 * (y2 - y1) * expand_ratio)
                gx1 = max(0, min(target_w - 1, math.floor((cx - half_w) * scale_x)))
                gy1 = max(0, min(target_h - 1, math.floor((cy - half_h) * scale_y)))
                gx2 = max(gx1 + 1, min(target_w, math.ceil((cx + half_w) * scale_x)))
                gy2 = max(gy1 + 1, min(target_h, math.ceil((cy + half_h) * scale_y)))
                support = torch.zeros(
                    (target_h, target_w), device=device, dtype=torch.bool
                )
                support[gy1:gy2, gx1:gx2] = True
                supports.append(support)
        if not supports:
            return torch.zeros(
                (0, target_h, target_w), device=device, dtype=torch.bool
            )
        return torch.stack(supports, dim=0)

    @staticmethod
    def roi_balanced_final_mask_loss(
        logits: Tensor,
        targets: Tensor,
        roi_support: Tensor,
        cfg: ConfigDict,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Balanced ROI BCE + ROI Dice + weak outside-background BCE."""
        if logits.shape != targets.shape or targets.shape != roi_support.shape:
            raise ValueError(
                "ROI-focused final-mask tensors must have identical [N,H,W] shapes: "
                f"logits={tuple(logits.shape)} targets={tuple(targets.shape)} "
                f"support={tuple(roi_support.shape)}"
            )
        eps = float(cfg.eps)
        target = targets.float().clamp(0, 1)
        inside = roi_support.to(dtype=logits.dtype)
        outside = 1.0 - inside
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        reduce_dims = (1, 2)

        positive = inside * target
        negative = inside * (1.0 - target)
        positive_count = positive.sum(dim=reduce_dims)
        negative_count = negative.sum(dim=reduce_dims)
        positive_bce = (bce * positive).sum(dim=reduce_dims) / positive_count.clamp_min(1.0)
        negative_bce = (bce * negative).sum(dim=reduce_dims) / negative_count.clamp_min(1.0)
        has_positive = positive_count > 0
        has_negative = negative_count > 0
        both = has_positive & has_negative
        roi_bce_per_roi = torch.where(
            both,
            0.5 * (positive_bce + negative_bce),
            torch.where(has_positive, positive_bce, negative_bce),
        )
        roi_bce = roi_bce_per_roi.mean()

        probability = logits.sigmoid() * inside
        target_inside = target * inside
        intersection = (probability * target_inside).sum(dim=reduce_dims)
        denominator = probability.sum(dim=reduce_dims) + target_inside.sum(dim=reduce_dims)
        roi_dice = (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()

        outside_count = outside.sum(dim=reduce_dims)
        outside_bce_per_roi = (bce * outside).sum(dim=reduce_dims) / outside_count.clamp_min(1.0)
        outside_bce_per_roi = torch.where(
            outside_count > 0,
            outside_bce_per_roi,
            torch.zeros_like(outside_bce_per_roi),
        )
        outside_bce = outside_bce_per_roi.mean()
        total = (
            float(cfg.roi_bce_weight) * roi_bce
            + float(cfg.roi_dice_weight) * roi_dice
            + float(cfg.outside_bce_weight) * outside_bce
        )
        stats = {
            "debug_final_mask_roi_bce": roi_bce.detach(),
            "debug_final_mask_roi_dice": roi_dice.detach(),
            "debug_final_mask_outside_bce": outside_bce.detach(),
            "debug_final_mask_roi_coverage": inside.detach().mean(),
            "debug_final_mask_roi_target_fill": (
                target_inside.sum() / inside.sum().clamp_min(1.0)
            ).detach(),
        }
        return total, stats

    def get_coarse_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        mask_size: Optional[Tuple[int, int]] = None,
        prompt_pos_priors: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """coarse mask（ShapePriorInjector 输出）的 ROI-local 监督 target。

        coarse mask 在 ROI 局部坐标系生成与监督。本方法按 positive proposal box
        crop GT mask 并 resize 到 [M,H,W]（pred 原生分辨率），供
        CoarseMaskLoss 使用。旧 B0/B1 的最终 SAM2 mask 仍由 MMDetection
        ``mask_target`` 独立生成；普通显式 coarse 路线使用 full-image target；
        C2-R 则复用本方法，在 MaskDecoder 原生 ROI 网格上生成 final-mask target。

        性能优化：先对去重后的 GT mask 调一次 to_tensor（避免重复渲染整图 mask），
        再逐 ROI crop+resize（crop/resize 本身轻量，瓶颈在 to_tensor 的整图渲染）。
        """
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        if prompt_pos_priors is not None:
            pos_priors_list = prompt_pos_priors
        else:
            pos_priors_list = [res.pos_priors for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        device = pos_priors_list[0].device if pos_priors_list else torch.device("cpu")
        if mask_size is None:
            roi_h = roi_w = self.roi_mask_size
        else:
            roi_h, roi_w = int(mask_size[0]), int(mask_size[1])
        coarse_targets_list = []
        for pos_priors, pos_gt_inds, gt_mask in zip(
            pos_priors_list, pos_assigned_gt_inds, gt_masks
        ):
            if len(pos_gt_inds) == 0:
                coarse_targets_list.append(
                    torch.zeros((0, roi_h, roi_w), device=device, dtype=torch.float32)
                )
                continue
            # 性能优化：去重 GT 索引，只渲染唯一的整图 mask 一次（避免重复 to_tensor）
            pos_gt_inds_cpu = pos_gt_inds.cpu()
            unique_gt_inds, inverse_idx = torch.unique(
                pos_gt_inds_cpu, return_inverse=True
            )
            # 只对唯一 GT 调一次 to_tensor（整图渲染是大头开销）
            gt_mask_tensor_unique = gt_mask[unique_gt_inds].to_tensor(
                dtype=torch.float32, device=device
            )  # [num_unique_gt, H_full, W_full]
            # 按 inverse_idx 映射回每个 positive proposal（索引复用，无重复渲染）
            gt_mask_tensor = gt_mask_tensor_unique[inverse_idx]  # [num_pos, H, W]
            boxes = pos_priors
            cropped = []
            for i in range(gt_mask_tensor.shape[0]):
                # 审查第七节：用 floor(x1/y1)/ceil(x2/y2) 量化，避免 to(long) 截断丢边界。
                import math as _math
                _bx = boxes[i].cpu()
                x1f, y1f, x2f, y2f = _bx[0].item(), _bx[1].item(), _bx[2].item(), _bx[3].item()
                x1 = _math.floor(x1f)
                y1 = _math.floor(y1f)
                x2 = _math.ceil(x2f)
                y2 = _math.ceil(y2f)
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(gt_mask_tensor.shape[-1], max(x1c + 1, x2))
                y2c = min(gt_mask_tensor.shape[-2], max(y1c + 1, y2))
                patch = gt_mask_tensor[i : i + 1, y1c:y2c, x1c:x2c]
                if patch.numel() == 0:
                    patch = torch.zeros(1, 1, device=device, dtype=torch.float32)
                patch = patch.unsqueeze(0)
                # Binary instance targets are resized with nearest-neighbor so
                # the coarse loss is supervised by a crisp mask at its native size.
                patch = F.interpolate(
                    patch, size=(roi_h, roi_w), mode="nearest"
                )
                patch = patch.squeeze(0).squeeze(0).clamp(0, 1)  # [H,W]
                cropped.append(patch)
            coarse_targets_list.append(torch.stack(cropped, dim=0))
        coarse_targets = torch.cat(coarse_targets_list)
        assert coarse_targets.dim() == 3, (
            f"coarse_targets must be [M,H,W], got {tuple(coarse_targets.shape)}"
        )
        return coarse_targets

    def coarse_mask_loss(
        self,
        shape_prior_mask_logits: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        prompt_pos_priors: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Dict]:
        """计算 coarse mask 独立 loss（替换旧 _shape_prior_aux_loss）。

        在预测原生分辨率上计算（target 用 nearest 对齐到 pred 尺寸）。
        prompt_pos_priors 用于 box jitter 后的 target crop（见审查 2.7）。
        """
        target_size = shape_prior_mask_logits.shape[-2:]
        coarse_targets = self.get_coarse_targets(
            sampling_results, batch_gt_instances,
            mask_size=target_size, prompt_pos_priors=prompt_pos_priors,
        )
        gt_point_stats = self._shape_point_gt_stats(coarse_targets)
        if not hasattr(self, "_last_uav_debug_stats") or self._last_uav_debug_stats is None:
            self._last_uav_debug_stats = {}
        self._last_uav_debug_stats.update(gt_point_stats)
        return self.coarse_mask_loss_module(shape_prior_mask_logits, coarse_targets)

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
        final_mask_pos_priors: Optional[List[Tensor]] = None,
    ) -> dict:
        if self.final_mask_coordinate_mode == "full_image":
            mask_targets = self.get_full_image_targets(
                sampling_results,
                batch_gt_instances,
                target_size=mask_preds.shape[-2:],
                device=mask_preds.device,
            )
        elif self.roi_sam_enabled:
            # ROI-SAM keeps the target on the decoder's native ROI grid instead
            # of collapsing the 128x128 prediction to the historical 28x28
            # Mask R-CNN target used by B0/B1.
            mask_targets = self.get_coarse_targets(
                sampling_results,
                batch_gt_instances,
                mask_size=mask_preds.shape[-2:],
                prompt_pos_priors=final_mask_pos_priors,
            )
        else:
            mask_targets = self.get_targets(
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
                rcnn_train_cfg=rcnn_train_cfg,
                pos_priors_override=final_mask_pos_priors,
            )
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        if mask_preds.shape[-2:] != mask_targets.shape[-2:]:
            mask_preds = F.interpolate(
                mask_preds,
                size=mask_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        loss = dict()
        debug_stats = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.final_mask_loss_cfg.mode == "roi_balanced_dice":
                if self.class_agnostic:
                    selected_logits = mask_preds[:, 0]
                else:
                    row = torch.arange(mask_preds.shape[0], device=mask_preds.device)
                    selected_logits = mask_preds[row, pos_labels]
                roi_support = self.get_final_mask_roi_support(
                    sampling_results,
                    batch_gt_instances,
                    target_size=mask_targets.shape[-2:],
                    device=mask_preds.device,
                )
                loss_mask, focused_stats = self.roi_balanced_final_mask_loss(
                    selected_logits,
                    mask_targets,
                    roi_support,
                    self.final_mask_loss_cfg,
                )
                debug_stats.update(focused_stats)
            elif self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
            mask_probs = mask_preds.sigmoid().detach()
            debug_stats.update(
                debug_mask_pos_rois=mask_preds.new_tensor(float(mask_preds.size(0))),
                debug_mask_logit_mean=mask_preds.detach().mean(),
                debug_mask_logit_std=mask_preds.detach().std(unbiased=False),
                debug_mask_prob_mean=mask_probs.mean(),
                debug_mask_prob_ge05=(mask_probs >= 0.5).float().mean(),
                debug_mask_target_fill=mask_targets.detach().float().mean(),
            )
        loss["loss_mask"] = loss_mask
        return dict(
            loss_mask=loss,
            mask_targets=mask_targets,
            debug_stats=debug_stats,
        )

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
