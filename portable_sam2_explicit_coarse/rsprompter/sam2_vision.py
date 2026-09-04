"""SAM2 checkpoint loaders, positional embedding and LoRA vision encoder (split from models_sam2)."""

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


