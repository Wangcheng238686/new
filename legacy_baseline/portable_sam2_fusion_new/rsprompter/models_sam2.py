"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

import os
from typing import Dict, List, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.models import MaskRCNN, StandardRoIHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask.mask_target import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType


SAM2_AVAILABLE = False
try:
    from sam2.modeling.backbones.hieradet import Hiera
    from sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    from sam2.modeling.sam.mask_decoder import MaskDecoder
    SAM2_AVAILABLE = True
except ImportError:
    pass


def _ensure_sam2_on_path():
    """Prepend the SAM2 source dir to sys.path so ``import sam2`` resolves.

    Reads ``SAM2_SOURCE_PATH`` (default: the editable-install source on this
    machine). If the dir does not exist we silently return — sam2 may already
    be importable via the active environment (editable/pip install).
    """
    import sys
    sam2_src = os.environ.get("SAM2_SOURCE_PATH", "/data/wangcheng/myproject/sam2-main")
    if os.path.isdir(sam2_src) and sam2_src not in sys.path:
        sys.path.insert(0, sam2_src)


def _load_sam2_checkpoint(checkpoint_path: str, map_location: str = "cpu") -> Dict:
    """Load SAM2 checkpoint from file."""
    from mmengine.runner.checkpoint import _load_checkpoint
    checkpoint = _load_checkpoint(checkpoint_path, map_location=map_location)
    # SAM2 checkpoint format is {"model": {...}}
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


@MODELS.register_module()
class RSSAM2PositionalEmbedding(BaseModule):
    """SAM2 uses a different positional embedding scheme."""

    def __init__(
        self,
        image_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 256,
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.grid_size, self.grid_size, embed_dim)
        )

        class PositionalEmbeddingWrapper:
            def __init__(self, pos_embed):
                self.positional_embedding = pos_embed

        self.shared_image_embedding = PositionalEmbeddingWrapper(self.pos_embed)

    def get_dense_pe(self) -> Tensor:
        """Returns positional encoding in SAM2 format: [1, C, H, W]"""
        pos = self.pos_embed
        if pos.shape[-1] == 256:
            pos = pos.permute(0, 3, 1, 2)
        return pos

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
    ):
        super().__init__(init_cfg=init_cfg)
        self.checkpoint_path = checkpoint_path
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.freeze_backbone = freeze_backbone
        self.freeze_backbone_stages = freeze_backbone_stages
        self.use_lora = use_lora
        self.lora_config = lora_config or {}
        self.model_size = model_size
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
        if self.use_lora and self.encoder is not None:
            self._apply_lora()
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
        _ensure_sam2_on_path()

        if not SAM2_AVAILABLE:
            raise ImportError("SAM2 is not properly installed.")

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

            msg = image_encoder.load_state_dict(new_state_dict, strict=False)
            if is_main_process():
                print(f"Loaded SAM2 VisionEncoder ({self.model_size}). Missing keys: {len(msg.missing_keys)}")

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
                
        except ImportError:
            if is_main_process():
                print("[RSSAM2VisionEncoder] Warning: peft not installed, skipping LoRA. "
                      "Install with: pip install peft")
        except Exception as e:
            if is_main_process():
                print(f"[RSSAM2VisionEncoder] Warning: Failed to apply LoRA: {e}")
                import traceback
                traceback.print_exc()
                traceback.print_exc()

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
    ):
        super().__init__(init_cfg=init_cfg)
        self.decoder = None
        self.use_high_res_features = use_high_res_features

        if init_cfg is not None and checkpoint_path is None:
            checkpoint_path = init_cfg.get("checkpoint")

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_sam2_decoder(checkpoint_path)

    def _load_sam2_decoder(self, checkpoint_path: str):
        """Load SAM2 decoder from checkpoint."""
        try:
            _ensure_sam2_on_path()

            from sam2.modeling.sam.transformer import TwoWayTransformer

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

            print(f'Decoder state dict keys: {len(decoder_state_dict)}')
            msg = mask_decoder.load_state_dict(decoder_state_dict, strict=False)
            if is_main_process():
                print(f"Loaded SAM2 MaskDecoder. Missing keys: {len(msg.missing_keys)}")
                if msg.missing_keys:
                    print(f"  Missing keys: {msg.missing_keys[:10]}...")
                if self.use_high_res_features:
                    print(f"  use_high_res_features: True")
                    print(f"  Has conv_s0: {hasattr(mask_decoder, 'conv_s0')}")
                    print(f"  Has conv_s1: {hasattr(mask_decoder, 'conv_s1')}")

            self.decoder = mask_decoder
        except Exception as e:
            if is_main_process():
                print(f"Failed to load SAM2 decoder: {e}")
            self.decoder = None

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
            raise RuntimeError(
                "SAM2 MaskDecoder not loaded. Check checkpoint path and loading logic."
            )

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

        try:
            # Process high_res_features: apply conv_s0 and conv_s1 if available
            # SAM2 decoder expects high_res_features to be already processed by conv_s0/conv_s1
            processed_high_res = None
            if high_res_features is not None and self.use_high_res_features:
                if hasattr(self.decoder, 'conv_s0') and hasattr(self.decoder, 'conv_s1'):
                    feat_s0, feat_s1 = high_res_features
                    # Apply conv_s0 and conv_s1 to project features to correct dimensions
                    # conv_s0: 256 -> 32 channels, conv_s1: 256 -> 64 channels
                    processed_high_res = [
                        self.decoder.conv_s0(feat_s0),  # [B, 32, H, W]
                        self.decoder.conv_s1(feat_s1),  # [B, 64, H, W]
                    ]
            
            # SAM2 decoder returns 4 values: (masks, iou_pred, sam_tokens_out, object_score_logits)
            low_res_masks, iou_predictions, _, _ = self.decoder(
                image_embeddings,
                image_positional_embeddings_pe,
                sparse_prompt_embeddings,
                dense_prompt_embeddings,
                multimask_output,
                repeat_image,
                high_res_features=processed_high_res,
            )
            attn_weights = None
        except Exception as e:
            import traceback
            if is_main_process():
                traceback.print_exc()
                print(f"[ERROR] SAM2 decoder forward failed: {e}")
            h = w = image_embeddings.shape[2]
            low_res_masks = torch.zeros(batch_size, 1, h, w, device=image_embeddings.device)
            iou_predictions = torch.zeros(batch_size, 1, device=image_embeddings.device)
            attn_weights = None

        return low_res_masks, iou_predictions, attn_weights


@MODELS.register_module()
class RSFeatureAggregatorSAM2(BaseModule):
    """Feature aggregator adapted for SAM2's output dimensions."""

    in_channels_dict = {
        "tiny": [192, 96, 48, 24],
        "small": [384, 192, 96, 48],
        "base": [768, 384, 192, 96],
        "large": [1024, 512, 256, 128],
        "base_plus": [896, 448, 224, 112],
        "sam2_hiera_base": [256, 256, 256, 256],
        "sam2_hiera_large": [256, 256, 256, 256],  # FPN输出都是256维
    }

    def __init__(
        self,
        in_channels: str = "sam2_hiera_base",
        hidden_channels: int = 64,
        out_channels: int = 256,
        select_layers: Optional[List[int]] = None,
        fpn_type: str = "fused",
        init_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        if "sam2" in in_channels.lower():
            model_size = in_channels
        else:
            model_size = "base_plus" if "base" in in_channels else "large" if "large" in in_channels else "small" if "small" in in_channels else "tiny"
        self.in_channels = self.in_channels_dict.get(model_size, self.in_channels_dict["sam2_hiera_base"])
        self.select_layers = select_layers or [0, 1, 2]
        self.fpn_type = str(fpn_type)
        self._neck_debug_logged_stages = set()

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
        if self.fpn_type == "topdown":
            self.output_convs = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(hidden_channels, out_channels, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    )
                    for _ in self.select_layers
                ]
            )
        else:
            self.output_convs = None

        if self.fpn_type == "lateral_enhanced":
            self.lateral_convs = nn.ModuleList([
                nn.Conv2d(
                    self.in_channels[i] if i < len(self.in_channels) else self.in_channels[-1],
                    out_channels, 1,
                )
                for i in range(4)
            ])
            self.lateral_smooth_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in range(4)
            ])
            for conv in self.lateral_convs:
                nn.init.normal_(conv.weight, std=0.001)
                nn.init.zeros_(conv.bias)
            for conv in self.lateral_smooth_convs:
                nn.init.zeros_(conv.weight)
                with torch.no_grad():
                    for i in range(out_channels):
                        conv.weight[i, i, 1, 1] = 1.0
                nn.init.zeros_(conv.bias)

        if self.fpn_type == "direct":
            self.direct_output_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels[i] if i < len(self.in_channels) else self.in_channels[-1],
                        out_channels, 1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for i in range(4)
            ])

        if self.fpn_type == "direct_fpn":
            self.direct_output_convs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels[i] if i < len(self.in_channels) else self.in_channels[-1],
                        out_channels, 1,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for i in range(4)
            ])
            self.topdown_smooth_convs = nn.ModuleList([
                nn.Conv2d(out_channels, out_channels, 3, padding=1)
                for _ in range(4)
            ])

    def forward(self, inputs) -> Tuple[Tensor, ...]:
        if isinstance(inputs, tuple) and len(inputs) == 2 and isinstance(inputs[1], tuple):
            inputs = inputs[1]
        
        if not inputs:
            raise ValueError("Empty inputs to neck")

        inputs = tuple(x.float() for x in inputs)

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
        self._log_neck_layout(
            stage="pre_forward",
            raw_inputs=inputs,
            features=None,
            outputs=None,
        )
        if self.fpn_type == "topdown" and len(inputs) >= 2:
            return self._forward_topdown(inputs)
        if self.fpn_type == "lateral_enhanced" and len(inputs) >= 2:
            return self._forward_lateral_enhanced(inputs)
        if self.fpn_type == "direct" and len(inputs) >= 2:
            return self._forward_direct(inputs)
        if self.fpn_type == "direct_fpn" and len(inputs) >= 2:
            return self._forward_direct_fpn(inputs)

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
        self._log_neck_layout(
            stage="fused_outputs",
            raw_inputs=inputs,
            features=aligned_features,
            outputs=(feat_64, feat_32, feat_16, feat_8, feat_4),
        )
        # return features in order of increasing stride (4, 8, 16, 32, 64)
        return (feat_64, feat_32, feat_16, feat_8, feat_4)

    def _forward_topdown(self, inputs) -> Tuple[Tensor, ...]:
        laterals = []
        for idx, i_layer in enumerate(self.select_layers):
            if i_layer < len(inputs):
                x = inputs[i_layer]
                if x.dim() == 3:
                    x = x.permute(0, 2, 1)
                laterals.append(self.downconvs[idx](x))

        if not laterals:
            raise ValueError("No features to process in topdown FPN")

        for i in range(len(laterals) - 2, -1, -1):
            upsampled = F.interpolate(
                laterals[i + 1],
                size=laterals[i].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i] = laterals[i] + upsampled

        smoothed = [
            self.hidden_convs[i](lateral) for i, lateral in enumerate(laterals)
        ]
        outputs = [
            self.output_convs[i](smoothed[i]) for i in range(len(smoothed))
        ]
        outputs.append(F.max_pool2d(outputs[-1], kernel_size=2, stride=2))
        self._log_neck_layout(
            stage="topdown_outputs",
            raw_inputs=inputs,
            features=smoothed,
            outputs=tuple(outputs),
        )
        return tuple(outputs)

    def _forward_lateral_enhanced(self, inputs) -> Tuple[Tensor, ...]:
        """Lateral-Enhanced FPN: fused base + SAM2 backbone_fpn lateral injection.

        Keeps the fused representation for downstream compatibility while injecting
        SAM2's original backbone_fpn multi-scale features via lateral convolutions.
        Lateral convs are initialized near zero so training starts from pure fused.
        """
        raw_fpn = inputs

        # Step 1: Compute fused features (same as fused path)
        features = []
        for idx, i_layer in enumerate(self.select_layers):
            if i_layer < len(inputs):
                x = inputs[i_layer]
                if x.dim() == 3:
                    x = x.permute(0, 2, 1)
                features.append(self.downconvs[idx](x))

        if not features:
            raise ValueError("No features to process in lateral_enhanced FPN")

        target_h, target_w = 0, 0
        for feat in features:
            if feat.shape[2] > target_h:
                target_h = feat.shape[2]
                target_w = feat.shape[3]
        target_size = (target_h, target_w)

        aligned_features = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned_features.append(feat)

        x = aligned_features[0]
        for idx, hidden_conv in enumerate(self.hidden_convs[1:], 1):
            if idx < len(aligned_features):
                x = x + aligned_features[idx]
            residual = hidden_conv(x)
            x = x + residual
        fused_feat = self.fusion_conv(x)

        # Step 2: Create fused pseudo-pyramid
        fused_p2 = fused_feat
        fused_p3 = F.max_pool2d(fused_p2, kernel_size=2, stride=2)
        fused_p4 = F.max_pool2d(fused_p3, kernel_size=2, stride=2)
        fused_p5 = F.max_pool2d(fused_p4, kernel_size=2, stride=2)

        # Step 3: Lateral injection from raw SAM2 backbone_fpn
        fpn_levels = list(raw_fpn[:4])
        while len(fpn_levels) < 4:
            fpn_levels.append(fpn_levels[-1])

        laterals = [self.lateral_convs[i](fpn_levels[i].float()) for i in range(4)]

        fused_pyramid = [fused_p2, fused_p3, fused_p4, fused_p5]
        for i in range(4):
            if laterals[i].shape[2:] != fused_pyramid[i].shape[2:]:
                laterals[i] = F.interpolate(
                    laterals[i], size=fused_pyramid[i].shape[2:],
                    mode='bilinear', align_corners=False,
                )

        # Step 4: Fuse + smooth
        P2 = self.lateral_smooth_convs[0](fused_p2 + laterals[0])
        P3 = self.lateral_smooth_convs[1](fused_p3 + laterals[1])
        P4 = self.lateral_smooth_convs[2](fused_p4 + laterals[2])
        P5 = self.lateral_smooth_convs[3](fused_p5 + laterals[3])
        P6 = F.max_pool2d(P5, kernel_size=2, stride=2)

        self._log_neck_layout(
            stage="lateral_enhanced_outputs",
            raw_inputs=inputs,
            features=laterals,
            outputs=(P2, P3, P4, P5, P6),
        )
        return (P2, P3, P4, P5, P6)

    def _forward_direct(self, inputs) -> Tuple[Tensor, ...]:
        """Direct backbone_fpn FPN: use SAM2's FpnNeck output directly.

        Skips the entire fused path (downconvs → upsample+add → fusion_conv → max_pool).
        Each backbone_fpn level is projected independently via output_conv, preserving
        true multi-scale semantics from SAM2's top-down FPN.
        """
        fpn_levels = list(inputs[:4])
        while len(fpn_levels) < 4:
            fpn_levels.append(fpn_levels[-1])

        outputs = []
        for i in range(4):
            x = fpn_levels[i].float()
            if x.dim() == 3:
                x = x.permute(0, 2, 1)
            outputs.append(self.direct_output_convs[i](x))

        P2, P3, P4, P5 = outputs
        P6 = F.max_pool2d(P5, kernel_size=2, stride=2)

        self._log_neck_layout(
            stage="direct_outputs",
            raw_inputs=inputs,
            features=[f.float() for f in fpn_levels],
            outputs=(P2, P3, P4, P5, P6),
        )
        return (P2, P3, P4, P5, P6)

    def _forward_direct_fpn(self, inputs) -> Tuple[Tensor, ...]:
        """Direct backbone_fpn with top-down fusion.

        Combines direct per-level projection with standard FPN top-down path:
        1. Project each backbone_fpn level independently (preserving true multi-scale)
        2. Top-down: coarser level upsampled and added to finer level
        3. Smooth conv per level
        """
        fpn_levels = list(inputs[:4])
        while len(fpn_levels) < 4:
            fpn_levels.append(fpn_levels[-1])

        # Step 1: Per-level projection
        laterals = []
        for i in range(4):
            x = fpn_levels[i].float()
            if x.dim() == 3:
                x = x.permute(0, 2, 1)
            laterals.append(self.direct_output_convs[i](x))

        # Step 2: Top-down fusion (coarsest -> finest)
        # laterals[0]=P2(stride=4), laterals[1]=P3(8), laterals[2]=P4(16), laterals[3]=P5(32)
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:],
                mode='bilinear', align_corners=False,
            )
            laterals[i] = laterals[i] + up

        # Step 3: Smooth
        P2 = self.topdown_smooth_convs[0](laterals[0])
        P3 = self.topdown_smooth_convs[1](laterals[1])
        P4 = self.topdown_smooth_convs[2](laterals[2])
        P5 = self.topdown_smooth_convs[3](laterals[3])
        P6 = F.max_pool2d(P5, kernel_size=2, stride=2)

        self._log_neck_layout(
            stage="direct_fpn_outputs",
            raw_inputs=inputs,
            features=[f.float() for f in fpn_levels],
            outputs=(P2, P3, P4, P5, P6),
        )
        return (P2, P3, P4, P5, P6)

    def _log_neck_layout(
        self,
        stage: str,
        raw_inputs: Optional[List[Tensor]],
        features: Optional[List[Tensor]],
        outputs: Optional[Tuple[Tensor, ...]],
    ) -> None:
        if os.environ.get("DEBUG_NECK_LAYOUT", "0") != "1":
            return
        if stage in self._neck_debug_logged_stages:
            return
        if not is_main_process():
            return

        def _shape_str(tensors):
            if tensors is None:
                return "None"
            parts = []
            for idx, tensor in enumerate(tensors):
                if tensor is None:
                    parts.append(f"{idx}:None")
                    continue
                parts.append(f"{idx}:{tuple(tensor.shape)}")
            return " | ".join(parts)

        def _spatial_str(tensors):
            if tensors is None:
                return "None"
            sizes = []
            for idx, tensor in enumerate(tensors):
                if tensor is None:
                    continue
                sizes.append((idx, tensor.shape[-2], tensor.shape[-1]))
            return str(sizes)

        order_hint = "unknown"
        if raw_inputs:
            spatial_heights = [tensor.shape[-2] for tensor in raw_inputs]
            if spatial_heights == sorted(spatial_heights):
                order_hint = "coarse_to_fine"
            elif spatial_heights == sorted(spatial_heights, reverse=True):
                order_hint = "fine_to_coarse"

        print(
            "[NECK DEBUG] "
            f"fpn_type={self.fpn_type} stage={stage} select_layers={list(self.select_layers)} "
            f"input_order_hint={order_hint}"
        )
        print(f"[NECK DEBUG] raw_inputs={_shape_str(raw_inputs)}")
        print(f"[NECK DEBUG] raw_spatial={_spatial_str(raw_inputs)}")
        if features is not None:
            print(f"[NECK DEBUG] features={_shape_str(features)}")
            print(f"[NECK DEBUG] feature_spatial={_spatial_str(features)}")
        if outputs is not None:
            print(f"[NECK DEBUG] outputs={_shape_str(list(outputs))}")
            print(f"[NECK DEBUG] output_spatial={_spatial_str(list(outputs))}")

        if self.fpn_type == "topdown" and raw_inputs and len(raw_inputs) >= 2:
            for i in range(len(raw_inputs) - 1):
                src = raw_inputs[i + 1].shape[-2:]
                dst = raw_inputs[i].shape[-2:]
                print(
                    "[NECK DEBUG] topdown_step "
                    f"{i + 1}->{i}: resize {src} -> {dst}"
                )

        self._neck_debug_logged_stages.add(stage)


@MODELS.register_module()
class RSPrompterAnchorMaskHeadSAM2(FCNMaskHead, BaseModule):
    """SAM2-compatible Mask Head for RSPrompter."""

    def __init__(
        self,
        sam2_mask_decoder,
        in_channels,
        roi_feat_size=14,
        dense_prompt_mode="none",
        per_pointset_point=5,
        with_sincos=True,
        multimask_output=False,
        attention_similarity=None,
        target_embedding=None,
        output_attentions=None,
        class_agnostic=False,
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

        self.mask_decoder = RSSAM2MaskDecoderWrapper(**sam2_mask_decoder)

        self.no_mask_embed = nn.Parameter(torch.zeros(1, 256, 1, 1))
        self.dense_prompt_mode = str(dense_prompt_mode)

        # SAM2 decoder expects sparse_prompt_embeddings with dim=256 (matching image_embeddings)
        # Output: (batch, per_pointset_point, 256)
        num_sincos = 2 if with_sincos else 1
        embed_dim = 256  # Match SAM2 decoder embedding dimension
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

        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic
        self.mask_logit_scale = 1.0
        self.mask_logit_bias = 0.0
        self._maskhead_input_grad_hook_registered = False
        self._maskhead_sparse_grad_hook_registered = False
        
        # Keep prompt channel width at 256; topology contributes extra tokens, not narrower channels.
        self.topology_token_proj = nn.Linear(256, 256)

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(
        self, 
        x, 
        image_embeddings, 
        image_positional_embeddings, 
        roi_img_ids=None, 
        high_res_features=None,
        sparse_prompt_embeddings=None,
        topology_tokens=None,
    ):
        """
        Forward pass for mask prediction.
        
        Args:
            x: ROI features [N, C, H, W]
            image_embeddings: Image embeddings from SAM2 encoder [B, C, H, W]
            image_positional_embeddings: Positional embeddings [B, C, H, W] or [C, H, W]
            roi_img_ids: ROI to image mapping [N]
            high_res_features: High resolution features from SAM2 encoder (optional)
            sparse_prompt_embeddings: External sparse prompt embeddings [N, num_points, C] (optional)
                                     If provided, will be used instead of internally generated ones.
            topology_tokens: Topology tokens from IIMR [N, num_tokens, C] (optional)
                           Will be concatenated with sparse embeddings.
        
        Returns:
            low_res_masks: Predicted masks [N, num_masks, H, W]
            iou_predictions: IoU predictions [N, num_masks]
        """
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        if (
            self.training
            and (not self._maskhead_input_grad_hook_registered)
            and x.requires_grad
            and is_main_process()
        ):
            def _maskhead_input_grad_hook(grad: Tensor):
                print(
                    "[SAM2 MaskHead Input Grad Hook] "
                    f"x_grad_abs_mean={float(grad.abs().mean().item()):.6e}"
                )

            x.register_hook(_maskhead_input_grad_hook)
            self._maskhead_input_grad_hook_registered = True
            print("[SAM2 MaskHead Hook Register] registered_on_input_x req_grad=True")

        # Always build ROI-conditioned point embeddings from x.
        # When external sparse prompts are provided, fuse them so mask head
        # still depends on current ROI features in iterative steps.
        point_embedings = self.point_emb(x)
        point_embedings = einops.rearrange(point_embedings, "b (n c) -> b n c", n=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        if sparse_prompt_embeddings is not None:
            sparse_embeddings = sparse_prompt_embeddings
            n_ext = sparse_embeddings.shape[1]
            n_gen = point_embedings.shape[1]
            n_fuse = min(n_ext, n_gen)
            if n_fuse > 0:
                sparse_embeddings = sparse_embeddings.clone()
                sparse_embeddings[:, :n_fuse, :] = sparse_embeddings[:, :n_fuse, :] + point_embedings[:, :n_fuse, :]
        else:
            sparse_embeddings = point_embedings
        
        # 如果有拓扑 token，拼接到 sparse embeddings
        # 使用投影层确保最终维度为 256（SAM2 decoder 期望的维度）
        if topology_tokens is not None:
            topology_tokens_proj = self.topology_token_proj(topology_tokens)
            sparse_embeddings = torch.cat([sparse_embeddings, topology_tokens_proj], dim=1)

        if (
            self.training
            and (not self._maskhead_sparse_grad_hook_registered)
            and sparse_embeddings.requires_grad
            and is_main_process()
        ):
            def _maskhead_sparse_grad_hook(grad: Tensor):
                print(
                    "[SAM2 MaskHead Sparse Grad Hook] "
                    f"sparse_grad_abs_mean={float(grad.abs().mean().item()):.6e}"
                )

            sparse_embeddings.register_hook(_maskhead_sparse_grad_hook)
            self._maskhead_sparse_grad_hook_registered = True
            print("[SAM2 MaskHead Hook Register] registered_on_sparse_embeddings req_grad=True")

        num_roi_per_image = torch.bincount(roi_img_ids.long())
        num_roi_per_image = torch.cat(
            [
                num_roi_per_image,
                torch.zeros(
                    img_bs - len(num_roi_per_image),
                    device=num_roi_per_image.device,
                    dtype=num_roi_per_image.dtype,
                ),
            ]
        )

        base_dense_embeddings = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
            roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
        )
        if self.dense_prompt_mode == "residual" and self.roi_dense_proj is not None:
            roi_dense_embeddings = self.roi_dense_proj(x)
            roi_dense_embeddings = F.interpolate(
                roi_dense_embeddings,
                size=image_embedding_size,
                mode="bilinear",
                align_corners=False,
            )
            dense_embeddings = (
                base_dense_embeddings
                + self.roi_dense_scale * roi_dense_embeddings
            )
        else:
            dense_embeddings = base_dense_embeddings
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        
        if image_positional_embeddings.dim() == 3:
            image_positional_embeddings = image_positional_embeddings.unsqueeze(0)
        if image_positional_embeddings.dim() == 4:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        else:
            image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        
        if high_res_features is not None:
            high_res_features_expanded = [
                feat.repeat_interleave(num_roi_per_image, dim=0)
                for feat in high_res_features
            ]
        else:
            high_res_features_expanded = None
        
        low_res_masks, iou_predictions, _ = self.mask_decoder(
            image_embeddings,
            image_positional_embeddings,
            sparse_embeddings,
            dense_embeddings,
            self.multimask_output,
            False,
            high_res_features=high_res_features_expanded,
        )
        low_res_masks = low_res_masks * self.mask_logit_scale + self.mask_logit_bias
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        return low_res_masks, iou_predictions

    def get_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        return mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
        mask_targets: Optional[Tensor] = None,
        pos_labels: Optional[Tensor] = None,
    ) -> dict:
        if mask_targets is None:
            mask_targets = self.get_targets(
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
                rcnn_train_cfg=rcnn_train_cfg,
            )
        if pos_labels is None:
            pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        mask_preds = F.interpolate(mask_preds, size=mask_targets.shape[-2:], mode="bilinear", align_corners=False)

        loss = dict()
        debug_stats = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
            mask_probs = mask_preds.sigmoid().detach()
            debug_stats = dict(
                debug_mask_pos_rois=mask_preds.new_tensor(float(mask_preds.size(0))),
                debug_mask_logit_mean=mask_preds.detach().mean(),
                debug_mask_logit_std=mask_preds.detach().std(unbiased=False),
                debug_mask_prob_mean=mask_probs.mean(),
                debug_mask_prob_ge05=(mask_probs >= 0.5).float().mean(),
                debug_mask_target_fill=mask_targets.detach().float().mean(),
            )
            if (
                is_main_process()
                and os.environ.get("DEBUG_MASK_TRAIN", "0") == "1"
                and not hasattr(self, "_printed_mask_debug")
            ):
                self._printed_mask_debug = True
                print(
                    "[MASK_DEBUG] "
                    f"pos_rois={debug_stats['debug_mask_pos_rois'].item():.0f} "
                    f"logit_mean={debug_stats['debug_mask_logit_mean'].item():.4f} "
                    f"logit_std={debug_stats['debug_mask_logit_std'].item():.4f} "
                    f"prob_mean={debug_stats['debug_mask_prob_mean'].item():.4f} "
                    f"prob_ge05={debug_stats['debug_mask_prob_ge05'].item():.4f} "
                    f"target_fill={debug_stats['debug_mask_target_fill'].item():.4f}"
                )
        loss["loss_mask"] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets, debug_stats=debug_stats)

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
        # Paste ROI masks back to the image canvas according to each bbox.
        # Directly resizing every ROI mask to the full image ignores bbox
        # geometry and severely hurts instance mask quality.
        return super()._predict_by_feat_single(
            mask_preds=mask_preds,
            bboxes=bboxes,
            labels=labels,
            img_meta=img_meta,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale,
            activate_map=activate_map,
        )
