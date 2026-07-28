import copy
import os
from typing import Dict, List, Optional, Tuple

import einops
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmengine import ConfigDict
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS
from torch import Tensor, nn
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamMaskDecoder,
    SamPositionalEmbedding,
    SamPromptEncoder,
    SamVisionEncoder,
    SamVisionEncoderOutput,
)

from mmdet.models import MaskRCNN, StandardRoIHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask.mask_target import mask_target
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptConfigType


@MODELS.register_module(force=True)
class LN2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


MMENGINE_MODELS.register_module(module=LN2d, name="LN2d", force=True)


@MODELS.register_module()
class RSSamPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    def __init__(self, hf_pretrain_name, extra_config=None, init_cfg=None, use_offline_mode=False):
        BaseModule.__init__(self, init_cfg=init_cfg)
        
        # Support offline mode by using local config instead of downloading
        if use_offline_mode:
            # Create a basic SAM config for offline mode
            from transformers import SamConfig
            sam_config = SamConfig()
            sam_config = sam_config.vision_config
        else:
            sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)
        
        if init_cfg is not None:
            checkpoint_path = init_cfg.get("checkpoint")
            from mmengine.runner.checkpoint import _load_checkpoint

            checkpoint = _load_checkpoint(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            # Revise keys
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith("module."):
                    new_k = new_k[7:]
                if new_k.startswith("shared_image_embedding."):
                    new_k = new_k[23:]
                new_state_dict[new_k] = v
            
            # Load
            msg = self.shared_image_embedding.load_state_dict(new_state_dict, strict=False)
            if is_main_process():
                print(f"Loaded SAM PositionalEmbedding. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


def interpolate_sam_pos_embed(state_dict, new_image_size, patch_size=16):
    """
    Interpolate SAM vision encoder positional embeddings for different image sizes.
    """
    if "pos_embed" in state_dict:
        pos_embed = state_dict["pos_embed"]  # [1, h, w, c]
        _, old_h, old_w, c = pos_embed.shape
        new_h, new_w = new_image_size // patch_size, new_image_size // patch_size
        if old_h != new_h or old_w != new_w:
            # [1, h, w, c] -> [1, c, h, w]
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            # [1, c, h, w] -> [1, h, w, c]
            state_dict["pos_embed"] = pos_embed.permute(0, 2, 3, 1)

    # Interpolate relative positional embeddings
    for k in list(state_dict.keys()):
        if "rel_pos_h" in k or "rel_pos_w" in k:
            rel_pos = state_dict[k]  # [L, c]
            old_L, c = rel_pos.shape
            # SAM uses 2*window_size - 1 or 2*grid_size - 1
            # If it's a global attention, L will be 2*old_grid_size - 1
            # If it's window attention, L will be 2*window_size - 1
            if old_L > 30:  # Heuristic for global attention (usually 127 for 64x64)
                old_grid_size = (old_L + 1) // 2
                new_grid_size = new_image_size // patch_size
                new_L = 2 * new_grid_size - 1
                if old_L != new_L:
                    # [L, c] -> [1, c, L]
                    rel_pos = rel_pos.reshape(1, old_L, c).permute(0, 2, 1)
                    rel_pos = F.interpolate(
                        rel_pos, size=new_L, mode="linear", align_corners=False
                    )
                    # [1, c, L] -> [L, c]
                    state_dict[k] = rel_pos.permute(0, 2, 1).reshape(new_L, c)
    return state_dict


@MODELS.register_module()
class RSSamVisionEncoder(BaseModule):
    def __init__(
        self,
        hf_pretrain_name,
        extra_config=None,
        peft_config=None,
        init_cfg=None,
        use_offline_mode=False,
        use_gradient_checkpointing=False,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        
        if use_offline_mode:
            from transformers import SamConfig
            sam_config = SamConfig()
            sam_config = sam_config.vision_config
        else:
            sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        
        self.image_size = sam_config.image_size
        self.use_gradient_checkpointing = use_gradient_checkpointing
        vision_encoder = SamVisionEncoder(sam_config)
        
        if init_cfg is not None:
            checkpoint_path = init_cfg.get("checkpoint")
            from mmengine.runner.checkpoint import _load_checkpoint

            checkpoint = _load_checkpoint(checkpoint_path, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith("module."):
                    new_k = new_k[7:]
                if new_k.startswith("vision_encoder."):
                    new_k = new_k[15:]
                new_state_dict[new_k] = v
            
            new_state_dict = interpolate_sam_pos_embed(new_state_dict, self.image_size)
            
            msg = vision_encoder.load_state_dict(new_state_dict, strict=False)
            if is_main_process():
                print(f"Loaded SAM VisionEncoder with interpolation. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

        if peft_config is not None:
            from peft import get_peft_config, get_peft_model

            if isinstance(peft_config, dict):
                config = {
                    "peft_type": "LORA",
                    "r": 16,
                    "target_modules": ["qkv"],
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "bias": "none",
                    "inference_mode": False,
                }
                config.update(peft_config)
                peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True
        
        if self.use_gradient_checkpointing and is_main_process():
            print("Enabled gradient checkpointing for SAM VisionEncoder (via forward wrapper)")

    def init_weights(self):
        if is_main_process():
            print("the vision encoder has been initialized")

    def forward(self, *args, **kwargs):
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.vision_encoder, *args, use_reentrant=False, **kwargs
            )
        return self.vision_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamPromptEncoder(SamPromptEncoder, BaseModule):
    def __init__(self, hf_pretrain_name, extra_config=None, init_cfg=None, use_offline_mode=False):
        BaseModule.__init__(self, init_cfg=init_cfg)
        
        # Support offline mode by using local config instead of downloading
        if use_offline_mode:
            # Create a basic SAM config for offline mode
            from transformers import SamConfig
            sam_config = SamConfig()
            sam_config = sam_config.prompt_encoder_config
        else:
            sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class RSSamMaskDecoder(SamMaskDecoder, BaseModule):
    def __init__(self, hf_pretrain_name, extra_config=None, init_cfg=None, use_offline_mode=False):
        BaseModule.__init__(self, init_cfg=init_cfg)
        
        # Support offline mode by using local config instead of downloading
        if use_offline_mode:
            # Create a basic SAM config for offline mode
            from transformers import SamConfig
            sam_config = SamConfig()
            sam_config = sam_config.mask_decoder_config
        else:
            sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)


@MODELS.register_module()
class RSFPN(BaseModule):
    def __init__(self, feature_aggregator=None, feature_spliter=None, init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        if hasattr(self, "feature_aggregator"):
            x = self.feature_aggregator(inputs)
        else:
            x = inputs
        if hasattr(self, "feature_spliter"):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x


@MODELS.register_module()
class RSFeatureAggregator(BaseModule):
    in_channels_dict = {
        "base": [768] * (12 + 1),
        "large": [1024] * (24 + 1),
        "huge": [1280] * (32 + 1),
    }

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=256,
        select_layers=range(1, 12, 2),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, str)
        model_arch = "base" if "base" in in_channels else "large" if "large" in in_channels else "huge"
        self.in_channels = self.in_channels_dict[model_arch]
        self.select_layers = select_layers

        self.downconvs = nn.ModuleList()
        for i_layer in self.select_layers:
            self.downconvs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels[i_layer], hidden_channels, 1),
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

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = [einops.rearrange(x, "b h w c -> b c h w") for x in inputs]

        features = []
        for idx, i_layer in enumerate(self.select_layers):
            features.append(self.downconvs[idx](inputs[i_layer]))

        x = None
        for hidden_state, hidden_conv in zip(features, self.hidden_convs):
            if x is not None:
                hidden_state = x + hidden_state
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
        x = self.fusion_conv(x)
        return x


@MODELS.register_module()
class RSSimpleFPN(BaseModule):
    def __init__(
        self,
        backbone_channel: int,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        init_cfg: MultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(self.backbone_channel // 2, self.backbone_channel // 4, 2, 2),
        )
        self.fpn2 = nn.Sequential(nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2))
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False,
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, input: Tensor) -> tuple:
        inputs = [self.fpn1(input), self.fpn2(input), self.fpn3(input), self.fpn4(input)]

        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        if self.num_outs > len(outs):
            for _ in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


@MODELS.register_module()
class RSPrompterAnchor(MaskRCNN):
    def __init__(self, shared_image_embedding, decoder_freeze=True, iimr=None, *args, **kwargs):
        peft_config = kwargs.get("backbone", {}).get("peft_config", {})
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.decoder_freeze = decoder_freeze
        
        self.iimr_enabled = False
        if iimr is not None:
            self.iimr_enabled = iimr.get("enabled", False)
            self.num_iterations = iimr.get("num_iterations", 3)
            self.use_topology_memory = iimr.get("use_topology_memory", True)
            self.use_dynamic_prompting = iimr.get("use_dynamic_prompting", True)
            if is_main_process():
                print(f"[RSPrompterAnchor] IIMR enabled: {self.iimr_enabled}")
                print(f"  - num_iterations: {self.num_iterations}")
                print(f"  - use_topology_memory: {self.use_topology_memory}")
                print(f"  - use_dynamic_prompting: {self.use_dynamic_prompting}")

        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.roi_head.mask_head.mask_decoder,
                self.roi_head.mask_head.no_mask_embed,
            ]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_image_wide_positional_embeddings(self, size):
        pos_embed = self.shared_image_embedding.get_dense_pe()
        
        if pos_embed is None:
            pos_embed = self.shared_image_embedding.shared_image_embedding.positional_embedding
        
        if pos_embed.dim() == 4:
            if pos_embed.shape[2] == size and pos_embed.shape[3] == size:
                return pos_embed
            pos_embed = torch.nn.functional.interpolate(
                pos_embed,
                size=(size, size),
                mode='bilinear',
                align_corners=False
            )
            return pos_embed
        
        return pos_embed

    def expand_features_for_iterations(
        self,
        features: Tuple[Tensor, ...],
        num_iterations: int
    ) -> Tuple[Tuple[Tensor, ...], int]:
        """
        Expand features for pseudo-temporal iterations.
        
        This method creates a pseudo-temporal stream by expanding single-frame features
        along a new time dimension, without actually copying the data (using expand).
        
        Args:
            features: Tuple of feature tensors at different scales
                     Each tensor has shape [B, C, H, W]
            num_iterations: Number of iterations T
            
        Returns:
            expanded_features: Tuple of expanded feature tensors
                              Each tensor has shape [B, T, C, H, W]
            T: Number of iterations (for convenience)
        """
        expanded_features = []
        for feat in features:
            if feat.dim() == 4:
                expanded_feat = feat.unsqueeze(1).expand(-1, num_iterations, -1, -1, -1)
                expanded_features.append(expanded_feat)
            else:
                expanded_features.append(feat)
        
        return tuple(expanded_features), num_iterations

    def expand_image_embeddings_for_iterations(
        self,
        image_embeddings: Tensor,
        num_iterations: int
    ) -> Tensor:
        """
        Expand image embeddings for pseudo-temporal iterations.
        
        Args:
            image_embeddings: Tensor of shape [B, C, H, W]
            num_iterations: Number of iterations T
            
        Returns:
            Expanded tensor of shape [B, T, C, H, W]
        """
        if image_embeddings.dim() == 4:
            return image_embeddings.unsqueeze(1).expand(-1, num_iterations, -1, -1, -1)
        return image_embeddings

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):
            image_embeddings = vision_outputs[0]
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError

        image_positional_embeddings = self.get_image_wide_positional_embeddings(size=image_embeddings.shape[-1])
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(batch_size, 1, 1, 1)
        x = self.neck(vision_hidden_states)
        
        high_res_features = None
        if isinstance(vision_outputs, tuple) and len(vision_outputs) > 1:
            if isinstance(vision_outputs[1], (list, tuple)) and len(vision_outputs[1]) >= 3:
                backbone_fpn = vision_outputs[1]
                target_h, target_w = image_embeddings.shape[-2:]
                feat_s0_h, feat_s0_w = target_h * 4, target_w * 4
                feat_s1_h, feat_s1_w = target_h * 2, target_w * 2
                feat_s0 = None
                feat_s1 = None
                
                for feat in backbone_fpn:
                    if feat.shape[-2:] == (feat_s0_h, feat_s0_w):
                        feat_s0 = feat
                    elif feat.shape[-2:] == (feat_s1_h, feat_s1_w):
                        feat_s1 = feat
                
                if feat_s0 is not None and feat_s1 is not None:
                    high_res_features = [feat_s0, feat_s1]
        
        return x, image_embeddings, image_positional_embeddings, high_res_features

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        x, image_embeddings, image_positional_embeddings, high_res_features = self.extract_feat(batch_inputs)
        losses = dict()

        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(x, rpn_data_samples, proposal_cfg=proposal_cfg)
        keys = rpn_losses.keys()
        for key in list(keys):
            if "loss" in key and "rpn" not in key:
                rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(
            x,
            rpn_results_list,
            batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            high_res_features=high_res_features,
        )
        losses.update(roi_losses)
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        x, image_embeddings, image_positional_embeddings, high_res_features = self.extract_feat(batch_inputs)

        if batch_data_samples[0].get("proposals", None) is None:
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

        results_list = self.roi_head.predict(
            x,
            rpn_results_list,
            batch_data_samples,
            rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            high_res_features=high_res_features,
        )
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples


@MODELS.register_module()
class RSPrompterAnchorRoIPromptHead(StandardRoIHead):
    def __init__(
        self,
        with_extra_pe=False,
        iimr=None,
        topdown_downstream_head="none",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.topdown_downstream_head = str(topdown_downstream_head)
        self._topdown_head_log_once = False
        if with_extra_pe:
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(num_feats=out_channels // 2, normalize=True)
            from mmdet.models import SinePositionalEncoding

            self.extra_pe = SinePositionalEncoding(**positional_encoding)

        if self.topdown_downstream_head == "dual_level_residual":
            bbox_levels = len(getattr(self.bbox_roi_extractor, "featmap_strides", []))
            bbox_channels = int(getattr(self.bbox_roi_extractor, "out_channels", 256))
            self.topdown_bbox_level_adapters = self._build_topdown_level_adapters(
                num_levels=bbox_levels,
                channels=bbox_channels,
            )
            self.topdown_bbox_level_scales = nn.Parameter(torch.zeros(bbox_levels))

            mask_levels = len(getattr(self.mask_roi_extractor, "featmap_strides", []))
            mask_channels = int(getattr(self.mask_roi_extractor, "out_channels", bbox_channels))
            self.topdown_mask_level_adapters = self._build_topdown_level_adapters(
                num_levels=mask_levels,
                channels=mask_channels,
            )
            self.topdown_mask_level_scales = nn.Parameter(torch.zeros(mask_levels))
        else:
            self.topdown_bbox_level_adapters = None
            self.register_parameter("topdown_bbox_level_scales", None)
            self.topdown_mask_level_adapters = None
            self.register_parameter("topdown_mask_level_scales", None)
        
        # IIMR (Iterative Instance Memory Reasoning) 配置
        self.iimr_enabled = False
        if iimr is not None and iimr.get("enabled", False):
            self.iimr_enabled = True
            self.num_iterations = iimr.get("num_iterations", 3)
            self.use_topology_memory = iimr.get("use_topology_memory", True)
            self.topology_to_decoder = iimr.get("topology_to_decoder", False)
            self.use_gt_topology_tokens = iimr.get("use_gt_topology_tokens", False)
            self.gt_topology_only_first_iter = iimr.get("gt_topology_only_first_iter", True)
            self.use_dynamic_prompting = iimr.get("use_dynamic_prompting", True)
            self.num_uncertainty_points = iimr.get("num_uncertainty_points", 8)
            self.dynamic_balance_ratio = iimr.get("dynamic_balance_ratio", 0.5)
            self.dynamic_min_point_distance = iimr.get("dynamic_min_point_distance", 0)
            self.dynamic_start_iteration = iimr.get("dynamic_start_iteration", 1)
            self.dynamic_point_mix = iimr.get("dynamic_point_mix", 1.0)
            self.dynamic_fusion_mode = iimr.get("dynamic_fusion_mode", "add")
            self.detach_mask_path = iimr.get("detach_mask_path", False)
            self.iimr_intermediate_loss_weight = float(
                iimr.get("intermediate_loss_weight", 0.0)
            )
            self.use_learnable_residual = bool(
                iimr.get("use_learnable_residual", False)
            )
            # 新增：Memory 残差融合系数
            self.memory_residual_weight = iimr.get("memory_residual_weight", 0.0)
            # ROI Spatial Position Encoding 开关
            self.use_roi_pos_encoding = iimr.get("use_roi_pos_encoding", True)
            # Geometry-Aware KV injection 开关
            self.use_geometry_aware_kv = iimr.get("use_geometry_aware_kv", False)
            # 边界感知不确定性采样
            self.use_boundary_uncertainty = iimr.get("use_boundary_uncertainty", False)
            self.boundary_kernel_size = iimr.get("boundary_kernel_size", 3)
            self.boundary_mix_weight = iimr.get("boundary_mix_weight", 0.1)
            # 自适应正负比
            self.adaptive_balance = iimr.get("adaptive_balance", False)

            # 实例化 IterativeMemoryReasoner
            from rsprompter.iterative_memory import IterativeMemoryReasoner
            self.iterative_reasoner = IterativeMemoryReasoner(
                embed_dim=256,
                num_iterations=self.num_iterations,
                use_topology_memory=self.use_topology_memory,
                topology_to_decoder=self.topology_to_decoder,
                use_dynamic_prompting=self.use_dynamic_prompting,
                num_uncertainty_points=self.num_uncertainty_points,
                dynamic_balance_ratio=self.dynamic_balance_ratio,
                dynamic_min_point_distance=self.dynamic_min_point_distance,
                dynamic_start_iteration=self.dynamic_start_iteration,
                use_roi_pos_encoding=self.use_roi_pos_encoding,
                use_geometry_aware_kv=self.use_geometry_aware_kv,
                in_channels=self.mask_head.in_channels,
                use_boundary_uncertainty=self.use_boundary_uncertainty,
                boundary_kernel_size=self.boundary_kernel_size,
                boundary_mix_weight=self.boundary_mix_weight,
                adaptive_balance=self.adaptive_balance,
                use_learnable_residual=self.use_learnable_residual,
                residual_weight_init=max(float(self.memory_residual_weight), 0.02),
            )

            if is_main_process():
                print(f"[RSPrompterAnchorRoIPromptHead] IIMR enabled: {self.iimr_enabled}")
                print(f"  - num_iterations: {self.num_iterations}")
                print(f"  - use_topology_memory: {self.use_topology_memory}")
                print(f"  - topology_to_decoder: {self.topology_to_decoder}")
                print(f"  - use_gt_topology_tokens: {self.use_gt_topology_tokens}")
                print(f"  - gt_topology_only_first_iter: {self.gt_topology_only_first_iter}")
                print(f"  - use_dynamic_prompting: {self.use_dynamic_prompting}")
                print(f"  - num_uncertainty_points: {self.num_uncertainty_points}")
                print(f"  - dynamic_balance_ratio: {self.dynamic_balance_ratio}")
                print(f"  - dynamic_min_point_distance: {self.dynamic_min_point_distance}")
                print(f"  - dynamic_start_iteration: {self.dynamic_start_iteration}")
                print(f"  - dynamic_point_mix: {self.dynamic_point_mix}")
                print(f"  - dynamic_fusion_mode: {self.dynamic_fusion_mode}")
                print(f"  - detach_mask_path: {self.detach_mask_path}")
                print(
                    f"  - intermediate_loss_weight: {self.iimr_intermediate_loss_weight}"
                )
                print(f"  - use_learnable_residual: {self.use_learnable_residual}")
                print(f"  - use_geometry_aware_kv: {self.use_geometry_aware_kv}")
                print(f"  - use_boundary_uncertainty: {self.use_boundary_uncertainty}")
                print(f"  - boundary_kernel_size: {self.boundary_kernel_size}")
                print(f"  - boundary_mix_weight: {self.boundary_mix_weight}")
                print(f"  - adaptive_balance: {self.adaptive_balance}")
            
            self.sparse_proj = nn.Linear(512, 256)
            self.dynamic_gate_proj = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Sigmoid(),
            )
            self._gate_log_counter = 0

    def _build_topdown_level_adapters(self, num_levels: int, channels: int) -> nn.ModuleList:
        adapters = nn.ModuleList()
        for _ in range(num_levels):
            adapters.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, channels, 1),
                    nn.BatchNorm2d(channels),
                )
            )
        return adapters

    def _apply_topdown_level_adapters(
        self,
        x: Tuple[Tensor],
        adapters: Optional[nn.ModuleList],
        scales: Optional[nn.Parameter],
        branch_name: str,
    ) -> Tuple[Tensor]:
        if adapters is None or scales is None:
            return x

        adapted_feats = []
        for level_idx, feat in enumerate(x):
            if level_idx >= len(adapters):
                adapted_feats.append(feat)
                continue
            delta = adapters[level_idx](feat)
            scale = torch.tanh(scales[level_idx]).to(device=feat.device, dtype=feat.dtype)
            adapted_feats.append(feat + scale.view(1, 1, 1, 1) * delta)

        if (
            os.environ.get("DEBUG_TOPDOWN_HEAD", "0") == "1"
            and not self._topdown_head_log_once
            and is_main_process()
        ):
            scale_values = torch.tanh(scales.detach()).cpu().tolist()
            print(
                "[TOPDOWN_HEAD DEBUG] "
                f"mode={self.topdown_downstream_head} branch={branch_name} "
                f"levels={len(adapters)} scales={[round(v, 6) for v in scale_values]}"
            )
        return tuple(adapted_feats)

    def _get_downstream_branch_features(self, x: Tuple[Tensor]) -> Tuple[Tuple[Tensor], Tuple[Tensor]]:
        if self.topdown_downstream_head != "dual_level_residual":
            return x, x

        bbox_x = self._apply_topdown_level_adapters(
            x,
            adapters=self.topdown_bbox_level_adapters,
            scales=self.topdown_bbox_level_scales,
            branch_name="bbox",
        )
        mask_x = self._apply_topdown_level_adapters(
            x,
            adapters=self.topdown_mask_level_adapters,
            scales=self.topdown_mask_level_scales,
            branch_name="mask",
        )
        self._topdown_head_log_once = True
        return bbox_x, mask_x

    def _mask_forward(
        self,
        x: Tuple[Tensor],
        rois: Tensor = None,
        pos_inds: Optional[Tensor] = None,
        bbox_feats: Optional[Tensor] = None,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
        sparse_prompt_embeddings=None,
        sampling_results: Optional[List[SamplingResult]] = None,
        batch_gt_instances: Optional[InstanceList] = None,
    ) -> dict:
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        mask_x = x
        mask_image_embeddings = image_embeddings
        mask_image_positional_embeddings = image_positional_embeddings
        mask_high_res_features = high_res_features
        mask_sparse_prompt_embeddings = sparse_prompt_embeddings

        # Optional isolation switch for debugging whether IIMR mask supervision
        # is destabilizing detection through shared backbone features.
        if self.iimr_enabled and getattr(self, "detach_mask_path", False):
            mask_x = tuple(feat.detach() for feat in x)
            if image_embeddings is not None:
                mask_image_embeddings = image_embeddings.detach()
            if image_positional_embeddings is not None:
                mask_image_positional_embeddings = image_positional_embeddings.detach()
            if high_res_features is not None:
                mask_high_res_features = [feat.detach() for feat in high_res_features]
            if sparse_prompt_embeddings is not None:
                mask_sparse_prompt_embeddings = sparse_prompt_embeddings.detach()

        if rois is not None:
            mask_feats = self.mask_roi_extractor(mask_x[: self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        # IIMR 迭代推理
        if self.iimr_enabled:
            all_mask_preds = []
            all_iou_predictions = []

            current_sparse_embeddings = mask_sparse_prompt_embeddings
            current_topology_tokens = None
            current_mask_feats = mask_feats
            gt_topology_masks = None

            for t in range(self.num_iterations):
                if t > 0:
                    original_feats = current_mask_feats
                    memory_enhanced = self.iterative_reasoner.forward_iteration(
                        current_mask_feats,
                        iteration_idx=t,
                        positional_embeddings=mask_image_positional_embeddings,
                    )
                    # Memory 残差融合：保留原始特征，用 memory 做增量
                    if self.use_learnable_residual:
                        learned_rw = torch.sigmoid(
                            self.iterative_reasoner.residual_weight
                        ).to(
                            device=original_feats.device,
                            dtype=original_feats.dtype,
                        )
                        schedule_weight = max(float(self.memory_residual_weight), 0.0)
                        if schedule_weight > 0:
                            warmup_scale = schedule_weight / max(
                                float(self.iterative_reasoner.residual_weight_init),
                                1e-6,
                            )
                            rw = learned_rw * warmup_scale
                            current_mask_feats = original_feats + rw * (
                                memory_enhanced - original_feats
                            )
                        else:
                            current_mask_feats = memory_enhanced
                    elif self.memory_residual_weight > 0:
                        current_mask_feats = original_feats + self.memory_residual_weight * (memory_enhanced - original_feats)
                    else:
                        current_mask_feats = memory_enhanced

                # 调用 mask head 进行预测
                mask_preds, iou_predictions = self.mask_head(
                    current_mask_feats,
                    image_embeddings=mask_image_embeddings,
                    image_positional_embeddings=mask_image_positional_embeddings,
                    roi_img_ids=rois[:, 0] if rois is not None else None,
                    high_res_features=mask_high_res_features,
                    sparse_prompt_embeddings=current_sparse_embeddings,
                    topology_tokens=current_topology_tokens,
                )
                
                all_mask_preds.append(mask_preds)
                all_iou_predictions.append(iou_predictions)
                
                # 第一次迭代时，动态设置 memory bank 尺寸（使用 mask_feats 的尺寸）
                if t == 0:
                    roi_bs = mask_feats.shape[0]
                    device = mask_feats.device
                    spatial_size = mask_feats.shape[-2:]
                    in_channels = mask_feats.shape[1]
                    self.iterative_reasoner.reset_memory(
                        batch_size=roi_bs,
                        spatial_size=spatial_size,
                        device=device,
                        in_channels=in_channels
                    )
                
                # 提取拓扑 token（如果启用）
                topology_tokens = None
                if self.use_topology_memory:
                    use_gt_topology = (
                        self.training
                        and self.use_gt_topology_tokens
                        and sampling_results is not None
                        and batch_gt_instances is not None
                        and (not self.gt_topology_only_first_iter or t == 0)
                    )
                    if use_gt_topology:
                        if gt_topology_masks is None or gt_topology_masks.shape[-2:] != mask_preds.shape[-2:]:
                            gt_topology_masks = self._get_gt_masks_for_progressive_loss(
                                sampling_results, batch_gt_instances, mask_preds
                            )
                        with torch.no_grad():
                            topology_tokens = self.iterative_reasoner.extract_topology_tokens(
                                gt_topology_masks.squeeze(1) if gt_topology_masks.dim() == 4 else gt_topology_masks
                            )
                    else:
                        mask_probs = torch.sigmoid(mask_preds)
                        topology_tokens = self.iterative_reasoner.extract_topology_tokens(
                            mask_probs.squeeze(1) if mask_preds.dim() == 4 else mask_probs
                        )
                    if self.topology_to_decoder:
                        current_topology_tokens = topology_tokens
                    else:
                        current_topology_tokens = None
                
                # 更新 memory bank（存储特征而不是 mask 预测）
                self.iterative_reasoner.update_memory(
                    t,
                    current_mask_feats,
                    mask_logits=mask_preds,
                    topology_features=topology_tokens,
                )
                
                # 生成下一轮的 prompt 和融合特征（如果不是最后一次迭代）
                if t < self.num_iterations - 1:
                    # 使用不确定性点生成器生成新的 prompt。
                    # dynamic_start_iteration refers to the target iteration index
                    # that will first consume uncertainty-driven prompts.
                    if self.use_dynamic_prompting and (t + 1) >= self.dynamic_start_iteration:
                        current_sparse_embeddings = self._generate_next_prompt(
                            mask_preds, mask_feats, rois
                        )
            
            # 返回最终迭代的预测，同时保存所有迭代的预测用于损失计算
            return dict(
                mask_preds=all_mask_preds[-1],
                mask_feats=mask_feats,
                iou_predictions=all_iou_predictions[-1],
                all_mask_preds=all_mask_preds,
                all_iou_predictions=all_iou_predictions,
            )
        else:
            # 原始单次推理
            mask_preds, iou_predictions = self.mask_head(
                mask_feats,
                image_embeddings=image_embeddings,
                image_positional_embeddings=mask_image_positional_embeddings,
                roi_img_ids=rois[:, 0] if rois is not None else None,
                high_res_features=mask_high_res_features,
            )
            return dict(mask_preds=mask_preds, mask_feats=mask_feats, iou_predictions=iou_predictions)
    
    def _generate_next_prompt(
        self,
        mask_preds: Tensor,
        mask_feats: Tensor,
        rois: Tensor,
    ) -> Tensor:
        """
        根据上一轮 mask 预测生成下一轮的 prompt embeddings。
        
        Args:
            mask_preds: 上一轮的 mask 预测 [N, H, W]
            mask_feats: ROI 特征 [N, C, H, W]
            rois: ROI 坐标 [N, 5]
        
        Returns:
            sparse_prompt_embeddings: [N, num_points, 256]
        """
        # 使用不确定性点生成器
        points, labels = self.iterative_reasoner.generate_dynamic_prompts(mask_preds)
        
        # 使用现有的 ROI prompt 分支作为基础，再叠加点位编码。
        new_prompt_embeddings = self.mask_head.point_emb(mask_feats)
        new_prompt_embeddings = einops.rearrange(
            new_prompt_embeddings, "b (n c) -> b n c", 
            n=self.mask_head.per_pointset_point
        )
        
        if getattr(self.mask_head, "with_sincos", False):
            new_prompt_embeddings = (
                torch.sin(new_prompt_embeddings[..., ::2])
                + new_prompt_embeddings[..., 1::2]
            )

        # Reshape for sparse_proj: [B, n, c] -> [B*n, c]
        B, n, c = new_prompt_embeddings.shape
        new_prompt_embeddings = new_prompt_embeddings.reshape(B * n, c)

        # Project to 256 channels when the generated prompt width is still larger.
        if c != 256:
            new_prompt_embeddings = self.sparse_proj(new_prompt_embeddings)

        point_embeddings = self._encode_prompt_points(
            points=points,
            labels=labels,
            spatial_shape=mask_preds.shape[-2:],
            target_num_points=n,
            device=new_prompt_embeddings.device,
            dtype=new_prompt_embeddings.dtype,
        )
        if point_embeddings.numel() > 0:
            point_embeddings = point_embeddings.reshape(B * n, -1)
            scaled_point_embeddings = self.dynamic_point_mix * point_embeddings
            if self.dynamic_fusion_mode == "gated":
                gate = self.dynamic_gate_proj(
                    torch.cat([new_prompt_embeddings, point_embeddings], dim=-1)
                )
                new_prompt_embeddings = (1 - gate) * new_prompt_embeddings + gate * scaled_point_embeddings
                if self.training and hasattr(self, '_gate_log_counter'):
                    self._gate_log_counter += 1
                    if self._gate_log_counter % 100 == 1:
                        print(f"[GATE] shape={gate.shape} mean={gate.mean().item():.4f} "
                              f"std={gate.std().item():.4f} min={gate.min().item():.4f} max={gate.max().item():.4f}")
            else:
                new_prompt_embeddings = new_prompt_embeddings + scaled_point_embeddings

        # Reshape back: [B*n, 256] -> [B, n, 256]
        new_prompt_embeddings = new_prompt_embeddings.reshape(B, n, -1)
        
        return new_prompt_embeddings

    def _encode_prompt_points(
        self,
        points: Optional[Tensor],
        labels: Optional[Tensor],
        spatial_shape: Tuple[int, int],
        target_num_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build deterministic embeddings from uncertainty points."""
        if points is None or labels is None:
            return torch.zeros(0, 256, device=device, dtype=dtype)

        height, width = spatial_shape
        points = points.to(device=device, dtype=dtype)
        labels = labels.to(device=device)

        norm_x = (points[..., 0] / max(width - 1, 1)) * 2 - 1
        norm_y = (points[..., 1] / max(height - 1, 1)) * 2 - 1

        freq_count = 64
        frequencies = torch.exp(
            torch.linspace(0, 1, freq_count, device=device, dtype=dtype)
        )
        x_angles = norm_x.unsqueeze(-1) * frequencies
        y_angles = norm_y.unsqueeze(-1) * frequencies
        coord_embedding = torch.cat(
            [
                torch.sin(x_angles),
                torch.cos(x_angles),
                torch.sin(y_angles),
                torch.cos(y_angles),
            ],
            dim=-1,
        )
        label_scale = torch.where(
            labels > 0,
            torch.ones_like(labels, dtype=dtype),
            torch.where(labels == 0, -torch.ones_like(labels, dtype=dtype), torch.zeros_like(labels, dtype=dtype)),
        ).unsqueeze(-1)
        point_embeddings = coord_embedding * label_scale

        current_num_points = point_embeddings.shape[1]
        if current_num_points > target_num_points:
            point_embeddings = point_embeddings[:, :target_num_points]
        elif current_num_points < target_num_points:
            pad = torch.zeros(
                point_embeddings.shape[0],
                target_num_points - current_num_points,
                point_embeddings.shape[2],
                device=device,
                dtype=dtype,
            )
            point_embeddings = torch.cat([point_embeddings, pad], dim=1)

        return point_embeddings

    def mask_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        bbox_feats: Tensor,
        batch_gt_instances: InstanceList,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            if len(pos_rois) == 0:
                return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
            mask_results = self._mask_forward(
                x,
                pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
            )
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(torch.ones(res.pos_priors.shape[0], device=device, dtype=torch.uint8))
                pos_inds.append(torch.zeros(res.neg_priors.shape[0], device=device, dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            mask_results = self._mask_forward(
                x,
                pos_inds=pos_inds,
                bbox_feats=bbox_feats,
                high_res_features=high_res_features,
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
            )

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg,
        )
        total_loss = mask_loss_and_target["loss_mask"]
        all_mask_preds = mask_results.get("all_mask_preds")
        intermediate_weight = float(
            getattr(self, "iimr_intermediate_loss_weight", 0.0)
        )
        if (
            all_mask_preds is not None
            and intermediate_weight > 0
            and len(all_mask_preds) > 1
        ):
            shared_mask_targets = mask_loss_and_target["mask_targets"]
            shared_pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results]
            )
            intermediate_losses = []
            for iter_idx, iter_mask_preds in enumerate(all_mask_preds[:-1]):
                iter_loss_and_target = self.mask_head.loss_and_target(
                    mask_preds=iter_mask_preds,
                    sampling_results=sampling_results,
                    batch_gt_instances=batch_gt_instances,
                    rcnn_train_cfg=self.train_cfg,
                    mask_targets=shared_mask_targets,
                    pos_labels=shared_pos_labels,
                )
                iter_loss = iter_loss_and_target["loss_mask"]["loss_mask"]
                total_loss["loss_mask"] = (
                    total_loss["loss_mask"] + intermediate_weight * iter_loss
                )
                intermediate_losses.append(iter_loss.detach())
                total_loss[f"loss_mask_iter{iter_idx}"] = iter_loss
            if intermediate_losses:
                total_loss["debug_mask_intermediate_loss_mean"] = torch.stack(
                    intermediate_losses
                ).mean()

        if all_mask_preds is not None and len(all_mask_preds) > 1:
            shared_mask_targets = mask_loss_and_target["mask_targets"]
            shared_pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results]
            )
            total_loss.update(
                self._compute_iimr_iteration_debug_stats(
                    all_mask_preds=all_mask_preds,
                    mask_targets=shared_mask_targets,
                    pos_labels=shared_pos_labels,
                )
            )

        if os.environ.get("DEBUG_SMALL_OBJECT_STATS", "0") == "1":
            small_debug_stats = self._get_small_object_debug_stats(
                sampling_results=sampling_results,
                batch_gt_instances=batch_gt_instances,
                mask_targets=mask_loss_and_target.get("mask_targets"),
                ref_tensor=mask_results["mask_preds"],
            )
            total_loss.update(small_debug_stats)

        mask_results.update(loss_mask=total_loss)
        if "debug_stats" in mask_loss_and_target:
            mask_results["loss_mask"].update(mask_loss_and_target["debug_stats"])
        return mask_results

    def _compute_iimr_iteration_debug_stats(
        self,
        all_mask_preds: List[Tensor],
        mask_targets: Tensor,
        pos_labels: Tensor,
    ) -> Dict[str, Tensor]:
        """Return detached per-iteration mask quality diagnostics."""
        stats: Dict[str, Tensor] = {}
        if mask_targets.numel() == 0 or len(all_mask_preds) == 0:
            return stats

        target = mask_targets.detach().float()
        target_flat = target.flatten(1)
        eps = 1e-6
        iter_bce_values: List[Tensor] = []
        iter_iou_values: List[Tensor] = []

        for iter_idx, iter_mask_preds in enumerate(all_mask_preds):
            pred = F.interpolate(
                iter_mask_preds,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            if getattr(self.mask_head, "class_agnostic", False):
                pred = pred[:, 0]
            else:
                gather_labels = pos_labels.to(device=pred.device, dtype=torch.long)
                pred = pred[torch.arange(pred.shape[0], device=pred.device), gather_labels]

            pred_detached = pred.detach()
            prob = pred_detached.sigmoid()
            prob_flat = prob.flatten(1)
            bce = F.binary_cross_entropy_with_logits(
                pred_detached,
                target.to(device=pred_detached.device, dtype=pred_detached.dtype),
                reduction="mean",
            )
            intersection = (prob_flat * target_flat.to(prob_flat.device)).sum(dim=1)
            union = (
                prob_flat
                + target_flat.to(prob_flat.device)
                - prob_flat * target_flat.to(prob_flat.device)
            ).sum(dim=1)
            soft_iou = ((intersection + eps) / (union + eps)).mean()

            iter_bce_values.append(bce)
            iter_iou_values.append(soft_iou)
            stats[f"debug_iimr_iter{iter_idx}_bce"] = bce
            stats[f"debug_iimr_iter{iter_idx}_soft_iou"] = soft_iou
            stats[f"debug_iimr_iter{iter_idx}_prob_ge05"] = (prob >= 0.5).float().mean()
            stats[f"debug_iimr_iter{iter_idx}_logit_mean"] = pred_detached.mean()

        if len(iter_bce_values) > 1:
            stats["debug_iimr_delta_final_minus_iter0_bce"] = (
                iter_bce_values[-1] - iter_bce_values[0]
            )
            stats["debug_iimr_delta_final_minus_iter0_soft_iou"] = (
                iter_iou_values[-1] - iter_iou_values[0]
            )

        return stats

    def _get_small_object_debug_stats(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        mask_targets: Optional[Tensor],
        ref_tensor: Tensor,
    ) -> Dict[str, Tensor]:
        stats = {
            "debug_small_pos_gt_count": ref_tensor.new_tensor(0.0),
            "debug_small_pos_gt_ratio": ref_tensor.new_tensor(0.0),
            "debug_small_gt_box_area_mean": ref_tensor.new_tensor(0.0),
            "debug_small_mask_target_fill": ref_tensor.new_tensor(0.0),
        }
        if mask_targets is None or mask_targets.numel() == 0:
            return stats

        gt_areas = []
        for img_idx, res in enumerate(sampling_results):
            if res.pos_assigned_gt_inds.numel() == 0:
                continue
            gt_bboxes = getattr(batch_gt_instances[img_idx], "bboxes", None)
            if gt_bboxes is None or gt_bboxes.numel() == 0:
                continue
            assigned_inds = res.pos_assigned_gt_inds.to(dtype=torch.long, device=gt_bboxes.device)
            assigned_inds = assigned_inds.clamp(min=0, max=max(gt_bboxes.size(0) - 1, 0))
            assigned_boxes = gt_bboxes[assigned_inds]
            wh = (assigned_boxes[:, 2:4] - assigned_boxes[:, 0:2]).clamp(min=0)
            gt_areas.append((wh[:, 0] * wh[:, 1]).to(device=ref_tensor.device))

        if not gt_areas:
            return stats

        gt_areas = torch.cat(gt_areas, dim=0)
        total_pos = int(gt_areas.numel())
        if total_pos == 0:
            return stats

        small_mask = gt_areas < float(32 ** 2)
        small_count = int(small_mask.sum().item())
        stats["debug_small_pos_gt_count"] = ref_tensor.new_tensor(float(small_count))
        stats["debug_small_pos_gt_ratio"] = ref_tensor.new_tensor(float(small_count) / float(total_pos))

        if small_count > 0:
            stats["debug_small_gt_box_area_mean"] = gt_areas[small_mask].mean()
            if mask_targets.size(0) == gt_areas.size(0):
                small_targets = mask_targets[small_mask]
                stats["debug_small_mask_target_fill"] = small_targets.float().mean()

        return stats
    
    def _get_gt_masks_for_progressive_loss(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        mask_preds: Tensor,
    ) -> Tensor:
        """
        获取用于渐进式损失计算的 GT masks。
        
        Args:
            sampling_results: 采样结果列表
            batch_gt_instances: GT 实例列表
            mask_preds: mask 预测 [N, H, W]
        
        Returns:
            gt_masks: GT masks [N, H, W]
        """
        gt_masks_list = []
        target_h, target_w = mask_preds.shape[-2:]
        
        for i, sampling_result in enumerate(sampling_results):
            gt_instances = batch_gt_instances[i]
            pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
            
            if hasattr(gt_instances, "masks") and gt_instances.masks is not None:
                masks = gt_instances.masks
                if hasattr(masks, "masks"):
                    masks = masks.masks
                for gt_ind in pos_assigned_gt_inds:
                    if gt_ind < len(masks):
                        mask = masks[gt_ind]
                        if isinstance(mask, np.ndarray):
                            mask = torch.from_numpy(mask).to(mask_preds.device)
                        if mask.dim() == 2:
                            mask = mask.unsqueeze(0).unsqueeze(0).float()
                        elif mask.dim() == 3:
                            mask = mask.unsqueeze(0).float()
                        else:
                            mask = mask.float()
                        if mask.shape[-2:] != (target_h, target_w):
                            mask = F.interpolate(mask, size=(target_h, target_w), mode='nearest')
                        gt_masks_list.append(mask.squeeze(0))
            else:
                for _ in pos_assigned_gt_inds:
                    gt_masks_list.append(torch.zeros(1, target_h, target_w, device=mask_preds.device))
        
        if len(gt_masks_list) > 0:
            gt_masks = torch.stack(gt_masks_list, dim=0)
        else:
            gt_masks = torch.zeros_like(mask_preds)
        
        return gt_masks

    def loss(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: List[DetDataSample],
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> dict:
        assert len(rpn_results_list) == len(batch_data_samples)
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        if hasattr(self, "extra_pe"):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode="bilinear", align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        bbox_x, mask_x = self._get_downstream_branch_features(x)

        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop("bboxes")

            assign_result = self.bbox_assigner.assign(rpn_results, batch_gt_instances[i], batch_gt_instances_ignore[i])
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in bbox_x],
            )
            sampling_results.append(sampling_result)

        losses = dict()
        if self.with_bbox:
            bbox_results = self.bbox_loss(bbox_x, sampling_results)
            losses.update(bbox_results["loss_bbox"])

        if self.with_mask:
            mask_results = self.mask_loss(
                mask_x,
                sampling_results,
                bbox_results["bbox_feats"],
                batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
            )
            losses.update(mask_results["loss_mask"])
        return losses

    def predict_mask(
        self,
        x: Tuple[Tensor],
        batch_img_metas: List[dict],
        results_list: InstanceList,
        rescale: bool = False,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> InstanceList:
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
            )
            return results_list

        mask_results = self._mask_forward(
            x,
            mask_rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            high_res_features=high_res_features,
        )

        mask_preds = mask_results["mask_preds"]
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_pred_splits = mask_preds.split(num_mask_rois_per_img, 0)

        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_pred_splits,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )
        for result, mask_pred in zip(results_list, mask_pred_splits):
            if mask_pred.numel() == 0:
                continue
            prob = mask_pred.sigmoid().squeeze(1)
            result.mask_prob_mean = prob.mean(dim=(-2, -1))
            result.mask_prob_ge_03 = (prob >= 0.3).float().mean(dim=(-2, -1))
            result.mask_prob_ge_05 = (prob >= 0.5).float().mean(dim=(-2, -1))
            result.mask_prob_ge_07 = (prob >= 0.7).float().mean(dim=(-2, -1))
        return results_list

    def predict(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: SampleList,
        rescale: bool = False,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
    ) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        if hasattr(self, "extra_pe"):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(img_feats_pe, size=x[i].shape[-2:], mode="bilinear", align_corners=False)
                outputs.append(output)
            x = tuple(outputs)

        bbox_x, mask_x = self._get_downstream_branch_features(x)

        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            bbox_x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale,
        )
        self._log_bbox_prediction_debug(results_list)

        if self.with_mask:
            results_list = self.predict_mask(
                mask_x,
                batch_img_metas,
                results_list,
                rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
            )
        return results_list

    def _log_bbox_prediction_debug(self, results_list: InstanceList) -> None:
        if os.environ.get("DEBUG_BBOX_PRED", "0") != "1":
            return
        if not is_main_process():
            return

        debug_lines = []
        total_boxes = 0
        for img_idx, result in enumerate(results_list):
            scores = getattr(result, "scores", None)
            labels = getattr(result, "labels", None)
            num_boxes = len(result)
            total_boxes += num_boxes

            if scores is None or scores.numel() == 0:
                debug_lines.append(f"[BBOX_DEBUG] img={img_idx} boxes=0")
                continue

            unique_labels = labels.unique().tolist() if labels is not None and labels.numel() > 0 else []
            debug_lines.append(
                "[BBOX_DEBUG] "
                f"img={img_idx} boxes={num_boxes} "
                f"score_min={scores.min().item():.4f} "
                f"score_mean={scores.mean().item():.4f} "
                f"score_max={scores.max().item():.4f} "
                f"top5={[round(v, 4) for v in scores[:5].tolist()]} "
                f"labels={unique_labels}"
            )

        print(f"[BBOX_DEBUG] total_boxes={total_boxes} num_images={len(results_list)}")
        for line in debug_lines:
            print(line)


@MODELS.register_module()
class RSPrompterAnchorMaskHead(FCNMaskHead, BaseModule):
    def __init__(
        self,
        mask_decoder,
        in_channels,
        roi_feat_size=14,
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

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type="RSSamPromptEncoder",
            hf_pretrain_name=copy.deepcopy(mask_decoder.get("hf_pretrain_name")),
            init_cfg=copy.deepcopy(mask_decoder.get("init_cfg")),
            use_offline_mode=mask_decoder.get("use_offline_mode", False),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed

        num_sincos = 2 if with_sincos else 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels * roi_feat_size**2 // 4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point),
        )

        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(self, x, image_embeddings, image_positional_embeddings, roi_img_ids=None):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embedings = self.point_emb(x)
        point_embedings = einops.rearrange(point_embedings, "b (n c) -> b n c", n=self.per_pointset_point)
        if self.with_sincos:
            point_embedings = torch.sin(point_embedings[..., ::2]) + point_embedings[..., 1::2]

        sparse_embeddings = point_embedings.unsqueeze(1)
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

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
        )
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(num_roi_per_image, dim=0)

        low_res_masks, iou_predictions, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
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
        # Align supervision with each positive ROI instead of supervising
        # against the full-image instance mask. This matches Mask R-CNN target
        # generation and avoids training the ROI decoder on globally resized GT.
        return mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg)

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> dict:
        mask_targets = self.get_targets(sampling_results=sampling_results, batch_gt_instances=batch_gt_instances, rcnn_train_cfg=rcnn_train_cfg)
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
        return super()._predict_by_feat_single(
            mask_preds=mask_preds,
            bboxes=bboxes,
            labels=labels,
            img_meta=img_meta,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale,
            activate_map=activate_map,
        )
