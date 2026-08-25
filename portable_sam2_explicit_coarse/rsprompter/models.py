import copy
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

from .ckpt_utils import load_module_state_dict_strict
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
from mmdet.models.roi_heads.bbox_heads import Shared2FCBBoxHead
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi, get_box_tensor
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
class AP75DualLossShared2FCBBoxHead(Shared2FCBBoxHead):
    """Shared2FC bbox head with encoded SmoothL1 and decoded IoU losses."""

    def __init__(self, loss_bbox_iou, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.reg_decoded_bbox:
            raise ValueError(
                "AP75DualLossShared2FCBBoxHead requires reg_decoded_bbox=False"
            )
        self.loss_bbox_iou = MODELS.build(loss_bbox_iou)

    def loss(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        reduction_override: Optional[str] = None,
    ) -> dict:
        losses = super().loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override,
        )
        if bbox_pred is None:
            return losses

        pos_inds = (labels >= 0) & (labels < self.num_classes)
        if not pos_inds.any():
            losses["loss_bbox_iou"] = bbox_pred.sum() * 0
            return losses

        if self.reg_class_agnostic:
            pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)[pos_inds]
        else:
            pos_bbox_pred = bbox_pred.view(
                bbox_pred.size(0), self.num_classes, -1
            )[pos_inds, labels[pos_inds]]

        pos_rois = rois[pos_inds, 1:]
        pred_boxes = get_box_tensor(
            self.bbox_coder.decode(pos_rois, pos_bbox_pred)
        )
        target_boxes = get_box_tensor(
            self.bbox_coder.decode(pos_rois, bbox_targets[pos_inds])
        )
        losses["loss_bbox_iou"] = self.loss_bbox_iou(
            pred_boxes,
            target_boxes,
            bbox_weights[pos_inds],
            avg_factor=bbox_targets.size(0),
            reduction_override=reduction_override,
        )
        return losses


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
            
            # 问题 13：用严格加载工具打印 missing/unexpected 明细
            load_module_state_dict_strict(
                self.shared_image_embedding,
                new_state_dict,
                module_name="PositionalEmbedding",
                strict_reproduction=os.environ.get("SAM2_STRICT_CKPT", "0") == "1",
            )

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

            load_module_state_dict_strict(
                vision_encoder,
                new_state_dict,
                module_name="VisionEncoder(interpolated)",
                strict_reproduction=os.environ.get("SAM2_STRICT_CKPT", "0") == "1",
            )

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
    def __init__(
        self,
        shared_image_embedding,
        decoder_freeze=True,
        sam_image_embedding_stride=32,
        *args,
        **kwargs,
    ):
        peft_config = kwargs.get("backbone", {}).get("peft_config", {})
        self.sam_image_embedding_stride = int(sam_image_embedding_stride)
        if self.sam_image_embedding_stride not in (16, 32):
            raise ValueError(
                "sam_image_embedding_stride must be 16 (official) or 32 "
                f"(legacy), got {self.sam_image_embedding_stride}"
            )
        super().__init__(*args, **kwargs)
        mask_head = getattr(getattr(self, "roi_head", None), "mask_head", None)
        uses_prompt_encoder_pe = getattr(mask_head, "prompt_encoder", None) is not None
        # Preserve the legacy initialization/RNG sequence before dropping the
        # positional embedding module that PromptEncoder always supersedes.
        legacy_image_embedding = MODELS.build(shared_image_embedding)
        self.shared_image_embedding = None if uses_prompt_encoder_pe else legacy_image_embedding
        self.decoder_freeze = decoder_freeze

        self.frozen_modules = []
        if peft_config is None:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            if self.shared_image_embedding is not None:
                self.frozen_modules.append(self.shared_image_embedding)
            self.frozen_modules.append(self.roi_head.mask_head.mask_decoder)
            if self.roi_head.mask_head.no_mask_embed is not None:
                self.frozen_modules.append(self.roi_head.mask_head.no_mask_embed)
        elif self.shared_image_embedding is not None and getattr(
            self.roi_head.mask_head, "prompt_encoder_enabled", False
        ):
            self.frozen_modules += [self.shared_image_embedding]
        self._set_grad_false(self.frozen_modules)

    def _set_grad_false(self, module_list=[]):
        for module in module_list:
            module.eval()
            if isinstance(module, nn.Parameter):
                module.requires_grad = False
            for param in module.parameters():
                param.requires_grad = False

    def get_prompt_debug_stats(self) -> Dict[str, object]:
        roi_head = getattr(self, "roi_head", None)
        if roi_head is None or not hasattr(roi_head, "get_prompt_debug_stats"):
            return {}
        return roi_head.get_prompt_debug_stats()

    # Backward-compatible training logger entry.  The canonical single-stream
    # shape-point model has no UAV branch, but the legacy logger calls this name.
    def get_uav_debug_stats(self) -> Dict[str, object]:
        return self.get_prompt_debug_stats()

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

    def _select_sam_image_embeddings(
        self, vision_outputs, batch_inputs: Tensor
    ) -> Tensor:
        """Select stride-16 official or stride-32 legacy SAM decoder features."""
        expected_hw = (
            int(batch_inputs.shape[-2]) // self.sam_image_embedding_stride,
            int(batch_inputs.shape[-1]) // self.sam_image_embedding_stride,
        )
        candidates = []
        if isinstance(vision_outputs, tuple):
            if torch.is_tensor(vision_outputs[0]):
                candidates.append(vision_outputs[0])
            if len(vision_outputs) > 1 and isinstance(
                vision_outputs[1], (list, tuple)
            ):
                candidates.extend(
                    feature
                    for feature in vision_outputs[1]
                    if torch.is_tensor(feature)
                )
        elif isinstance(vision_outputs, SamVisionEncoderOutput):
            candidates.append(vision_outputs[0])
        matches = [
            feature for feature in candidates if feature.shape[-2:] == expected_hw
        ]
        if not matches:
            available = [tuple(feature.shape[-2:]) for feature in candidates]
            raise RuntimeError(
                "SAM image embedding selection failed: expected spatial size "
                f"{expected_hw} for stride={self.sam_image_embedding_stride}, "
                f"available={available}"
            )
        return matches[0]

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if isinstance(vision_outputs, SamVisionEncoderOutput):
            vision_hidden_states = vision_outputs[1]
        elif isinstance(vision_outputs, tuple):
            vision_hidden_states = vision_outputs
        else:
            raise NotImplementedError
        image_embeddings = self._select_sam_image_embeddings(
            vision_outputs, batch_inputs
        )

        mask_head = getattr(getattr(self, "roi_head", None), "mask_head", None)
        if getattr(mask_head, "prompt_encoder", None) is not None:
            image_positional_embeddings = None
        else:
            image_positional_embeddings = self.get_image_wide_positional_embeddings(
                size=image_embeddings.shape[-1]
            )
            batch_size = image_embeddings.shape[0]
            image_positional_embeddings = image_positional_embeddings.repeat(
                batch_size, 1, 1, 1
            )
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
        bbox_train_fp32=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.bbox_train_fp32 = bool(bbox_train_fp32)
        if with_extra_pe:
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(num_feats=out_channels // 2, normalize=True)
            from mmdet.models import SinePositionalEncoding

            self.extra_pe = SinePositionalEncoding(**positional_encoding)

    def get_prompt_debug_stats(self) -> Dict[str, object]:
        stats: Dict[str, object] = {}
        mask_head = getattr(self, "mask_head", None)
        if mask_head is not None and hasattr(mask_head, "get_uav_debug_stats"):
            stats.update(mask_head.get_uav_debug_stats())
        for key, value in (getattr(self, "_last_coarse_stats", {}) or {}).items():
            stats[f"COARSE/{key}"] = value
        stats.update(getattr(self, "_last_box_jitter_stats", {}) or {})
        stats.update(self._last_prompt_box_stats or {})

        # Convert count/sum diagnostics into interpretable local-rank ratios.
        def _ratio(out_key: str, num_key: str, den_key: str):
            if num_key not in stats or den_key not in stats:
                return
            num, den = stats[num_key], stats[den_key]
            if torch.is_tensor(num) and torch.is_tensor(den):
                stats[out_key] = num.detach().float() / den.detach().float().clamp_min(1.0)

        _ratio("SP/pos_in_pred_fg", "SP/pos_in_pred_fg_count", "SP/valid_pos_count")
        _ratio("SP/neg_in_pred_bg", "SP/neg_in_pred_bg_count", "SP/valid_neg_count")
        _ratio("SP/mean_point_distance", "SP/point_distance_sum", "SP/point_pair_count")
        _ratio("SP/point_overlap_ratio", "SP/point_overlap_count", "SP/point_pair_count")
        _ratio("JITTER/area_ratio_mean", "JITTER/area_ratio_sum", "JITTER/area_ratio_count")
        _ratio("BOX/fallback_ratio", "BOX/fallback_count", "BOX/num_rois")
        _ratio("BOX/refined_proposal_iou", "BOX/refined_proposal_iou_sum", "BOX/refined_valid_count")
        _ratio("SP/p1_valid_ratio", "SP/p1_valid_count", "SP/roi_count")
        _ratio("SP/p2_valid_ratio", "SP/p2_valid_count", "SP/roi_count")
        _ratio("SP/n1_valid_ratio", "SP/n1_valid_count", "SP/roi_count")
        _ratio("SP/n2_valid_ratio", "SP/n2_valid_count", "SP/roi_count")
        _ratio("SP/empty_ratio", "SP/empty_count", "SP/roi_count")
        _ratio("SP/full_ratio", "SP/full_count", "SP/roi_count")
        _ratio("SP/all_invalid_ratio", "SP/all_invalid_count", "SP/roi_count")
        _ratio("SP/p1_in_gt_fg", "SP/p1_in_gt_fg_count", "SP/p1_gt_valid_count")
        _ratio("SP/p2_in_gt_fg", "SP/p2_in_gt_fg_count", "SP/p2_gt_valid_count")
        _ratio("SP/n1_in_gt_bg", "SP/n1_in_gt_bg_count", "SP/n1_gt_valid_count")
        _ratio("SP/n2_in_gt_bg", "SP/n2_in_gt_bg_count", "SP/n2_gt_valid_count")
        return stats

    def bbox_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
    ) -> dict:
        if not self.bbox_train_fp32:
            return super().bbox_loss(x, sampling_results)

        rois = bbox2roi([res.priors for res in sampling_results])
        with torch.autocast(device_type=x[0].device.type, enabled=False):
            fp32_x = tuple(feat.float() for feat in x)
            bbox_results = self._bbox_forward(fp32_x, rois.float())
            bbox_loss_and_target = self.bbox_head.loss_and_target(
                cls_score=bbox_results["cls_score"],
                bbox_pred=bbox_results["bbox_pred"],
                rois=rois.float(),
                sampling_results=sampling_results,
                rcnn_train_cfg=self.train_cfg,
            )
        bbox_results.update(loss_bbox=bbox_loss_and_target["loss_bbox"])
        return bbox_results

    @staticmethod
    def _aligned_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
        if boxes1.shape != boxes2.shape or boxes1.ndim != 2 or boxes1.shape[1] != 4:
            raise ValueError("aligned IoU requires matching [N,4] tensors")
        lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
        rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp_min(0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
        return inter / (area1 + area2 - inter).clamp_min(1e-6)

    def _decode_positive_refined_boxes(
        self,
        bbox_results: dict,
        sampling_results: List[SamplingResult],
        image_size: int,
    ) -> Tensor:
        """Decode SABL positive boxes in bbox-forward order and detach them."""
        bbox_pred = bbox_results.get("bbox_pred") if bbox_results is not None else None
        if not isinstance(bbox_pred, (tuple, list)) or len(bbox_pred) != 2:
            raise RuntimeError(
                "sabl_refined_detach requires SABL tuple bbox predictions"
            )
        positive_indices = []
        offset = 0
        for result in sampling_results:
            count = result.pos_priors.shape[0]
            positive_indices.append(
                torch.arange(offset, offset + count, device=bbox_pred[0].device)
            )
            offset += result.priors.shape[0]
        if not positive_indices:
            return bbox_pred[0].new_zeros((0, 4))
        positive_indices = torch.cat(positive_indices, dim=0)
        pos_rois = bbox2roi([result.pos_priors for result in sampling_results])
        positive_pred = tuple(pred[positive_indices].float() for pred in bbox_pred)
        with torch.autocast(device_type=pos_rois.device.type, enabled=False):
            decoded_boxes, _ = self.bbox_head.bbox_coder.decode(
                pos_rois[:, 1:].float(),
                positive_pred,
                max_shape=(int(image_size), int(image_size)),
            )
        return decoded_boxes.detach()

    def _mask_forward(
        self,
        x: Tuple[Tensor],
        rois: Tensor = None,
        pos_inds: Optional[Tensor] = None,
        bbox_feats: Optional[Tensor] = None,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
        roi_img_ids_override: Optional[Tensor] = None,
        boxes_override: Optional[Tensor] = None,
    ) -> dict:
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(x[: self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        if rois is not None:
            prompt_rois = rois
        elif roi_img_ids_override is not None and boxes_override is not None:
            prompt_rois = torch.cat(
                [roi_img_ids_override[:, None].to(boxes_override.dtype), boxes_override],
                dim=1,
            )
        else:
            prompt_rois = None

        mask_head_out = self.mask_head(
            mask_feats,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=(
                rois[:, 0]
                if rois is not None
                else roi_img_ids_override
            ),
            high_res_features=high_res_features,
            boxes=rois[:, 1:] if rois is not None else boxes_override,
            p2_feature=x[0],
            prompt_rois=prompt_rois,
        )
        if len(mask_head_out) != 6:
            raise RuntimeError(
                "Canonical mask head must return 6 values, got "
                f"{len(mask_head_out)}"
            )
        (
            mask_preds,
            iou_predictions,
            coarse_outputs,
            uav_aux_losses,
            quality_predictions,
            mask_tokens,
        ) = mask_head_out
        return dict(
            mask_preds=mask_preds,
            mask_feats=mask_feats,
            iou_predictions=iou_predictions,
            quality_predictions=quality_predictions,
            mask_tokens=mask_tokens,
            coarse_outputs=coarse_outputs,
            uav_aux_losses=uav_aux_losses,
        )

    def mask_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        bbox_feats: Tensor,
        bbox_results: Optional[dict],
        batch_gt_instances: InstanceList,
        image_embeddings=None,
        image_positional_embeddings=None,
        high_res_features=None,
        batch_img_metas: Optional[List[dict]] = None,
    ) -> dict:
        original_pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
        if len(original_pos_rois) == 0:
            # Do not leak diagnostics from the previous batch when this rank has
            # no positive proposals.
            zero = x[0].new_zeros(())
            self._last_coarse_stats = {
                "dice": zero,
                "iou": zero,
                "boundary_tp": zero,
                "boundary_fp": zero,
                "boundary_fn": zero,
            }
            self._last_prompt_box_stats = {
                "BOX/num_rois": zero,
                "BOX/fallback_count": zero,
                "BOX/refined_proposal_iou_sum": zero,
                "BOX/refined_valid_count": zero,
            }
            self._last_box_jitter_stats = {
                "JITTER/num_rois": zero,
                "JITTER/num_jittered": zero,
                "JITTER/mean_iou": zero,
                "JITTER/fallback_count": zero,
                "JITTER/area_ratio_sum": zero,
                "JITTER/area_ratio_count": zero,
            }
            return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))

        prompt_rois = original_pos_rois
        prompt_pos_priors = None
        prompt_box_cfg = dict(
            getattr(self.mask_head, "mask_prompt_box_cfg", {}) or {}
        )
        train_box_source = str(
            prompt_box_cfg.get("train_box_source", "proposal")
        )
        # mask_loss can be invoked in eval mode (e.g. EMA/loss-monitor paths).
        # sabl_refined_detach is a training-only box source; in eval mode fall
        # back to the proposal box so the sabl_refined branch is not entered.
        if not self.training and train_box_source == "sabl_refined_detach":
            train_box_source = "proposal"
        if self.training and train_box_source == "sabl_refined_detach":
            if self.share_roi_extractor:
                raise RuntimeError(
                    "sabl_refined_detach requires a dedicated mask_roi_extractor "
                    "so Mask RoIAlign and all prompts share the same prompt_rois"
                )
            if bbox_results is None:
                raise RuntimeError(
                    "sabl_refined_detach requires bbox_results from the SABL branch"
                )
            image_size = int(getattr(self.mask_head, "prompt_encoder_image_size", 1024))
            decoded = self._decode_positive_refined_boxes(
                bbox_results, sampling_results, image_size=image_size
            )
            proposals = original_pos_rois[:, 1:].detach().float()
            decoded = decoded.to(proposals.device, dtype=proposals.dtype)
            if bool(prompt_box_cfg.get("clamp_to_image", True)):
                decoded[:, 0::2].clamp_(0.0, float(image_size))
                decoded[:, 1::2].clamp_(0.0, float(image_size))
            widths = decoded[:, 2] - decoded[:, 0]
            heights = decoded[:, 3] - decoded[:, 1]
            aligned_iou = self._aligned_box_iou(decoded, proposals)
            valid = (
                torch.isfinite(decoded).all(dim=1)
                & (widths >= float(prompt_box_cfg.get("min_box_size", 2.0)))
                & (heights >= float(prompt_box_cfg.get("min_box_size", 2.0)))
                & (aligned_iou >= float(
                    prompt_box_cfg.get("min_refined_proposal_iou", 0.30)
                ))
            )
            fallback = ~valid
            if not bool(prompt_box_cfg.get("fallback_to_proposal", True)) and fallback.any():
                raise RuntimeError(
                    "Invalid SABL refined prompt boxes and fallback_to_proposal=False"
                )
            selected = torch.where(valid[:, None], decoded, proposals)
            prompt_rois = torch.cat([original_pos_rois[:, :1], selected], dim=1)
            self._last_prompt_box_stats = {
                "BOX/num_rois": proposals.new_tensor(float(proposals.shape[0])),
                "BOX/fallback_count": fallback.sum().detach(),
                "BOX/refined_proposal_iou_sum": (aligned_iou * valid.float()).sum().detach(),
                "BOX/refined_valid_count": valid.sum().detach(),
            }
        elif train_box_source == "proposal":
            zero = original_pos_rois.new_zeros(())
            self._last_prompt_box_stats = {
                "BOX/num_rois": original_pos_rois.new_tensor(float(len(original_pos_rois))),
                "BOX/fallback_count": zero,
                "BOX/refined_proposal_iou_sum": zero,
                "BOX/refined_valid_count": zero,
            }
        else:
            raise ValueError(
                f"Unsupported train_box_source={train_box_source!r}; "
                "expected proposal or sabl_refined_detach"
            )

        jitter_cfg = dict(getattr(self.mask_head, "box_prompt_cfg", {}) or {})
        jitter_prob = float(jitter_cfg.get("jitter_prob", 0.0))
        if self.training and jitter_prob > 0.0:
            if batch_img_metas is None:
                raise ValueError("box prompt jitter requires batch_img_metas")
            if self.share_roi_extractor:
                raise RuntimeError(
                    "box prompt jitter requires a dedicated mask_roi_extractor so "
                    "RoIAlign and prompt boxes use the same jittered RoIs"
                )
            from .box_jitter import jitter_rois
            prompt_rois, self._last_box_jitter_stats = jitter_rois(
                prompt_rois,
                batch_img_metas,
                jitter_prob=jitter_prob,
                jitter_scale=float(jitter_cfg.get("jitter_scale", 0.05)),
                min_box_size=float(jitter_cfg.get("min_box_size", 2.0)),
                min_iou_with_original=float(
                    jitter_cfg.get("min_iou_with_original", 0.5)
                ),
            )
        else:
            zero = original_pos_rois.new_zeros(())
            self._last_box_jitter_stats = {
                "JITTER/num_rois": original_pos_rois.new_tensor(float(len(original_pos_rois))),
                "JITTER/num_jittered": zero,
                "JITTER/mean_iou": original_pos_rois.new_tensor(1.0),
                "JITTER/fallback_count": zero,
                "JITTER/area_ratio_sum": original_pos_rois.new_tensor(float(len(original_pos_rois))),
                "JITTER/area_ratio_count": original_pos_rois.new_tensor(float(len(original_pos_rois))),
            }

        pos_counts = [res.pos_priors.shape[0] for res in sampling_results]
        prompt_pos_priors = list(prompt_rois[:, 1:].split(pos_counts, dim=0))
        # From this point onward every mask/prompt operation uses prompt_rois.
        # Sampling assignments and bbox losses remain tied to sampling_results.
        pos_rois = prompt_rois

        if not self.share_roi_extractor:
            mask_results = self._mask_forward(
                x,
                pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
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
                roi_img_ids_override=pos_rois[:, 0] if len(pos_rois) > 0 else None,
                boxes_override=pos_rois[:, 1:] if len(pos_rois) > 0 else None,
            )

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg,
            final_mask_pos_priors=prompt_pos_priors,
        )
        mask_results.update(loss_mask=mask_loss_and_target["loss_mask"])
        mask_results["loss_mask"].update(
            mask_loss_and_target.get("debug_stats", {})
        )
        if getattr(self.mask_head, "quality_head_enabled", False):
            mask_targets = mask_loss_and_target["mask_targets"]
            quality_preds = mask_results["quality_predictions"]
            resized_mask_preds = F.interpolate(
                mask_results["mask_preds"],
                size=mask_targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results]
            )
            if self.mask_head.class_agnostic:
                selected_mask_preds = resized_mask_preds[:, 0]
                selected_quality_preds = quality_preds[:, 0]
            else:
                row = torch.arange(pos_labels.shape[0], device=pos_labels.device)
                mask_columns = pos_labels.clamp_max(resized_mask_preds.shape[1] - 1)
                quality_columns = pos_labels.clamp_max(quality_preds.shape[1] - 1)
                selected_mask_preds = resized_mask_preds[row, mask_columns]
                selected_quality_preds = quality_preds[row, quality_columns]

            with torch.no_grad():
                target = mask_targets.float()
                mask_probability = torch.sigmoid(selected_mask_preds.float())
                target_mask = mask_probability
                if self.mask_head.quality_head_type == "ap_ordinal":
                    target_mask = (mask_probability >= 0.5).to(target.dtype)
                intersection = (target_mask * target).flatten(1).sum(1)
                union = (
                    target_mask.flatten(1).sum(1)
                    + target.flatten(1).sum(1)
                    - intersection
                )
                quality_targets = (intersection / union.clamp_min(1e-6)).clamp(0, 1)

            if self.mask_head.quality_head_type == "ap_ordinal":
                thresholds = selected_quality_preds.new_tensor(
                    self.mask_head.quality_head.thresholds
                )
                ordinal_targets = (
                    quality_targets[:, None] >= thresholds[None]
                ).to(selected_quality_preds.dtype)
                quality_loss = F.binary_cross_entropy_with_logits(
                    selected_quality_preds.float(), ordinal_targets.float()
                )
            else:
                quality_loss = F.smooth_l1_loss(
                    selected_quality_preds.float(),
                    quality_targets,
                    beta=self.mask_head.quality_head_loss_beta,
                )
            mask_results["loss_mask_quality"] = {
                "loss_mask_quality": self.mask_head.quality_head_loss_weight
                * quality_loss
            }

        # === ROI-local coarse-mask supervision ===
        # CoarseMaskLoss operates at the prediction's native resolution. Binary
        # GT crops are resized with nearest-neighbour interpolation, and the
        # same prompt_pos_priors used by mask RoIAlign/prompts are used for the
        # crop when training-time box jitter is enabled.
        coarse_outputs = mask_results.get("coarse_outputs")
        if coarse_outputs is not None:
            raw_coarse_logits = coarse_outputs["raw_logits"]
            refined_coarse_logits = coarse_outputs["refined_logits"]
            sp_weight = getattr(self.mask_head, "shape_prior_loss_weight", 0.0)
            if sp_weight > 0 and hasattr(self.mask_head, "coarse_mask_loss"):
                coarse_loss, coarse_stats = self.mask_head.coarse_mask_loss(
                    raw_coarse_logits, sampling_results, batch_gt_instances,
                    prompt_pos_priors=prompt_pos_priors,
                )
                mask_results["loss_shape_prior"] = {"loss_shape_prior": coarse_loss}
                if not hasattr(self, "_last_coarse_stats") or self._last_coarse_stats is None:
                    self._last_coarse_stats = {}
                raw_stats = {k: v.detach() for k, v in coarse_stats.items()}
                self._last_coarse_stats.update(raw_stats)
                self._last_coarse_stats.update({
                    "raw_dice_score": 1.0 - raw_stats["dice"],
                    "raw_iou": raw_stats["iou"],
                    "raw_boundary_tp": raw_stats["boundary_tp"],
                    "raw_boundary_fp": raw_stats["boundary_fp"],
                    "raw_boundary_fn": raw_stats["boundary_fn"],
                })

                if self.mask_head.p2_boundary_refiner is not None:
                    coarse_targets = self.mask_head.get_coarse_targets(
                        sampling_results,
                        batch_gt_instances,
                        mask_size=raw_coarse_logits.shape[-2:],
                        prompt_pos_priors=prompt_pos_priors,
                    )
                    local_sum, local_count, boundary_stats = (
                        self.mask_head.p2_boundary_refiner.boundary_loss(
                            raw_logits=raw_coarse_logits,
                            delta_logits=coarse_outputs["delta_logits"],
                            raw_boundary_band=coarse_outputs["raw_boundary_band"],
                            search_support=coarse_outputs["search_support"],
                            target_mask=coarse_targets,
                        )
                    )
                    mask_results["_p2br_local_sum"] = local_sum
                    mask_results["_p2br_valid_count"] = local_count
                    with torch.no_grad():
                        _, refined_stats = self.mask_head.coarse_mask_loss_module(
                            refined_coarse_logits.detach(), coarse_targets
                        )
                    self._last_coarse_stats.update({
                        "refined_dice_score": 1.0 - refined_stats["dice"],
                        "refined_iou": refined_stats["iou"],
                        "refined_boundary_tp": refined_stats["boundary_tp"],
                        "refined_boundary_fp": refined_stats["boundary_fp"],
                        "refined_boundary_fn": refined_stats["boundary_fn"],
                        "boundary_loss_local_sum": local_sum.detach(),
                        **{k: v.detach() for k, v in boundary_stats.items()},
                    })

        uav_aux_losses = mask_results.get("uav_aux_losses") or {}
        for loss_name, loss_value in uav_aux_losses.items():
            if loss_value is not None:
                mask_results[loss_name] = {loss_name: loss_value}

        return mask_results

    def _shape_prior_aux_loss(
        self,
        shape_prior_mask_logits: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
    ) -> Optional[Tensor]:
        """Auxiliary loss for ROI-local shape prior logits.

        The shape-prior branch predicts a mask in ROI coordinates, so supervise it
        with the assigned GT mask cropped by the same positive proposal box.
        """
        pred = shape_prior_mask_logits  # [N,1,64,64]
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)  # [N,64,64]

        tgt = self._shape_prior_targets(
            shape_prior_mask_logits.shape[-2:],
            sampling_results,
            batch_gt_instances,
            pred.device,
        )
        if tgt is None:
            return None
        if tgt.shape[0] != pred.shape[0]:
            n = min(tgt.shape[0], pred.shape[0])
            pred = pred[:n]
            tgt = tgt[:n]

        pred_prob = torch.sigmoid(pred)
        # dice
        inter = (pred_prob * tgt).flatten(1).sum(dim=1)
        union = pred_prob.flatten(1).sum(dim=1) + tgt.flatten(1).sum(dim=1) + 1e-6
        dice_loss = 1.0 - 2.0 * inter / union
        # bce
        bce_loss = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none").flatten(1).mean(dim=1)
        return (dice_loss + bce_loss).mean()

    def _shape_prior_targets(
        self,
        target_size,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        device,
    ) -> Optional[Tensor]:
        targets_64 = []
        for res, gt in zip(sampling_results, batch_gt_instances):
            pos_gt_inds = res.pos_assigned_gt_inds
            if len(pos_gt_inds) == 0:
                continue
            pos_priors = res.pos_priors
            gt_masks = gt.masks
            if hasattr(gt_masks, "to_tensor"):
                mask_tensor = gt_masks.to_tensor(
                    dtype=torch.float32, device=device,
                )  # [num_gt, H, W]
            else:
                mask_tensor = torch.as_tensor(
                    gt_masks, dtype=torch.float32, device=device,
                )
            for box, idx in zip(pos_priors, pos_gt_inds):
                if idx < mask_tensor.shape[0]:
                    gt_mask = mask_tensor[int(idx)].unsqueeze(0).unsqueeze(0)
                    h, w = gt_mask.shape[-2:]
                    x1, y1, x2, y2 = box.detach()
                    x1i = int(torch.floor(x1).clamp(0, w - 1).item())
                    y1i = int(torch.floor(y1).clamp(0, h - 1).item())
                    x2i = int(torch.ceil(x2).clamp(x1i + 1, w).item())
                    y2i = int(torch.ceil(y2).clamp(y1i + 1, h).item())
                    m = gt_mask[:, :, y1i:y2i, x1i:x2i]
                    m = F.interpolate(
                        m,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    m = m.squeeze(0).squeeze(0).clamp(0, 1)  # [64,64]
                    targets_64.append(m)

        if not targets_64:
            return None
        return torch.stack(targets_64, dim=0)

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
                feats=[lvl_feat[i][None] for lvl_feat in x],
            )
            sampling_results.append(sampling_result)

        losses = dict()
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results["loss_bbox"])

        if self.with_mask:
            mask_results = self.mask_loss(
                x,
                sampling_results,
                bbox_results["bbox_feats"],
                bbox_results,
                batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
                batch_img_metas=[sample.metainfo for sample in batch_data_samples],
            )
            losses.update(mask_results["loss_mask"])
            if "loss_shape_prior" in mask_results:
                losses.update(mask_results["loss_shape_prior"])
            for loss_name in (
                "loss_mask_quality",
            ):
                if loss_name in mask_results:
                    losses.update(mask_results[loss_name])
            for loss_name, loss_value in mask_results.items():
                if (
                    loss_name.startswith("loss_roi_")
                    and isinstance(loss_value, dict)
                ):
                    losses.update(loss_value)
            refiner = getattr(self.mask_head, "p2_boundary_refiner", None)
            if refiner is not None:
                zero = x[0].sum() * 0.0
                losses["_p2br_local_sum"] = mask_results.get(
                    "_p2br_local_sum", zero
                )
                losses["_p2br_valid_count"] = mask_results.get(
                    "_p2br_valid_count", zero.detach()
                )
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
        quality_preds = mask_results["quality_predictions"]
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)
        quality_preds = quality_preds.split(num_mask_rois_per_img, 0)

        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )
        if getattr(self.mask_head, "quality_head_enabled", False):
            alpha = self.mask_head.quality_head_score_alpha
            for result, quality in zip(results_list, quality_preds):
                quality = quality[:, 0].float()
                if self.mask_head.quality_head_type == "ap_ordinal":
                    quality_levels = torch.sigmoid(quality)
                    result.mask_quality_levels = quality_levels
                    quality = quality_levels.mean(dim=-1)
                else:
                    quality = quality.clamp(0.0, 1.0)
                result.mask_quality = quality
                result.mask_scores = result.scores * quality.pow(alpha)
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

        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale,
        )

        if self.with_mask:
            results_list = self.predict_mask(
                x,
                batch_img_metas,
                results_list,
                rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
                high_res_features=high_res_features,
            )
        return results_list


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
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros((0,) + mask_size, device=device, dtype=torch.float32)
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(dtype=torch.float32, device=device)
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

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
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_preds, mask_targets, torch.zeros_like(pos_labels))
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
        loss["loss_mask"] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

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
        _ = labels
        scale_factor = bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))
        img_h, img_w = img_meta["ori_shape"][:2]
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:
            bboxes /= scale_factor
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(
            mask_preds,
            size=img_meta["batch_input_shape"],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        scale_factor_w, scale_factor_h = img_meta["scale_factor"]
        ori_rescaled_size = (img_h * scale_factor_h, img_w * scale_factor_w)
        im_mask = im_mask[:, : int(ori_rescaled_size[0]), : int(ori_rescaled_size[1])]

        h, w = img_meta["ori_shape"]
        im_mask = F.interpolate(im_mask.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask
