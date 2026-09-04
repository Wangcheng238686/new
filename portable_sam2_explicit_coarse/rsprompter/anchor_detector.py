"""WHU/VHR-10 single-stream detector: SAM2 features + two-stage RPN/ROI."""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from torch import Tensor, nn
from transformers.models.sam.modeling_sam import SamVisionEncoderOutput

from mmdet.models import MaskRCNN, StandardRoIHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import InstanceList

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


