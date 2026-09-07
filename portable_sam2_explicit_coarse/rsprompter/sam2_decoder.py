"""SAM2 mask-decoder wrapper (split from models_sam2)."""

"""
SAM2 Adapter for Portable SAM Fusion - Simplified Version
直接使用SAM2的build_sam2函数加载模型
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from torch import Tensor


from .ckpt_utils import load_module_state_dict_strict
from .sam2_vision import SAM2_REPO, MaskDecoder, _load_sam2_checkpoint



class RSSAM2MaskDecoderWrapper(BaseModule):
    """Wrapper class to make RSSAM2MaskDecoder compatible with existing RSPrompterAnchorMaskHead."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        use_high_res_features: bool = False,
        capture_upscaled_embedding: bool = False,
        init_cfg: Optional[Dict] = None,
        checkpoint_load_cfg: Optional[Dict] = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.decoder = None
        self.use_high_res_features = use_high_res_features
        self.capture_upscaled_embedding = bool(capture_upscaled_embedding)
        self._captured_upscaled_embedding = None
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
            if self.capture_upscaled_embedding:
                # SAM2's public forward discards this tensor.  Keep the vendor
                # API untouched; the optional wrapper route owns this capture.
                # High-res SAM2 manually unpacks the Sequential, so a hook on
                # the container never fires.  Its final activation is the
                # exact upscaled_embedding in both high-res and plain paths.
                self.decoder.output_upscaling[-1].register_forward_hook(
                    self._capture_upscaled_embedding
                )
        except Exception as e:
            if is_main_process():
                print(f"Failed to load SAM2 decoder: {e}")
            if self.allow_decoder_fallback:
                self.decoder = None
            else:
                raise

    def _capture_upscaled_embedding(self, _module, _inputs, output):
        self._captured_upscaled_embedding = output

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
            if self.capture_upscaled_embedding:
                return low_res_masks, iou_predictions, None, None
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

        self._captured_upscaled_embedding = None
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

        if self.capture_upscaled_embedding:
            upscaled = self._captured_upscaled_embedding
            if upscaled is None:
                raise RuntimeError("SAM2 output_upscaling hook did not capture an embedding")
            if upscaled.shape[0] != low_res_masks.shape[0] or upscaled.shape[-2:] != low_res_masks.shape[-2:]:
                raise RuntimeError("Captured SAM2 upscaled embedding does not match native mask logits")
            return low_res_masks, iou_predictions, mask_tokens_out, upscaled
        return low_res_masks, iou_predictions, mask_tokens_out
