"""Same-pass native RoI evidence refinement; old numerical sources unchanged.

Post-decoder act2 and logits share the inherited ROI output lattice. Global
RGB/geometry evidence is sampled THROUGH boxes, never directly resized. The
SAM wrapper still returns its original three tensors; capture is call-local.
"""
from contextlib import contextmanager
from copy import deepcopy
import weakref
import torch
from torch import nn
from .candidate_detail_interaction import CandidateDetailInteraction
from .native_roi_refiner import NativeRoIRefiner, sample_roi_evidence
from ..cspm.mask_head import RSPrompterAnchorMaskHeadSAM2
from ..detection.anchor_roi_head import RSPrompterAnchorRoIPromptHead


def native_model_config(config):
    config=deepcopy(config)
    config['roi_head']['mask_head']['decoder_tail_refiner_cfg']={'enabled':False}
    if tuple(config['train_cfg']['rcnn']['mask_size']) != (512,512):
        raise ValueError('Preserve original512 ROI mask supervision; do not change target resolution')
    return config


class NativeEvidenceG(CandidateDetailInteraction):
    def encode(self,geometry):
        self.native_geometry=geometry
        self.native_aligned_rgb=self.native_rgb_valid=None
        return super().encode(geometry)
    def reduce_rgb(self,satellite,selected,valid,geometry):
        aligned=super().reduce_rgb(satellite,selected,valid,geometry)
        # EXACT existing pre-rgb_fusion result, not a recomputation or satellite copy.
        self.native_aligned_rgb=aligned.T.reshape(1,32,128,128)
        self.native_rgb_valid=valid.any(1).reshape(1,1,128,128)
        return aligned
    def clear(self):
        super().clear()
        self.native_aligned_rgb=None
        self.native_rgb_valid=None
        self.native_geometry=None


class NativeMaskHead(RSPrompterAnchorMaskHeadSAM2):
    def forward(self,*args,**kwargs):
        shim=self.mask_decoder
        if not isinstance(shim,NativeDecoderShim):
            raise RuntimeError('NativeMaskHead requires its own local decoder shim')
        if shim.call_context is not None:
            raise RuntimeError('Native mask head is not reentrant')
        shim.call_context=(kwargs.get('boxes'),kwargs.get('roi_img_ids'))
        try:
            return super().forward(*args,**kwargs)
        finally:
            shim.call_context=None
            shim.captured=None


class NativeRoIHead(RSPrompterAnchorRoIPromptHead):
    def _empty_rank_losses(self,x):
        losses,telemetry=super()._empty_rank_losses(x)
        head=getattr(self.mask_head,'native_refiner',None)
        if head is not None:
            losses['loss_mask']=losses['loss_mask']+sum(p.sum()*0 for p in head.parameters())
        return losses,telemetry


class NativeDecoderShim(nn.Module):
    def __init__(self,base,mask_head,adapter,zero_evidence=False):
        super().__init__();self.base=base
        # Weak references avoid duplicate parameter registration and ownership.
        object.__setattr__(self,'_head',weakref.ref(mask_head))
        object.__setattr__(self,'_adapter',weakref.ref(adapter))
        self.zero_evidence=zero_evidence;self.call_context=None;self.captured=None
        self.calls=0;self.observation={};self.observe=False
    def forward(self,*args,**kwargs):
        if self.call_context is None or self.captured is not None:
            raise RuntimeError('Missing/native decoder context or stale activation')
        boxes,image_ids=self.call_context
        if boxes is None or image_ids is None:
            raise RuntimeError('Native evidence requires explicit proposal boxes and image IDs')
        def capture(module,inputs,output):
            if self.captured is not None:raise RuntimeError('More than one act2 in a decoder call')
            self.captured=output
        handle=self.base.decoder.output_upscaling[-1].register_forward_hook(capture)
        try:
            result=self.base(*args,**kwargs)
            if len(result)!=3:raise RuntimeError('Original decoder must return3; UDPR capture disabled')
            logits,iou,tokens=result;feature=self.captured
            if feature is None or feature.shape!=(len(boxes),32,*logits.shape[-2:]):
                raise RuntimeError('Captured act2 must share native mask lattice and ROI order')
            if not torch.isfinite(image_ids).all() or not torch.equal(image_ids,image_ids.long().to(image_ids.dtype)):
                raise ValueError('RoI image IDs must be finite integer values')
            image_ids=image_ids.long()
            adapter=self._adapter();head=self._head()
            if self.zero_evidence:
                rgb=geometry=None
            else:
                if adapter.native_aligned_rgb is None or adapter.native_geometry is None:
                    raise RuntimeError('Evidence must come from current G encode/extract, never a previous image')
                rgb,geometry=sample_roi_evidence(boxes,image_ids,logits.shape[-2:],
                    aligned_rgb=adapter.native_aligned_rgb,rgb_valid=adapter.native_rgb_valid,
                    geometry=adapter.native_geometry,image_size=(512,512))
            refined=head.native_refiner(feature,logits,rgb,geometry)
            self.calls+=1
            self.observation=dict(native_hw=list(logits.shape[-2:]),instances=len(logits),
                                  zero_evidence=self.zero_evidence,decoder_calls=self.calls)
            if self.observe:
                self.observation.update(raw=logits.detach().clone(),refined=refined.detach().clone(),
                    boxes=boxes.detach().clone(),roi_image_ids=image_ids.detach().clone(),
                    rgb=None if rgb is None else rgb.detach().clone(),
                    geometry=None if geometry is None else geometry.detach().clone())
            return refined,iou,tokens
        finally:
            handle.remove();self.captured=None


def install_native_refiner(satellite,adapter,zero_evidence=False,seed=2080):
    head=satellite.roi_head.mask_head
    if head.decoder_tail_enabled or head.decoder_tail_refiner is not None:
        raise ValueError('Disable UDPR before constructing/loading model')
    if head.final_mask_coordinate_mode!='roi_local':raise ValueError('native RoI contract required')
    if head.mask_decoder.capture_upscaled_embedding:raise ValueError('Original3tuple decoder required')
    if isinstance(head,NativeMaskHead):raise ValueError('Already installed')
    adapter.__class__=NativeEvidenceG
    adapter.native_aligned_rgb=adapter.native_rgb_valid=adapter.native_geometry=None
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(seed);head.add_module('native_refiner',NativeRoIRefiner().to(next(head.parameters()).device))
    original=head.mask_decoder
    head.mask_decoder=NativeDecoderShim(original,head,adapter,zero_evidence)
    head.__class__=NativeMaskHead
    satellite.roi_head.__class__=NativeRoIHead
    return head.mask_decoder
