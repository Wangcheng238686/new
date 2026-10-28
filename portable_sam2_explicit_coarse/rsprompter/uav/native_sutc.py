"""Independent S/U/T/C switches and tail-only satellite wiring.

S=(0,0), U=(1,0), T=(0,1), C=(1,1). U means the existing upstream
G + instance objective + candidate path; T means real evidence at native UBR.
ALL arms retain the same native UBR head. S/T install no upstream UAV hooks.
Legacy numerical modules are reused without changing their source files.
"""
from dataclasses import dataclass
import weakref
import torch
from torch import nn
from torch.nn import functional as F
from .attention_ladder import sample_source_continuous
from .local_adapter_probe import YAWS
from .view_subset import canonical_views, source_tokens, check_index_views
from .native_roi_refiner import NativeRoIRefiner
from .ubr_native_integration import NativeDecoderShim, NativeMaskHead, NativeRoIHead


@dataclass(frozen=True)
class SUTCSwitches:
    U: bool
    T: bool

    def __post_init__(self):
        if type(self.U) is not bool or type(self.T) is not bool:
            raise TypeError('U/T must be boolean switches')

    @classmethod
    def for_arm(cls,arm):
        try:return cls(*{'S':(False,False),'U':(True,False),'T':(False,True),'C':(True,True)}[arm])
        except KeyError:raise ValueError('Expected S/U/T/C arm') from None

    @property
    def upstream(self):return self.U

    @property
    def tail_evidence(self):return self.T

    @property
    def arm(self):return {(False,False):'S',(True,False):'U',(False,True):'T',(True,True):'C'}[(self.U,self.T)]


class ZeroEvidenceProvider:
    """No modules, tensors, files or UAV input reader; weakref-able sentinel."""
    enabled=False  # The legacy formal consumer must skip UAV proposal decoding.
    native_aligned_rgb=None
    native_rgb_valid=None
    native_geometry=None

    def clear(self):
        self.native_aligned_rgb=self.native_rgb_valid=self.native_geometry=None


def install_native_tail(satellite,provider,*,zero_evidence=False,seed=2080):
    """Same native head/shim contract; never change provider type or add G hooks.

    As in the established native installer, head/RoI types are adapted in place.
    The provider's lifetime is owned by SatelliteTailJoint, not this weakref shim.
    """
    head=satellite.roi_head.mask_head
    if isinstance(head,NativeMaskHead):raise ValueError('Native tail already installed')
    if head.decoder_tail_enabled or head.decoder_tail_refiner is not None:
        raise ValueError('Disable UDPR in config BEFORE model construction')
    if head.final_mask_coordinate_mode!='roi_local' or head.mask_decoder.capture_upscaled_embedding:
        raise ValueError('Require original roi_local three-tensor decoder contract')
    weakref.ref(provider)
    for name in ('native_aligned_rgb','native_rgb_valid','native_geometry'):
        if not hasattr(provider,name):raise TypeError('Missing evidence interface '+name)
    device=next(head.parameters()).device
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []):
        # Head is initialized on CPU, as in the old installer, before transfer.
        torch.random.default_generator.manual_seed(seed)
        head.add_module('native_refiner',NativeRoIRefiner().to(device))
    head.mask_decoder=NativeDecoderShim(head.mask_decoder,head,provider,zero_evidence)
    head.__class__=NativeMaskHead
    satellite.roi_head.__class__=NativeRoIHead
    return head.mask_decoder


def _make_p0(foundation):
    # Only the generic visual encoder is temporarily built; never instantiate
    # CompactInteraction, geometry CNN, instance head, query or satellite hooks.
    from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
    from .compact_joint import P0Source
    source=RSSAM2VisionEncoder(checkpoint_path=str(foundation),model_size='base_plus',
        use_gradient_checkpointing=True,use_lora=True,freeze_backbone=False,
        lora_config=dict(r=16,lora_alpha=32,target_modules=['qkv','proj'],lora_dropout=.05,bias='none',modules_to_save=[]))
    return P0Source(source)


class TailP0Evidence(nn.Module):
    """Pure P0 RGB source + projection + measured correspondence mean + geo5.

    Common UAV state is an input dictionary, not a task-trained checkpoint.
    Runtime owns verification that it is the generic common initializer. All
    shared source keys must match exactly; unrelated upstream keys are excluded.
    """
    enabled=False
    def __init__(self,foundation,common_state,source_seed=2045,*,p0_factory=None,view_ids=YAWS):
        super().__init__()
        self.view_ids=canonical_views(view_ids)
        factory=_make_p0 if p0_factory is None else p0_factory
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(source_seed)
            self.rgb_encoder=factory(foundation)
            self.rgb_project=nn.Conv2d(256,32,1)
        self.register_buffer('rgb_mean',torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer('rgb_std',torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))
        expected=set(self.state_dict())
        selected={k:v for k,v in common_state.items() if k.startswith(('rgb_encoder.','rgb_project.')) or k in ('rgb_mean','rgb_std')}
        if set(selected)!=expected:
            raise RuntimeError('Shared P0 state mismatch missing=%s extra=%s'%(sorted(expected-set(selected)),sorted(set(selected)-expected)))
        self.load_state_dict(selected,strict=True)
        self.shared_source_keys=tuple(sorted(selected))
        self.native_aligned_rgb=self.native_rgb_valid=self.native_geometry=None
        self.train(True)

    def train(self,mode=True):
        super().train(mode)
        self.rgb_encoder.eval()
        for name,module in self.rgb_encoder.named_modules():
            if 'lora_dropout' in name:module.train(mode)
        return self

    def encode(self,geometry,rgb,index,anchor):
        self.clear()
        if geometry.shape!=(1,5,512,512) or rgb.shape!=(5,3,512,512):raise ValueError('Expected geo5 and five512 RGB views')
        if index.shape!=(16384,8) or anchor.shape!=(16384,8,6) or anchor.dtype!=torch.float64:
            raise ValueError('Retain original eight candidate indices and float64 anchors')
        check_index_views(index,self.view_ids)
        features=source_tokens(self,rgb,self.view_ids)
        selected=sample_source_continuous(features,index,anchor)
        valid=index>=0
        aligned=(selected*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        self.native_aligned_rgb=aligned.T.reshape(1,32,128,128)
        self.native_rgb_valid=valid.any(1).reshape(1,1,128,128)
        self.native_geometry=geometry  # Independent raw512 support; no CNN/pooling.

    def clear(self):
        self.native_aligned_rgb=self.native_rgb_valid=self.native_geometry=None


class SatelliteTailJoint(nn.Module):
    """S/T original satellite loss plus same-pass UBR; no auxiliary objective."""
    def __init__(self,satellite,provider=None):
        super().__init__();self.satellite=satellite
        self.provider=ZeroEvidenceProvider() if provider is None else provider
        if self.provider.enabled:raise ValueError('S/T provider must not generate UAV proposals')
        shim=install_native_tail(satellite,self.provider,zero_evidence=provider is None)
        # Satellite owns this module; the joint keeps only a non-registering reference.
        object.__setattr__(self,'shim',shim)

    @property
    def uav(self):return self.provider

    def prepare_uav(self,**payload):
        if isinstance(self.provider,ZeroEvidenceProvider):
            if payload:raise ValueError('S must not read or receive UAV payload')
            self.provider.clear()
        else:
            if set(payload)!={'geometry','rgb','index','anchor'}:raise ValueError('T requires only measured RGB/depth correspondence inputs')
            self.provider.encode(**payload)

    def forward(self,sample,**payload):
        try:
            self.prepare_uav(**payload)
            features=self.satellite.extract_feat(sample['inputs'])
            original=self.satellite.extract_feat
            self.satellite.extract_feat=lambda inputs:features
            try:return self.satellite.loss(sample['inputs'],sample['data_samples'])
            finally:self.satellite.extract_feat=original
        except Exception:
            self.clear()
            raise

    def clear(self):self.provider.clear()

    def cuda(self,device=None):
        self.satellite.cuda(device=device)
        if isinstance(self.provider,nn.Module):self.provider.cuda(device=device)
        return self

    def cpu(self):
        self.satellite.cpu()
        if isinstance(self.provider,nn.Module):self.provider.cpu()
        return self

    def to(self,*args,**kwargs):
        self.satellite.to(*args,**kwargs)
        if isinstance(self.provider,nn.Module):self.provider.to(*args,**kwargs)
        return self
