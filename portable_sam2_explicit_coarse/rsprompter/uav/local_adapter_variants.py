"""Single-variable consumption/source variants for the frozen v1 probe.

Default highres/fine remains the original LocalUAVAdapter, unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .local_adapter_probe import LocalUAVAdapter


class VariantLocalUAVAdapter(LocalUAVAdapter):
    def __init__(self, pretrained, route='highres', source_mode='fine'):
        super().__init__(pretrained)
        assert route in ('highres','early') and source_mode in ('fine','contextual')
        self.route = route
        self.source_mode = source_mode
        if route == 'early':
            self.output = nn.Linear(32,256)
        if source_mode == 'contextual':
            self.context_source = nn.Conv2d(256,32,1)

    def attach(self,model):
        if self.route == 'highres':
            return super().attach(model)
        head=model.roi_head.mask_head
        self.handles.append(head.register_forward_pre_hook(self._boxes_hook,with_kwargs=True))
        self.handles.append(head.mask_decoder.register_forward_pre_hook(self._early_hook,with_kwargs=True))

    def _early_hook(self,module,args,kwargs):
        if not self.enabled or self.context is None:
            return
        # The wrapper receives one repeated image embedding for each RoI.
        # Residual is RoI-local and precedes the frozen two-way transformer.
        assert args[0].shape[0] == len(self.boxes) and args[0].shape[1:] == (256,32,32)
        enhanced=super()._inject(module,(),args[0])
        return (enhanced,*args[1:]),kwargs

    def encode(self,rgb,index,geom,satellite_high):
        if self.source_mode == 'fine':
            return super().encode(rgb,index,geom,satellite_high)
        features=[]
        for image in rgb.split(1):
            _,pyramid=self.encoder((image-self.mean)/self.std)
            fine=self.source(pyramid[0])
            context=F.interpolate(self.context_source(pyramid[2]),size=fine.shape[-2:],mode='bilinear',align_corners=False)
            features.append(fine+context)
        f=torch.cat(features).permute(0,2,3,1).reshape(-1,32)
        self.context=(f,index,geom,satellite_high)
        self.stats={}


class UnitLocalUAVAdapter(LocalUAVAdapter):
    """Bound the source and residual vector RMS without adding a learned scale."""
    def __init__(self,pretrained):
        super().__init__(pretrained)
        # Reuse the same initialized projection, then remove its free amplitude.
        self.output=nn.Sequential(self.output,nn.LayerNorm(32,elementwise_affine=False))

    def encode(self,rgb,index,geom,satellite_high):
        super().encode(rgb,index,geom,satellite_high)
        f,ix,g,high=self.context
        self.context=(F.layer_norm(f,(32,)),ix,g,high)

    def _chunk(self,qimage,keys,values,index,boxes,size):
        # LayerNorm uses FP32 under AMP; preserve the receiver dtype at zero gain.
        return super()._chunk(qimage,keys,values,index,boxes,size).to(qimage.dtype)
