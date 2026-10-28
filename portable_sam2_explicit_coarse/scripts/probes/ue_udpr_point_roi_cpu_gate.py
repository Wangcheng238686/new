"""Analytical full-image coordinates and sparse-write contract on real UDPR."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from rsprompter.uav.udpr_point_roi_probe import UDPRPointROIAdapter as UDPRPointAdapter,mask_indices_to_satellite_uv
from rsprompter.decoder.decoder_tail_refiner import DecoderTailPointRefiner


def main():
    torch.set_num_threads(2);torch.manual_seed(45)
    for size in [128,256]:
        ids=torch.arange(size*size).reshape(1,-1)
        uv=mask_indices_to_satellite_uv(ids,size,size,torch.tensor([[0.,0.,512.,512.]]))
        expected=torch.stack(((ids%size+.5)*512/size,(ids//size+.5)*512/size),-1)
        assert torch.equal(uv,expected)
        cells=(uv[...,1]//4).long()*128+(uv[...,0]//4).long()
        if size==128:assert torch.equal(cells,ids)
        assert cells.min()==0 and cells.max()==16383
    boxes=torch.tensor([[32.,64.,160.,320.],[300.,80.,480.,200.]])
    for size in [128,256]:
        idx=torch.arange(size*size).expand(2,-1)
        uv=mask_indices_to_satellite_uv(idx,size,size,boxes)
        local=torch.stack(((idx%size+.5)/size,(idx//size+.5)/size),-1)
        assert torch.equal(uv,boxes[:,None,:2]+local*(boxes[:,None,2:]-boxes[:,None,:2]))
        shifted=mask_indices_to_satellite_uv(idx,size,size,boxes+torch.tensor([5.,7.,5.,7.]))
        assert torch.allclose(shifted,uv+torch.tensor([5.,7.]),atol=3e-5)
        scaled=mask_indices_to_satellite_uv(idx,size,size,boxes*1.25)
        assert torch.allclose(scaled,uv*1.25,atol=3e-5)
    # Independent target-crop convention: a linear x/y image must crop to the
    # same bin centers using the aligned RoIAlign used by mask targets.
    from mmcv.ops import roi_align
    yy,xx=torch.meshgrid(torch.arange(512)+.5,torch.arange(512)+.5,indexing='ij')
    ramp=torch.stack([xx,yy])[None]
    rois=torch.cat([torch.zeros(2,1),boxes],1)
    cropped=roi_align(ramp,rois,(128,128),1.,1,'avg',True)
    uv=mask_indices_to_satellite_uv(torch.arange(16384).expand(2,-1),128,128,boxes)
    assert torch.allclose(cropped.flatten(2).transpose(1,2),uv,atol=6e-5)
    a=UDPRPointAdapter.__new__(UDPRPointAdapter);nn.Module.__init__(a)
    a.enabled=True;a.observe=False;a.point_calls=0;a.boxes=boxes
    for name,di,do in [('point_query',32,32),('key',32,32),('value',32,32),('geom_key',6,32),('geom_value',6,32),('output',32,32)]:
        setattr(a,name,nn.Linear(di,do))
    a.alpha=nn.Parameter(torch.ones(()),requires_grad=False)
    f=torch.randn(81920,32,requires_grad=True)
    index=torch.full((16384,8),-1,dtype=torch.long);index[::2,0]=torch.arange(8192)
    index[::2,1]=torch.arange(8192)+16384
    geom=torch.zeros(16384,8,6);a.context=(f,index,geom,None)
    tail=DecoderTailPointRefiner(num_points=64)
    # Nonzero original MLP ensures feature-path gradients are testable.
    nn.init.normal_(tail.mlp[-1].weight,std=.01)
    logits=torch.randn(2,1,128,128);high=torch.randn(2,32,128,128);tokens=torch.randn(2,1,256)
    nn.init.zeros_(a.output.weight);nn.init.zeros_(a.output.bias)
    unchanged=a._point_hook(tail,(logits,high,tokens))
    assert torch.equal(unchanged[1],high)
    nn.init.normal_(a.output.weight,std=.01);nn.init.constant_(a.output.bias,.02)
    changed=a._point_hook(tail,(logits,high,tokens))
    assert changed[0] is logits and changed[2] is tokens
    selected,_=tail._select(logits)
    selected_mask=torch.zeros(2,16384,dtype=torch.bool).scatter_(1,selected,True)
    delta=(changed[1]-high).flatten(2).transpose(1,2)
    assert not delta[~selected_mask].any() and delta[selected_mask].abs().max()>0
    assert torch.equal(tail._select(changed[0])[0],selected)
    refined,_=tail.forward(*changed);refined.square().mean().backward()
    assert f.grad is not None and f.grad.abs().max()>0
    assert a.point_query.weight.grad.abs().max()>0
    a.context=(f,torch.full_like(index,-1),geom,None)
    unsupported=a._point_hook(tail,(logits,high,tokens))
    assert torch.equal(unsupported[1],high)
    print(json.dumps(dict(passed=True,full_frame_and_roi_sizes=[128,256],translation_scale_and_target_crop_verified=True,zero_projection_exact=True,
        no_support_exact_with_nonzero_bias=True,changes_only_selected_points=True,raw_logits_and_tokens_identity=True,
        selected_indices_exact=True,source_and_query_gradient_nonzero=True)))


if __name__=='__main__':main()
