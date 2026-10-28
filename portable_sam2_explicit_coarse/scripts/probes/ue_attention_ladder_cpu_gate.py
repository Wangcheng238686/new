"""Analytic CPU gates for L0--L6; no encoder download or CUDA allocation."""
import argparse
import json
import sys
import types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

import torch
from torch import nn
from torch.nn import functional as F
from rsprompter.uav.local_adapter_variants import UnitLocalUAVAdapter
from rsprompter.uav.attention_ladder import (
    AttentionLadderAdapter, source_uv_grid, sample_source_continuous,
    near_surface_bias, masked_weights)


def run():
    torch.set_num_threads(1)
    # Replace only the constructor's heavyweight encoder import; all adapter
    # initialization and forward mathematics remain real production code.
    fake=types.ModuleType('rsprompter.backbone.sam2_vision')
    class Encoder(nn.Module):
        def __init__(self,**kwargs):
            super().__init__()
            self.fake=nn.Linear(2,2)
    fake.RSSAM2VisionEncoder=Encoder
    previous=sys.modules.get(fake.__name__)
    sys.modules[fake.__name__]=fake
    try:
        torch.manual_seed(44)
        base=UnitLocalUAVAdapter('unused')
        base_rng=torch.get_rng_state().clone()
        models={}
        for direction in ('L0','L1','L2','L3','L4','L5','L6'):
            torch.manual_seed(44)
            models[direction]=AttentionLadderAdapter('unused',direction)
            assert torch.equal(base_rng,torch.get_rng_state()),'extra module changed common RNG'
    finally:
        if previous is None:
            sys.modules.pop(fake.__name__,None)
        else:
            sys.modules[fake.__name__]=previous
    baseline=base.state_dict()
    unified=models['L0'].state_dict()
    for direction,m in models.items():
        assert set(unified)==set(m.state_dict())
        m.load_state_dict(unified,strict=True)
        assert all(torch.equal(value,m.state_dict()[name]) for name,value in baseline.items())
        assert m.view_output.weight.requires_grad == (direction=='L2')
        assert m.null_logit.requires_grad == (direction=='L3')
        assert m.geometry_score.weight.requires_grad == (direction=='L4')
        assert m.geom_key.weight.requires_grad == (direction!='L4')
        assert m.output[0].weight.requires_grad == (direction!='L2')

    torch.manual_seed(81)
    q=torch.randn(1,32,128,128)
    k=torch.randn(16384,8,32)
    v=torch.randn_like(k)
    ix=torch.full((16384,8),-1,dtype=torch.long)
    ix[:,:3]=torch.tensor([0,16384,32768])
    ix[0]=-1
    boxes=torch.tensor([[0.,0.,32.,32.],[-32.,-32.,32.,32.],[480.,480.,512.,512.]])
    expected=base._chunk(q,k,v,ix,boxes,8)
    assert torch.equal(expected,models['L0']._chunk(q,k,v,ix,boxes,8))
    assert torch.equal(expected,models['L6']._chunk(q,k,v,ix,boxes,8))
    for direction,m in models.items():
        m.candidate_bias=torch.zeros(16384,8)
        empty=m._chunk(q,k,v,torch.full_like(ix,-1),boxes,8)
        assert torch.isfinite(empty).all() and torch.count_nonzero(empty)==0,direction
        empty=m._chunk(q,k,v,ix,torch.tensor([[-100.,-100.,-50.,-50.]]),8)
        assert torch.count_nonzero(empty)==0,direction

    # Real Hiera PatchEmbed with a center-delta kernel establishes the lattice,
    # independently of grid_sample: output i samples input integer center 4i.
    from sam2.modeling.backbones.utils import PatchEmbed
    patch=PatchEmbed(in_chans=1,embed_dim=1)
    assert patch.proj.kernel_size==(7,7) and patch.proj.stride==(4,4) and patch.proj.padding==(3,3)
    with torch.no_grad():
        patch.proj.weight.zero_();patch.proj.bias.zero_()
        patch.proj.weight[0,0,3,3]=1.
    iy,ixx=torch.meshgrid(torch.arange(512.),torch.arange(512.),indexing='ij')
    image_ramp=3*ixx+7*iy
    lattice=patch(image_ramp[None,None])[0,:,:,0]
    yy,xx=torch.meshgrid(torch.arange(128)*4.,torch.arange(128)*4.,indexing='ij')
    assert torch.equal(lattice,3*xx+7*yy)
    maps=[]
    for view in range(5):
        maps.append((lattice+10000*view)[...,None].expand(128,128,32))
    features=torch.stack(maps).reshape(-1,32)
    source_uv=torch.tensor([[1.,1.],[31.,55.],[511.,511.],[100.,200.],[200.,100.]])
    anchor=torch.zeros(1,8,6)
    anchor[0,:5,:2]=source_uv
    index=torch.full((1,8),-1,dtype=torch.long)
    index[0,:5]=torch.arange(5)*16384+(source_uv[:,1].long()//4)*128+source_uv[:,0].long()//4
    sampled=sample_source_continuous(features,index,anchor)
    centers=source_uv.clamp(0,508)
    analytic=3*centers[:,0]+7*centers[:,1]+10000*torch.arange(5)
    assert torch.allclose(sampled[0,:5,0],analytic,atol=.02,rtol=0)
    assert torch.count_nonzero(sampled[0,5:])==0
    # At the actual token centers u=4i, the continuous read is exact token i.
    anchor[0,:5,:2]=torch.tensor([[0.,0.],[4.,8.],[508.,508.],[80.,48.],[12.,92.]])
    token_xy=(anchor[0,:5,:2]/4).long()
    index[0,:5]=torch.arange(5)*16384+token_xy[:,1]*128+token_xy[:,0]
    features=features.detach().requires_grad_(True)
    sampled=sample_source_continuous(features,index,anchor)
    assert torch.equal(sampled[0,:5],features[index[0,:5]])
    sampled.square().mean().backward()
    assert features.grad is not None and bool((features.grad!=0).any())
    assert torch.equal(source_uv_grid(torch.tensor([0.,508.])),torch.tensor([-127/128,127/128]))

    logits=torch.tensor([[.2,-.3,.8,.1,-.2,.9,-.4,.6]])
    valid=torch.tensor([[True,True,True,False,False,False,False,False]])
    ids=torch.tensor([[0,1,16384,-1,-1,-1,-1,-1]])
    values=torch.randn(1,8,32)
    # Exactly preserve per-view candidate identity and ordering, absent views 0.
    l2=models['L2']
    got=l2._aggregate(logits,values,valid,ids,torch.float32)
    per=[]
    for view in range(5):
        ww=masked_weights(logits,valid&(ids//16384==view))
        per.append((ww[...,None]*values).sum(-2))
    manual=l2.output[1](l2.view_output(torch.cat(per,-1)))
    assert torch.equal(got,manual)
    assert torch.count_nonzero(per[2])==0
    assert torch.equal(per[1],values[:,2])

    l3=models['L3']
    l3.observe=True
    with torch.no_grad():
        l3.null_logit.fill_(-100)
    accepted=l3._aggregate(logits,values,valid,ids,torch.float32)
    manual=l3.output((masked_weights(logits,valid)[...,None]*values).sum(-2))
    assert torch.equal(accepted,manual)
    with torch.no_grad():
        l3.null_logit.fill_(100)
    rejected=l3._aggregate(logits,values,valid,ids,torch.float32)
    assert float(rejected.detach().abs().max()) < 1e-30,'output LN cancelled rejection'
    assert l3.observation['null_weight_mean']==1.
    with torch.no_grad():
        l3.null_logit.zero_()
    result=l3._aggregate(logits,values,valid,ids,torch.float32)
    result.square().sum().backward()
    assert l3.null_logit.grad is not None and torch.isfinite(l3.null_logit.grad)
    assert float(l3.null_logit.grad.abs())>0

    f=torch.randn(5*16384,32)
    fullgeom=torch.randn(16384,8,6)
    l4=models['L4']
    l4k,l4v=l4._candidate_projections(f,ix,fullgeom)
    assert torch.equal(l4k,l4.key(f)[ix.clamp_min(0)])
    assert torch.equal(l4v,l4.value(f)[ix.clamp_min(0)]+l4.geom_value(fullgeom))
    assert torch.count_nonzero(l4.candidate_bias)==0
    l4.candidate_bias.sum().backward()
    assert l4.geometry_score.weight.grad is not None

    za=torch.zeros(3,8,6)
    zi=torch.full((3,8),-1,dtype=torch.long)
    zi[0,:4]=torch.tensor([0,16384,32768,49152])
    za[0,:4,5]=torch.tensor([10.,10.5,11.,13.])
    zi[2,0]=1;za[2,0,5]=77
    bias=near_surface_bias(zi,za)
    assert torch.equal(bias[0,:4],torch.tensor([0.,0.,0.,-2.]))
    assert torch.count_nonzero(bias[1:])==0 and torch.isfinite(bias).all()
    return dict(status='passed',device='cpu',directions=list(models),
        gates=['shared initialization and RNG','strict unified state and active parameters',
            'L0/L6 exact Unit chunk identity','all-empty and out-of-image zero',
            'L1 real PatchEmbed center-delta ramp; exact integer 4i token centers; border',
            'L2 view identity and missing-view zero','L3 actual residual rejection after LN',
            'L4 appearance-only K, geometry V retained, scalar bias gradient',
            'L5 satellite camera metre near-surface bias'],
        source_coordinate_contract='grid=(source_uv+2)/256-1; real Hiera 7/4/pad3 center lattice; align_corners=False; border',
        inactive_parameters={d:[n for n,p in m.named_parameters() if not p.requires_grad] for d,m in models.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output')
    args=parser.parse_args()
    result=run()
    if args.output:
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
