"""Analytical candidate grouping, neighborhood layout and J1 gradient closure."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from rsprompter.uav.joint_encoding_probe import JointEncodingAdapter,group_four_cells,positional_xy


def main():
    torch.set_num_threads(2);torch.manual_seed(45)
    ids=torch.arange(16384*8).reshape(16384,8)
    grouped=group_four_cells(ids)
    for y,x in [(0,0),(0,63),(31,17),(63,63)]:
        expected=torch.cat([ids[(2*y+dy)*128+2*x+dx] for dy in range(2) for dx in range(2)])
        assert torch.equal(grouped[y*64+x],expected)
    # Build the exact interaction without loading an unrelated source backbone.
    a=JointEncodingAdapter.__new__(JointEncodingAdapter);nn.Module.__init__(a)
    a.satellite=nn.Linear(224,32)
    for name in ('reverse_q','reverse_k','reverse_v','reverse_out','query','key','value'):
        setattr(a,name,nn.Linear(32,32))
    a.geom_key=nn.Linear(6,32,bias=False);a.geom_value=nn.Linear(6,32,bias=False);a.output=nn.Linear(32,224)
    xy=torch.tensor([[x,y] for y in (-1.,0.,1.) for x in (-1.,0.,1.)]);a.neighbor_position=positional_xy(xy)
    xy=torch.tensor([[x,y] for y in (-.5,.5) for x in (-.5,.5)]).repeat_interleave(8,0);a.child_position=positional_xy(xy)
    u=torch.randn(4096,32,32,requires_grad=True);g=torch.randn(4096,32,6)
    valid=torch.ones(4096,32,dtype=torch.bool);valid[0]=False
    a.context=(u,valid,g);base=torch.randn(1,64,64,224)
    nn.init.zeros_(a.output.weight);nn.init.zeros_(a.output.bias)
    d,_=a.interact(base);assert torch.equal(base+d,base)
    nn.init.normal_(a.output.weight,std=.01);nn.init.constant_(a.output.bias,.02)
    d,_=a.interact(base);assert not d[0,0,0].any()
    d.square().mean().backward()
    for key in ['reverse_q','reverse_k','reverse_v','reverse_out','query','key','value']:
        assert getattr(a,key).weight.grad.abs().max()>0,key
    assert u.grad.abs().max()>0
    a.context=(u,torch.zeros_like(valid),g);d,_=a.interact(base);assert not d.any()
    print(json.dumps(dict(passed=True,candidate_order='dy dx candidate; 4x8=32',
        target_shape=[64,64,224],zero_output_exact=True,nonzero_bias_no_support_exact=True,
        forward_reverse_and_source_gradient_nonzero=True)))


if __name__=='__main__':main()
