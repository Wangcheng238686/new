"""Geometry-conditioned bounded RPN objectness residual; original heads stay frozen."""
import weakref
import torch
from torch import nn
from torch.nn import functional as F


class RPNGeometryAdapter(nn.Module):
    def __init__(self,hidden_channels=512,anchors=3):
        super().__init__()
        self.query=nn.Conv2d(hidden_channels,32,1)
        self.mixer=nn.Sequential(nn.Conv2d(37,32,1),nn.GELU(),nn.Conv2d(32,anchors,1))
        self.alpha=nn.Parameter(torch.zeros(()))
        self.enabled=True
        self.context=None
        self.handles=[]
        self.stats=[]
        self.anchors=anchors

    def attach(self,model):
        assert not self.handles
        rpn=model.rpn_head
        assert rpn.use_sigmoid_cls and rpn.cls_out_channels==1
        assert all(a==self.anchors for a in rpn.prior_generator.num_base_priors)
        self._rpn_ref=weakref.ref(rpn)
        self.handles=[rpn.register_forward_pre_hook(self._begin),rpn.rpn_cls.register_forward_hook(self._inject)]

    def encode(self,unused_rgb,support,fields,unused_high=None):
        assert support.shape==(1,1,512,512) and fields.shape==(1,4,512,512)
        self.context=(support,fields)

    def _begin(self,module,args):
        self.level=0;self.stats=[]
        x=args[0]
        self.priors=module.prior_generator.grid_priors([f.shape[-2:] for f in x],dtype=torch.float32,device=x[0].device)

    def sample(self,prior,shape):
        h,w=shape
        centers=((prior[:,:2]+prior[:,2:])/2).reshape(h,w,self.anchors,2)
        assert torch.allclose(centers,centers[:,:,:1].expand_as(centers),atol=1e-4,rtol=0)
        uv=centers[:,:,0]
        # Native cache uses floor-binned satellite pixel indices. Current anchor
        # centers are integer indices; nearest sampling reads that exact bin.
        assert torch.allclose(uv,uv.round(),atol=1e-4,rtol=0)
        grid=((uv+.5)/256-1)[None]
        support,fields=self.context
        valid=F.grid_sample(support.float(),grid,mode='nearest',align_corners=False)>0
        geo=F.grid_sample(fields.float(),grid,mode='nearest',align_corners=False)
        return geo,valid

    def _inject(self,module,args,base):
        level=self.level;self.level+=1
        if not self.enabled or self.context is None:return base
        assert base.shape[0]==1 and base.shape[1]==self.anchors
        geo,valid=self.sample(self.priors[level],base.shape[-2:])
        q=self.query(args[0].detach())
        q=F.layer_norm(q.float().permute(0,2,3,1),(32,)).permute(0,3,1,2).to(q.dtype)
        raw=self.mixer(torch.cat((q,geo.to(q.dtype),valid.to(q.dtype)),1))
        delta=(2*self.alpha.tanh()*raw.float().tanh()*valid).to(base.dtype)
        with torch.no_grad():
            self.stats.append(dict(level=level,locations=valid.numel(),supported=int(valid.sum()),
                elements=base.numel(),delta_square_sum=float(delta.float().square().sum()),
                base_square_sum=float(base.float().square().sum()),alpha=float(self.alpha)))
        return base+delta

    def clear(self):
        self.context=None;self.priors=None
