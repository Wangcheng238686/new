"""J1: local round-trip UAV/SAT interaction inside Hiera Base+ block4."""
import torch
from torch import nn
from torch.nn import functional as F
from .attention_ladder import sample_source_continuous


def group_four_cells(tensor):
    """[128*128,8,...] -> [64*64,32,...], dy then dx then candidate."""
    assert tensor.shape[:2] == (16384,8)
    rest = tensor.shape[2:]
    value = tensor.reshape(64,2,64,2,8,*rest)
    return value.permute(0,2,1,3,4,*range(5,value.ndim)).reshape(4096,32,*rest)


def positional_xy(xy):
    frequency = torch.exp(torch.linspace(0.,-9.210340372,8,device=xy.device))
    phases = xy[...,None]*frequency
    return torch.cat((phases[...,0,:].sin(),phases[...,0,:].cos(),
                      phases[...,1,:].sin(),phases[...,1,:].cos()),-1)


class JointEncodingAdapter(nn.Module):
    def __init__(self,pretrained):
        super().__init__()
        from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
        self.encoder=RSSAM2VisionEncoder(checkpoint_path=pretrained,model_size='base_plus',
            use_gradient_checkpointing=True,use_lora=True,freeze_backbone=False,
            lora_config=dict(r=16,lora_alpha=32,target_modules=['qkv','proj'],lora_dropout=.05,bias='none',modules_to_save=[]))
        self.source=nn.Conv2d(256,32,1)
        self.satellite=nn.Linear(224,32)
        for name in ('reverse_q','reverse_k','reverse_v','reverse_out','query','key','value'):
            setattr(self,name,nn.Linear(32,32))
        self.geom_key=nn.Linear(6,32,bias=False);self.geom_value=nn.Linear(6,32,bias=False)
        self.output=nn.Linear(32,224)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.register_buffer('mean',torch.tensor([123.675,116.28,103.53]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([58.395,57.12,57.375]).view(1,3,1,1))
        xy=torch.tensor([[x,y] for y in (-1.,0.,1.) for x in (-1.,0.,1.)])
        self.register_buffer('neighbor_position',positional_xy(xy))
        child=torch.tensor([[x,y] for y in (-.5,.5) for x in (-.5,.5)]).repeat_interleave(8,0)
        self.register_buffer('child_position',positional_xy(child))
        self.enabled=True;self.observe=False;self.context=None;self.stats={};self.observation={};self.handles=[];self.calls=0

    def attach(self,model):
        assert not self.handles
        trunks=[m for m in model.backbone.modules() if hasattr(m,'stage_ends') and hasattr(m,'blocks')]
        assert len(trunks)==1
        trunk=trunks[0]
        assert list(trunk.stage_ends)==[1,4,20,23] and len(trunk.blocks)==24
        assert trunk.blocks[4].dim_out==224
        self.handles.append(trunk.blocks[4].register_forward_hook(self._inject))

    def encode(self,rgb,index,geom,anchor):
        features=[]
        for image in rgb.split(1):
            _,pyramid=self.encoder((image-self.mean)/self.std)
            features.append(self.source(pyramid[0]))
        f=torch.cat(features).permute(0,2,3,1).reshape(-1,32)
        f=F.layer_norm(f,(32,))
        selected=sample_source_continuous(f,index,anchor)
        self.context=(group_four_cells(selected),group_four_cells(index>=0),group_four_cells(geom))
        self.stats={};self.observation={}

    def interact(self,base):
        assert base.shape==(1,64,64,224),base.shape
        u,valid,geom=self.context
        s=self.satellite(F.layer_norm(base,(224,))).reshape(4096,32)
        neighborhood=F.unfold(s.T.reshape(1,32,64,64),3,padding=1).reshape(32,9,4096).permute(2,1,0)
        neighbors=F.unfold(torch.ones(1,1,64,64,device=base.device),3,padding=1)[0].T>0
        rq=F.normalize(self.reverse_q(u).float(),dim=-1)
        rk=F.normalize((self.reverse_k(neighborhood)+self.neighbor_position).float(),dim=-1)
        logits=torch.bmm(rq,rk.transpose(1,2))*4
        weights=logits.masked_fill(~neighbors[:,None,:],-1e4).softmax(-1)
        reverse=torch.bmm(weights,self.reverse_v(neighborhood).float())
        updated=u+.2*self.reverse_out(reverse.to(u.dtype))
        k=self.key(updated)+self.geom_key(geom.to(u.dtype))+self.child_position
        v=self.value(updated)+self.geom_value(geom.to(u.dtype))
        q=F.normalize(self.query(s).float(),dim=-1)
        score=torch.bmm(F.normalize(k.float(),dim=-1),q[...,None]).squeeze(-1)*4
        attn=score.masked_fill(~valid,-1e4).softmax(-1)*valid
        attn=attn/attn.sum(-1,keepdim=True).clamp_min(1e-6)
        pooled=(attn[...,None]*v).sum(1)
        delta=self.output(pooled.to(s.dtype))*valid.any(-1)[:,None]
        return delta.reshape_as(base).to(base.dtype),valid

    def _inject(self,module,inputs,base):
        if not self.enabled or self.context is None:return base
        delta,valid=self.interact(base)
        self.calls+=1
        with torch.no_grad():
            self.stats=dict(target_support_fraction=float(valid.any(-1).float().mean()),
                source_delta_rms=float(delta.float().square().mean().sqrt()),
                base_stage_rms=float(base.float().square().mean().sqrt()),target_tokens=4096,candidates_per_token=32)
        return base+delta
