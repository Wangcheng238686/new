"""Bounded geometry-anchored RGB-D reading and RoI-local residual study.

Mechanism borrowing, not an S2DB-Net reproduction. Satellite modules are not owned
or unfrozen here. Local offsets sample appearance context, not new 3D anchors.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .local_adapter_variants import UnitLocalUAVAdapter


class S2DBAdapter(UnitLocalUAVAdapter):
    def __init__(self,pretrained,read='fixed',fusion='normalized',content='correct',aux_weight=0.):
        super().__init__(pretrained)
        assert read in ('fixed','local') and fusion in ('normalized','concat')
        assert content in ('correct','constant','zero')
        self.read,self.fusion,self.content,self.aux_weight=read,fusion,content,aux_weight
        # All arms instantiate every extra module with an isolated initialization
        # stream. Neither extras nor absent interventions perturb LoRA dropout RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026102201)
            self.offset=nn.Sequential(nn.Linear(70,32),nn.GELU(),nn.Linear(32,2))
            nn.init.zeros_(self.offset[-1].weight);nn.init.zeros_(self.offset[-1].bias)
            self.joint=nn.Sequential(nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,32,1))
            self.aux_head=nn.Sequential(nn.Conv2d(32,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        for p in self.offset.parameters():p.requires_grad_(read=='local')
        for p in self.joint.parameters():p.requires_grad_(fusion=='concat')
        for p in self.output.parameters():p.requires_grad_(fusion=='normalized')
        for p in self.aux_head.parameters():p.requires_grad_(aux_weight>0)
        self.observe=False
        self.observation={}
        self.aux_logits=None
        self.support=None

    def encode(self,rgb,index,geom,anchor,depth,support,satellite_high):
        super().encode(rgb,index,geom,satellite_high)
        self.anchor,self.depth,self.support=anchor,depth,support
        f,_,_,high=self.context
        q=self.query(high)
        self.prepared=(q,*self._candidates(f,index,geom,q))
        self.aux_logits=None
        if self.aux_weight>0:
            # Recompute the read path with detached query, including offsets.
            # Merely detaching the final attention query leaks through offsets.
            aq=q.detach()
            keys,values=self._candidates(f,index,geom,aq,observe=False)
            query=aq.permute(0,2,3,1).reshape(16384,32)
            pooled=self._pool(query,keys,values,index>=0)
            if self.content=='zero':pooled=pooled*0
            self.aux_logits=self.aux_head(pooled.T.reshape(1,32,128,128).to(q.dtype))
        self.stats={}

    def _candidates(self,f,index,geom,q,observe=True):
        valid=index>=0
        safe=index.clamp_min(0)
        sampled=f[safe]
        if self.read=='local':
            query=q.permute(0,2,3,1).reshape(16384,32)[:,None,:].expand(-1,8,-1)
            offset=self.offset(torch.cat((query.to(sampled.dtype),sampled,geom.to(sampled.dtype)),-1)).float().tanh()
            # Token centers exactly reproduce gather at offset=0 (binary-exact grids).
            token=safe%16384
            xy=torch.stack((token%128,token//128),-1).float()+.5+offset
            source=f.reshape(5,128,128,32).permute(0,3,1,2)
            flat=valid.flatten().nonzero().flatten()
            view=(safe.flatten()[flat]//16384)
            gathered=[];locations=[];accepts=[];jumps=[]
            for v in range(5):
                take=flat[view==v]
                if not take.numel():continue
                grid=(xy.reshape(-1,2)[take]/64-1).view(1,-1,1,2)
                feature=F.grid_sample(source[v:v+1].float(),grid,align_corners=False).squeeze(0).squeeze(-1).T
                uv=self.anchor.reshape(-1,6)[take,:2]+4*offset.reshape(-1,2)[take]
                dg=((uv+.5)/256-1).view(1,-1,1,2)
                raw=self.depth[v:v+1,None]
                dv=torch.isfinite(raw)&(raw>0)&(raw<10000)
                cleaned=torch.where(dv,raw,torch.zeros_like(raw))
                dz=F.grid_sample(cleaned,dg,align_corners=False).flatten()
                coverage=F.grid_sample(dv.float(),dg,align_corners=False).flatten()
                source_xy=xy.reshape(-1,2)[take]
                ok=(uv>=0).all(-1)&(uv<=511).all(-1)&(source_xy>=.5).all(-1)&(source_xy<=127.5).all(-1)
                ok=ok&(coverage>=1-1e-6)&torch.isfinite(feature).all(-1)
                anchor_z=self.anchor.reshape(-1,6)[take,2]
                jump=(dz-anchor_z).abs()>torch.maximum(torch.ones_like(anchor_z),.05*anchor_z)
                feature=torch.where(ok[:,None],feature,sampled.reshape(-1,32)[take])
                gathered.append(feature);locations.append(take);accepts.append(ok);jumps.append(jump&ok)
            if locations:
                loc=torch.cat(locations)
                sampled=sampled.reshape(-1,32).index_copy(0,loc,torch.cat(gathered)).reshape(16384,8,32)
            if self.observe and observe:
                off=offset[valid].detach();n=int(valid.sum())
                self.observation=dict(valid_anchors=n,offset_unit='source feature pixel; stride 4 RGB pixels',
                    offset_mean_abs=off.abs().mean(0).tolist() if n else [None,None],
                    offset_p95_abs=torch.quantile(off.abs(),.95,dim=0).tolist() if n else [None,None],
                    axis_at_95pct_count=(off.abs()>=.95).sum(0).tolist(),
                    accepted=int(torch.cat(accepts).sum()) if accepts else 0,
                    fallback=n-int(torch.cat(accepts).sum()) if accepts else 0,
                    depth_jump_accepted=int(torch.cat(jumps).sum()) if jumps else 0,
                    depth_jump_rule='abs(z_shift-z_anchor)>max(1m,.05*z_anchor); not a rejection criterion')
        return self.key(sampled)+self.geom_key(geom),self.value(sampled)+self.geom_value(geom)

    @staticmethod
    def _pool(query,keys,values,valid):
        logits=(F.normalize(query.float(),dim=-1)[...,None,:]*F.normalize(keys.float(),dim=-1)).sum(-1)*4
        weights=logits.masked_fill(~valid,-1e4).softmax(-1)*valid
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-6)
        return (weights[...,None]*values).sum(-2)

    def _chunk(self,qimage,keys,values,index,boxes,size):
        lin=(torch.arange(size,device=boxes.device,dtype=torch.float32)+.5)/size
        yy,xx=torch.meshgrid(lin,lin,indexing='ij')
        uv=boxes[:,None,None,:2]+torch.stack((xx,yy),-1)[None]*(boxes[:,None,None,2:]-boxes[:,None,None,:2])
        q=F.grid_sample(qimage.expand(len(boxes),-1,-1,-1),(uv/256-1).to(qimage.dtype),align_corners=False).permute(0,2,3,1)
        cell=(uv[...,1]//4).long().clamp(0,127)*128+(uv[...,0]//4).long().clamp(0,127)
        valid=(index[cell]>=0)&((uv>=0)&(uv<512)).all(-1)[...,None]
        pooled=self._pool(q,keys[cell],values[cell],valid)
        if self.content=='zero':pooled=pooled*0
        if self.fusion=='normalized':
            delta=self.output(pooled.to(q.dtype))
        else:
            joint=torch.cat((F.layer_norm(q.float(),(32,)),F.layer_norm(pooled.float(),(32,))),-1)
            delta=self.joint(joint.permute(0,3,1,2).to(q.dtype)).permute(0,2,3,1)
            delta=F.layer_norm(delta.float(),(32,))
        # Mask after affine-free normalization; no query-only unsupported residual.
        return (delta*valid.any(-1)[...,None]).permute(0,3,1,2).to(qimage.dtype)

    def _inject(self,module,args,base):
        if not self.enabled or self.context is None:return base
        q,k,v=self.prepared
        index=self.context[1]
        chunks=[]
        for boxes in self.boxes.split(4):
            def fn(qi,kk,vv,ix,bb):return self._chunk(qi,kk,vv,ix,bb,base.shape[-1])
            if self.training and torch.is_grad_enabled():d=checkpoint(fn,q,k,v,index,boxes,use_reentrant=False)
            else:d=fn(q,k,v,index,boxes)
            chunks.append(d)
        delta=torch.cat(chunks) if chunks else torch.zeros_like(base)
        applied=(self.alpha*delta).to(base.dtype)
        with torch.no_grad():
            self.stats=dict(alpha=float(self.alpha),support_fraction=float((index>=0).any(-1).float().mean()),
                source_delta_rms=float(delta.float().square().mean().sqrt()),
                applied_delta_rms=float(applied.float().square().mean().sqrt()),
                base_rms=float(base.float().square().mean().sqrt()),rois=len(base))
        return base+applied

    def auxiliary_loss(self,target):
        assert self.aux_logits is not None
        logits=F.interpolate(self.aux_logits.float(),(512,512),mode='bilinear',align_corners=False)[0,0]
        valid=self.support.bool()
        count=valid.sum()
        loss=(F.binary_cross_entropy_with_logits(logits,target.float(),reduction='none')*valid).sum()/count.clamp_min(1)
        return loss,count

    def clear(self):
        self.context=None;self.prepared=None;self.aux_logits=None
        self.anchor=None;self.depth=None;self.support=None
