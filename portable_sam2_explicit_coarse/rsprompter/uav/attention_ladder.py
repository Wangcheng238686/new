"""One-change-at-a-time attention probes; the production model is untouched.

L0 and L6 exactly reuse UnitLocalUAVAdapter. L6 decoder adaptation belongs to
its runner. The anchor contract is integer pixel-center source u/v, metres
source-Z, satellite u/v, satellite-Z. The L1 feature coordinate contract treats
feature centers as integer source pixel indices 0,4,...,508: the actual
Hiera PatchEmbed is kernel7/stride4/padding3. Stage0 preserves this lattice,
and FPN0 is lateral-only stride1. Border padding extends the final feature
center to source indices 509..511; candidate identity/support is unchanged.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .local_adapter_variants import UnitLocalUAVAdapter


def source_uv_grid(uv):
    """Integer-center image indices -> normalized stride-4 feature coordinates."""
    return (uv + 2.) / 256. - 1.


def sample_source_continuous(features, index, anchor):
    """Sample normalized 5x128x128 token maps at unchanged retained anchors."""
    valid = index >= 0
    result = features.new_zeros((*index.shape, 32))
    maps = features.reshape(5,128,128,32).permute(0,3,1,2)
    # Each selected source token identifies the view; its original continuous
    # pixel anchor is retained by s2db_geometry. No resampling of the candidates.
    for view in range(5):
        take = valid & (index // 16384 == view)
        if not bool(take.any()):
            continue
        uv = anchor[take][:,:2]
        grid = source_uv_grid(uv).reshape(1,-1,1,2).to(maps.dtype)
        result[take] = F.grid_sample(maps[view:view+1],grid,
            mode='bilinear',padding_mode='border',align_corners=False)[0,:,:,0].T
    return result


def masked_weights(logits, valid):
    weight = logits.masked_fill(~valid,-1e4).softmax(-1) * valid
    return weight / weight.sum(-1,keepdim=True).clamp_min(1e-6)


def near_surface_bias(index, anchor, tau=1.):
    """Soft nearest-observed-surface bias in SAT camera metres, per target cell."""
    valid = index >= 0
    z = anchor[...,5].float()
    nearest = z.masked_fill(~valid,float('inf')).amin(-1,keepdim=True)
    nearest = torch.where(valid.any(-1,keepdim=True),nearest,torch.zeros_like(nearest))
    return (-F.relu(z-nearest-tau)/tau).masked_fill(~valid,0)


class AttentionLadderAdapter(UnitLocalUAVAdapter):
    def __init__(self,pretrained,direction='L0'):
        if direction not in tuple('L%d'%i for i in range(7)):
            raise ValueError(direction)
        super().__init__(pretrained)
        self.direction = direction
        self.anchor = None
        self.candidate_bias = None
        self.observe = False
        self.observation = {}
        # New modules must not consume the common/global RNG stream. All common
        # parameters have exactly the L0 initialization and state-dict names.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261022)
            self.view_output = nn.Linear(5*32,32)
            self.null_logit = nn.Parameter(torch.zeros(()))
            self.geometry_score = nn.Linear(6,1,bias=False)
            nn.init.zeros_(self.geometry_score.weight)
        self.view_output.requires_grad_(direction == 'L2')
        self.null_logit.requires_grad_(direction == 'L3')
        self.geometry_score.requires_grad_(direction == 'L4')
        # L2 replaces the old 32->32 output projection with 160->32, followed by
        # the identical affine-free LN; no orphan output parameters are trained.
        if direction == 'L2':
            self.output[0].requires_grad_(False)
        if direction == 'L4':
            self.geom_key.requires_grad_(False)

    def encode(self,rgb,index,geom,anchor,satellite_high):
        super().encode(rgb,index,geom,satellite_high)
        self.anchor = anchor
        self.candidate_bias = None
        self.observation = {}

    def _candidate_projections(self,f,index,geom):
        ids = index.clamp_min(0)
        if self.direction == 'L1':
            selected = sample_source_continuous(f,index,self.anchor)
            k = self.key(selected) + self.geom_key(geom)
            v = self.value(selected) + self.geom_value(geom)
        else:
            k = self.key(f)[ids]
            if self.direction != 'L4':
                k = k + self.geom_key(geom)
            v = self.value(f)[ids] + self.geom_value(geom)
        if self.direction == 'L4':
            self.candidate_bias = self.geometry_score(geom).squeeze(-1)
        elif self.direction == 'L5':
            self.candidate_bias = near_surface_bias(index,self.anchor,tau=1.)
        return k,v

    def _chunk(self,qimage,keys,values,index,boxes,size):
        if self.direction in ('L0','L1','L6'):
            return super()._chunk(qimage,keys,values,index,boxes,size)
        lin = (torch.arange(size,device=boxes.device,dtype=torch.float32)+.5)/size
        yy,xx = torch.meshgrid(lin,lin,indexing='ij')
        uv = boxes[:,None,None,:2]+torch.stack((xx,yy),-1)[None]*(boxes[:,None,None,2:]-boxes[:,None,None,:2])
        grid = uv/256-1
        q = F.grid_sample(qimage.expand(len(boxes),-1,-1,-1),grid.to(qimage.dtype),align_corners=False).permute(0,2,3,1)
        cell = ((uv[...,1]//4).long().clamp(0,127)*128+(uv[...,0]//4).long().clamp(0,127))
        ids = index[cell]
        valid = (ids>=0)&((uv>=0)&(uv<512)).all(-1)[...,None]
        k,v = keys[cell],values[cell]
        logits = (F.normalize(q.float(),dim=-1)[...,None,:]*F.normalize(k.float(),dim=-1)).sum(-1)*4
        if self.direction in ('L4','L5'):
            logits = logits+self.candidate_bias[cell].float()
        delta = self._aggregate(logits,v,valid,ids,q.dtype)
        return delta.permute(0,3,1,2).to(qimage.dtype)

    def _aggregate(self,logits,values,valid,ids,query_dtype):
        """Pure attention/output stage, also exercised by the CPU contract gate."""
        if self.direction == 'L2':
            per_view = []
            for view in range(5):
                weight = masked_weights(logits,valid&(ids//16384==view))
                per_view.append((weight[...,None]*values).sum(-2))
            pooled = torch.cat(per_view,-1).to(query_dtype)
            delta = self.output[1](self.view_output(pooled))
        elif self.direction == 'L3':
            # Factor conditional nonnull attention and its total probability.
            # Applying this mass AFTER output LN prevents LN cancelling rejection
            # and attenuates the output bias as well as the pooled value.
            masked = logits.masked_fill(~valid,-float('inf'))
            all_scores = torch.cat((masked,self.null_logit.float().expand(*logits.shape[:-1],1)),-1)
            probabilities = all_scores.softmax(-1)
            nonnull_mass = probabilities[...,:-1].sum(-1,keepdim=True)
            weight = masked_weights(logits,valid)
            pooled = (weight[...,None]*values).sum(-2)
            delta = self.output(pooled.to(query_dtype))*nonnull_mass
            if self.observe:
                with torch.no_grad():
                    p = probabilities[...,-1].detach().float()
                    self.observation = dict(null_weight_mean=float(p.mean()),
                        null_weight_supported_mean=float(p[valid.any(-1)].mean()) if bool(valid.any()) else None,
                        null_weight_max=float(p.max()),
                        scope='latest adapter chunk, may be overwritten by checkpoint recomputation')
        else:
            weight = masked_weights(logits,valid)
            pooled = (weight[...,None]*values).sum(-2)
            delta = self.output(pooled.to(query_dtype))
        return delta*valid.any(-1)[...,None]

    def _inject(self,module,args,base):
        if self.direction in ('L0','L6'):
            return super()._inject(module,args,base)
        if not self.enabled or self.context is None:
            return base
        f,index,geom,high = self.context
        k,v = self._candidate_projections(f,index,geom)
        qimage = self.query(high)
        chunks=[]
        for boxes in self.boxes.split(4):
            fn=lambda qi,kk,vv,ix,bb:self._chunk(qi,kk,vv,ix,bb,base.shape[-1])
            if self.training and torch.is_grad_enabled():
                delta=checkpoint(fn,qimage,k,v,index,boxes,use_reentrant=False)
            else:
                delta=fn(qimage,k,v,index,boxes)
            chunks.append(delta)
        delta=torch.cat(chunks)
        applied=self.alpha*delta
        with torch.no_grad():
            self.stats=dict(alpha=float(self.alpha),support_fraction=float((index>=0).any(-1).float().mean()),
                source_delta_rms=float(delta.float().square().mean().sqrt()),
                applied_delta_rms=float(applied.float().square().mean().sqrt()),
                base_rms=float(base.float().square().mean().sqrt()),rois=len(base))
        return base+applied
