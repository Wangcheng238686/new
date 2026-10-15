"""UNEXECUTED engineering sketch. Root review required before any GPU use.
No model construction, fit, or CLI. Existing strict EMA extractor owns lifecycle.
"""
import math
import torch
import torch.nn.functional as F

def candidate_uniform_query(tokens, payload, chunk=256):
    """Equal weight per valid production candidate slot; not GT-selected views.
    Tokens are DepthCVT.source output [5*64*64,256], after Conv/GN/SiLU.
    No anchor position/view embedding/query learned bias/relative bias is added.
    Distinct anchor slots may share visual token identity; preserve production
    candidate definition and do not deduplicate based on a new heuristic.
    """
    indices=payload.candidate_index
    if not len(payload.anchor_token):
        assert not bool((indices>=0).any())
        return tokens.new_zeros(1,256,128,128)
    output=[]
    for start in range(0,len(indices),chunk):
        ids=indices[start:start+chunk]
        valid=ids>=0
        token_ids=payload.anchor_token[ids.clamp_min(0)]
        value=tokens[token_ids]*valid[...,None]
        output.append(value.sum(1)/valid.sum(1).clamp_min(1)[:,None])
    query=torch.cat(output)
    assert torch.isfinite(query).all()
    assert bool(query[(indices>=0).any(1).logical_not()].eq(0).all())
    return query.T.reshape(1,256,128,128)

class StageCapture:
    """Read-only hooks, return None; caller verifies on/off and params unchanged."""
    def __init__(self,model):self.model=model;self.record={}
    def first(self,module,args):
        query,tokens,payload,chunk,checkpoint=args
        assert not module.training and tokens.shape==(20480,256)
        self.record['uniform_bev']=candidate_uniform_query(tokens,payload)
        # Do not retain all-token state or create alternative per-view reads.
        self.record['payload']=payload
    def last(self,module,args,output):
        self.record['cvt_bev']=output
        assert output.shape==(1,256,128,128)
    def __enter__(self):
        self.handles=[self.model.uav_bev.layers[0].register_forward_pre_hook(self.first),self.model.uav_bev.register_forward_hook(self.last)]
        return self
    def __exit__(self,*unused):
        for h in self.handles:h.remove()

def same_projection_and_roi(model,stage_capture,rois,roi_operator_16,rp_matrix):
    """Both stages consume same payload, sampler, weights and 256->32 matrix.
    roi_operator_16 must be explicitly configured spatial_scale=1/16 and match
    production aligned/sampling_ratio; do not reuse stride4 operator by mistake.
    Output-size16, same sampling/aligned flags in both arms.
    """
    assert rp_matrix.shape==(256,32)
    payload=stage_capture.record['payload'];result={};common=None
    for name in ('uniform_bev','cvt_bev'):
        image,q,c=model.bev_projection(stage_capture.record[name],[payload],(32,32))
        if common is None:common=(q,c)
        else:assert torch.equal(q,common[0]) and torch.equal(c,common[1])
        aligned=roi_operator_16(image,rois)
        shifted=roi_operator_16(image.roll((8,8),(-2,-1)),rois)
        for suffix,v in [('aligned',aligned),('shift',shifted)]:
            # Must be same resampling for the two arms and their shift controls.
            assert v.shape[-2:]==(16,16)
            x=v.float()
            full=x.permute(0,2,3,1).reshape(len(rois),256,256)
            result[name+'_'+suffix+'32']=full@rp_matrix.to(full)
            if suffix=='aligned':result[name+'_full256_for_train_covariance']=full
    result['quality'],result['coverage']=common
    return result

class StreamingCovariance:
    """Train-only uncentered second moments; no PCA feature fit or val selection."""
    def __init__(self,c=256):
        self.n=0;self.total=torch.zeros(c,dtype=torch.float64);self.cross=torch.zeros(c,c,dtype=torch.float64)
    def add(self,x):
        x=x.detach().reshape(-1,x.shape[-1]).double().cpu()
        self.n+=len(x);self.total+=x.sum(0);self.cross+=x.T@x
    def finish(self):
        mean=self.total/max(self.n,1);cov=self.cross/max(self.n,1)-mean[:,None]*mean[None,:]
        eig=torch.linalg.eigvalsh((cov+cov.T)/2).clamp_min(0);p=eig/eig.sum().clamp_min(1e-30)
        return dict(n=self.n,mean=mean.tolist(),std=cov.diag().clamp_min(0).sqrt().tolist(),covariance_effective_rank=float(torch.exp(-(p*p.clamp_min(1e-30).log()).sum())),eigenvalues=eig.tolist())
