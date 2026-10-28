"""Leakage and reader contracts for independently trained view-count studies."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
import torch
from torch import nn
from rsprompter.uav.local_adapter_probe import YAWS
from rsprompter.uav.geometry_inputs import candidate_arrays, native_arrays
from rsprompter.uav.view_subset import (canonical_views, subset_key, subset_arrays,
                                      check_index_views, source_tokens, configure_upstream_views)
from rsprompter.uav.ubr_native_integration import NativeEvidenceG


def cameras():
    camera = dict(intrinsics=dict(fx=256,fy=256,cx=255.5,cy=255.5,width=512,height=512),
                  world_to_camera_cv=np.eye(4).tolist(),camera_to_world_cv=np.eye(4).tolist())
    return dict(center_ue_cm=[0,0,0],satellite=camera,uav={'yaw_%03d'%v:camera for v in YAWS})


def depths():
    d=np.full((5,512,512),np.nan)
    # Two source tokens per target cell are not generally possible under this
    # identity projection; sparse samples still verify independent support/Z/count.
    d[:,1,1]=np.arange(5)+10
    d[:,9,9]=20
    d[4,17,17]=30
    return d


def test_ids_are_explicit_canonical_not_arbitrary_prefix():
    assert canonical_views([288,0,144])==(0,144,288)
    assert subset_key([144,0])=='views_000_144'
    for v in ([],[0,0],[1],[False],[0.,72]):
        with pytest.raises(ValueError):canonical_views(v)


def test_five_view_products_equal_original_and_removed_depth_cannot_leak():
    c=cameras();d=depths()
    five=subset_arrays(c,d,YAWS)
    old=candidate_arrays(c,d);fields,support=native_arrays(c,d)
    for name,value in zip(('index','geom','anchor'),old):assert np.array_equal(five[name],value)
    assert np.array_equal(five['fields'],fields) and np.array_equal(five['support'],support)
    one=subset_arrays(c,d,[0]);changed=d.copy();changed[1:]=777
    replay=subset_arrays(c,changed,[0])
    assert all(np.array_equal(one[k],replay[k]) for k in one)
    assert not one['support'][17,17] and five['support'][17,17]
    # Fixed /5 scaling: one actual observer is .2 in every subset, not 1.
    assert one['fields'][3,1,1]==np.float16(.2)
    assert np.all(one['index'][one['index']>=0]//16384==0)


def test_rebuild_recovers_candidate_previously_removed_by_global_cap():
    c=cameras()
    # SAT fx/ fy .5 compresses source pixels: two distinct tokens/view land in
    # the same target cell. Full5 top8 excludes fifth view's rank1 candidate.
    c['satellite']={**c['satellite'],'intrinsics':{**c['satellite']['intrinsics'],'fx':.5,'fy':.5}}
    d=np.full((5,512,512),np.nan);d[:,1,1]=10;d[:,1,9]=11
    five=subset_arrays(c,d,YAWS);single=subset_arrays(c,d,[288])
    before=five['index'][five['index']>=0];after=single['index'][single['index']>=0]
    assert len(before)==8 and len(after)==2
    assert set(after)-set(before)  # Cannot obtain this by filtering the old top8.


def test_candidate_slots_reject_removed_views():
    ids=torch.full((2,8),-1,dtype=torch.long);ids[0,0]=4*16384
    check_index_views(ids,[288])
    with pytest.raises(ValueError,match='removed'):check_index_views(ids,[0])


class FakeEncoder(nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,image):
        self.calls.append(image.detach().mean().item())
        value=image.mean().expand(1,256,128,128)
        return value,(value,)


def test_only_selected_RGB_is_encoded_and_has_gradients():
    encoder=FakeEncoder()
    projection=nn.Conv2d(256,32,1)
    adapter=SimpleNamespace(rgb_encoder=encoder,rgb_project=projection,rgb_mean=0,rgb_std=1)
    rgb=torch.stack([torch.full((3,512,512),float(i+1)) for i in range(5)]).requires_grad_()
    result=source_tokens(adapter,rgb,[0,144,288])
    assert encoder.calls==[1,3,5] and result.shape==(5*16384,32)
    assert not result[16384:32768].any() and not result[3*16384:4*16384].any()
    result.square().mean().backward()
    assert not rgb.grad[1].any() and not rgb.grad[3].any()
    assert rgb.grad[0].abs().sum()>0


def test_upstream_configuration_changes_no_parameters_or_hooks():
    adapter=NativeEvidenceG();adapter.variant='detail'
    before={k:id(v) for k,v in adapter.named_parameters()};handles=list(adapter.handles)
    configure_upstream_views(adapter,[0,144,288])
    assert before=={k:id(v) for k,v in adapter.named_parameters()} and adapter.handles==handles
    assert adapter.view_ids==(0,144,288)


def test_upstream_subset_encode_matches_original_for_same_retained_anchors():
    import copy
    adapter=NativeEvidenceG();adapter.variant='detail'
    adapter.rgb_encoder=FakeEncoder();adapter.rgb_project=nn.Conv2d(256,32,1)
    adapter.register_buffer('rgb_mean',torch.zeros(1,3,1,1));adapter.register_buffer('rgb_std',torch.ones(1,3,1,1))
    rgb=torch.randn(5,3,512,512)
    index=torch.full((16384,8),-1,dtype=torch.long);index[0,0]=0
    anchor=torch.zeros(16384,8,6,dtype=torch.float64)
    adapter.rgb_context=(rgb,index,anchor);adapter.candidate_geometry=torch.zeros(16384,8,7)
    reduced=copy.deepcopy(adapter);configure_upstream_views(reduced,[0])
    geometry=torch.zeros(1,5,512,512)
    with torch.no_grad():
        adapter.encode(geometry);reduced.encode(geometry)
        for a,b in zip(adapter.read_context,reduced.read_context):assert torch.equal(a,b)
    assert len(adapter.rgb_encoder.calls)==5 and len(reduced.rgb_encoder.calls)==1
