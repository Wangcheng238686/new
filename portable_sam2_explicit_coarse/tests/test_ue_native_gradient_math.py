import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from ue_native_gradient_math_20261028 import direction,partition_losses,summarize


def test_global_direction_not_local_cosine_mean():
 a=torch.tensor([[1.,0.],[-1.,2.] ]);b=torch.tensor([[1.,0.],[1.,0.]])
 assert direction(a.mean(0),b.mean(0))['cosine']==0
 assert direction(a[0],b[0])['cosine']!=0


def test_zero_is_undefined_not_conflict():
 assert direction(torch.zeros(2),torch.ones(2))['cosine'] is None
 assert direction(torch.tensor([1.,2.]),torch.tensor([-1.,-2.]))['cosine']==pytest.approx(-1)


def test_union_includes_unilateral_gradient():
 named=[('satellite.neck.x',torch.nn.Parameter(torch.ones(1))),('satellite.neck.y',torch.nn.Parameter(torch.ones(1)))]
 v=torch.tensor([[1.,10.],[-1.,0.],[0.,0.] ]);support=torch.tensor([[1,1],[1,0],[0,0]],dtype=torch.bool)
 r=summarize(named,v,support)['fpn']
 assert r['cosine']==pytest.approx(-1/(101**.5))
 assert r['joint_connected_only']['cosine']==-1
 assert r['support']['both']['elements']==1 and r['support']['det_only']['elements']==1


def test_partition_preserves_existing_weights_and_rejects_new_loss():
 x=torch.tensor(2.,requires_grad=True)
 losses={k:x for k in ('loss_rpn_cls','loss_rpn_bbox','loss_cls','loss_bbox','loss_mask','loss_uav_instances')};losses['loss_shape_prior']=x*.1
 tasks,keys=partition_losses(losses);assert float(tasks['detection'].detach())==8;assert float(tasks['mask'].detach())==pytest.approx(2.2)
 losses['loss_unknown']=x
 with pytest.raises(ValueError):partition_losses(losses)
