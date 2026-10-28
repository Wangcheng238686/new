"""CPU fake-source/model contracts; real SAM and four-rank gates are separate."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from rsprompter.uav.native_sutc import SUTCSwitches,ZeroEvidenceProvider,TailP0Evidence,SatelliteTailJoint,install_native_tail
from rsprompter.uav.ubr_native_integration import NativeMaskHead

torch.set_num_threads(1)

class FakeP0(nn.Module):
    def __init__(self):
        super().__init__();self.scale=nn.Parameter(torch.linspace(.1,2,256).reshape(1,256,1,1));self.lora_dropout=nn.Dropout(.2);self.other_dropout=nn.Dropout(.9)
    def forward(self,image):
        value=F.avg_pool2d(image.mean(1,keepdim=True),4)*self.scale
        return value,(self.lora_dropout(value),)


def common():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(345)
        encoder=FakeP0();project=nn.Conv2d(256,32,1)
    state={'rgb_encoder.'+k:v.clone() for k,v in encoder.state_dict().items()}
    state.update({'rgb_project.'+k:v.clone() for k,v in project.state_dict().items()})
    state.update(rgb_mean=torch.tensor([123.675,116.28,103.53]).reshape(1,3,1,1),rgb_std=torch.tensor([58.395,57.12,57.375]).reshape(1,3,1,1))
    state['a.unused_geometry']=torch.zeros(1)
    return state


def provider(state=None):return TailP0Evidence('unused',common() if state is None else state,p0_factory=lambda _:FakeP0())


class FakeDecoder(nn.Module):
    capture_upscaled_embedding=False
    def __init__(self):
        super().__init__();self.decoder=nn.Module();self.decoder.output_upscaling=nn.Sequential(nn.Identity(),nn.SiLU());self.weight=nn.Parameter(torch.ones(1));self.calls=0
    def forward(self,x):
        self.calls+=1;x=self.decoder.output_upscaling(x)*self.weight
        return x.mean(1,keepdim=True),x.mean((1,2,3)),x.mean((2,3))

class FakeSatellite(nn.Module):
    def __init__(self):
        super().__init__();self.roi_head=nn.Module();self.roi_head.mask_head=nn.Module()
        head=self.roi_head.mask_head;head.decoder_tail_enabled=False;head.decoder_tail_refiner=None
        head.final_mask_coordinate_mode='roi_local';head.mask_decoder=FakeDecoder()
        self.extractions=0;self.fail=False
    def extract_feat(self,x):self.extractions+=1;return x*2
    def loss(self,inputs,samples):
        if self.fail:raise RuntimeError('fake loss error')
        return {'loss_mask':self.extract_feat(inputs).sum()}


def test_switches_immutable_and_boolean():
    assert [(SUTCSwitches.for_arm(a).U,SUTCSwitches.for_arm(a).T) for a in 'SUTC']==[(False,False),(True,False),(False,True),(True,True)]
    assert [SUTCSwitches.for_arm(a).arm for a in 'SUTC']==list('SUTC')
    with pytest.raises(FrozenInstanceError):SUTCSwitches.for_arm('S').U=True
    with pytest.raises(TypeError):SUTCSwitches(1,False)
    with pytest.raises(ValueError):SUTCSwitches.for_arm('wrong')


def test_strict_shared_subset_rng_and_training_mode():
    state=common();torch.manual_seed(12);before=torch.get_rng_state();p=provider(state)
    assert torch.equal(before,torch.get_rng_state())
    assert set(p.state_dict())==set(state)-{'a.unused_geometry'}
    assert all(torch.equal(v,state[k]) for k,v in p.state_dict().items())
    p.train();assert p.training and not p.rgb_encoder.training and p.rgb_encoder.lora_dropout.training
    assert not p.rgb_encoder.other_dropout.training
    p.eval();assert not p.rgb_encoder.lora_dropout.training
    bad=dict(state);bad.pop('rgb_project.bias')
    with pytest.raises(RuntimeError,match='missing'):provider(bad)
    bad=dict(state);bad['rgb_encoder.extra']=torch.ones(1)
    with pytest.raises(RuntimeError,match='extra'):provider(bad)
    assert all(k.startswith(('rgb_encoder.','rgb_project.')) for k,_ in p.named_parameters())


def test_measured_reader_exact_mean_independent_support_gradient_and_clear():
    p=provider().eval();rgb=torch.randn(5,3,512,512)
    index=torch.full((16384,8),-1,dtype=torch.long);index[0,:2]=torch.tensor([0,16384])
    anchor=torch.zeros(16384,8,6,dtype=torch.float64);geo=torch.zeros(1,5,512,512)
    p.encode(geo,rgb,index,anchor)
    with torch.no_grad():
        values=[]
        for image in rgb.split(1):
            _,pyramid=p.rgb_encoder((image-p.rgb_mean)/p.rgb_std)
            z=p.rgb_project(pyramid[0])[0,:,0,0]
            values.append(F.layer_norm(z,(32,)))
        expected=(values[0]+values[1])/2
    torch.testing.assert_close(p.native_aligned_rgb[0,:,0,0],expected)
    assert p.native_rgb_valid.sum()==1 and p.native_geometry is geo and geo[:,4].sum()==0
    assert not p.native_aligned_rgb[0,:,1:].any()
    p.native_aligned_rgb[0,:,0,0].square().sum().backward()
    assert p.rgb_project.weight.grad.abs().sum()>0 and p.rgb_encoder.scale.grad.abs().sum()>0
    p.clear();assert p.native_aligned_rgb is p.native_rgb_valid is p.native_geometry is None


def test_installer_preserves_provider_class_rng_and_single_ownership():
    sat=FakeSatellite();sentinel=ZeroEvidenceProvider();kind=type(sentinel)
    before=torch.get_rng_state();shim=install_native_tail(sat,sentinel,zero_evidence=True)
    assert type(sentinel) is kind and torch.equal(before,torch.get_rng_state())
    assert isinstance(sat.roi_head.mask_head,NativeMaskHead)
    shim.call_context=(torch.tensor([[0.,0.,512.,512.]]),torch.tensor([0.]))
    x=torch.randn(1,32,4,4);y=shim(x)
    assert len(y)==3 and shim.base.calls==1 and torch.equal(y[0],F.silu(x).mean(1,keepdim=True))
    assert not shim.base.decoder.output_upscaling[-1]._forward_hooks
    with pytest.raises(ValueError,match='already'):install_native_tail(sat,sentinel,zero_evidence=True)


def test_same_seed_native_heads_and_zero_provider_has_no_parameters():
    s=SatelliteTailJoint(FakeSatellite());t=SatelliteTailJoint(FakeSatellite(),provider())
    assert type(t.provider) is TailP0Evidence and not s.uav.enabled and not t.uav.enabled
    for a,b in zip(s.satellite.roi_head.mask_head.native_refiner.parameters(),t.satellite.roi_head.mask_head.native_refiner.parameters()):assert torch.equal(a,b)
    names=list(s.state_dict());assert names and all(k.startswith('satellite.') for k in names)
    assert not any(k.startswith(('shim.','uav.')) for k in t.state_dict())
    assert not isinstance(s.provider,nn.Module)
    with pytest.raises(ValueError,match='must not'):s.prepare_uav(geometry=torch.zeros(1))
    with pytest.raises(ValueError,match='requires'):t.prepare_uav()


def test_satellite_loss_single_extraction_no_aux_and_exception_restore():
    sat=FakeSatellite();joint=SatelliteTailJoint(sat);sample={'inputs':torch.ones(1),'data_samples':[]}
    before=sat.extract_feat;loss=joint(sample)
    assert sat.extractions==1 and set(loss)=={'loss_mask'} and loss['loss_mask']==2
    assert sat.extract_feat==before
    sat.fail=True
    with pytest.raises(RuntimeError,match='fake loss'):joint(sample)
    assert sat.extract_feat==before
