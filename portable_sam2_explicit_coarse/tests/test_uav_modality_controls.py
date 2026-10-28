"""CPU content ablations: values, untouched correspondence, and runtime routing."""
from dataclasses import FrozenInstanceError
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from rsprompter.uav.modality_controls import ModalityControls,RGB_MEAN


def payload():
    return dict(rgb=torch.randn(5,3,4,4),geometry=torch.arange(80,dtype=torch.float32).reshape(1,5,4,4),
                index=torch.tensor([[0,3,-1]]),anchor=torch.arange(18,dtype=torch.float64).reshape(1,3,6))


def test_defaults_are_value_and_reference_identity_no_side_effects():
    x=payload();control=ModalityControls();y=control.apply_payload(x)
    assert y is not x and all(y[k] is x[k] for k in x)
    candidates=torch.randn(2,8,7);assert control.apply_candidate_geometry(candidates) is candidates
    assert control.audit()['extra_losses']==control.audit()['extra_parameters']==0


def test_constant_RGB_and_geometry_off_preserve_raw_inputs_and_correspondence():
    x=payload();before={k:v.clone() for k,v in x.items()};control=ModalityControls('constant','off');y=control.apply_payload(x)
    mean=torch.tensor(RGB_MEAN).reshape(1,3,1,1)
    assert torch.equal(y['rgb'],mean.expand_as(x['rgb']))
    assert ((y['rgb']-mean)/torch.tensor([58.395,57.12,57.375]).reshape(1,3,1,1)).count_nonzero()==0
    assert not y['geometry'][:,:4].any() and torch.equal(y['geometry'][:,4],x['geometry'][:,4])
    assert y['index'] is x['index'] and y['anchor'] is x['anchor']
    assert all(torch.equal(x[k],before[k]) for k in x)
    assert y['rgb'].data_ptr()!=x['rgb'].data_ptr() and y['geometry'].data_ptr()!=x['geometry'].data_ptr()


def test_candidate_descriptor_mask_exact_dimensions_and_gradients():
    x=torch.arange(112,dtype=torch.float32).reshape(2,8,7).requires_grad_();before=x.detach().clone()
    y=ModalityControls(geometry_content='off').apply_candidate_geometry(x)
    assert not y[...,[2,3,6]].any() and torch.equal(y[...,[0,1,4,5]],before[...,[0,1,4,5]])
    assert torch.equal(x,before);y.sum().backward()
    assert not x.grad[...,[2,3,6]].any() and torch.equal(x.grad[...,[0,1,4,5]],torch.ones(2,8,4))


def test_frozen_config_invalid_controls_and_S_rejected():
    with pytest.raises(FrozenInstanceError):ModalityControls().rgb_content='constant'
    for kwargs in [dict(rgb_content='zero'),dict(geometry_content='none')]:
        with pytest.raises(ValueError):ModalityControls(**kwargs)
    ModalityControls().validate_arm('S')
    for controls in [ModalityControls('constant'),ModalityControls(geometry_content='off')]:
        with pytest.raises(ValueError,match='S has no UAV'):controls.validate_arm('S')
    with pytest.raises(ValueError):ModalityControls(geometry_content='off').apply_candidate_geometry(torch.zeros(1,6))
    with pytest.raises(ValueError):ModalityControls('constant').apply_payload(dict(rgb=torch.ones(5,3,2,2,dtype=torch.uint8)))
    with pytest.raises(ValueError):ModalityControls(geometry_content='off').apply_payload(dict(geometry=torch.zeros(1,4,2,2)))


@pytest.fixture
def runtime_module():
    path=Path(__file__).resolve().parents[1]/'scripts/probes/ue_ubr_native_sutc_runtime_20261028.py'
    spec=importlib.util.spec_from_file_location('modality_sutc_runtime',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('arm',['U','C','T'])
def test_full_path_payload_loss_and_inference_receive_same_controls(runtime_module,monkeypatch,arm):
    m=runtime_module;rt=m.Runtime.__new__(m.Runtime);rt.arm=arm;rt.switches=m.SUTCSwitches.for_arm(arm)
    rt.modalities=ModalityControls('constant','off');source=payload();candidate=torch.ones(1,3,7)
    loader=Mock(return_value=candidate);monkeypatch.setattr(m.legacy,'load_candidate_geometry',loader)
    monkeypatch.setattr(m.base.Runtime,'payload',lambda self,ident:dict(source))
    rt.uav_inputs=SimpleNamespace(load=Mock(side_effect=lambda *a,**kw:dict(source,provenance={'measured_depth':True})))
    rt.joint=SimpleNamespace(uav=SimpleNamespace(enabled=rt.switches.upstream),prepare_uav=Mock())
    rt.network=Mock(return_value={'loss_mask':torch.tensor(1.,requires_grad=True)})
    sample=dict(inputs='sat',data_samples=[SimpleNamespace(gt_instances=SimpleNamespace(bboxes=torch.tensor([[0.,0.,2.,2.]])))])
    m.Runtime.losses(rt,sample,10)
    used=rt.network.call_args.kwargs
    assert used['geometry'][:,:4].count_nonzero()==0 and torch.equal(used['geometry'][:,4],source['geometry'][:,4])
    assert torch.equal(used['rgb'],torch.tensor(RGB_MEAN).reshape(1,3,1,1).expand_as(source['rgb']))
    assert used['anchor'] is source['anchor'] and used['index'] is source['index']
    assert ('instance_target' in used)==rt.switches.upstream
    if rt.switches.upstream:
        loader.assert_called_once();assert loader.call_args.args[-1] is source['anchor']
        assert not rt.joint.uav.candidate_geometry[...,[2,3,6]].any()
        assert torch.equal(rt.joint.uav.candidate_geometry[...,[0,1,4,5]],torch.ones(1,3,4))
    else:loader.assert_not_called()
    rt.bare=SimpleNamespace(training=False);rt.satellite=SimpleNamespace(extract_feat=Mock(return_value='features'))
    infer=Mock(return_value=('prefix','combined',0));monkeypatch.setattr(m.core,'inference_outputs',infer)
    rt.inference(sample,10);prepared=rt.joint.prepare_uav.call_args.kwargs
    assert torch.equal(prepared['geometry'],used['geometry']) and torch.equal(prepared['rgb'],used['rgb'])
    assert infer.call_args.args[-1] is prepared['geometry']
    assert torch.equal(source['geometry'][:,:4],torch.arange(64).reshape(1,4,4,4).float())


def test_S_rejection_before_construction_and_build_keywords(runtime_module,monkeypatch):
    m=runtime_module;construct=Mock(side_effect=AssertionError('GPU allocation forbidden'));monkeypatch.setattr(m.legacy,'build_runtime',construct)
    with pytest.raises(ValueError,match='S has no UAV'):m.Runtime('S',rgb_content='constant')
    with pytest.raises(ValueError,match='S has no UAV'):m.Runtime('S',geometry_content='off')
    construct.assert_not_called()
    target=Mock(return_value='planned');monkeypatch.setattr(m,'Runtime',target)
    assert m.build_runtime('C',rgb_content='constant',geometry_content='off')=='planned'
    assert target.call_args.kwargs['rgb_content']=='constant' and target.call_args.kwargs['geometry_content']=='off'
