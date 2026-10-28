"""CPU runtime routing tests: switches must cover loaders, losses and consumers."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / 'scripts/probes/ue_ubr_native_sutc_runtime_20261028.py'


@pytest.fixture
def runtime_module():
    spec = importlib.util.spec_from_file_location('native_sutc_runtime_test', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bare_runtime(module, arm):
    rt = module.Runtime.__new__(module.Runtime)
    rt.arm = arm
    rt.switches = module.SUTCSwitches.for_arm(arm)
    rt.uav_inputs = Mock()
    return rt


def test_S_payload_never_calls_UAV_loader(runtime_module):
    rt = bare_runtime(runtime_module, 'S')
    assert rt.payload(123) == {}
    rt.uav_inputs.load.assert_not_called()


@pytest.mark.parametrize('arm', ['S', 'T'])
def test_no_upstream_means_no_instance_target_or_loss(runtime_module, monkeypatch, arm):
    rt = bare_runtime(runtime_module, arm)
    rt.payload = Mock(return_value={} if arm == 'S' else {'rgb': 'test'})
    rt.network = Mock(return_value={'loss_mask': torch.tensor(1., requires_grad=True)})
    monkeypatch.setattr(runtime_module, 'center_targets', Mock(side_effect=AssertionError('unexpected instance target')))
    sample = {'marker': 3}
    loss = rt.objective(sample, 123)
    assert loss.item() == 1
    rt.network.assert_called_once_with(sample, **rt.payload.return_value)
    rt.network.return_value['loss_uav_instances'] = loss
    with pytest.raises(RuntimeError, match='only original satellite'):
        rt.losses(sample, 123)


def test_T_loader_uses_measured_correspondence_without_geometry7(runtime_module):
    rt = bare_runtime(runtime_module, 'T')
    rt.uav_inputs.load.return_value = {'geometry': 'geo5', 'index': 'depth_index',
                                     'anchor': 'depth_anchor', 'rgb': 'RGB',
                                     'provenance': {'measured_depth': True}}
    result = rt.payload(9)
    assert set(result) == {'geometry', 'index', 'anchor', 'rgb'}
    rt.uav_inputs.load.assert_called_once_with(9, geometry_mode='measured', rgb_mode='correct')


@pytest.mark.parametrize('arm', ['S', 'T'])
def test_formal_inference_keeps_satellite_consumer_and_disables_UAV_candidates(runtime_module, monkeypatch, arm):
    rt = bare_runtime(runtime_module, arm)
    adapter = SimpleNamespace(enabled=False)
    rt.joint = SimpleNamespace(uav=adapter, prepare_uav=Mock())
    rt.bare = SimpleNamespace(training=False)
    rt.satellite = SimpleNamespace(extract_feat=Mock(return_value='features'))
    rt.payload = Mock(return_value={} if arm == 'S' else {'geometry': 'geo5'})
    consumer = Mock(return_value=('prefix', 'combined', 0))
    monkeypatch.setattr(runtime_module.core, 'inference_outputs', consumer)
    assert rt.inference({'inputs': 'satellite'}, 9) == ('prefix', 'combined', 0)
    consumer.assert_called_once_with(rt.satellite, adapter, {'inputs': 'satellite'}, 'features',
                                    None if arm == 'S' else 'geo5')


@pytest.mark.parametrize('arm', ['U', 'C'])
def test_upstream_routes_instance_loss_inside_network(runtime_module, monkeypatch, arm):
    rt = bare_runtime(runtime_module, arm)
    rt.payload = Mock(return_value={'geometry': 'geo5'})
    rt.network = Mock(return_value={'loss_mask': torch.tensor(1., requires_grad=True),
                                  'loss_uav_instances': torch.tensor(2., requires_grad=True)})
    target = object()
    factory = Mock(return_value=target)
    monkeypatch.setattr(runtime_module, 'center_targets', factory)
    sample = {'data_samples': [SimpleNamespace(gt_instances=SimpleNamespace(bboxes=torch.ones(2, 4)))]}
    assert rt.objective(sample, 3).item() == 3
    factory.assert_called_once()
    rt.network.assert_called_once_with(sample, geometry='geo5', instance_target=target,
                                       center_loss_fn=runtime_module.core.pilot.loss_fn)


def test_view_count_factory_resolves_explicit_ids_and_rejects_ambiguous_arguments(runtime_module, monkeypatch):
    constructor=Mock(return_value='runtime')
    monkeypatch.setattr(runtime_module,'Runtime',constructor)
    assert runtime_module.build_runtime('C',view_count=3,subset_cache_root='/cache')=='runtime'
    constructor.assert_called_once_with('C',uav_view_ids=(0,144,288),subset_cache_root='/cache')
    with pytest.raises(ValueError):runtime_module.build_runtime('C',view_count=3,uav_view_ids=[0])
    with pytest.raises(ValueError):runtime_module.build_runtime('C',view_count=2)


def test_reduced_views_without_cache_fail_before_GPU_allocation(runtime_module):
    with pytest.raises(ValueError,match='rebuilt subset cache'):
        runtime_module.Runtime('C',uav_view_ids=[0])
