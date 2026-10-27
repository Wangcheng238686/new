"""Efficiency parity tests; real dense SAM/DDP profiling is a separate gate."""
import importlib.util
from pathlib import Path

import pytest
import torch

from rsprompter.uav import ubr_detector_fast as fast
from rsprompter.uav.unified_boundary_refiner import boundary_loss

spec = importlib.util.spec_from_file_location('fast_test_fixture', Path(__file__).with_name('test_ubr_detector_integration.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
torch.set_num_threads(1)


def test_context_neighbors_exact_and_decoder_only_once_for_eval():
    satellite, old, prepared, pool, _ = fixtures.setup_model()
    controller = fast.FastUBRDetectorIntegration.from_controller(old)
    old_context = old.prediction_context(prepared, pool)
    context = controller.prediction_context(prepared, pool)
    assert torch.equal(context.retained_masks, old_context.retained_masks)
    for cid in pool.candidate_ids.tolist():
        assert torch.equal(context.neighbors(cid), old_context.neighbors(cid))
    satellite.eval()
    baseline = old.predict(prepared, pool)
    satellite.roi_head.decode_batches.clear()
    output = controller.predict(prepared, pool)
    assert len(satellite.roi_head.decode_batches) == 2  # two source segments, ONE pass
    assert torch.equal(output['probabilities'], baseline['probabilities'])
    assert controller.last_stats['native_shape'] == [4, 4]
    with pytest.raises(KeyError):
        context.neighbors(999)


def test_nonempty_instance_sum_matches_original_loss_not_chunk_mean():
    z = torch.randn(5, 1, 3, 4, requires_grad=True)
    target = torch.randint(0, 2, z.shape).float()
    band = torch.zeros_like(z, dtype=torch.bool)
    band[0, :, :1] = True
    band[1, :, :2] = True
    band[4, :, :3] = True
    original = boundary_loss(z, target, band, torch.zeros(5, dtype=torch.long), 1)
    values = sum(fast._band_instance_sums(z[i:i+2], target[i:i+2], band[i:i+2]) for i in range(0, 5, 2)) / 3
    assert torch.allclose(values, original, atol=2e-7)
    expected_grad, = torch.autograd.grad(original, z, retain_graph=True)
    actual_grad, = torch.autograd.grad(values, z)
    assert torch.allclose(actual_grad, expected_grad, atol=1e-7)


@pytest.mark.parametrize('microbatch', [1, 2, 4])
def test_train_loss_gradients_parity_and_sdt_only_first_forward(microbatch, monkeypatch):
    torch.manual_seed(44)
    satellite, old, prepared, pool, embedding = fixtures.setup_model()
    with torch.no_grad():
        old.refiner.out.weight.normal_(std=.03)
        old.refiner.out.bias.fill_(.2)
    satellite.train()
    controller = fast.FastUBRDetectorIntegration.from_controller(old, microbatch)
    gt = torch.zeros(1, 12, 12)
    gt[:, 3:9, 3:9] = 1
    old_loss = old.training_loss(prepared, pool, pool.boxes[:1], gt)['loss_ubr']
    old_loss.backward()
    parameters = list(satellite.parameters())
    expected = [p.grad.clone() if p.grad is not None else None for p in parameters]
    old_embedding_grad = embedding.grad.clone()
    for p in parameters:
        p.grad = None
    embedding.grad = None
    calls = []
    edt = fast.signed_distance_transform
    def observed(mask):
        calls.append(mask.shape[0])
        return edt(mask)
    monkeypatch.setattr(fast, 'signed_distance_transform', observed)
    new_loss = controller.training_loss(prepared, pool, pool.boxes[:1], gt)['loss_ubr']
    active_calls = satellite.roi_head.decode_batches[-controller.last_stats['active_count']:]
    assert all(len(call) == 1 for call in active_calls)  # native singleton contract
    before_backward = list(calls)
    assert calls and all(n <= microbatch for n in calls)
    new_loss.backward()
    assert calls == before_backward  # no EDT repeats on checkpoint recomputation
    assert torch.allclose(new_loss, old_loss, atol=2e-7, rtol=1e-6)
    assert torch.allclose(embedding.grad, old_embedding_grad, atol=2e-7, rtol=1e-5)
    for actual, reference in zip(parameters, expected):
        if reference is None:
            assert actual.grad is None
        else:
            assert torch.allclose(actual.grad, reference, atol=2e-7, rtol=1e-5)
    assert old.refiner.checkpoint_blocks is False  # setup fixture default, restored


def test_empty_loss_all_parameters_connected():
    satellite, old, prepared, pool, _ = fixtures.setup_model()
    controller = fast.FastUBRDetectorIntegration.from_controller(old)
    loss = controller.training_loss(prepared, pool, torch.empty(0, 4), torch.empty(0, 12, 12))['loss_ubr']
    assert loss == 0
    loss.backward()
    assert controller.last_stats['active_count'] == 0
    assert all(p.grad is not None and not p.grad.any() for p in controller.refiner.parameters())


def test_restore_inner_checkpoint_flag_and_new_installer():
    satellite, old, prepared, pool, _ = fixtures.setup_model()
    old.refiner.checkpoint_blocks = True
    controller = fast.FastUBRDetectorIntegration.from_controller(old)
    satellite.eval()
    controller.predict(prepared, pool)
    assert old.refiner.checkpoint_blocks is True
    fresh = torch.nn.Module()
    fresh.roi_head = fixtures.FakeHead()
    installed = fast.install_fast_ubr(fresh)
    assert installed.refiner is fresh.ubr_refiner
    assert installed.active_microbatch == 1
    assert installed.inference_microbatch == 4


def test_eval_native_chunk_unchanged_but_refiner_uses_microbatches():
    satellite, old, prepared, _, _ = fixtures.setup_model()
    satellite.eval()
    satellite.roi_head.mask_roi_chunk_size = 5
    pool = fixtures.PredictionPool(torch.tensor([[1., 1., 10., 10.]] * 9),
                                  torch.ones(9), torch.arange(9), ('sat',)*7 + ('uav',)*2)
    old_output = old.predict(prepared, pool)
    satellite.roi_head.decode_batches.clear()
    controller = fast.FastUBRDetectorIntegration.from_controller(old, active_microbatch=1,
                                                               inference_microbatch=4)
    batch_sizes = []
    handle = old.refiner.register_forward_pre_hook(lambda module, args: batch_sizes.append(len(args[0])))
    result = controller.predict(prepared, pool)
    handle.remove()
    assert [len(batch) for batch in satellite.roi_head.decode_batches] == [5, 2, 2]
    assert batch_sizes == [4, 1, 2, 2]
    assert torch.equal(result['probabilities'], old_output['probabilities'])
