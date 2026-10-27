"""Small real-controller contract tests; real SAM/data gate is separate."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rsprompter.uav.ubr_detector_integration import (
    PredictionPool, active_prediction_indices, disable_udpr_config, install_ubr)
from rsprompter.uav.unified_boundary_refiner import UnifiedBoundaryRefiner

torch.set_num_threads(1)


class FakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        decoder = nn.Module()
        decoder.output_upscaling = nn.Sequential(nn.Conv2d(32, 32, 1), nn.SiLU())
        wrapper = nn.Module()
        wrapper.decoder = decoder
        wrapper.capture_upscaled_embedding = False
        self.mask_head = nn.Module()
        self.mask_head.mask_decoder = wrapper
        self.mask_head.decoder_tail_enabled = False
        self.mask_head.decoder_tail_refiner = None
        self.mask_head.final_mask_coordinate_mode = 'roi_local'
        self.mask_head.class_agnostic = True
        self.test_cfg = SimpleNamespace(mask_thr_binary=.4)
        self.extra_count = 0
        self.decode_grad_flags = []
        self.decode_batches = []

    def _inject_extra_pe(self, features):
        self.extra_count += 1
        return tuple(f + .125 for f in features)

    def _mask_forward(self, features, rois, image_embeddings=None, **kwargs):
        self.decode_grad_flags.append(torch.is_grad_enabled())
        self.decode_batches.append(rois[:, 1:].detach().clone())
        n = len(rois)
        source = image_embeddings.expand(n, -1, -1, -1)
        upscaled = self.mask_head.mask_decoder.decoder.output_upscaling(source)
        yy = torch.arange(source.shape[-2], device=source.device)[:, None]
        xx = torch.arange(source.shape[-1], device=source.device)[None]
        pattern = (((yy+xx) % 2) * 4 - 2).to(source.dtype)
        return dict(mask_preds=upscaled.mean(1, keepdim=True) + pattern)


def setup_model():
    satellite = nn.Module()
    satellite.roi_head = FakeHead()
    control = install_ubr(satellite, UnifiedBoundaryRefiner(hidden=8, depth=1, band_width=2))
    embedding = torch.randn(1, 32, 4, 4, requires_grad=True)
    features = ((embedding,), embedding, None, None)
    prepared = control.prepare(features, dict(ori_shape=(12, 12), scale_factor=(1., 1.)))
    pool = PredictionPool(torch.tensor([[2., 2., 10., 10.], [1., 1., 8., 8.]]),
                          torch.tensor([.9, .8]), torch.tensor([101, 202]), ('rpn', 'uav'))
    return satellite, control, prepared, pool, embedding


def test_disable_config_no_mutation_and_install_guard():
    cfg = dict(roi_head=dict(mask_head=dict(decoder_tail_refiner_cfg=dict(enabled=True, mode='old'))))
    new = disable_udpr_config(cfg)
    assert cfg['roi_head']['mask_head']['decoder_tail_refiner_cfg']['enabled']
    assert new['roi_head']['mask_head']['decoder_tail_refiner_cfg'] == dict(enabled=False)
    satellite, control, _, _, _ = setup_model()
    with pytest.raises(ValueError):
        install_ubr(satellite)
    assert any(name.startswith('ubr_refiner.') for name, _ in satellite.named_parameters())


def test_context_rng_modes_capture_and_zero_prediction_identity():
    satellite, control, prepared, pool, _ = setup_model()
    satellite.train()
    satellite.roi_head.mask_head.mask_decoder.eval()  # heterogeneous flags restored exactly
    modes = [m.training for m in satellite.roi_head.modules()]
    rng = torch.get_rng_state().clone()
    context = control.prediction_context(prepared, pool)
    assert torch.equal(torch.get_rng_state(), rng)
    assert modes == [m.training for m in satellite.roi_head.modules()]
    assert control._capture is None and not control._capturing
    assert not any(satellite.roi_head.decode_grad_flags)
    assert context.neighbors(101).shape == (12, 12)
    satellite.eval()
    output = control.predict(prepared, pool)
    from rsprompter.uav.ubr_coordinate_adapter import adapt_native_roi
    for i in range(2):
        native, feature = control.decode_native(prepared, pool.boxes[i:i+1])
        state = adapt_native_roi(native, feature, pool.boxes[i:i+1], prepared.meta)
        assert torch.equal(output['probabilities'][i], state.probabilities[0, 0])
    assert torch.equal(output['scores'], pool.scores)
    assert satellite.roi_head.extra_count == 1


def test_active_iou_and_ordering_cap():
    pool = PredictionPool(torch.tensor([[0., 0., 4., 4.]] * 70 + [[9., 9., 10., 10.]]),
                          torch.ones(71), torch.arange(70, -1, -1), tuple('rpn' for _ in range(71)))
    active, gt = active_prediction_indices(pool, torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]]))
    assert len(active) == 64
    assert torch.equal(pool.candidate_ids[active], torch.arange(1, 65))
    assert not gt.any()  # first GT wins exact IoU tie; nonmatch never forced


def test_checkpointed_active_decoder_gradients_and_empty_parameter_connection():
    satellite, control, prepared, pool, embedding = setup_model()
    satellite.train()
    gt = torch.zeros(1, 12, 12)
    gt[:, 3:9, 3:9] = 1
    optimizer = torch.optim.SGD(satellite.parameters(), lr=.1)
    for step in range(2):
        loss = control.training_loss(prepared, pool, pool.boxes[:1], gt)['loss_ubr']
        assert loss.requires_grad and torch.isfinite(loss)
        loss.backward()
        assert control.refiner.out.weight.grad.abs().sum() > 0
        assert embedding.grad is not None and torch.isfinite(embedding.grad).all()
        if step == 1:
            assert control.refiner.input_proj.weight.grad.abs().sum() > 0
        optimizer.step()
        optimizer.zero_grad()
        embedding.grad = None
    assert control.last_stats['active_count'] >= 1
    assert any(satellite.roi_head.decode_grad_flags) and not all(satellite.roi_head.decode_grad_flags)
    loss = control.training_loss(prepared, pool, torch.empty(0, 4), torch.empty(0, 12, 12))['loss_ubr']
    loss.backward()
    for parameter in control.refiner.parameters():
        assert parameter.grad is not None and parameter.grad.count_nonzero() == 0


def test_no_silent_geometry_resize():
    satellite, control, prepared, pool, _ = setup_model()
    satellite.eval()
    with pytest.raises(ValueError, match='geometry'):
        control.predict(prepared, pool, torch.zeros(1, 5, 6, 6))


@pytest.mark.parametrize('chunk_size,expected_lengths', [(3, [3, 2, 2, 2]),
                                                        (0, [5, 2, 2])])
def test_eval_decoder_calls_preserve_source_segments_and_legacy_chunks(chunk_size, expected_lengths):
    satellite, control, prepared, _, _ = setup_model()
    satellite.eval()
    satellite.roi_head.mask_roi_chunk_size = chunk_size
    boxes = torch.tensor([[float(i)/10, 1., 10., 11.] for i in range(9)])
    pool = PredictionPool(boxes, torch.ones(9), torch.arange(9),
                          ('sat',)*5 + ('sat_supplement',)*2 + ('uav',)*2)
    context = control.prediction_context(prepared, pool)
    batches = satellite.roi_head.decode_batches
    assert [len(batch) for batch in batches] == expected_lengths
    assert torch.equal(torch.cat(batches), boxes)
    assert len(context.retained_ids) == 9
    satellite.roi_head.decode_batches.clear()
    control.predict(prepared, pool)
    # Both context pass and final decode pass preserve precisely the same calls.
    batches = satellite.roi_head.decode_batches
    assert [len(batch) for batch in batches] == expected_lengths * 2
    half = len(expected_lengths)
    for left, right in zip(batches[:half], batches[half:]):
        assert torch.equal(left, right)
