"""Compare the standalone bridge against the installed real MMDet paste."""
import importlib.util
from pathlib import Path

import pytest
import torch
from mmengine import ConfigDict
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead, _do_paste_mask

SPEC = importlib.util.spec_from_file_location('ubr_bridge', Path(__file__).resolve().parents[1] /
                                            'rsprompter/uav/ubr_coordinate_adapter.py')
import sys
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)
torch.set_num_threads(1)


def fixture():
    boxes = torch.tensor([[2., 3., 18., 15.], [1.25, 2.75, 19.3, 16.1],
                          [-4.2, -1.1, 10.8, 11.9], [15.7, 11.2, 25., 23.]])
    yy, xx = torch.meshgrid(torch.arange(7), torch.arange(9), indexing='ij')
    checker = ((xx + yy) % 2).float() * 12 - 6
    z = checker[None, None].expand(4, 1, -1, -1).clone()
    # Landmarks distinguish image resize from actual bbox paste.
    feat = torch.zeros(4, 32, 7, 9)
    feat[:, 0, 2, 3] = 1
    feat[:, 1] = xx
    return z, feat, boxes


@pytest.mark.parametrize('rescale', [False, True])
def test_real_fcn_binary_visualization_and_score_identity(rescale):
    z, feat, boxes = fixture()
    original_boxes = boxes.clone()
    meta = dict(ori_shape=(17, 21), img_shape=(24, 28), scale_factor=(1.25, 1.5))
    state = bridge.adapt_native_roi(z, feat, boxes, meta, rescale)
    assert torch.equal(boxes, original_boxes)
    restored = bridge.residual_probabilities(state, state.logits)
    assert torch.equal(restored, state.probabilities)
    class Head:
        class_agnostic = True
    for threshold in [.4, .5, -1.]:
        expected = FCNMaskHead._predict_by_feat_single(
            Head(), z, boxes.clone(), torch.zeros(4, dtype=torch.long), meta,
            ConfigDict(mask_thr_binary=threshold), rescale=rescale)
        actual = (restored[:, 0] >= threshold if threshold >= 0 else
                  (restored[:, 0] * 255).to(torch.uint8))
        assert torch.equal(actual, expected)
    # Score identity for arbitrary probability-based mask scoring at zero residual.
    assert torch.equal(restored.flatten(1).mean(1), state.probabilities.flatten(1).mean(1))


def test_checkerboard_probability_paste_and_feature_landmarks():
    z, feat, boxes = fixture()
    state = bridge.adapt_native_roi(z, feat, boxes,
                                   dict(ori_shape=(19, 23), scale_factor=(1., 1.)))
    reference, _ = _do_paste_mask(z.sigmoid(), boxes, 19, 23, skip_empty=False)
    assert torch.equal(state.probabilities[:, 0], reference)
    for channel in (0, 1):
        expected, _ = _do_paste_mask(feat[:, channel:channel+1], boxes, 19, 23, skip_empty=False)
        assert torch.equal(state.decoder_features[:, channel], expected)
    wrong, _ = _do_paste_mask(z, boxes, 19, 23, skip_empty=False)
    assert (wrong.sigmoid() - reference).abs().max() > .1
    assert not torch.equal(state.decoder_features,
                           torch.nn.functional.interpolate(feat, (19, 23), mode='bilinear', align_corners=False))


def test_native_pixel_center_landmark():
    value = torch.zeros(1, 1, 4, 4)
    value[0, 0, 1, 2] = 1
    pasted = bridge.paste_native_roi(value, torch.tensor([[2., 3., 6., 7.]]), (10, 10))
    assert pasted[0, 0, 4, 4] == 1
    assert pasted.sum() == 1


def test_cpu_skip_empty_halo_matches_real_fcn():
    z = torch.full((1, 1, 2, 2), 4.)
    boxes = torch.tensor([[15., 15., 45., 45.]])
    state = bridge.adapt_native_roi(z, torch.zeros(1, 32, 2, 2), boxes,
                                   dict(ori_shape=(60, 60), scale_factor=(1., 1.)))
    local, slices = _do_paste_mask(z.sigmoid(), boxes, 60, 60, skip_empty=True)
    expected = torch.zeros(1, 60, 60)
    expected[(slice(None),) + slices] = local
    assert torch.equal(state.probabilities[:, 0], expected)
    full = bridge.paste_native_roi(z.sigmoid(), boxes, (60, 60))
    assert not torch.equal(full, state.probabilities)  # real upstream CPU/GPU difference


def test_finite_inverse_and_exact_saturated_identity():
    z = torch.tensor([[[[-1000., 1000.], [-1000., 1000.]]]])
    state = bridge.adapt_native_roi(z, torch.zeros(1, 32, 2, 2),
                                   torch.tensor([[0., 0., 2., 2.]]),
                                   dict(ori_shape=(2, 2), scale_factor=(1., 1.)))
    assert torch.isfinite(state.logits).all()
    assert (state.logits.sigmoid() - state.probabilities).abs().max() <= 1.1e-6
    assert torch.equal(bridge.residual_probabilities(state, state.logits), state.probabilities)
    changed = state.logits + 1
    assert torch.equal(bridge.residual_probabilities(state, changed), changed.sigmoid())
    for invalid_epsilon in (0., 1e-12, .5, float('nan')):
        with pytest.raises(ValueError):
            bridge.finite_probability_logits(torch.tensor([0., 1.]), invalid_epsilon)


def test_empty_and_invalid_boxes():
    state = bridge.adapt_native_roi(torch.empty(0, 1, 4, 4), torch.empty(0, 32, 4, 4),
                                   torch.empty(0, 4), dict(ori_shape=(9, 11), scale_factor=(1., 1.)))
    assert state.probabilities.shape == (0, 1, 9, 11)
    assert state.decoder_features.shape == (0, 32, 9, 11)
    for box in ([1., 1., 1., 2.], [float('nan'), 0., 2., 2.]):
        with pytest.raises(ValueError):
            bridge.paste_native_roi(torch.ones(1, 1, 4, 4), torch.tensor([box]), (5, 5))


def test_gradients_from_canvas_and_mixed_precision():
    z, feat, boxes = fixture()
    z.requires_grad_()
    feat.requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        state = bridge.adapt_native_roi(z, feat, boxes,
                                       dict(ori_shape=(19, 23), scale_factor=(1., 1.)))
        loss = state.logits.square().mean() + state.decoder_features.square().mean()
    assert state.probabilities.dtype == torch.float32
    assert state.decoder_features.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    assert torch.isfinite(feat.grad).all() and feat.grad.abs().sum() > 0


def test_image_size_uses_original_scale_not_padding():
    boxes = torch.tensor([[0., 0., 8., 8.]])
    result, size = bridge.fcn_canvas(boxes, dict(ori_shape=(9, 11), scale_factor=(.5, .5),
                                               img_shape=(5, 6), pad_shape=(32, 32)))
    assert size == (4, 6)  # numpy banker's rounding mirrors actual FCN
    assert torch.equal(result, boxes)


def test_exact_and_adjacent_threshold_probabilities():
    # Threshold ties must survive the finite-logit bridge at deployment.
    threshold = torch.tensor(.4)
    p = torch.stack((torch.nextafter(threshold, torch.tensor(0.)), threshold,
                     torch.nextafter(threshold, torch.tensor(1.)),
                     torch.tensor(0.), torch.tensor(1.))).reshape(1, 1, 1, 5)
    state = bridge.RoIImageState(p, bridge.finite_probability_logits(p),
                                torch.zeros(1, 32, 1, 5),
                                torch.tensor([[0., 0., 5., 1.]]), (1, 5), 1e-6)
    restored = bridge.residual_probabilities(state, state.logits.clone())
    assert torch.equal(restored, p)
    assert torch.equal(restored >= .4, p >= .4)
