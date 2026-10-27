import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('native_refiner_core', Path(__file__).resolve().parents[1] /
                                            'rsprompter/uav/native_roi_refiner.py')
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
torch.set_num_threads(1)


def test_full_image_rgb_bucket_centers_exact_and_geometry_pixel_centers():
    yy, xx = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
    rgb = (xx + 128*yy).float()[None, None].expand(1, 32, -1, -1).clone()
    geometry = torch.zeros(1, 5, 512, 512)
    gy, gx = torch.meshgrid(torch.arange(512), torch.arange(512), indexing='ij')
    geometry[:, 0] = gx + gy*512
    geometry[:, 4] = 1
    boxes = torch.tensor([[0., 0., 512., 512.]])
    sampled, geo = core.sample_roi_evidence(boxes, torch.tensor([0]), (128, 128),
                                           rgb, torch.ones(1, 1, 128, 128), geometry)
    assert torch.equal(sampled, rgb)
    # Grid at image edge-coordinates2,6,... is halfway between image pixel
    # centers1.5/2.5; torch nearest tie behavior picks even source index.
    assert geo[0, 0, 0, 0] == geometry[0, 0, 2, 2]
    assert geo[0, 0, 1, 1] == geometry[0, 0, 6, 6]


def test_fractional_roi_and_explicit_origin_landmarks():
    rgb = torch.zeros(1, 32, 2, 2)
    rgb[0, :, 0, 0] = 2
    rgb[0, :, 0, 1] = 6
    valid = torch.ones(1, 1, 2, 2)
    # ROI center(3,2) lies 1/4 of the way from bucket center x2 to x6.
    sampled, _ = core.sample_roi_evidence(torch.tensor([[2.5, 1.5, 3.5, 2.5]]),
        torch.tensor([0]), (1, 1), rgb, valid, image_size=(8, 8))
    assert torch.allclose(sampled, torch.full_like(sampled, 3.))
    wrong, _ = core.sample_roi_evidence(torch.tensor([[2.5, 1.5, 3.5, 2.5]]),
        torch.tensor([0]), (1, 1), rgb, valid, rgb_origin=(0., 0.), image_size=(8, 8))
    assert not torch.equal(sampled, wrong)


def test_invalid_holes_nan_and_independent_support():
    rgb = torch.full((1, 32, 2, 2), float('nan'))
    rgb[:, :, 0, 0] = 7
    valid = torch.zeros(1, 1, 2, 2)
    valid[:, :, 0, 0] = 1
    geo = torch.zeros(1, 5, 8, 8)
    geo[:, :4] = 99  # unsupported geometry must not leak
    sampled, geometry = core.sample_roi_evidence(torch.tensor([[3., 3., 5., 5.]]),
        torch.tensor([0]), (1, 1), rgb, valid, geo, image_size=(8, 8))
    assert torch.equal(sampled, torch.full_like(sampled, 7.))
    assert not geometry.any()  # RGB validity is not geometry validity
    valid.zero_()
    sampled, _ = core.sample_roi_evidence(torch.tensor([[3., 3., 5., 5.]]),
        torch.tensor([0]), (1, 1), rgb, valid, image_size=(8, 8))
    assert not sampled.any()
    # Explicit image bound rejects surviving interpolation support outside image.
    sampled, _ = core.sample_roi_evidence(torch.tensor([[-1., 1., -.1, 3.]]),
        torch.tensor([0]), (1, 1), torch.ones_like(rgb), torch.ones_like(valid), image_size=(8, 8))
    assert not sampled.any()


def test_geometry_nearest_not_bilinear_and_image_batch_indices():
    geo = torch.zeros(2, 5, 8, 8)
    geo[1, :, 2, 3] = torch.tensor([5., 4., 3., 2., 1.])
    _, sampled = core.sample_roi_evidence(torch.tensor([[3., 2., 4., 3.]]),
        torch.tensor([1]), (1, 1), geometry=geo, image_size=(8, 8))
    assert torch.equal(sampled[0, :, 0, 0], geo[1, :, 2, 3])


def test_zero_identity_empty_connected_and_satellite_only_updates():
    model = core.NativeRoIRefiner()
    decoder = torch.randn(2, 32, 7, 9, requires_grad=True)
    logits = torch.randn(2, 1, 7, 9, requires_grad=True)
    result = model(decoder, logits)
    assert torch.equal(result, logits)
    result.square().mean().backward()
    assert model.out.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.step()
    optimizer.zero_grad()
    decoder.grad = None
    model(decoder, logits).square().mean().backward()
    assert decoder.grad.abs().sum() > 0
    assert model.input_proj.weight.grad.abs().sum() > 0
    optimizer.zero_grad()
    model(torch.empty(0, 32, 7, 9), torch.empty(0, 1, 7, 9)).sum().backward()
    assert all(p.grad is not None and not p.grad.any() for p in model.parameters())


def test_nonzero_head_gradients_reach_only_valid_rgb_and_probability_detached():
    torch.manual_seed(44)
    model = core.NativeRoIRefiner()
    with torch.no_grad():
        model.out.weight.normal_(std=.1)
    rgb = torch.randn(1, 32, 2, 2, requires_grad=True)
    valid = torch.zeros(1, 1, 2, 2)
    valid[:, :, 0, 0] = 1
    sampled, geo = core.sample_roi_evidence(torch.tensor([[0., 0., 8., 8.]]),
        torch.tensor([0]), (5, 5), rgb, valid, image_size=(8, 8))
    decoder = torch.randn(1, 32, 5, 5, requires_grad=True)
    logits = torch.randn(1, 1, 5, 5, requires_grad=True)
    result = model(decoder, logits, sampled, geo)
    result.sum().backward()
    assert torch.equal(logits.grad, torch.ones_like(logits))
    assert rgb.grad[:, :, 0, 0].abs().sum() > 0
    assert not rgb.grad[:, :, 1].any() and not rgb.grad[:, :, 0, 1].any()


def test_missing_evidence_empty_and_invalid_contract():
    rgb, geo = core.sample_roi_evidence(torch.empty(0, 4), torch.empty(0, dtype=torch.long), (7, 9))
    assert rgb.shape == (0, 32, 7, 9) and geo.shape == (0, 5, 7, 9)
    with pytest.raises(ValueError):
        core.sample_roi_evidence(torch.tensor([[0., 0., 2., 2.]]), torch.tensor([0]), (2, 2),
                                 aligned_rgb=torch.ones(1, 32, 2, 2))


def test_fp32_head_island_under_autocast_preserves_interface_and_hidden_gradients():
    torch.manual_seed(44)
    model = core.NativeRoIRefiner()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005)
    decoder = torch.randn(2, 32, 16, 16, dtype=torch.bfloat16)
    logits = torch.randn(2, 1, 16, 16, dtype=torch.bfloat16)
    target = torch.randint(0, 2, logits.shape).float()
    seen = []
    hook = model.out.register_forward_hook(lambda module, args, out: seen.append((args[0].dtype, out.dtype)))
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            refined = model(decoder, logits)
            assert refined.dtype == logits.dtype
            if step == 0:
                assert torch.equal(refined, logits)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(refined.float(), target)
        (loss*32).backward()
        assert model.out.weight.grad.abs().sum() > 0
        if step == 1:
            assert model.input_proj.weight.grad.abs().sum() > 0
        optimizer.step()
    hook.remove()
    assert seen == [(torch.float32, torch.float32)]*2
