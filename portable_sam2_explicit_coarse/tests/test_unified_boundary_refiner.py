"""CPU contract tests; loads standalone module without mmdet package imports."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

SPEC = importlib.util.spec_from_file_location(
    'standalone_ubr', Path(__file__).resolve().parents[1] /
    'rsprompter/uav/unified_boundary_refiner.py')
ubr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ubr)
torch.set_num_threads(1)


def inputs(n=2, side=24):
    z = torch.full((n, 1, side, side), -3.)
    z[:, :, 6:18, 6:18] = 3.
    feat = torch.randn(n, 32, side, side)
    sdt, band = ubr.prediction_boundary_state(z, width=2)
    return feat, z, sdt, torch.zeros_like(z), band


def test_identity_and_outside_binary_unchanged():
    model = ubr.UnifiedBoundaryRefiner(band_width=2)
    feat, z, sdt, neighbors, band = inputs()
    assert torch.equal(model(feat, z, sdt, neighbors), z)
    with torch.no_grad():
        model.out.bias.fill_(8)
    result = model(feat, z, sdt, neighbors)
    assert not torch.equal(result[band], z[band])
    assert torch.equal(result[~band], z[~band])
    assert torch.equal((result.sigmoid() >= .4)[~band], (z.sigmoid() >= .4)[~band])
    # A zero UAV port is NOT a gate disabling the refiner.
    assert torch.equal(model(feat, z, sdt, neighbors, torch.zeros(2, 5, 24, 24)), result)


def test_gradients_and_detached_state_after_first_update():
    model = ubr.UnifiedBoundaryRefiner(band_width=2)
    feat, z, sdt, neighbors, band = inputs()
    feat.requires_grad_()
    z.requires_grad_()
    sdt.requires_grad_()
    neighbors.requires_grad_()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    model(feat, z, sdt, neighbors).sum().backward()
    assert torch.equal(z.grad, torch.ones_like(z))
    assert sdt.grad is None and neighbors.grad is None
    assert torch.count_nonzero(model.out.weight.grad) > 0
    assert torch.count_nonzero(feat.grad) == 0  # expected zero-last-layer first step
    optimizer.step()
    optimizer.zero_grad()
    feat.grad = None
    z.grad = None
    model(feat, z, sdt, neighbors).square().mean().backward()
    assert torch.count_nonzero(feat.grad) > 0
    assert torch.count_nonzero(model.input_proj.weight.grad) > 0
    assert sdt.grad is None and neighbors.grad is None
    # Detached probability does not introduce an indirect logit gradient.
    assert torch.allclose(z.grad, 2 * model(feat, z, sdt, neighbors).detach() / z.numel())


def test_empty_full_and_deployment_threshold():
    z = torch.stack([torch.full((1, 8, 8), -20.), torch.full((1, 8, 8), 20.)])
    sdt, band = ubr.prediction_boundary_state(z, width=100000)
    assert not band.any()
    model = ubr.UnifiedBoundaryRefiner(band_width=100000)
    with torch.no_grad():
        model.out.bias.fill_(100)
    assert torch.equal(model(torch.randn(2, 32, 8, 8), z, sdt, torch.zeros_like(z)), z)
    test_z = torch.tensor([[[[-1., -.3], [-1., -.3]]]])
    threshold_state, _ = ubr.prediction_boundary_state(test_z, threshold=.4, width=1)
    assert (threshold_state[0, 0, :, 1] > 0).all()  # p(-.3)>.4 but <.5
    equality = torch.tensor([[[[0., -1.], [0., -1.]]]])
    eq_state, _ = ubr.prediction_boundary_state(equality, threshold=.5, width=1)
    assert (eq_state[0, 0, :, 0] > 0).all()  # sigmoid(0) exactly equals .5


def test_signed_distance_pixels_and_edges():
    mask = torch.zeros(1, 1, 7, 7, dtype=torch.bool)
    mask[:, :, :4, :4] = True
    sdt = ubr.signed_distance_transform(mask)
    assert sdt[0, 0, 3, 3] == 1
    assert sdt[0, 0, 4, 3] == -1
    assert sdt[0, 0, 2, 2] == 2
    assert ubr.band_mask(sdt, 1)[0, 0, 3, 3]
    assert not ubr.band_mask(sdt, 1)[0, 0, 2, 2]


def test_loss_uses_instance_then_image_denominators():
    z = torch.tensor([[[[0., 1., 9.]]], [[[2., -1., 9.]]], [[[1., 3., 9.]]]], requires_grad=True)
    target = torch.tensor([[[[1., 0., 0.]]], [[[0., 1., 0.]]], [[[1., 1., 0.]]]])
    band = torch.tensor([[[[1, 0, 0]]], [[[1, 1, 0]]], [[[1, 1, 0]]]], dtype=torch.bool)
    ids = torch.tensor([0, 0, 1])
    components = ubr.boundary_loss_components(z, target, band, ids, num_images=3)
    focal, dice = [], []
    for i in range(3):
        zi, yi = z[i][band[i]], target[i][band[i]]
        p = zi.sigmoid()
        pt = p * yi + (1 - p) * (1 - yi)
        focal.append(((1 - pt).square() * F.binary_cross_entropy_with_logits(zi, yi, reduction='none')).mean())
        dice.append(1 - (2 * (p * yi).sum()) / (p.sum() + yi.sum() + 1e-6))
    assert torch.allclose(components['focal'], ((focal[0] + focal[1]) / 2 + focal[2]) / 3)
    assert torch.allclose(components['dice'], ((dice[0] + dice[1]) / 2 + dice[2]) / 3)
    loss = ubr.boundary_loss(z, target, band, ids, num_images=3)
    assert torch.allclose(loss, sum(components.values()))  # raw, NOT preweighted
    loss.backward()
    assert torch.count_nonzero(z.grad[~band]) == 0
    assert torch.count_nonzero(z.grad[band]) == 5
    # Empty-band instances do not dilute valid instances in the same image.
    extra = torch.zeros(1, 1, 1, 3)
    augmented = ubr.boundary_loss(torch.cat([z.detach(), extra]), torch.cat([target, extra]),
                                 torch.cat([band, extra.bool()]), torch.tensor([0, 0, 1, 0]), 3)
    assert torch.allclose(loss, augmented)


def test_empty_batch_and_all_empty_bands_are_differentiable_zero():
    model = ubr.UnifiedBoundaryRefiner()
    z = torch.empty(0, 1, 4, 4, requires_grad=True)
    final = model(torch.empty(0, 32, 4, 4), z, z, z)
    loss = ubr.boundary_loss(final, z.detach(), z.bool(), torch.empty(0, dtype=torch.long), 2)
    assert loss == 0 and loss.requires_grad
    loss.backward()
    assert all(p.grad is not None and not p.grad.any() for p in model.parameters())
    z = torch.randn(2, 1, 4, 4, requires_grad=True)
    loss = ubr.boundary_loss(z, torch.zeros_like(z), torch.zeros_like(z, dtype=torch.bool), torch.tensor([0, 1]), 2)
    loss.backward()
    assert loss == 0 and not z.grad.any()


def test_checkpoint_matches_outputs_and_parameter_gradients():
    a = ubr.UnifiedBoundaryRefiner(band_width=2)
    b = ubr.UnifiedBoundaryRefiner(band_width=2, checkpoint_blocks=True)
    with torch.no_grad():
        a.out.weight.normal_(std=.01)
    b.load_state_dict(a.state_dict())
    feat, z, sdt, neighbors, _ = inputs(1)
    ra, rb = a(feat, z, sdt, neighbors), b(feat, z, sdt, neighbors)
    assert torch.equal(ra, rb)
    ra.sum().backward()
    rb.sum().backward()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa.grad, pb.grad)


def test_invalid_shapes_and_loss_contract():
    model = ubr.UnifiedBoundaryRefiner()
    feat, z, sdt, neighbors, band = inputs()
    with pytest.raises(ValueError, match='grid'):
        model(feat[:, :, :-1], z, sdt, neighbors)
    with pytest.raises(ValueError, match='geometry'):
        model(feat, z, sdt, neighbors, torch.zeros(2, 4, 24, 24))
    with pytest.raises(ValueError, match='integer'):
        ubr.band_mask(sdt, .5)
    with pytest.raises(ValueError, match='outside'):
        ubr.boundary_loss(z, z.sigmoid(), band, torch.tensor([0, 3]), 2)
    with pytest.raises(ValueError, match='boolean'):
        ubr.boundary_loss(z, z.sigmoid(), band.float(), torch.tensor([0, 1]), 2)
