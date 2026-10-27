"""CPU fake-shim contracts; real SAM/G geometry and four-rank gates are separate."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from rsprompter.uav.native_roi_refiner import NativeRoIRefiner
from rsprompter.uav.ubr_native_integration import (
    NativeDecoderShim, NativeEvidenceG, NativeMaskHead, NativeRoIHead,
    RSPrompterAnchorMaskHeadSAM2, RSPrompterAnchorRoIPromptHead,
)

torch.set_num_threads(1)


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.output_upscaling = nn.Sequential(nn.Identity(), nn.SiLU())
        self.calls = 0
        self.raise_after_capture = False

    def forward(self, feature):
        self.calls += 1
        value = self.decoder.output_upscaling(feature)
        if self.raise_after_capture:
            raise RuntimeError('fake decode failure after act2')
        return value.mean(1, keepdim=True), value.mean((1, 2, 3)), value.mean((2, 3))


def setup_shim(zero=False):
    base = FakeBase()
    head = nn.Module()
    head.native_refiner = NativeRoIRefiner()
    class Evidence:
        pass
    adapter = Evidence()
    adapter.native_aligned_rgb = torch.randn(1, 32, 128, 128, requires_grad=True)
    adapter.native_rgb_valid = torch.ones(1, 1, 128, 128, dtype=torch.bool)
    adapter.native_geometry = torch.zeros(1, 5, 512, 512)
    shim = NativeDecoderShim(base, head, adapter, zero)
    shim.call_context = (torch.tensor([[0., 0., 512., 512.]]), torch.tensor([0.]))
    return base, head, adapter, shim


def test_same_pass_three_tuple_identity_and_nonzero_gradient():
    base, head, adapter, shim = setup_shim()
    value = torch.randn(1, 32, 5, 7, requires_grad=True)
    output = shim(value)
    assert len(output) == 3 and base.calls == 1 and shim.calls == 1
    assert torch.equal(output[0], torch.nn.functional.silu(value).mean(1, keepdim=True))
    assert shim.captured is None
    assert not base.decoder.output_upscaling[-1]._forward_hooks
    with torch.no_grad():
        head.native_refiner.out.weight.normal_(std=.1)
    result = shim(value)
    result[0].square().mean().backward()
    assert base.calls == 2  # exactly one real decoder call per mask pass
    assert value.grad is not None and value.grad.abs().sum() > 0
    assert adapter.native_aligned_rgb.grad is not None
    assert adapter.native_aligned_rgb.grad.abs().sum() > 0
    assert shim.captured is None


@pytest.mark.parametrize('failure', ['decoder', 'bad_id', 'missing_evidence'])
def test_shim_exception_cleans_hook_and_activation(failure):
    base, head, adapter, shim = setup_shim()
    if failure == 'decoder':
        base.raise_after_capture = True
    elif failure == 'bad_id':
        shim.call_context = (shim.call_context[0], torch.tensor([.25]))
    else:
        adapter.native_aligned_rgb = None
    with pytest.raises((RuntimeError, ValueError)):
        shim(torch.randn(1, 32, 4, 4))
    assert shim.captured is None
    assert not base.decoder.output_upscaling[-1]._forward_hooks


def test_parent_forward_exception_cleans_whole_call_context():
    base, head, adapter, shim = setup_shim()
    fake_head = NativeMaskHead.__new__(NativeMaskHead)
    nn.Module.__init__(fake_head)
    fake_head.mask_decoder = shim
    shim.call_context = None
    def parent(self, *args, **kwargs):
        assert self.mask_decoder.call_context is not None
        self.mask_decoder.captured = torch.ones(1)
        raise RuntimeError('fake parent failure')
    with patch.object(RSPrompterAnchorMaskHeadSAM2, 'forward', parent):
        with pytest.raises(RuntimeError, match='fake parent'):
            NativeMaskHead.forward(fake_head, boxes=torch.zeros(1, 4), roi_img_ids=torch.zeros(1))
    assert shim.call_context is None and shim.captured is None


def test_empty_positive_rank_connects_all_native_parameters():
    refiner = NativeRoIRefiner()
    fake_roi_head = NativeRoIHead.__new__(NativeRoIHead)
    nn.Module.__init__(fake_roi_head)
    fake_roi_head.mask_head = nn.Module()
    fake_roi_head.mask_head.native_refiner = refiner
    feature = torch.randn(1, 2, 3, 3, requires_grad=True)
    def empty_parent(self, x):
        return {'loss_mask': x[0].sum() * 0}, {'marker': 1}
    with patch.object(RSPrompterAnchorRoIPromptHead, '_empty_rank_losses', empty_parent):
        losses, telemetry = NativeRoIHead._empty_rank_losses(fake_roi_head, (feature,))
    assert telemetry == {'marker': 1}
    losses['loss_mask'].backward()
    assert all(p.grad is not None and not p.grad.any() for p in refiner.parameters())


def test_prefusion_rgb_is_original_math_differentiable_and_clear():
    adapter = NativeEvidenceG()
    adapter.variant = 'detail'
    selected = torch.randn(16384, 8, 32, requires_grad=True)
    valid = torch.zeros(16384, 8, dtype=torch.bool)
    valid[10, :2] = True
    valid[25, 3] = True
    satellite = torch.randn(16384, 112)
    geometry = torch.zeros(16384, 8, 7)
    expected = (selected * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    actual = adapter.reduce_rgb(satellite, selected, valid, geometry)
    assert torch.equal(actual, expected)
    assert torch.equal(adapter.native_aligned_rgb, actual.T.reshape(1, 32, 128, 128))
    assert torch.equal(adapter.native_rgb_valid.reshape(-1), valid.any(1))
    adapter.native_aligned_rgb.sum().backward()
    assert selected.grad.abs().sum() > 0
    assert not selected.grad[~valid].any()
    adapter.native_geometry = torch.ones(1, 5, 512, 512)
    adapter.clear()
    assert adapter.native_aligned_rgb is None and adapter.native_rgb_valid is None
    assert adapter.native_geometry is None and adapter.read_context is None
