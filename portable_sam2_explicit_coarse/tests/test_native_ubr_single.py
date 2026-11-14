"""Unit tests for the standalone native-UBR tail port (EXP-0046/0047).

Light tests cover the refiner module (param count / zero-init identity),
installer guards and receipts, RNG locality, and the decoupled-aux-clip
partition. Heavy tests (skipped when SAM2 assets are missing) build the two
paper delta configs on CPU, install the tail, and check the shimmed
state-dict layout plus the paper-fingerprint parity gate.

Run: /data2/wangcheng/envs/cvt2/bin/python -m pytest tests/test_native_ubr_single.py -v
(from portable_sam2_explicit_coarse/)
"""
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from rsprompter.native_ubr_refiner import NativeRoIRefiner
from rsprompter.native_ubr_integration import (  # noqa: E402
    UBR_EXPECTED_PARAMS,
    NativeDecoderShim,
    NativeMaskHead,
    install_native_ubr_tail,
    verify_ubr_ready,
)


# ---------------------------------------------------------------------------
# light: refiner module
# ---------------------------------------------------------------------------

def test_refiner_param_count_and_zero_identity():
    torch.manual_seed(0)
    refiner = NativeRoIRefiner()
    assert sum(p.numel() for p in refiner.parameters()) == UBR_EXPECTED_PARAMS == 6625
    assert float(refiner.out.weight.abs().sum()) == 0.0
    assert float(refiner.out.bias.abs().sum()) == 0.0
    logits = torch.randn(4, 1, 64, 64)
    feature = torch.randn(4, 32, 64, 64)
    out = refiner(feature, logits, None, None)
    assert out.shape == logits.shape
    # Zero-initialised output conv: exact native logit identity at install.
    assert torch.equal(out, logits)


def test_refiner_zero_identity_batched_lattice():
    refiner = NativeRoIRefiner()
    for n, h in ((0, 64), (3, 128), (2, 256)):
        logits = torch.randn(n, 1, h, h)
        feature = torch.randn(n, 32, h, h)
        out = refiner(feature, logits, None, None)
        assert torch.equal(out, logits)


def test_refiner_rejects_bad_shapes():
    refiner = NativeRoIRefiner()
    logits = torch.randn(4, 1, 64, 64)
    with pytest.raises(ValueError):
        refiner(torch.randn(4, 32, 32, 32), logits, None, None)
    with pytest.raises(ValueError):
        refiner(torch.randn(4, 32, 64, 64), torch.randn(4, 2, 64, 64), None, None)


# ---------------------------------------------------------------------------
# light: installer guards / receipt / RNG locality
# ---------------------------------------------------------------------------

class _FakeDecoder:
    def __init__(self, capture=False):
        self.capture_upscaled_embedding = capture


class _FakeHead(nn.Module):
    def __init__(self, mode="roi_local", udpr=False, capture=False):
        super().__init__()
        self.decoder_tail_enabled = udpr
        self.decoder_tail_refiner = object() if udpr else None
        self.final_mask_coordinate_mode = mode
        self.mask_decoder = _FakeDecoder(capture)
        self.dummy = nn.Parameter(torch.zeros(1))


class _FakeRoIHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.mask_head = head


class _FakeModel(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.roi_head = _FakeRoIHead(head)


def test_install_receipt_records_mode_and_keys():
    for mode in ("roi_local", "full_image"):
        model = _FakeModel(_FakeHead(mode=mode))
        receipt = install_native_ubr_tail(model, seed=123)
        assert receipt["params"] == 6625
        assert receipt["coordinate_mode"] == mode
        assert receipt["seed"] == 123
        assert receipt["keys"] and all(
            key.startswith("native_refiner.") for key in receipt["keys"]
        )
        assert isinstance(model.roi_head.mask_head, NativeMaskHead)
        with pytest.raises(ValueError):
            verify_ubr_ready(model.roi_head.mask_head)  # already installed


def test_install_guards():
    with pytest.raises(ValueError):
        verify_ubr_ready(_FakeHead(udpr=True))
    with pytest.raises(ValueError):
        verify_ubr_ready(_FakeHead(mode="weird_mode"))
    with pytest.raises(ValueError):
        verify_ubr_ready(_FakeHead(capture=True))


def test_install_seed_is_rng_local():
    torch.manual_seed(7)
    state_before = torch.get_rng_state()
    model_a = _FakeModel(_FakeHead())
    install_native_ubr_tail(model_a, seed=2080)
    assert torch.equal(state_before, torch.get_rng_state())
    # Same seed -> identical refiner weights.
    model_b = _FakeModel(_FakeHead())
    install_native_ubr_tail(model_b, seed=2080)
    for (ka, va), (kb, vb) in zip(
        model_a.roi_head.mask_head.native_refiner.state_dict().items(),
        model_b.roi_head.mask_head.native_refiner.state_dict().items(),
    ):
        assert ka == kb and torch.equal(va, vb)


def test_aux_clip_partition_separates_refiner():
    sys.path.insert(0, str(PKG / "train"))
    from optimizer import _AUX_CLIP_MODULE_PREFIXES, _partition_aux_clip_params

    model = _FakeModel(_FakeHead())
    install_native_ubr_tail(model, seed=2080)
    assert any("native_refiner." in prefix for prefix in _AUX_CLIP_MODULE_PREFIXES)
    base, aux_groups = _partition_aux_clip_params(model)
    refiner_ids = {
        id(p) for p in model.roi_head.mask_head.native_refiner.parameters()
    }
    grouped_ids = {id(p) for group in aux_groups for p in group}
    assert refiner_ids == grouped_ids
    assert not refiner_ids & {id(p) for p in base}


def test_shim_context_guards():
    # A NativeMaskHead without its own shim refuses to forward.
    head = _FakeHead()
    head.__class__ = NativeMaskHead
    with pytest.raises(RuntimeError):
        head.forward()
    # The shim refuses to forward without an active call context.
    head2 = _FakeHead()
    shim = NativeDecoderShim(head2.mask_decoder, head2)
    with pytest.raises(RuntimeError):
        shim.forward()


# ---------------------------------------------------------------------------
# heavy: full CPU build of both delta configs + parity gate
# ---------------------------------------------------------------------------

_SAM2 = os.environ.get(
    "SAM2_CKPT", "/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt"
)
_SAM2_REPO = os.environ.get("SAM2_REPO", str(PKG.parent / "sam2"))
_HEAVY_OK = Path(_SAM2).exists() and Path(_SAM2_REPO).exists()

_DELTAS = [
    (
        "configs/ablations/whu1024_ubr_fast150_20261114.py",
        "configs/a3sgm_whu1024.py",
        "5d5f78ec947fba1c5a92fb28a8486c9b8d7050c66c0034abf8425d1034bce17b",
        "roi_local",
    ),
    (
        "configs/ablations/vhr10_ubr_matrix300_20261114.py",
        "configs/a3sgm_vhr10.py",
        "ed7a422ddf368874a261e7b176d32da4208b1972c9cac9cc2ca1208a3a90dd23",
        "full_image",
    ),
]


@pytest.mark.skipif(not _HEAVY_OK, reason="SAM2 assets unavailable on this host")
@pytest.mark.parametrize("delta,paper,paper_fp,mode", _DELTAS)
def test_full_build_install_and_layout(delta, paper, paper_fp, mode):
    os.environ.setdefault("SAM2_REPO", _SAM2_REPO)
    import mmdet.models  # noqa: F401  DetDataPreprocessor registration
    from mmengine.config import Config
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from mmdet.registry import MODELS

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor
        )
    import rsprompter  # noqa: F401  registry registration
    from rsprompter.architecture_contract import architecture_fingerprint

    # Parity gate: the PAPER config on this code must reproduce its recorded
    # fingerprint (no silent model drift besides UDPR->UBR).
    assert architecture_fingerprint(Config.fromfile(str(PKG / paper)).model.to_dict()) == paper_fp

    cfg = Config.fromfile(str(PKG / delta))
    model = MODELS.build(cfg.model)
    head = model.roi_head.mask_head
    assert head.decoder_tail_enabled is False
    assert head.decoder_tail_refiner is None
    assert not head.mask_decoder.capture_upscaled_embedding
    assert head.final_mask_coordinate_mode == mode

    receipt = install_native_ubr_tail(model)
    assert receipt["params"] == 6625
    assert receipt["coordinate_mode"] == mode

    names = set(model.state_dict())
    assert any(".native_refiner." in name for name in names)
    # Shim wraps the decoder: keys live under mask_decoder.base.*, and no
    # unshimmed decoder keys remain (strict-load layout round-trip).
    assert any(
        name.startswith("roi_head.mask_head.mask_decoder.base.") for name in names
    )
    assert not any(
        name.startswith("roi_head.mask_head.mask_decoder.")
        and ".base." not in name
        for name in names
    )


@pytest.mark.skipif(not _HEAVY_OK, reason="SAM2 assets unavailable on this host")
def test_delta_fingerprint_touches_only_udpr_key():
    from copy import deepcopy

    from mmengine.config import Config
    from rsprompter.architecture_contract import architecture_fingerprint

    for delta, paper, _fp, _mode in _DELTAS:
        delta_model = Config.fromfile(str(PKG / delta)).model.to_dict()
        paper_model = Config.fromfile(str(PKG / paper)).model.to_dict()
        udpr_key = "decoder_tail_refiner_cfg"
        # The disabled tail cfg must be exactly {enabled: False} (head guard).
        assert delta_model["roi_head"]["mask_head"][udpr_key] == {"enabled": False}
        # Restoring the paper's UDPR cfg into the delta must reproduce the
        # paper fingerprint: the delta differs from the paper ONLY at this key
        # (checkpoint paths are fingerprint-normalized, schema v3).
        restored = deepcopy(delta_model)
        restored["roi_head"]["mask_head"][udpr_key] = deepcopy(
            paper_model["roi_head"]["mask_head"][udpr_key]
        )
        assert (
            architecture_fingerprint(restored)
            == architecture_fingerprint(paper_model)
        )


@pytest.mark.skipif(not _HEAVY_OK, reason="SAM2 assets unavailable on this host")
def test_empty_rank_losses_tuple_contract_and_refiner_graph():
    # F1 regression: the host hook returns (losses, telemetry) despite its
    # stale "-> dict" annotation; the override must keep that tuple contract
    # and keep the refiner in the DDP graph on empty-positive ranks.
    os.environ.setdefault("SAM2_REPO", _SAM2_REPO)
    import mmdet.models  # noqa: F401
    from mmengine.config import Config
    from mmengine.registry import MODELS as MMENGINE_MODELS
    from mmdet.models.data_preprocessors import DetDataPreprocessor
    from mmdet.registry import MODELS

    if "DetDataPreprocessor" not in MMENGINE_MODELS:
        MMENGINE_MODELS.register_module(
            name="DetDataPreprocessor", module=DetDataPreprocessor
        )
    import rsprompter  # noqa: F401

    cfg = Config.fromfile(str(PKG / "configs/ablations/vhr10_ubr_matrix300_20261114.py"))
    model = MODELS.build(cfg.model)
    install_native_ubr_tail(model)

    x = (torch.zeros(1, 512, 8, 8),)
    out = model.roi_head._empty_rank_losses(x)
    assert isinstance(out, tuple) and len(out) == 2
    losses, telemetry = out
    assert "loss_mask" in losses and torch.is_tensor(losses["loss_mask"])
    losses["loss_mask"].backward()
    refiner_grads = [
        p.grad for p in model.roi_head.mask_head.native_refiner.parameters()
    ]
    assert any(grad is not None for grad in refiner_grads)
