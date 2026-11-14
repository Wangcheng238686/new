"""Standalone native-UBR tail install for the paper single-stream model.

Port lineage (2026-11-14): ``rsprompter/uav/native_roi_refiner.py`` (byte
copy -> ``rsprompter/native_ubr_refiner.py``, sha256 aa057e3e..., commit
ed8bd4c6) and ``rsprompter/uav/ubr_native_integration.py`` (commit abc51d74)
from branch codex/uav-bev-dual. The installed tail is the single-stream
S-arm zero-evidence form used by the dual-stream comparison family: the
6,625-param ``NativeRoIRefiner`` post-hoc residual head, seeded 2080, with
its zero-initialised output conv giving exact logit identity at install.

Adaptations for this post-restructure single-stream package:
- No UAV evidence adapter: zero-evidence only (rgb/geometry stay None).
- ``final_mask_coordinate_mode`` follows each dataset's PAPER config
  (WHU fast150 = roi_local, NWPU matrix300 = full_image). The per-RoI
  decoder pass and its native lattice are structurally identical in both
  modes (see cspm/mask_head.py forward: the mode only changes downstream
  mask consumption), so one shim serves both; the mode is RECORDED, not
  asserted to roi_local like the dual-side installer.
- ``RSPrompterAnchorMaskHeadSAM2.forward`` here receives ``boxes`` /
  ``roi_img_ids`` as head kwargs and returns ``MaskHeadOutput``; the shim
  context is captured from those kwargs around ``super().forward``.
- ``RSPrompterAnchorRoIPromptHead._empty_rank_losses`` returns a
  ``(losses, telemetry)`` tuple in this package (its ``-> dict`` annotation
  is stale); the empty-rank graph keeper mirrors the dual-side tuple
  contract.
- The 512 mask-size guard is widened to the paper 1024 family (the refiner
  is fully convolutional; 1024 input -> native lattice 256).

Default behaviour of every existing entry point is unchanged: installation
only happens when explicitly requested (trainer ``--native-ubr`` flag or
the inference marker branch).
"""
import weakref

import torch
from torch import nn

from .cspm.mask_head import RSPrompterAnchorMaskHeadSAM2
from .detection.anchor_roi_head import RSPrompterAnchorRoIPromptHead
from .native_ubr_refiner import NativeRoIRefiner

UBR_TAIL_SEED = 2080
UBR_EXPECTED_PARAMS = 6625
UBR_MARKER = "native_ubr_v1"


class NativeMaskHead(RSPrompterAnchorMaskHeadSAM2):
    """Head wrapper that feeds proposal context to the decoder shim."""

    def forward(self, *args, **kwargs):
        shim = self.mask_decoder
        if not isinstance(shim, NativeDecoderShim):
            raise RuntimeError("NativeMaskHead requires its own local decoder shim")
        if shim.call_context is not None:
            raise RuntimeError("Native mask head is not reentrant")
        shim.call_context = (kwargs.get("boxes"), kwargs.get("roi_img_ids"))
        try:
            return super().forward(*args, **kwargs)
        finally:
            shim.call_context = None
            shim.captured = None


class NativeRoIHead(RSPrompterAnchorRoIPromptHead):
    """Keep the refiner parameters in the DDP graph on empty-positive ranks."""

    def _empty_rank_losses(self, x):
        # Host contract (anchor_roi_head.py): returns (losses, telemetry) —
        # the "-> dict" annotation there is stale. Mirror the dual-side
        # installer: add a zero-graph term on the existing loss_mask entry.
        losses, telemetry = super()._empty_rank_losses(x)
        head = getattr(self.mask_head, "native_refiner", None)
        if head is not None:
            losses["loss_mask"] = losses["loss_mask"] + sum(
                p.sum() * 0 for p in head.parameters()
            )
        return losses, telemetry


class NativeDecoderShim(nn.Module):
    """Capture the decoder's own act2 and add the zero-evidence UBR residual.

    The wrapped decoder keeps its original three-tensor contract (UDPR's
    ``capture_upscaled_embedding`` must stay off). Capture is call-local via
    a forward hook on ``output_upscaling[-1]``; act2 shares the native mask
    lattice and RoI order of the returned logits, exactly like upstream.
    """

    def __init__(self, base, mask_head):
        super().__init__()
        self.base = base
        # Weak references avoid duplicate parameter registration and ownership.
        object.__setattr__(self, "_head", weakref.ref(mask_head))
        # Expose the decoder contract flag the installer/verifier probes: the
        # shim preserves the original three-tuple (non-capturing) contract.
        self.capture_upscaled_embedding = False
        self.call_context = None
        self.captured = None
        self.calls = 0
        self.observation = {}
        self.observe = False

    def forward(self, *args, **kwargs):
        if self.call_context is None or self.captured is not None:
            raise RuntimeError("Missing native decoder context or stale activation")
        boxes, image_ids = self.call_context

        def capture(module, inputs, output):
            if self.captured is not None:
                raise RuntimeError("More than one act2 in a decoder call")
            self.captured = output

        handle = self.base.decoder.output_upscaling[-1].register_forward_hook(capture)
        try:
            result = self.base(*args, **kwargs)
            if len(result) != 3:
                raise RuntimeError("Original decoder must return3; UDPR capture disabled")
            logits, iou, tokens = result
            feature = self.captured
            n = logits.shape[0]
            if feature is None or feature.shape != (n, 32, *logits.shape[-2:]):
                raise RuntimeError(
                    "Captured act2 must share native mask lattice and ROI order"
                )
            if boxes is not None and len(boxes) != n:
                raise RuntimeError("Proposal context must match decoder RoI order")
            if image_ids is not None and (
                not torch.isfinite(image_ids).all()
                or not torch.equal(image_ids, image_ids.long().to(image_ids.dtype))
            ):
                raise ValueError("RoI image IDs must be finite integer values")
            head = self._head()
            # Zero-evidence S form: rgb/geometry absent -> refiner zero-fills.
            refined = head.native_refiner(feature, logits, None, None)
            self.calls += 1
            self.observation = dict(
                native_hw=list(logits.shape[-2:]),
                instances=n,
                zero_evidence=True,
                decoder_calls=self.calls,
            )
            if self.observe:
                self.observation.update(
                    raw=logits.detach().clone(), refined=refined.detach().clone()
                )
            return refined, iou, tokens
        finally:
            handle.remove()
            self.captured = None


def verify_ubr_ready(head):
    """Fail fast on any contract violation BEFORE touching the model."""
    # Installed first: after install the head is a NativeMaskHead whose
    # mask_decoder is the shim (it reports capture_upscaled_embedding=False).
    if isinstance(head, NativeMaskHead):
        raise ValueError("Already installed")
    if head.decoder_tail_enabled or head.decoder_tail_refiner is not None:
        raise ValueError("Disable UDPR (decoder_tail_refiner_cfg.enabled=False) before install")
    mode = head.final_mask_coordinate_mode
    if mode not in ("roi_local", "full_image"):
        raise ValueError(f"Unsupported final_mask_coordinate_mode: {mode!r}")
    if head.mask_decoder.capture_upscaled_embedding:
        raise ValueError("Original 3-tuple decoder contract required")
    return mode


def install_native_ubr_tail(model, seed=UBR_TAIL_SEED):
    """Install the S-arm zero-evidence native UBR tail on a built model.

    Must run after ``MODELS.build`` + ``.to(device)`` and before DDP wrap /
    optimizer build. Returns a receipt dict for logging and audits.
    """
    head = model.roi_head.mask_head
    mode = verify_ubr_ready(head)
    device = next(head.parameters()).device
    devices = [torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        head.add_module("native_refiner", NativeRoIRefiner().to(device))
    params = sum(p.numel() for p in head.native_refiner.parameters())
    if params != UBR_EXPECTED_PARAMS:
        raise RuntimeError(f"native UBR param count {params} != {UBR_EXPECTED_PARAMS}")
    head.mask_decoder = NativeDecoderShim(head.mask_decoder, head)
    head.__class__ = NativeMaskHead
    model.roi_head.__class__ = NativeRoIHead
    return dict(
        marker=UBR_MARKER,
        architecture="native_ubr_zero_evidence_std_v1",
        params=params,
        keys=sorted(k for k in head.state_dict() if k.startswith("native_refiner.")),
        coordinate_mode=mode,
        seed=int(seed),
        zero_init_out_conv=True,
    )
