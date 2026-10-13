"""Tracking-only exponential moving average of model weights."""

from typing import Dict

import torch

from distributed import _get_model_core
from optimizer import _assert_no_mask_embedding_contract


class ExponentialMovingAverage:
    """EMA shadow weights used only for eval/export, never for backprop."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"EMA decay must be in (0,1), got {decay}")
        self.decay = float(decay)
        model_core = _get_model_core(model)
        state = model_core.state_dict()
        self.ema_state = {k: v.detach().clone() for k, v in state.items()}
        self.backup_state = None
        # Frozen tensors (params with requires_grad=False, plus buffers) must
        # be copied verbatim, never averaged: averaging a constant across tens
        # of thousands of steps accumulates fp32 rounding drift (~1e-6 per
        # the fast-150 run), which trips strict pretrained-value contract
        # checks (no_mask_embed) when the EMA state is loaded for inference.
        _frozen = {n for n, p in model_core.named_parameters() if not p.requires_grad}
        _frozen.update(n for n, _ in model_core.named_buffers())
        self._frozen_keys = frozenset(k for k in state if k in _frozen)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        model_state = _get_model_core(model).state_dict()
        for key, value in model_state.items():
            value_detached = value.detach()
            if key not in self.ema_state:
                self.ema_state[key] = value_detached.clone()
                continue
            ema_value = self.ema_state[key]
            if (
                torch.is_floating_point(value_detached)
                and key not in self._frozen_keys
            ):
                ema_value.mul_(self.decay).add_(value_detached, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(value_detached)

        stale_keys = [k for k in self.ema_state.keys() if k not in model_state]
        for stale_key in stale_keys:
            self.ema_state.pop(stale_key, None)

    def apply_to(self, model: torch.nn.Module):
        model_core = _get_model_core(model)
        current_state = model_core.state_dict()
        self.backup_state = {k: v.detach().clone() for k, v in current_state.items()}
        model_core.load_state_dict(self.ema_state, strict=False)
        _assert_no_mask_embedding_contract(model_core, "EMA state load")

    def restore(self, model: torch.nn.Module):
        if self.backup_state is None:
            return
        model_core = _get_model_core(model)
        model_core.load_state_dict(self.backup_state, strict=False)
        _assert_no_mask_embedding_contract(model_core, "EMA state restore")
        self.backup_state = None

    def state_dict(self) -> Dict:
        return {
            "decay": self.decay,
            "ema_state": {k: v.detach().clone() for k, v in self.ema_state.items()},
        }

    def load_state_dict(self, state: Dict):
        self.decay = float(state.get("decay", self.decay))
        loaded = state.get("ema_state", {})
        default_device = None
        for tensor in self.ema_state.values():
            default_device = tensor.device
            break

        for key in list(self.ema_state.keys()):
            if key in loaded:
                target = self.ema_state[key]
                source = loaded[key].detach().to(device=target.device, dtype=target.dtype)
                target.copy_(source)
        for key, value in loaded.items():
            if key not in self.ema_state:
                source = value.detach()
                if default_device is not None:
                    source = source.to(device=default_device)
                self.ema_state[key] = source.clone()
