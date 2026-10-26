"""Opt-in compact internal interaction for foundation-initialized joint training.

No task checkpoint is loaded here. Existing frozen probe sources are unchanged.
The caller owns data identity, optimizer, selection and resume contracts.
"""
from pathlib import Path

import torch
from torch import nn

from .instance_interaction_probe import InstanceInteraction
from .rgb_p0_encoder import RGBP0Encoder


class P0Source(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.p0 = RGBP0Encoder(source)

    def forward(self, image):
        value = self.p0(image)
        return value, (value,)


class CompactInteraction(InstanceInteraction):
    def add_rgb(self, pretrained, seed):
        super().add_rgb(pretrained, seed)
        # Later source stages are discarded before an optimizer is constructed.
        self.rgb_encoder = P0Source(self.rgb_encoder)


class CompactJoint(nn.Module):
    def __init__(self, satellite, foundation, source_seed=2045):
        super().__init__()
        self.satellite = satellite
        self.uav = CompactInteraction()
        self.uav.add_rgb(str(foundation), source_seed)
        self.uav.internal = True
        self.uav.attach(self.satellite)

    def cuda(self, device=None):
        # MMEngine BaseModel updates preprocessor._device in its own transfer
        # methods; a parent nn.Module recursive _apply bypasses that override.
        self.satellite.cuda(device=device)
        self.uav.cuda(device=device)
        return self

    def cpu(self):
        self.satellite.cpu()
        self.uav.cpu()
        return self

    def to(self, *args, **kwargs):
        self.satellite.to(*args, **kwargs)
        self.uav.to(*args, **kwargs)
        return self

    def prepare_uav(self, geometry, rgb, index, anchor):
        assert geometry.shape == (1, 5, 512, 512)
        assert rgb.shape == (5, 3, 512, 512)
        assert index.shape == (16384, 8) and anchor.shape == (16384, 8, 6)
        assert anchor.dtype == torch.float64
        self.uav.rgb_context = (rgb, index, anchor)
        self.uav.encode(geometry)

    def loss(self, sample, geometry, rgb, index, anchor, instance_target, center_loss_fn):
        self.prepare_uav(geometry, rgb, index, anchor)
        # No no_grad: the satellite LoRA, FPN/heads and decoder retain the
        # existing single-stream trainable scope, while UAV gradients cross S'.
        features = self.satellite.extract_feat(sample['inputs'])
        original_extract = self.satellite.extract_feat
        self.satellite.extract_feat = lambda inputs: features
        try:
            losses = self.satellite.loss(sample['inputs'], sample['data_samples'])
        finally:
            self.satellite.extract_feat = original_extract
        direct, _ = center_loss_fn(self.uav.outputs, instance_target)
        assert 'loss_uav_instances' not in losses
        return dict(losses, loss_uav_instances=direct)

    def forward(self, sample, geometry, rgb, index, anchor, instance_target, center_loss_fn):
        return self.loss(sample, geometry, rgb, index, anchor, instance_target, center_loss_fn)

    def clear(self):
        self.uav.clear()
        self.uav.rgb_context = None


def build_foundation_satellite(register_and_build, config_path, foundation):
    """Fresh sealed architecture, generic SAM2 only; caller fixes initialization RNG."""
    from mmengine import Config
    cfg = Config.fromfile(str(config_path)).model.to_dict()
    foundation = str(Path(foundation).resolve(strict=True))
    cfg['backbone']['checkpoint_path'] = foundation
    cfg['backbone']['init_cfg'] = dict(type='Pretrained', checkpoint=foundation)
    cfg['roi_head']['mask_head']['sam2_mask_decoder']['checkpoint_path'] = foundation
    # PromptEncoder/no-mask embedding read the same generic source via the
    # existing mask-head constructor. The gate audits actual torch.load paths.
    model = register_and_build(cfg)
    assert model.test_cfg.rcnn.max_per_img == 250
    return model, cfg
