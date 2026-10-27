#!/usr/bin/env python3
"""Four-GPU synthetic coordinate bridge gate against real installed MMDet.

Not a detector training, dataset-alignment, geometry, AP or resource gate.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from mmengine import ConfigDict
from mmdet.models.roi_heads.mask_heads.fcn_mask_head import FCNMaskHead


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get('WORLD_SIZE', 0)) != 4:
        raise RuntimeError('requires four torchrun processes')
    rank = int(os.environ['RANK'])
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', timeout=timedelta(minutes=3))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    source = Path(__file__).resolve().parents[2] / 'rsprompter/uav/ubr_coordinate_adapter.py'
    spec = importlib.util.spec_from_file_location('ubr_bridge_gate', source)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    torch.manual_seed(44 + rank)
    boxes = torch.tensor([[1.25, 3.75, 110.5, 120.1], [-5., -3., 24.8, 38.2],
                          [107., 110., 130., 132.]], device=device)
    meta = dict(ori_shape=(128, 128), scale_factor=(1., 1.))
    checks = []
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        z = torch.randn(3, 1, 16, 16, device=device, dtype=dtype).requires_grad_()
        features = torch.randn(3, 32, 16, 16, device=device, dtype=dtype).requires_grad_()
        with torch.autocast('cuda', dtype=torch.float16):
            state = bridge.adapt_native_roi(z, features, boxes, meta)
        assert state.logits.dtype == state.decoder_features.dtype == torch.float32
        restored = bridge.residual_probabilities(state, state.logits.clone())
        class Head:
            class_agnostic = True
        for threshold in (.4, .5, -1.):
            expected = FCNMaskHead._predict_by_feat_single(
                Head(), z.detach(), boxes.clone(), torch.zeros(3, dtype=torch.long, device=device),
                meta, ConfigDict(mask_thr_binary=threshold), rescale=False)
            actual = (restored[:, 0] >= threshold if threshold >= 0 else
                      (restored[:, 0] * 255).to(torch.uint8))
            assert torch.equal(actual, expected), (rank, dtype, threshold)
        error = (state.logits.sigmoid() - state.probabilities).abs().max().item()
        assert error <= 1.1e-6
        (state.logits.square().mean() + state.decoder_features.square().mean()).backward()
        for gradient in (z.grad, features.grad):
            assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        checks.append(dict(dtype=str(dtype), max_roundtrip_error=error,
                           exact_binary_and_uint8=True, gradient_finite_nonzero=True))
    t = torch.tensor(.4, device=device)
    p = torch.stack((torch.nextafter(t, t.new_tensor(0.)), t,
                     torch.nextafter(t, t.new_tensor(1.)), t.new_tensor(0.),
                     t.new_tensor(1.))).reshape(1, 1, 1, 5)
    state = bridge.RoIImageState(p, bridge.finite_probability_logits(p),
                               torch.zeros(1, 32, 1, 5, device=device), boxes[:1], (1, 5), 1e-6)
    restored = bridge.residual_probabilities(state, state.logits)
    assert torch.equal(restored, p) and torch.equal(restored >= .4, p >= .4)
    report = dict(rank=rank, checks=checks, threshold_ties_exact=True,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='synthetic_adapter_only_not_detector_or_AP')
    reports = [None] * 4
    dist.all_gather_object(reports, report)
    if rank == 0:
        (args.output / 'complete.json').write_text(json.dumps(reports, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
