#!/usr/bin/env python
"""AMP/checkpoint regression: detached diagnostics preserve outputs and gradients."""
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ["RSPROMPTER_LIGHT_IMPORT"] = "1"
import numpy as np
import torch
from mmengine.structures import BaseDataElement
from rsprompter.uav.contract import geometry_config, geometry_signature
from rsprompter.uav.geometry import build_geometry
import rsprompter.uav.cvt as cvt


def main():
    torch.manual_seed(42)
    config = geometry_config(
        dict(side_m=64.0, grid_size=16, image_size=64, radius_m=12.0)
    )
    k = np.array([[100.0, 0, 32], [0, 100.0, 32], [0, 0, 1.0]])
    e = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 100.0], [0, 0, 0, 1.0]])
    camera = (k, e, np.linalg.inv(e), (64, 64))
    arrays, _ = build_geometry(
        [np.full((64, 64), 100.0)] * 5, [camera] * 5, camera, np.zeros(3), config
    )
    payload = BaseDataElement(
        metainfo={"geometry_signature": geometry_signature(config)}
    )
    for key, value in arrays.items():
        setattr(payload, key, torch.from_numpy(value.copy()).cuda())
    first = cvt.DepthCVT(config, layers=2, query_chunk=32).cuda().train()
    second = copy.deepcopy(first)
    features = torch.randn(1, 5, 256, 8, 8, device="cuda")
    original = cvt.sampled_attention_stats
    outputs = []
    gradients = []
    for model, enabled in [(first, True), (second, False)]:
        cvt.sampled_attention_stats = original if enabled else (lambda *args: {})
        source = features.clone().requires_grad_(True)
        with torch.amp.autocast("cuda"):
            out = model(source, [payload])
            loss = out.float().square().mean() * 512.0
        loss.backward()
        outputs.append(out.detach())
        gradients.append(
            {
                name: p.grad.detach().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
        )
        gradients[-1]["input"] = source.grad.detach().clone()
    cvt.sampled_attention_stats = original
    assert torch.equal(outputs[0], outputs[1])
    assert gradients[0].keys() == gradients[1].keys()
    error = max(
        float((gradients[0][key] - gradients[1][key]).abs().max())
        for key in gradients[0]
    )
    for key in gradients[0]:
        assert torch.allclose(
            gradients[0][key], gradients[1][key], atol=1e-5, rtol=1e-4
        ), key
    assert any(
        "relative" in key and gradients[0][key].count_nonzero() for key in gradients[0]
    )
    result = dict(
        outputs_identical=True,
        gradient_keys_identical=True,
        max_gradient_abs_difference=error,
        relative_mlp_gradients_nonzero=True,
        amp_checkpoint=True,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
