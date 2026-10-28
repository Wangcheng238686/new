"""Deterministic tensor gates; no data, checkpoint, GPU, or training launch."""
import json
import os
from pathlib import Path
import sys

os.environ['RSPROMPTER_LIGHT_IMPORT'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from rsprompter.uav.native_geometry_probe import NativeGeometryAdapter
from rsprompter.uav.native_geometry_mechanism import GeometryMechanismAdapter


def main():
    torch.set_num_threads(1)
    torch.manual_seed(3044)
    old = NativeGeometryAdapter().eval()
    with torch.no_grad():
        old.alpha.fill_(.37)
    q = torch.randn(1, 32, 16, 16)
    high = torch.randn(1, 256, 16, 16)
    support = (torch.rand(1, 1, 512, 512) > .25).float()
    fields = torch.randn(1, 4, 512, 512) * support
    boxes = torch.tensor([[0., 0., 512., 512.], [-10., -10., 8., 8.],
                          [504., 504., 520., 520.], [100., 110., 170., 210.],
                          [-500., -500., -300., -300.]])
    base = torch.randn(len(boxes), 32, 8, 8)
    tests = {}
    new = GeometryMechanismAdapter().eval()
    new.load_state_dict(old.state_dict(), strict=True)
    assert tuple(new.state_dict()) == tuple(old.state_dict())
    assert sum(p.numel() for p in new.parameters()) == 19681
    assert torch.equal(old._chunk(q, fields, support, boxes, 8),
                       new._chunk(q, fields, support, boxes, 8))
    for model in (old, new):
        model.encode(None, support, fields, high)
        model.boxes = boxes
    assert torch.equal(old._inject(None, None, base), new._inject(None, None, base))
    assert old.stats == new.stats
    tests['legacy_bilinear_concat_state_chunk_injection_stats_exact'] = True

    for sampling in ('bilinear', 'nearest'):
        for fusion in ('concat', 'difference'):
            model = GeometryMechanismAdapter(sampling, fusion).eval()
            model.load_state_dict(old.state_dict(), strict=True)
            zeros = torch.zeros_like(fields)
            assert torch.count_nonzero(model._chunk(q, zeros, torch.zeros_like(support), boxes, 8)) == 0
            model.encode(None, torch.zeros_like(support), zeros, high)
            model.boxes = boxes
            assert torch.equal(model._inject(None, None, base), base)
            # A fully out-of-bounds box is unsupported in both reading modes.
            assert torch.count_nonzero(model._chunk(q, fields, support, boxes[-1:], 8)) == 0
            if fusion == 'difference':
                for amp in (False, True):
                    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                        delta = model._chunk(q, zeros, support, boxes, 8)
                        model.encode(None, support, zeros, high)
                        result = model._inject(None, None, base)
                    assert torch.count_nonzero(delta) == 0
                    assert torch.equal(result, base)
                # Constant nonzero geometry must not be silently zeroed: this
                # is why the experimental matrix needs a trained content control.
                constant = torch.full_like(fields, .5) * support
                assert torch.count_nonzero(model._chunk(q, constant, support, boxes, 8)) > 0
                model.train()
                model.encode(None, support, zeros, high)
                model.boxes = boxes
                # Exercise inherited checkpointing, nonzero alpha, and gradients.
                result = model._inject(None, None, base)
                assert torch.equal(result, base)
                result.sum().backward()
                assert model.alpha.grad is not None and model.alpha.grad.item() == 0
            tests[f'{sampling}_{fusion}_support_bounds_and_identity'] = True

    nearest = GeometryMechanismAdapter('nearest', 'difference')
    bilinear = GeometryMechanismAdapter('bilinear', 'difference')
    step = torch.zeros(1, 4, 512, 512)
    step[:, 0, :, 256:] = 1.
    full = torch.ones_like(support)
    # Pixel-edge coordinate x=256 lies halfway between raster centers 255/256.
    crossing = torch.tensor([[255.9, 100., 256.1, 101.]])
    nq, ng, nv = nearest._sample_inputs(q, step, full, crossing, 1)
    bq, bg, bv = bilinear._sample_inputs(q, step, full, crossing, 1)
    assert torch.equal(nq, bq)
    assert ng[0, 0, 0, 0].item() in (0., 1.)
    assert abs(bg[0, 0, 0, 0].item() - .5) < 1e-5
    assert nv.all() and bv.all()
    tests['height_step_nearest_no_average_sat_query_unchanged'] = True
    # A zero-height supported point remains valid. An unknown neighbor does not.
    support_hole = full.clone()
    support_hole[:, :, :, 256:] = 0
    points = torch.tensor([[254.5, 100., 255.5, 101.], [256.5, 100., 257.5, 101.]])
    _, sampled, valid = nearest._sample_inputs(q, torch.zeros_like(step), support_hole, points, 1)
    assert torch.count_nonzero(sampled) == 0
    assert valid[0].all() and not valid[1].any()
    tests['valid_zero_height_distinguished_from_unknown'] = True
    for bad in ({'sampling': 'bad'}, {'fusion': 'bad'}):
        try:
            GeometryMechanismAdapter(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid configuration accepted')
    tests['invalid_config_rejected'] = True
    print(json.dumps({'passed': True, 'device': 'cpu', 'amp_dtype': 'bfloat16',
                      'parameters': 19681, 'tests': tests}, indent=2))


if __name__ == '__main__':
    main()
