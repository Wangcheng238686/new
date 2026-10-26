"""Factorial geometry-reading and attribution probes, using the native v1 cache.

This changes neither the camera projection nor the four cached fields.  The
default bilinear/concat path delegates to the archived implementation exactly.
"""
import torch
from torch.nn import functional as F

from .native_geometry_probe import NativeGeometryAdapter


class GeometryMechanismAdapter(NativeGeometryAdapter):
    """One existing injection site with independent sampling/fusion choices.

    ``nearest`` means nearest raster sample, not a search for nearby support.
    An unsupported nearest pixel remains unsupported and contributes no delta.
    ``difference`` removes the unconditional satellite-only residual, but a
    nonzero constant field can still condition the satellite feature transform.
    The choices are metadata; callers must save them in the experiment contract.
    Parameter names/shapes and state_dict are identical to the native adapter.
    """

    def __init__(self, sampling='bilinear', fusion='concat'):
        if sampling not in ('bilinear', 'nearest'):
            raise ValueError(f'Unknown geometry sampling: {sampling}')
        if fusion not in ('concat', 'difference'):
            raise ValueError(f'Unknown geometry fusion: {fusion}')
        super().__init__()
        self.sampling = sampling
        self.fusion = fusion

    def _sample_inputs(self, qimage, fields, support, boxes, size):
        line = (torch.arange(size, device=boxes.device, dtype=torch.float32) + .5) / size
        yy, xx = torch.meshgrid(line, line, indexing='ij')
        uv = boxes[:, None, None, :2] + torch.stack((xx, yy), -1)[None] * (
            boxes[:, None, None, 2:] - boxes[:, None, None, :2])
        grid = uv / 256 - 1
        q = F.grid_sample(qimage.expand(len(boxes), -1, -1, -1),
                          grid.to(qimage.dtype), align_corners=False)
        if self.sampling == 'bilinear':
            cov = F.grid_sample(support.expand(len(boxes), -1, -1, -1),
                                grid, align_corners=False)
            geo = F.grid_sample(fields.expand(len(boxes), -1, -1, -1),
                                grid.to(fields.dtype), align_corners=False) / cov.clamp_min(1e-6)
        else:
            geo = F.grid_sample(fields.expand(len(boxes), -1, -1, -1),
                                grid.to(fields.dtype), mode='nearest', align_corners=False)
        valid = F.grid_sample(support.expand(len(boxes), -1, -1, -1),
                              grid, mode='nearest', align_corners=False) > 0
        # Keep the mask separate from height: valid zero-height ground is not
        # equivalent to unknown support. No nearest-neighbor hole filling.
        return q, geo.to(q.dtype), valid

    def _chunk(self, qimage, fields, support, boxes, size):
        if self.sampling == 'bilinear' and self.fusion == 'concat':
            return super()._chunk(qimage, fields, support, boxes, size)
        q, geo, valid = self._sample_inputs(qimage, fields, support, boxes, size)
        residual = self.mixer(torch.cat((q, geo), 1))
        if self.fusion == 'difference':
            reference = self.mixer(torch.cat((q, torch.zeros_like(geo)), 1))
            residual = residual - reference
        return residual * valid
