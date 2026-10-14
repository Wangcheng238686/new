"""Sparse feature pooling after fine-pixel geometric conflict filtering."""

import torch
from torch import nn


class BEVToSatellite(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = int(image_size)

    def forward(self, bev, payloads, target_hw):
        h, w = target_hw
        if self.image_size % h or self.image_size % w:
            raise ValueError("Satellite feature grid must divide the image grid")
        sy, sx = self.image_size // h, self.image_size // w
        outputs, qualities, coverages = [], [], []
        for features, payload in zip(bev, payloads):
            # Accumulate geometry and mixed-precision features in float32.
            source = payload.project_source
            pixel = payload.project_pixel
            target = (pixel // self.image_size // sy) * w + (
                pixel % self.image_size // sx
            )
            mass, quality = payload.project_mass, payload.project_quality
            weight = mass * quality
            values = features.flatten(1).T[source].float()
            denominator = values.new_zeros(h * w)
            raw_mass = values.new_zeros(h * w)
            summed = values.new_zeros(h * w, features.shape[0])
            if len(source):
                # Atomic index_add has order-dependent CUDA rounding. Sort
                # once, reduce each segment, then write unique target ids.
                target, order = target.sort(stable=True)
                unique, lengths = torch.unique_consecutive(target, return_counts=True)
                denominator.index_copy_(
                    0,
                    unique,
                    torch.segment_reduce(weight[order], "sum", lengths=lengths),
                )
                raw_mass.index_copy_(
                    0, unique, torch.segment_reduce(mass[order], "sum", lengths=lengths)
                )
                reduced = torch.segment_reduce(
                    (values * weight[:, None])[order], "sum", lengths=lengths
                )
                summed.index_copy_(0, unique, reduced)
            else:
                # Keep the zero-support result differentiable without
                # sending an empty tensor to CUDA's segmented reduction.
                summed = summed + features.float().sum() * 0
            mean = summed / denominator.clamp_min(1e-6)[:, None]
            mean_quality = denominator / raw_mass.clamp_min(1e-6)
            coverage = (raw_mass / (sx * sy)).clamp(0, 1)
            outputs.append(mean.T.reshape(features.shape[0], h, w).to(features.dtype))
            qualities.append(mean_quality.reshape(1, h, w))
            coverages.append(coverage.reshape(1, h, w))
        return torch.stack(outputs), torch.stack(qualities), torch.stack(coverages)
