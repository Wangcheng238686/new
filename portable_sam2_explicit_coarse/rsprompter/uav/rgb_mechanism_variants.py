"""Independent single-change variants of the sealed compact interaction."""
import torch
from torch import nn
from torch.nn import functional as F

from .compact_joint import CompactInteraction


class RGBMechanismVariant(CompactInteraction):
    def configure(self, variant):
        assert variant in ('view_preserve', 'spatial_preserve', 'no_sat_copy')
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2055)
            if variant == 'view_preserve':
                self.view_mix = nn.Linear(160, 32, bias=False)
                with torch.no_grad():
                    self.view_mix.weight.copy_(torch.cat([torch.eye(32)] * 5, 1))
            if variant == 'spatial_preserve':
                self.spatial_mix = nn.Conv2d(128, 32, 1, bias=False)
                with torch.no_grad():
                    # pixel_unshuffle order: channel, dy, dx.
                    self.spatial_mix.weight.zero_()
                    for c in range(32):
                        self.spatial_mix.weight[c, c * 4:c * 4 + 4, 0, 0] = .25

    def view_reduce(self, selected, index):
        valid = index >= 0
        denom = valid.sum(1, keepdim=True).clamp_min(1)
        parts = [((selected * (valid & (index // 16384 == view))[..., None]).sum(1) / denom)
                 for view in range(5)]
        # Sum/count convention gives exactly the original candidate-weighted
        # mean at initialization, including missing or unequal-support views.
        return self.view_mix(torch.cat(parts, -1))

    def fuse_rgb(self, u):
        if self.variant != 'view_preserve':
            return super().fuse_rgb(u)
        from .attention_ladder import sample_source_continuous
        rgb, index, anchor = self.rgb_context
        maps = []
        for image in rgb.split(1):
            _, pyramid = self.rgb_encoder((image - self.rgb_mean) / self.rgb_std)
            maps.append(self.rgb_project(pyramid[0]))
        features = torch.cat(maps).permute(0, 2, 3, 1).reshape(-1, 32)
        features = F.layer_norm(features, (32,))
        selected = sample_source_continuous(features, index, anchor)
        aligned = self.view_reduce(selected, index).T.reshape(1, 32, 128, 128)
        return self.rgb_fusion(torch.cat((u, aligned), 1))

    def pool_spatial(self, u):
        return self.spatial_mix(F.pixel_unshuffle(u, 2))

    def interact(self, base):
        if self.variant == 'no_sat_copy':
            # This also removes the ur contribution to the instance head.
            # No claim that head_satellite or all SAT-dependent paths vanish.
            handle = self.uout.register_forward_hook(lambda module, args, output: output * 0)
            try:
                return super().interact(base)
            finally:
                handle.remove()
        if self.variant != 'spatial_preserve':
            return super().interact(base)
        # Unchanged InstanceInteraction.interact, except its first avg_pool.
        assert base.shape == (1, 64, 64, 224)
        u128, support128 = self.context
        support64 = F.max_pool2d(support128.float(), 2, 2) > 0
        u64 = self.pool_spatial(u128)
        u = F.layer_norm(u64.permute(0, 2, 3, 1).reshape(4096, 32), (32,))
        s = self.satellite(F.layer_norm(base, (224,))).reshape(4096, 32)
        sn = self.neighbors(s.T.reshape(1, 32, 64, 64))
        border = F.unfold(torch.ones_like(support64, dtype=torch.float32), 3, padding=1)[0].T > 0
        q = F.normalize(self.uq(u).float(), dim=-1)
        k = F.normalize((self.sk(sn) + self.position).float(), dim=-1)
        w = (q[:, None] * k).sum(-1).mul(4).masked_fill(~border, -1e4).softmax(-1)
        upd = self.uout((w[..., None] * self.sv(sn).float()).sum(1).to(u.dtype))
        valid = support64.reshape(-1, 1)
        updated_u = u + .2 * upd * valid
        un = self.neighbors(updated_u.T.reshape(1, 32, 64, 64))
        supported_neighbors = F.unfold(support64.float(), 3, padding=1)[0].T > 0
        q = F.normalize(self.sq(s).float(), dim=-1)
        k = F.normalize((self.uk(un) + self.position).float(), dim=-1)
        logits = (q[:, None] * k).sum(-1).mul(4)
        w = logits.masked_fill(~supported_neighbors, -1e4).softmax(-1) * supported_neighbors
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        evidence = (w[..., None] * self.uv(un).float()).sum(1)
        delta = self.output(evidence.to(s.dtype)) * valid
        updated_sat = base + delta.reshape_as(base).to(base.dtype)
        ur = (.2 * upd * valid).T.reshape(1, 32, 64, 64)
        high_u = u128 + F.interpolate(ur.to(u128.dtype), size=(128, 128), mode='bilinear', align_corners=False)
        high_s = self.head_satellite(updated_sat.permute(0, 3, 1, 2))
        high_s = F.interpolate(high_s, size=(128, 128), mode='bilinear', align_corners=False)
        h = self.head_fusion(torch.cat((high_u, high_s), 1))
        self.outputs = (self.heat(h), self.reg(h))
        if self.observe:
            self.observation = dict(support_tokens=int(valid.sum().detach()), target_tokens=4096,
                delta_square_sum=float(delta.detach().float().square().sum()),
                base_square_sum=float(base.detach().float().square().sum()), elements=base.numel())
        return updated_sat


def install(adapter, variant):
    assert isinstance(adapter, CompactInteraction)
    adapter.__class__ = RGBMechanismVariant
    adapter.configure(variant)
    adapter.cuda()
    return adapter
