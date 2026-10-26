"""Depth-content interaction with a directly supervised instance discovery head.

Experimental only. Original satellite parameters and module state names are untouched.
"""
import torch
from torch import nn
from torch.nn import functional as F


def conv_block(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(4, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU())


class InstanceInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = conv_block(5, 16, 2)
        self.b = conv_block(16, 32, 2)
        self.c = conv_block(32, 64, 2)
        self.d = conv_block(64, 64, 2)
        self.up = conv_block(160, 32)
        self.satellite = nn.Linear(224, 32)
        for name in ('uq', 'sk', 'sv', 'uout', 'sq', 'uk', 'uv'):
            setattr(self, name, nn.Linear(32, 32))
        self.output = nn.Linear(32, 224)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.head_satellite = nn.Conv2d(224, 32, 1)
        self.head_fusion = conv_block(64, 32)
        self.heat = nn.Conv2d(32, 1, 1)
        self.reg = nn.Conv2d(32, 4, 1)
        nn.init.constant_(self.heat.bias, -2.19)
        self.position = nn.Parameter(torch.zeros(9, 32))
        self.internal = True
        self.enabled = True
        self.context = None
        self.outputs = None
        self.handles = []
        self.observation = {}
        self.observe = False
        self.rgb_context = None

    def add_rgb(self, pretrained, seed):
        from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
        # Additional capacity must not perturb shared initialization/RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.rgb_encoder = RSSAM2VisionEncoder(
                checkpoint_path=pretrained, model_size='base_plus',
                use_gradient_checkpointing=True, use_lora=True, freeze_backbone=False,
                lora_config=dict(r=16, lora_alpha=32, target_modules=['qkv', 'proj'],
                                 lora_dropout=.05, bias='none', modules_to_save=[]))
            self.rgb_project = nn.Conv2d(256, 32, 1)
            self.rgb_fusion = conv_block(64, 32)
        self.register_buffer('rgb_mean', torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer('rgb_std', torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'rgb_encoder'):
            self.rgb_encoder.eval()
            for name, module in self.rgb_encoder.named_modules():
                if 'lora_dropout' in name:
                    module.train(mode)
        return self

    def fuse_rgb(self, u):
        if not hasattr(self, 'rgb_encoder'):
            return u
        from .attention_ladder import sample_source_continuous
        rgb, index, anchor = self.rgb_context
        maps = []
        for image in rgb.split(1):
            _, pyramid = self.rgb_encoder((image - self.rgb_mean) / self.rgb_std)
            maps.append(self.rgb_project(pyramid[0]))
        features = torch.cat(maps).permute(0, 2, 3, 1).reshape(-1, 32)
        features = F.layer_norm(features, (32,))
        selected = sample_source_continuous(features, index, anchor)
        valid = index >= 0
        pooled = (selected * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        aligned = pooled.T.reshape(1, 32, 128, 128)
        return self.rgb_fusion(torch.cat((u, aligned), 1))

    def attach(self, model):
        assert not self.handles
        trunks = [m for m in model.backbone.modules()
                  if hasattr(m, 'stage_ends') and hasattr(m, 'blocks')]
        assert len(trunks) == 1
        trunk = trunks[0]
        assert list(trunk.stage_ends) == [1, 4, 20, 23]
        assert trunk.blocks[4].dim_out == 224
        assert tuple(trunk.blocks[2].pool.stride) == (2, 2)
        self.handles.append(trunk.blocks[4].register_forward_hook(self._inject))

    def encode(self, geometry):
        assert geometry.shape == (1, 5, 512, 512)
        b = self.b(self.a(geometry))
        c = self.c(b)
        d = self.d(c)
        u = self.up(torch.cat((b, F.interpolate(c, size=(128, 128), mode='bilinear', align_corners=False),
                               F.interpolate(d, size=(128, 128), mode='bilinear', align_corners=False)), 1))
        u = self.fuse_rgb(u)
        support = F.max_pool2d(geometry[:, 4:5], 4, 4) > 0
        self.context = (u, support)
        self.outputs = None

    @staticmethod
    def neighbors(x):
        return F.unfold(x, 3, padding=1).reshape(32, 9, 4096).permute(2, 1, 0)

    def interact(self, base):
        assert base.shape == (1, 64, 64, 224)
        u128, support128 = self.context
        support64 = F.max_pool2d(support128.float(), 2, 2) > 0
        # Nominal CNN centers: U128 at 4*i, pooled U64 at 8*j+2;
        # Hiera's first 2x2 Q-pool has the same nominal lattice. These
        # feature footprints are not certified visible 3-D surface points.
        u64 = F.avg_pool2d(u128, 2, 2)
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
        # Center/box supervision must cross the satellite write-back branch.
        # Reading U' alone would leave that branch without this direct loss.
        ur = (.2 * upd * valid).T.reshape(1, 32, 64, 64)
        high_u = u128 + F.interpolate(ur.to(u128.dtype), size=(128, 128), mode='bilinear', align_corners=False)
        high_s = self.head_satellite(updated_sat.permute(0, 3, 1, 2))
        high_s = F.interpolate(high_s, size=(128, 128), mode='bilinear', align_corners=False)
        h = self.head_fusion(torch.cat((high_u, high_s), 1))
        self.outputs = (self.heat(h), self.reg(h))
        if self.observe:
            self.observation = dict(
                support_tokens=int(valid.sum().detach()), target_tokens=4096,
                delta_square_sum=float(delta.detach().float().square().sum()),
                base_square_sum=float(base.detach().float().square().sum()), elements=base.numel())
        return updated_sat

    def _inject(self, module, inputs, base):
        if not self.enabled or self.context is None:
            return base
        updated = self.interact(base)
        return updated if self.internal else base

    def clear(self):
        self.context = None
        self.outputs = None

