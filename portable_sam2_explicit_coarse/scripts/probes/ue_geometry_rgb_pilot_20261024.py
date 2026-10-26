"""RGB-augmented geometry proposal pilot (20261024 prereg part 1).

Runs the LOCKED pilot trainer verbatim via module substitution - identical
loop, seeding, checkpointing and val recall - with three single-factor
substitutions: (1) input layout [fields4, support1, rgb3/255] (support kept
at channel 4 so every pilot-internal indexing stays valid); (2) Net stage-a
5->8 channels; (3) content control 'constrgb' = fields real + support real +
RGB replaced by train-only per-channel mean under support (the geometry
fields are real in BOTH arms; only RGB content is the controlled factor).

The pilot source file is never modified; its sha is recorded by its own
protocol. This launcher additionally records rgb_protocol.json.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import ue_geometry_proposal_pilot_20261024 as pilot
import ue_geometry_mechanism_probe as g

_L = Path('/data2/wangcheng/remote-lthpc')
pilot.OUT = _L / 'checkpoint/ue_geometry_rgb_pilot_20261024'
pilot.CACHE = _L / 'cache/ue_v1_rgb_geometry'
pilot.CONSTANTS = _L / 'checkpoint/ue_geometry_mechanism_20261022/common_constants.json'
g.DATA = _L / 'dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
OUT = pilot.OUT
CONSTANTS = pilot.CONSTANTS
import math


class Data(pilot.Data):
    def __getitem__(self, index):
        ident = self.ids[index]
        with np.load(pilot.CACHE / f'{ident:06d}.npz') as z:
            x = np.concatenate((z['fields'].astype(np.float32),
                                z['support'][None].astype(np.float32),
                                z['rgb'].astype(np.float32) / 255.0))
        heat = np.zeros((1, 128, 128), np.float32)
        reg = np.zeros((4, 128, 128), np.float32)
        valid = np.zeros((1, 128, 128), np.float32)
        boxes = self.boxes[index]
        order = np.argsort(-np.prod(boxes[:, 2:] - boxes[:, :2], axis=1), kind='stable')
        for box in boxes[order]:
            wh = (box[2:] - box[:2]) / 4
            center = (box[:2] + box[2:]) / 8
            ix, iy = np.clip(np.floor(center).astype(int), 0, 127)
            sigma = float(np.clip(min(wh) / 6, 1, 4))
            radius = int(math.ceil(3 * sigma))
            x0, x1 = max(0, ix - radius), min(128, ix + radius + 1)
            y0, y1 = max(0, iy - radius), min(128, iy + radius + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gauss = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma ** 2))
            heat[0, y0:y1, x0:x1] = np.maximum(heat[0, y0:y1, x0:x1], gauss)
            reg[:2, iy, ix] = center - [ix, iy]
            reg[2:, iy, ix] = np.log(np.maximum(wh, 1e-3))
            valid[0, iy, ix] = 1
        return dict(x=torch.from_numpy(x), heat=torch.from_numpy(heat),
                    reg=torch.from_numpy(reg), valid=torch.from_numpy(valid),
                    image_id=ident, index=index)


class Net(pilot.Net):
    def __init__(self):
        super().__init__()
        self.a = pilot.block(8, 16, 2)


_RGB_MEANS = None


def _rgb_means():
    global _RGB_MEANS
    if _RGB_MEANS is None:
        p = OUT / 'rgb_constants.json'
        if p.exists():
            _RGB_MEANS = torch.tensor(json.loads(p.read_text())['mean_rgb'], device='cuda')[None, :, None, None]
        else:
            _RGB_MEANS = torch.full((1, 3, 1, 1), 0.35, device='cuda')
    return _RGB_MEANS


def content(x, kind, means):
    if kind == 'constrgb':
        support = x[:, 4:5]
        return torch.cat((x[:, :4], support, _rgb_means() * support), 1)
    return x


def compute_rgb_constants():
    """Train-only per-channel RGB mean under support (scan order = split list)."""
    ids = json.loads((g.DATA / 'splits/train.json').read_text())
    sums = np.zeros(3, np.float64)
    cnt = 0
    for i in ids:
        with np.load(pilot.CACHE / f'{i:06d}.npz') as z:
            rgb = z['rgb'].astype(np.float64) / 255.0
            sup = z['support']
        sums += (rgb * sup[None]).sum((1, 2))
        cnt += int(sup.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'rgb_constants.json').write_text(json.dumps(
        dict(mean_rgb=(sums / max(cnt, 1)).tolist(), support_pixels=cnt, points=len(ids))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['init', 'gate', 'train', 'constants'])
    p.add_argument('--content', choices=['geometry', 'constant', 'real', 'constrgb'], default='real')
    a = p.parse_args()
    if a.mode == 'constants':
        compute_rgb_constants()
        print('RGB CONSTANTS DONE')
        return
    # substitute into the locked trainer
    pilot.Data = Data
    pilot.Net = Net
    pilot.content = content
    if a.content == 'real':
        a.content = 'geometry'      # passthrough content kind in the pilot loop
    elif a.content == 'constrgb':
        a.content = 'constant'      # pilot loop name; our content() does RGB-only replacement
    sys.argv = [sys.argv[0], a.mode, '--content', a.content]
    if a.mode == 'init':
        pilot.main()
        g.dump(OUT / 'rgb_protocol.json', dict(
            launcher_sha256=g.file_sha(__file__), layout='fields4|support1|rgb3/255',
            control='constrgb: fields real, support real, rgb=train-only mean under support',
            rgb_cache_version='native512_rgb_v1',
            note='pilot trainer run verbatim via module substitution; pilot protocol records pilot source sha'))
        return
    pilot.main()


if __name__ == '__main__':
    main()
