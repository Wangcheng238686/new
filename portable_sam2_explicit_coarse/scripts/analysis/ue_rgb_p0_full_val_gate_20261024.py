"""All-val exact UAV read/aggregation gate, no detector training or AP selection."""
import argparse
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_instance_interaction_20261024 as probe
from rsprompter.uav.rgb_p0_encoder import RGBP0Encoder
import torch
from torch import nn


class P0Interface(nn.Module):
    def __init__(self, p0):
        super().__init__()
        self.p0 = p0

    def forward(self, x):
        p0 = self.p0(x)
        return p0, (p0,)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'complete.json').exists()
    checkpoint = probe.OUT / 'seed44_rgb_internal_geometry_correct/last.pth'
    cp_sha = probe.C.file_sha(checkpoint)
    frozen = probe.source_hashes()
    sources = {name: probe.C.file_sha(WORK / name) for name in (
        'rsprompter/uav/rgb_p0_encoder.py',
        'scripts/analysis/ue_rgb_p0_full_val_gate_20261024.py')}
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['epoch'] == 10 and cp['step'] == 2670
    assert cp['contract']['source_hashes'] == frozen
    adapter = probe.InstanceInteraction()
    adapter.add_rgb('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt', 2045)
    adapter.load_state_dict(cp['adapter'], strict=True)
    del cp
    adapter.cuda().eval()
    before = probe.previous.tensor_digest(adapter.state_dict())
    full = adapter.rgb_encoder
    slim = RGBP0Encoder(full).cuda().eval()
    interface = P0Interface(slim)
    ds = probe.C.MinimalSatelliteDataset(probe.previous.DATA, 'val')
    payload_args = argparse.Namespace(cache=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    ids = []
    with torch.inference_mode():
        for i, row in enumerate(ds.samples):
            folder = probe.C.point_folder(ds, i, probe.previous.DATA)
            rgb, index, _, anchor = probe.previous.payload(payload_args, folder, 'correct', None)
            images = (rgb - adapter.rgb_mean) / adapter.rgb_std
            for view, image in enumerate(images.split(1)):
                assert torch.equal(full(image)[1][0], slim(image)), (row['image_id'], view)
            adapter.rgb_context = (rgb, index, anchor)
            geometry = probe.load_geometry(row['image_id'])
            adapter.rgb_encoder = full
            adapter.encode(geometry)
            expected, support = adapter.context
            adapter.rgb_encoder = interface
            adapter.encode(geometry)
            actual, new_support = adapter.context
            assert torch.equal(expected, actual) and torch.equal(support, new_support), row['image_id']
            adapter.clear()
            adapter.rgb_encoder = full
            ids.append(row['image_id'])
            if (i + 1) % 25 == 0:
                probe.C.dump(args.out / 'progress.json', dict(images=i + 1, total=len(ds)))
                print('P0 ALL-VAL', i + 1, flush=True)
    assert len(ids) == 459 and len(set(ids)) == 459
    assert probe.previous.tensor_digest(adapter.state_dict()) == before
    assert probe.C.file_sha(checkpoint) == cp_sha and probe.source_hashes() == frozen
    assert all(probe.C.file_sha(WORK / name) == sh for name, sh in sources.items())
    probe.C.dump(args.out / 'complete.json', dict(complete=True, images=len(ids), image_ids=ids,
        views_per_image=5, source_P0_exact=True, fused_U128_exact=True, support_exact=True,
        checkpoint_sha256=cp_sha, source_hashes=sources, frozen_source_hashes=frozen,
        adapter_unchanged=True, precision='FP32',
        scope='All459val P0 and final UAV U128 inputs to internal interaction; not final detection/AP or training-trajectory equivalence'))
    print('P0 FULL VAL GATE PASS', flush=True)


if __name__ == '__main__':
    main()
