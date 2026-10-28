"""Read-only decomposition of a fixed interaction checkpoint's write-back.

Observed attention weights are held fixed. This is algebraic attribution, not
an intervention, an AP test, or a new training arm. Stops after Hiera block4.
"""
import argparse
import itertools
import math
import os
from pathlib import Path
import sys
import types

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(WORK / 'scripts/probes'))
import ue_instance_interaction_20261024 as probe
import torch
from torch.nn import functional as F


class Block4Complete(Exception):
    pass


def decompose(adapter, captured):
    u128, support128 = adapter.context
    u = F.layer_norm(F.avg_pool2d(u128, 2).permute(0, 2, 3, 1).reshape(4096, 32), (32,))
    original_neighbors = adapter.neighbors(u.T.reshape(1, 32, 64, 64))
    support = F.max_pool2d(support128.float(), 2) > 0
    neighbors_valid = F.unfold(support.float(), 3, padding=1)[0].T > 0
    q = F.normalize(captured['sq'].float(), dim=-1)
    k = F.normalize((captured['uk'] + adapter.position).float(), dim=-1)
    weights = ((q[:, None] * k).sum(-1) * 4).masked_fill(~neighbors_valid, -1e4).softmax(-1)
    weights = weights * neighbors_valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
    increment = captured['uv_input'] - original_neighbors
    evidence = {}
    for name, values in (('uav', original_neighbors), ('satellite_return', increment)):
        projected = F.linear(values, adapter.uv.weight, None)
        evidence[name] = (weights[..., None] * projected.float()).sum(1)
    evidence['bias'] = weights.sum(1, keepdim=True) * adapter.uv.bias.float()
    valid = support.reshape(-1, 1)
    parts = {name: F.linear(value, adapter.output.weight, None) * valid
             for name, value in evidence.items()}
    parts['bias'] = parts['bias'] + adapter.output.bias * valid
    parts['actual'] = captured['output'] * valid
    parts['base'] = captured['base'].reshape(4096, 224)
    reconstructed = parts['uav'] + parts['satellite_return'] + parts['bias']
    error = (reconstructed - parts['actual']).abs().max().item()
    # FP32 splitting changes the order of floating-point sums; never claim bit exact.
    tolerance = 2e-6 + 2e-5 * parts['actual'].abs().max().item()
    assert error <= tolerance, (error, tolerance)
    sums = {name: float(value.double().square().sum()) for name, value in parts.items()}
    dots = {f'{a}__{b}': float((parts[a].double() * parts[b].double()).sum())
            for a, b in itertools.combinations(('uav', 'satellite_return', 'bias', 'actual'), 2)}
    return dict(square_sums=sums, dot_sums=dots, elements=4096 * 224,
                supported_tokens=int(valid.sum()), target_tokens=4096,
                max_reconstruction_error=error, reconstruction_tolerance=tolerance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=probe.OUT)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not (args.out / 'complete.json').exists(), 'Use a distinct directory for each invocation'
    args.out.mkdir(parents=True, exist_ok=True)
    args.seed, args.family, args.rgb = 44, 'geom', 'correct'
    args.route, args.content = 'internal', 'geometry'
    checkpoint = args.root / 'seed44_geom_internal_geometry_correct/last.pth'
    cp_sha = probe.C.file_sha(checkpoint)
    source = WORK / 'scripts/analysis/ue_instance_interaction_value_audit_20261024.py'
    source_sha = probe.C.file_sha(source)
    frozen_sources = probe.source_hashes()
    model, adapter, _, _ = probe.build(args)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['epoch'] == 10 and cp['step'] == 2670
    assert cp['contract']['source_hashes'] == frozen_sources
    adapter.load_state_dict(cp['adapter'], strict=True)
    del cp
    adapter.eval()
    adapter_sha = probe.previous.tensor_digest(adapter.state_dict())
    dataset = probe.C.MinimalSatelliteDataset(probe.previous.DATA, args.split)
    ids = [row['image_id'] for row in dataset.samples]
    count = min(args.limit, len(ids)) if args.limit else len(ids)
    assert count > 0
    protocol = dict(checkpoint_sha256=cp_sha, source_sha256=source_sha,
        frozen_source_hashes=frozen_sources, split=args.split, image_ids=ids[:count],
        precision='FP32', modes=['geometry', 'constant', 'donor'], pid=os.getpid(),
        denominator='all selected images x 4096 target tokens x 224 channels; unsupported writes zero',
        interpretation='Fixed observed attention; UAV term includes geometry CNN support/context. Satellite-return term is post-read token increment. Bias includes both linear biases. RMS terms are not additive percentages or causal AP effects.',
        scope='Exploratory read-only algebra audit; no changes to original 14 arms/119 jobs')
    probe.C.dump(args.out / 'protocol.json', protocol)
    captured = {}
    handles = []
    def output_hook(name):
        def hook(module, inputs, output):
            captured[name] = output
            if name == 'uv':
                captured['uv_input'] = inputs[0]
        return hook
    for name in ('sq', 'uk', 'uv', 'output'):
        handles.append(getattr(adapter, name).register_forward_hook(output_hook(name)))
    original = adapter.interact
    observations = []
    def inspect(self, base):
        captured['base'] = base
        original(base)
        observations.append(decompose(self, captured))
        raise Block4Complete()
    adapter.interact = types.MethodType(inspect, adapter)
    constants = probe.means()
    summaries = {}
    with torch.inference_mode():
        for mode in protocol['modes']:
            rows = []
            for index in range(count):
                batch = probe.C.rtmdet_collate_fn([dataset[index]])
                sample = probe.C.processed(model, batch)
                x = probe.geometry_content(probe.load_geometry(ids[index]), mode, ids[index], ids, constants)
                try:
                    probe.forward(model, adapter, sample, x)
                    raise AssertionError('block4 audit hook did not execute')
                except Block4Complete:
                    pass
                rows.append(dict(image_id=ids[index], **observations.pop()))
                captured.clear()
                adapter.clear()
                if (index + 1) % 25 == 0:
                    probe.C.dump(args.out / 'progress.json', dict(mode=mode, images=index + 1, pid=os.getpid()))
                    print(mode, index + 1, flush=True)
            probe.C.dump(args.out / f'{mode}_per_image.json', rows)
            n = sum(row['elements'] for row in rows)
            ss = {key: sum(row['square_sums'][key] for row in rows) for key in rows[0]['square_sums']}
            dd = {key: sum(row['dot_sums'][key] for row in rows) for key in rows[0]['dot_sums']}
            cosines = {}
            for pair, dot in dd.items():
                a, b = pair.split('__')
                denom = math.sqrt(ss[a] * ss[b])
                cosines[pair] = dot / denom if denom else None
            summaries[mode] = dict(images=len(rows), elements=n,
                supported_tokens=sum(row['supported_tokens'] for row in rows), target_tokens=4096 * len(rows),
                pooled_rms={key: math.sqrt(value / n) for key, value in ss.items()},
                pooled_cosines=cosines,
                max_reconstruction_error=max(row['max_reconstruction_error'] for row in rows))
    for handle in handles:
        handle.remove()
    assert probe.C.state_digest(model) == probe.previous.FROZEN_SHA
    assert probe.previous.tensor_digest(adapter.state_dict()) == adapter_sha
    assert probe.C.file_sha(checkpoint) == cp_sha and probe.C.file_sha(source) == source_sha
    assert probe.source_hashes() == frozen_sources
    probe.C.dump(args.out / 'summary.json', summaries)
    probe.C.dump(args.out / 'complete.json', dict(complete=True, images_per_mode=count,
        all_split_images=count == len(dataset), checkpoint_sha256=cp_sha, source_sha256=source_sha,
        original_model_and_adapter_unchanged=True, frozen_source_hashes=frozen_sources))
    print('VALUE AUDIT COMPLETE', summaries, flush=True)


if __name__ == '__main__':
    main()
