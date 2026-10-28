"""Fixed-train-sample offline gradient directions at old RGB-D E1/E3/E10.

No optimizer or training log changes. Eval-mode FP32 gradients describe these
fixed functions, not the distribution of stochastic augmented training updates.
"""
import argparse
import math
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_instance_interaction_20261024 as core
from data.ue_instance_targets import center_targets
import numpy as np
import torch


def group(name):
    if name.startswith('output.'):
        return 'writeback'
    if name.startswith('rgb_encoder.') and 'lora_' in name:
        return 'source_lora'
    if name.split('.')[0] in ('a', 'b', 'c', 'd', 'up'):
        return 'geometry'
    if name == 'position' or name.split('.')[0] in ('uq', 'sk', 'sv', 'uout', 'sq', 'uk', 'uv'):
        return 'attention'
    return None


def compare(named, left, right):
    values = {}
    for (name, parameter), a, b in zip(named, left, right):
        key = group(name)
        row = values.setdefault(key, dict(main_square=0., auxiliary_square=0., dot=0.,
            declared_parameters=0, main_active_parameters=0, auxiliary_active_parameters=0))
        row['declared_parameters'] += parameter.numel()
        for grad in (a, b):
            assert grad is None or torch.isfinite(grad).all(), name
        if a is not None:
            row['main_square'] += float(a.double().square().sum())
            row['main_active_parameters'] += parameter.numel()
        if b is not None:
            row['auxiliary_square'] += float(b.double().square().sum())
            row['auxiliary_active_parameters'] += parameter.numel()
        if a is not None and b is not None:
            row['dot'] += float((a.double() * b.double()).sum())
    for row in values.values():
        row['main_norm'] = math.sqrt(row['main_square'])
        row['auxiliary_norm'] = math.sqrt(row['auxiliary_square'])
        denominator = row['main_norm'] * row['auxiliary_norm']
        row['cosine'] = row['dot'] / denominator if denominator else None
        row['auxiliary_over_main_norm'] = row['auxiliary_norm'] / row['main_norm'] if row['main_norm'] else None
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert not (args.out / 'complete.json').exists()
    config = argparse.Namespace(root=core.OUT, seed=44, family='rgb', rgb='correct', route='internal', content='geometry')
    model, adapter, _, _ = core.build(config)
    model.eval(); adapter.eval()
    named = [(name, parameter) for name, parameter in adapter.named_parameters()
             if parameter.requires_grad and group(name) is not None]
    parameters = [parameter for _, parameter in named]
    ds = core.C.MinimalSatelliteDataset(core.previous.DATA, 'train', augment=False)
    indices = np.linspace(0, len(ds) - 1, 8, dtype=int).tolist()
    source = {**core.source_hashes(), 'data/ue_instance_targets.py': core.C.file_sha(WORK / 'data/ue_instance_targets.py'),
              'scripts/analysis/ue_interaction_loss_direction_20261024.py': core.C.file_sha(__file__)}
    core.C.dump(args.out / 'protocol.json', dict(source_hashes=source, split='train', fixed_indices=indices,
        image_ids=[ds.samples[i]['image_id'] for i in indices], epochs=[1, 3, 10], precision='FP32',
        mode='satellite eval/frozen, adapter eval; no input augmentation/dropout, no optimizer',
        group_vector='Union of declared group parameters; unused gradients are zero. Cosine undefined if either norm zero.',
        per_original_loss_writeback='only first fixed training image of each epoch checkpoint',
        scope='post-observation diagnostic, does not select architecture, test epoch or loss weight'))
    rows, bindings = [], {}
    for epoch in (1, 3, 10):
        checkpoint = core.OUT / 'seed44_rgb_internal_geometry_correct' / f'epoch{epoch}.pth'
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert cp['epoch'] == epoch and cp['step'] == epoch * 267
        assert cp['contract']['source_hashes'] == core.source_hashes()
        adapter.load_state_dict(cp['adapter'], strict=True)
        del cp
        binding = core.C.file_sha(checkpoint)
        bindings[str(epoch)] = binding
        state_sha = core.previous.tensor_digest(adapter.state_dict())
        for position, index in enumerate(indices):
            core.C.seed(44000 + index)
            ident = ds.samples[index]['image_id']
            sample = core.C.processed(model, core.C.rtmdet_collate_fn([ds[index]]), True)
            target = center_targets(sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy())
            core.bind_rgb(config, adapter, ds, index)
            features = core.forward(model, adapter, sample, core.load_geometry(ident))
            with core.C.cached_extract(model, features):
                original = model.loss(sample['inputs'], sample['data_samples'])
            primary = core.previous.objective(adapter, original)
            auxiliary, _ = core.pilot.loss_fn(adapter.outputs, target)
            left = torch.autograd.grad(primary, parameters, retain_graph=True, allow_unused=True)
            right = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
            row = dict(epoch=epoch, image_id=ident, original_loss=float(primary.detach()),
                auxiliary_loss=float(auxiliary.detach()), shared_groups=compare(named, left, right))
            if position == 0:
                per_loss = {}
                for name, value in original.items():
                    if 'loss' not in name:
                        continue
                    terms = value if isinstance(value, (list, tuple)) else [value]
                    terms = [x for x in terms if torch.is_tensor(x) and x.requires_grad]
                    if not terms:
                        continue
                    grad = torch.autograd.grad(sum(terms), [adapter.output.weight], retain_graph=True, allow_unused=True)[0]
                    per_loss[name] = None if grad is None else float(grad.float().norm())
                row['original_loss_writeback_weight_norms'] = per_loss
            rows.append(row)
            del features, original, primary, auxiliary, left, right
            adapter.clear(); adapter.rgb_context = None
            core.C.dump(args.out / 'progress.json', dict(rows=len(rows), total=24, epoch=epoch, image_id=ident))
            print('LOSS DIRECTION', epoch, ident, row['shared_groups']['writeback'], flush=True)
        assert core.previous.tensor_digest(adapter.state_dict()) == state_sha
        assert core.C.file_sha(checkpoint) == binding
    assert core.C.state_digest(model) == core.previous.FROZEN_SHA
    assert all(core.C.file_sha(WORK / name) == sha for name, sha in source.items())
    summary = {}
    for epoch in (1, 3, 10):
        summary[str(epoch)] = {}
        for name in ('writeback', 'geometry', 'attention', 'source_lora'):
            selected = [r['shared_groups'][name] for r in rows if r['epoch'] == epoch]
            defined = [r['cosine'] for r in selected if r['cosine'] is not None]
            summary[str(epoch)][name] = dict(samples=len(selected), defined_cosines=len(defined),
                negative_cosines=sum(x < 0 for x in defined),
                mean_cosine=float(np.mean(defined)) if defined else None,
                mean_main_norm=float(np.mean([r['main_norm'] for r in selected])),
                mean_auxiliary_norm=float(np.mean([r['auxiliary_norm'] for r in selected])))
    core.C.dump(args.out / 'rows.json', rows)
    core.C.dump(args.out / 'complete.json', dict(complete=True, rows=24, checkpoints=bindings,
        source_hashes=source, frozen_satellite_exact=True, adapter_parameters_unchanged=True,
        summary=summary, scope='Fixed train functions and union-parameter norms; conflict is not proof of AP harm or causal bottleneck'))
    print('LOSS DIRECTION AUDIT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
