"""Fixed train-function gradients at joint RGB-D anchor checkpoints.

FP32/eval, no augmentation/dropout or optimizer updates. Reports descriptive
parameter-vector directions; no automatic loss weighting or AP-causality claim.
"""
import argparse
import math
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_compact_joint_run_20261024 as run
import numpy as np
import torch


def group(name):
    if name.startswith('satellite.'):
        if name.startswith('satellite.backbone.') and 'lora_' in name:
            return 'satellite_lora'
        if name.startswith('satellite.neck.'):
            return 'satellite_neck'
        if name.startswith(('satellite.rpn_head.', 'satellite.roi_head.bbox_head.')):
            return 'satellite_detection_heads'
        if name.startswith('satellite.roi_head.mask_head.mask_decoder.'):
            return 'satellite_mask_decoder'
        return 'satellite_mask_other'
    name = name.removeprefix('uav.') if hasattr(str, 'removeprefix') else name[4:]
    if name.startswith('rgb_encoder.'):
        return 'source_lora'
    if name.split('.')[0] in ('a', 'b', 'c', 'd', 'up'):
        return 'geometry'
    if name.startswith('output.'):
        return 'writeback'
    if name.startswith(('rgb_project.', 'rgb_fusion.')):
        return 'rgb_alignment_fusion'
    if name.startswith(('head_satellite.', 'head_fusion.', 'heat.', 'reg.')):
        return 'instance_head'
    return 'attention'


def compare(named, left, right):
    values = {}
    for (name, parameter), a, b in zip(named, left, right):
        row = values.setdefault(group(name), dict(left_square=0., right_square=0., dot=0.,
            parameters=0, left_active_parameters=0, right_active_parameters=0))
        row['parameters'] += parameter.numel()
        for value in (a, b):
            assert value is None or torch.isfinite(value).all(), name
        if a is not None:
            row['left_square'] += float(a.double().square().sum())
            row['left_active_parameters'] += parameter.numel()
        if b is not None:
            row['right_square'] += float(b.double().square().sum())
            row['right_active_parameters'] += parameter.numel()
        if a is not None and b is not None:
            row['dot'] += float((a.double() * b.double()).sum())
    for row in values.values():
        row['left_norm'] = math.sqrt(row['left_square'])
        row['right_norm'] = math.sqrt(row['right_square'])
        den = row['left_norm'] * row['right_norm']
        row['cosine'] = row['dot'] / den if den else None
        row['right_over_left_norm'] = row['right_norm'] / row['left_norm'] if row['left_norm'] else None
    return values


def terms(value):
    if torch.is_tensor(value):
        return [value] if value.requires_grad else []
    return [t for v in value for t in terms(v)] if isinstance(value, (tuple, list)) else []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=run.ROOT)
    p.add_argument('--epoch', type=int, choices=[1, 3, 10], required=True)
    args = p.parse_args()
    out = args.root / 'joint_loss_direction' / ('epoch%d' % args.epoch)
    assert not out.exists(), 'Preserve started diagnostics, including failed attempts'
    checkpoint = args.root / 'rgbd' / ('epoch%d.pth' % args.epoch)
    assert checkpoint.exists()
    out.mkdir(parents=True)
    # Bound this process's allocator while co-renting with live training.
    torch.cuda.set_per_process_memory_fraction(14 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    runtime = run.Runtime(args.root, 'rgbd')
    contract = run.contract(runtime)
    cp = runtime.restore(checkpoint, contract, training=False)
    assert cp['epoch'] == args.epoch and cp['step'] == args.epoch * 267
    del cp
    runtime.network.eval()
    initial = run.core.C.state_digest(runtime.network)
    checkpoint_sha = run.core.C.file_sha(checkpoint)
    named = [(n, t) for n, t in runtime.network.named_parameters() if t.requires_grad]
    parameters = [t for _, t in named]
    ds = run.core.C.MinimalSatelliteDataset(run.core.previous.DATA, 'train', augment=False)
    indices = np.linspace(0, len(ds) - 1, 8, dtype=int).tolist()
    protocol = dict(epoch=args.epoch, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        fixed_indices=indices, image_ids=[ds.samples[i]['image_id'] for i in indices],
        split='train', precision='FP32', mode='eval; no augmentation/dropout; no optimizer updates',
        loss_partition=dict(detection=['loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox'],
            segmentation=['loss_mask', 'loss_decoder_tail', 'loss_shape_prior'], auxiliary='loss_uav_instances'),
        group_parameters={g: sum(t.numel() for n, t in named if group(n) == g) for g in {group(n) for n, _ in named}},
        source_hashes=run.sources(), diagnostic_source_sha256=run.core.C.file_sha(Path(__file__)),
        allocator_cap_GiB=14, nominal_successful_training_microbatch_factor=1/8,
        scope='fixed post-observation train functions, not stochastic training gradients or loss-weight selection')
    run.core.C.dump(out / 'protocol.json', protocol)
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for position, index in enumerate(indices):
        run.core.C.seed(44000 + index)
        ident = ds.samples[index]['image_id']
        sample = run.core.C.processed(runtime.satellite, run.core.C.rtmdet_collate_fn([ds[index]]), True)
        runtime.joint.prepare_uav(**runtime.payload(ident))
        features = runtime.satellite.extract_feat(sample['inputs'])
        with run.core.C.cached_extract(runtime.satellite, features):
            original = runtime.satellite.loss(sample['inputs'], sample['data_samples'])
        actual = {k: sum(terms(v)) for k, v in original.items() if 'loss' in k and terms(v)}
        det_keys = protocol['loss_partition']['detection']
        seg_keys = protocol['loss_partition']['segmentation']
        assert set(actual) == set(det_keys + seg_keys), sorted(actual)
        det, seg = sum(actual[k] for k in det_keys), sum(actual[k] for k in seg_keys)
        target = run.center_targets(sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy())
        auxiliary, _ = run.core.pilot.loss_fn(runtime.joint.uav.outputs, target)
        # Three VJPs from one identical forward; main gradient is exactly det+seg.
        gd = torch.autograd.grad(det, parameters, retain_graph=True, allow_unused=True)
        gs = torch.autograd.grad(seg, parameters, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
        gm = [a + b if a is not None and b is not None else a if b is None else b for a, b in zip(gd, gs)]
        pairs = {name: compare(named, a, b) for name, a, b in [
            ('main_vs_auxiliary', gm, ga), ('detection_vs_segmentation', gd, gs),
            ('detection_vs_auxiliary', gd, ga), ('segmentation_vs_auxiliary', gs, ga)]}
        row = dict(image_id=ident, losses={k: float(v.detach()) for k, v in actual.items()},
            auxiliary_loss=float(auxiliary.detach()), comparisons=pairs)
        rows.append(row)
        run.core.C.dump(out / 'rows.json', rows)
        run.core.C.dump(out / 'progress.json', dict(rows=len(rows), total=8, image_id=ident))
        print('JOINT GRADIENT', args.epoch, ident, pairs['main_vs_auxiliary']['writeback'], flush=True)
        del original, actual, det, seg, auxiliary, features, gd, gs, ga, gm, target, sample
        runtime.clear()
    assert run.core.C.state_digest(runtime.network) == initial
    assert not runtime.optimizer.state
    assert all(t.grad is None for t in parameters)
    assert run.core.C.file_sha(checkpoint) == checkpoint_sha and run.sources() == protocol['source_hashes']
    summary = {}
    for comparison in rows[0]['comparisons']:
        summary[comparison] = {}
        for g in rows[0]['comparisons'][comparison]:
            selected = [r['comparisons'][comparison][g] for r in rows]
            cosines = [v['cosine'] for v in selected if v['cosine'] is not None]
            summary[comparison][g] = dict(samples=8, defined_cosines=len(cosines),
                negative_cosines=sum(c < 0 for c in cosines), mean_cosine=float(np.mean(cosines)) if cosines else None,
                mean_left_norm=float(np.mean([v['left_norm'] for v in selected])),
                mean_right_norm=float(np.mean([v['right_norm'] for v in selected])))
    run.core.C.dump(out / 'complete.json', dict(complete=True, rows=8, protocol=protocol,
        summary=summary, model_parameters_and_persistent_buffers_unchanged=True,
        no_optimizer_updates=True, peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        limitations='8 eval/no-augmentation train functions; sample mean cosines, not pooled gradients; zero norms undefined; norms before AMP/clipping/AdamW do not equal effective update influence; neither conflict nor large norm proves AP harm'))


if __name__ == '__main__':
    main()
