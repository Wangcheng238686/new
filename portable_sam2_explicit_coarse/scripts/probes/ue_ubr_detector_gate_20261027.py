#!/usr/bin/env python3
"""Real-detector four-rank P1 engineering gate; no AP or long training.

Run only after source review with torchrun --standalone --nproc_per_node=4.
Never launches training subprocesses. A report with insufficient natural active
predictions is PARTIAL, not a passed 64-positive resource acceptance. No GT boxes
are inserted as predicted candidates. Existing experiment sources are read only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import copy
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def gather(value, cpu):
    out = [None] * 4
    dist.all_gather_object(out, value, group=cpu)
    return out


def digest(module, frozen_only=False):
    h = hashlib.sha256()
    for name, value in sorted(module.named_parameters()):
        if frozen_only and value.requires_grad:
            continue
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def gradients(module):
    groups = {'ubr': [], 'satellite_decoder': [], 'uav': [], 'G': []}
    for name, parameter in module.named_parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        if 'ubr_refiner' in name:
            groups['ubr'].append(value)
        elif 'mask_decoder' in name:
            groups['satellite_decoder'].append(value)
        if '.uav.' in '.' + name:
            groups['uav'].append(value)
        if 'detail_' in name:
            groups['G'].append(value)
    return {name: float(torch.stack(values).sum().sqrt()) if values else 0.
            for name, values in groups.items()}


def source_inventory():
    paths = {Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path:
            path = Path(path).resolve()
            if path.is_file() and WORK in path.parents and path.suffix == '.py':
                paths.add(path)
    return {str(path.relative_to(WORK)): sha(path) for path in sorted(paths)}


def tensor_gt(sample):
    gt = sample['data_samples'][0].gt_instances
    boxes = getattr(gt.bboxes, 'tensor', gt.bboxes)
    masks = gt.masks.to_tensor(dtype=torch.float32, device=boxes.device)
    return boxes, masks


class EngineeringForward(nn.Module):
    """All trainable use occurs inside ONE DDP forward."""
    def __init__(self, rt, base):
        super().__init__()
        self.model = rt.network
        # Runtime/controller are non-Module owners of references to self.model.
        object.__setattr__(self, 'rt', rt)
        object.__setattr__(self, 'base', base)
        self.stats = {}

    def forward(self, sample, ident):
        from data.ue_instance_targets import center_targets
        rt, base = self.rt, self.base
        boxes, masks = tensor_gt(sample)
        losses = self.model(sample, **rt.payload(ident),
            instance_target=center_targets(boxes.detach().cpu().numpy()),
            center_loss_fn=base.core.pilot.loss_fn)
        if 'loss_decoder_tail' in losses:
            raise AssertionError('UDPR loss must not exist')
        self.stats = dict(rt.controller.last_stats, gt_count=len(boxes), sample_image_id=int(ident))
        terms = [v for k, value in losses.items() if 'loss' in k
                 for v in (value if isinstance(value, (list, tuple)) else [value])
                 if torch.is_tensor(v) and v.requires_grad]
        self.stats['loss_terms'] = {k: sum(float(v.detach()) for v in
            (value if isinstance(value, (list, tuple)) else [value]) if torch.is_tensor(v))
            for k, value in losses.items() if 'loss' in k}
        return sum(terms)


def parity(rt, sample, ident):
    """Identical model, boxes and weights, original FCN deployment vs zero UBR."""
    from rsprompter.uav.ubr_coordinate_adapter import adapt_native_roi
    rt.network.eval()
    with torch.no_grad():
        payload = rt.payload(ident)
        rt.joint.prepare_uav(**payload)
        features = rt.satellite.extract_feat(sample['inputs'])
        prepared = rt.controller.prepare(features, sample['data_samples'][0].metainfo)
        pool = rt.select_pool(features, payload['geometry'], prepared=prepared)
        if not len(pool.boxes):
            return {'passed': False, 'reason': 'no natural predicted boxes; no GT inserted'}
        # Preserve the ORIGINAL per-source call and 32-RoI chunk boundaries.
        # Build independent oracle through head.predict_mask, not this controller.
        from mmengine.structures import InstanceData
        raw_exact = True
        native_chunks = []
        probabilities = []
        legacy_masks = []
        chunk_size = int(getattr(rt.satellite.roi_head, 'mask_roi_chunk_size', 32) or 0)
        start = 0
        while start < len(pool.boxes):
            end = start + 1
            while end < len(pool.boxes) and pool.source_ids[end] == pool.source_ids[start]:
                end += 1
            inst = InstanceData(bboxes=pool.boxes[start:end].clone(), scores=pool.scores[start:end].clone(),
                labels=torch.zeros(end-start, device='cuda', dtype=torch.long))
            legacy = rt.satellite.roi_head.predict_mask(prepared.roi_features, [prepared.meta], [inst],
                rescale=False, image_embeddings=prepared.image_embeddings,
                image_positional_embeddings=prepared.image_positional_embeddings,
                high_res_features=prepared.high_res_features)[0]
            legacy_masks.append(legacy.masks)
            step = chunk_size if chunk_size > 0 else end-start
            for begin in range(start,end,step):
                finish = min(begin+step,end)
                native_chunks.append([begin,finish])
                raw, feature = rt.controller.decode_native(prepared, pool.boxes[begin:finish])
                raw_again, _ = rt.controller.decode_native(prepared, pool.boxes[begin:finish])
                raw_exact = raw_exact and torch.equal(raw, raw_again)
                state = adapt_native_roi(raw, feature, pool.boxes[begin:finish], prepared.meta)
                probabilities.append(state.probabilities[:,0])
            start = end
        legacy_binary = torch.cat(legacy_masks)
        original_probabilities = torch.cat(probabilities)
        # Context and active inference decode sequentially, as formal deployment.
        prediction = rt.controller.predict(prepared, pool, payload['geometry'])
        error = float((prediction['probabilities'] - original_probabilities).abs().max())
        checks = dict(native_raw_exact=raw_exact, native_chunks=native_chunks,
                      probability_max_error=error,
                      legacy_binary_exact=torch.equal(legacy_binary, prediction['masks']),
                      scores_exact=torch.equal(pool.scores, prediction['scores']),
                      boxes_exact=torch.equal(pool.boxes, prediction['boxes']))
        checks['passed'] = bool(checks['native_raw_exact'] and error <= 1.1e-6 and
                                checks['legacy_binary_exact'] and checks['scores_exact'] and checks['boxes_exact'])
        checks['natural_prediction_count'] = len(pool.boxes)
    rt.clear()
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--updates', type=int, default=3, choices=[3, 4])
    p.add_argument('--engineering-checkpoint', type=Path)
    p.add_argument('--force-inference-250', action='store_true',
                   help='repeat a real predicted box; resource fixture only, not AP or predicted population')
    args = p.parse_args()
    if int(os.environ.get('WORLD_SIZE', 0)) != 4:
        raise RuntimeError('Use torchrun with exactly 4 ranks')
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    cpu = dist.new_group(backend='gloo', timeout=timedelta(minutes=60))
    ready = [None]
    if rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            ready[0] = {'ok': True}
        except Exception:
            ready[0] = {'ok': False, 'error': traceback.format_exc()}
    dist.broadcast_object_list(ready, 0, group=cpu)
    if not ready[0]['ok']:
        raise RuntimeError(ready[0]['error'])
    report = {'rank': rank, 'pid': os.getpid(), 'world_size': 4,
              'scope': 'real detector P1 engineering; no AP; not a training experiment',
              'forced_empty_gt': 'rank3 update1; same real image/RGB-D, labels explicitly cleared',
              'updates': [], 'engineering_checkpoint': str(args.engineering_checkpoint) if args.engineering_checkpoint else None}
    try:
        from ue_ubr_runtime_20261027 import build_runtime
        import ue_compact_joint_run_20261024 as base
        rt = build_runtime(content='correct', variant='detail', enabled=True)
        assert rt.joint is not None
        assert not any('decoder_tail_refiner' in name for name, _ in rt.network.named_parameters())
        assert not rt.satellite.roi_head.mask_head.decoder_tail_enabled
        if args.engineering_checkpoint:
            receipt = json.loads((args.engineering_checkpoint.parent / 'training_complete.json').read_text())
            checksum = [sha(args.engineering_checkpoint) if rank == 0 else None]
            dist.broadcast_object_list(checksum, 0, group=cpu)
            assert receipt['complete'] and receipt['epoch'] == 10 and receipt['updates'] == 2670
            assert receipt['world_size'] == 4 and receipt['checkpoint_sha256'] == checksum[0]
            assert receipt['contract']['variant'] == 'detail' and receipt['contract']['content'] == 'correct'
            for name, expected in receipt['contract']['source_hashes'].items():
                assert sha(WORK / name) == expected, ('engineering checkpoint source mismatch', name)
            cp = torch.load(args.engineering_checkpoint, map_location='cpu', weights_only=False)
            old = cp['model']
            state = {k: v for k, v in old.items() if 'decoder_tail_refiner' not in k}
            missing, unexpected = rt.joint.load_state_dict(state, strict=False)
            assert not unexpected and all('ubr_refiner' in k for k in missing), (missing, unexpected)
            report['engineering_checkpoint_sha256'] = checksum[0]
            report['initialization_role'] = 'task checkpoint for engineering only; not proposed new training initialization'
        else:
            report['initialization_role'] = 'generic foundation plus shared common initialization'
        assert all(float(p.abs().max()) == 0 for p in rt.controller.refiner.out.parameters())
        ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train', augment=False)
        ordered = sorted(range(len(ds)), key=lambda i: len(ds.samples[i]['annotations']))
        selected = [ordered[-1], ordered[-2], ordered[-3], ordered[0]][rank]
        ident = int(ds.samples[selected]['image_id'])
        batch = base.core.C.rtmdet_collate_fn([ds[selected]])
        sample = base.core.C.processed(rt.satellite, batch, False)
        report['sample_image_id'] = ident
        report['parity'] = parity(rt, sample, ident)
        torch.cuda.empty_cache()
        wrapper = EngineeringForward(rt, base).cuda()
        ddp = DistributedDataParallel(wrapper, device_ids=[local], output_device=local,
            find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
        scaler = torch.amp.GradScaler('cuda', init_scale=512.)
        report['frozen_before_sha'] = digest(wrapper, frozen_only=True)
        report['sources_before'] = source_inventory()
        generator = torch.Generator().manual_seed(44 + rank)
        torch.cuda.reset_peak_memory_stats()
        began = time.perf_counter()
        for step in range(1, args.updates + 1):
            ddp.train()
            rt.set_epoch(1)
            current = copy.deepcopy(sample)
            if rank == 3 and step == 1:
                current['data_samples'][0].gt_instances = current['data_samples'][0].gt_instances[:0]
            # Retry SAME prepared inputs with exact local RNG/buffer restoration.
            rng_before = dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(), loader=generator.get_state())
            buffers_before = {name: value.detach().clone() for name, value in wrapper.named_buffers()}
            attempts = []
            first_loss = None
            for attempt in range(9):
                if attempt:
                    random.setstate(rng_before['python'])
                    np.random.set_state(rng_before['numpy'])
                    torch.set_rng_state(rng_before['torch'])
                    torch.cuda.set_rng_state(rng_before['cuda'])
                    generator.set_state(rng_before['loader'])
                    with torch.no_grad():
                        for name,value in wrapper.named_buffers():
                            value.copy_(buffers_before[name])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = ddp(current, ident)
                finite_loss = torch.tensor(int(torch.isfinite(loss)), device='cuda')
                dist.all_reduce(finite_loss, op=dist.ReduceOp.MIN)
                assert finite_loss.item(), 'nonfinite forward loss; do not conceal by AMP backoff'
                if first_loss is None:
                    first_loss = float(loss.detach())
                else:
                    assert float(loss.detach()) == first_loss, 'restored retry changed forward loss'
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                bad = [name for name,parameter in wrapper.named_parameters()
                       if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
                bad_all = gather(bad[:12], cpu)
                norms = gradients(wrapper)
                old_scale = scaler.get_scale()
                attempts.append(dict(attempt=attempt, scale=old_scale,
                    nonfinite_first_names_by_rank=bad_all, gradient_norms=norms))
                if any(bad_all):
                    # No optimizer call, hence no parameters/AdamW state/decay change.
                    scaler.update(new_scale=old_scale / 2)
                    rt.clear()
                    if attempt == 8:
                        report['overflow_attempts'] = attempts
                        raise RuntimeError('synchronized AMP backoff exhausted')
                    continue
                norm = torch.nn.utils.clip_grad_norm_(wrapper.parameters(), .5, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                assert scaler.get_scale() >= old_scale, 'unexpected optimizer skip after finite all-rank gate'
                break
            del buffers_before
            digests = gather(digest(wrapper), cpu)
            assert len(set(digests)) == 1, 'replicas diverged'
            report['updates'].append(dict(step=step, loss=float(loss.detach()),
                grad_norm_before_clip=float(norm), gradient_norms=norms, amp_attempts=attempts,
                amp_scale=scaler.get_scale(), replica_sha=digests[0], **wrapper.stats))
            rt.clear()
        report['training_seconds'] = time.perf_counter() - began
        report['peak_allocated_mib'] = torch.cuda.max_memory_allocated() / 2**20
        report['peak_reserved_mib'] = torch.cuda.max_memory_reserved() / 2**20
        report['frozen_after_sha'] = digest(wrapper, frozen_only=True)
        assert report['frozen_before_sha'] == report['frozen_after_sha']
        report['sources_after'] = source_inventory()
        assert all(report['sources_after'].get(k) == v for k,v in report['sources_before'].items()), 'loaded local source changed during gate'
        # Rank-local checkpoint records all FIVE RNG families without touching other GPUs.
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state(), loader_generator=generator.get_state())
        torch.save(dict(rng=rng, sample_image_id=ident, optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict()), args.output / ('rank%d_state.pth' % rank))
        report['rng_families_saved'] = list(rng)
        if args.force_inference_250:
            # Explicit decoder resource stress only: not a candidate-generation claim.
            from rsprompter.uav.ubr_detector_integration import PredictionPool
            rt.network.eval()
            with torch.no_grad():
                payload = rt.payload(ident)
                rt.joint.prepare_uav(**payload)
                features = rt.satellite.extract_feat(sample['inputs'])
                prepared = rt.controller.prepare(features, sample['data_samples'][0].metainfo)
                natural = rt.select_pool(features, payload['geometry'], prepared=prepared)
                if not len(natural.boxes):
                    report['inference250'] = {'passed': False, 'reason': 'no real predicted box to repeat'}
                else:
                    fixture = PredictionPool(natural.boxes[:1].repeat(250, 1), natural.scores[:1].repeat(250),
                        torch.arange(250, device='cuda'), tuple('resource_repeat' for _ in range(250)))
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    pred = rt.controller.predict(prepared, fixture, payload['geometry'])
                    torch.cuda.synchronize()
                    assert len(pred['masks']) == 250 and torch.isfinite(pred['probabilities']).all()
                    report['inference250'] = dict(passed=True, fixture='one natural predicted box repeated; unique IDs',
                        seconds=time.perf_counter()-start, peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                rt.clear()
        report['max_natural_active'] = max(x['active_count'] for x in report['updates'])
        report['active_gradient_observed'] = any(x['active_count'] > 0 and x['gradient_norms']['ubr'] > 0
                                                   for x in report['updates'])
        report['engineering_completed'] = True
        report['passed_rank_checks'] = bool(report['parity']['passed'] and report['active_gradient_observed']
            and report.get('inference250', {}).get('passed'))
        write(args.output / ('rank%d.json' % rank), report)
        reports = gather(report, cpu)
        if rank == 0:
            accepted = (all(x['passed_rank_checks'] for x in reports) and
                        max(x['max_natural_active'] for x in reports) >= 64 and
                        reports[3]['updates'][0]['active_count'] == 0 and
                        reports[3]['updates'][0]['gt_count'] == 0)
            write(args.output / ('complete.json' if accepted else 'partial.json'),
                  dict(passed=accepted, world_size=4, ranks=reports,
                       meaning='engineering only; no AP, no long training',
                       caveat='64 real prediction-matched active instances on at least one same-spec GPU; rank3 intentionally sparse; no GT prediction insertion'))
    except Exception:
        report['failed'] = traceback.format_exc()
        write(args.output / ('rank%d_failed.json' % rank), report)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
