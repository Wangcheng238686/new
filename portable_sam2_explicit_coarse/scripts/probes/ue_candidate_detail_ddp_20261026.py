"""Every training experiment is one four-rank DDP job, effective batch eight."""
import argparse
import contextlib
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK/'scripts/probes')]
import ue_candidate_detail_run_20261026 as single
from data.ue_instance_targets import center_targets
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

base = single.base
ROOT = Path('/data/wangcheng/checkpoint/ue_candidate_detail_ddp_20261026')
CONFIG = WORK/'configs/ablations/ue_candidate_detail_ddp_protocol.json'
PROTOCOL = json.loads(CONFIG.read_text())
NEW = ['scripts/probes/ue_candidate_detail_ddp_20261026.py',
       'configs/ablations/ue_candidate_detail_ddp_protocol.json']


def hashes():
    return {**single.hashes(), **{n:base.core.C.file_sha(WORK/n) for n in NEW}}


def setup():
    assert int(os.environ.get('WORLD_SIZE', '0')) == 4, 'Use torchrun --nproc_per_node=4'
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(PROTOCOL['allocator_cap_MiB']*2**20/
        torch.cuda.get_device_properties(local).total_memory, local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=90))
    cpu = dist.new_group(backend='gloo', timeout=timedelta(minutes=90))
    return dist.get_rank(), cpu


def runtime(variant, content, training):
    if variant == 'satellite':
        rt = base.Runtime(base.ROOT, 'satellite')
    elif variant == 'baseline':
        rt = base.Runtime(base.ROOT, {'correct':'rgbd', 'constant':'constant_rgb'}[content])
    else:
        rt = single.runtime(variant, content)
    rt.bare = rt.network
    if training:
        rt.network = DistributedDataParallel(rt.bare, device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(), find_unused_parameters=True,
            broadcast_buffers=False)
    rt.generator = torch.Generator().manual_seed(44+dist.get_rank())
    base.core.C.seed(44+dist.get_rank())
    return rt


def identity(args):
    return dict(variant=args.variant, content=args.content, protocol=PROTOCOL,
        source_hashes=hashes(), shared_training_protocol=base.PROTOCOL,
        initialization=json.loads((base.ROOT/'initialization.json').read_text()),
        data_identity=base.core.C.archive_identity(base.core.previous.DATA),
        measured_cache_manifest_sha256=base.core.C.file_sha(base.core.OUT/'input_cache_sha256.json'),
        reference_build_sha256=base.core.C.file_sha(base.PLANE/'BUILD_COMPLETE.json'))


def rank_rng(rt):
    # One active CUDA device per rank. get_rng_state_all would initialize
    # unnecessary CUDA contexts on the other three visible GPUs in every rank.
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(), loader_generator=rt.generator.get_state())


def restore_rank_rng(rt, state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])
    rt.generator.set_state(state['loader_generator'])


def objective(rt, sample, ident):
    if rt.joint is None:
        losses = rt.network(sample['inputs'], sample['data_samples'], mode='loss')
    else:
        boxes = sample['data_samples'][0].gt_instances.bboxes.detach().cpu().numpy()
        losses = rt.network(sample, **rt.payload(ident), instance_target=center_targets(boxes),
            center_loss_fn=base.core.pilot.loss_fn)
    terms = []
    for name,value in losses.items():
        if 'loss' not in name:
            continue
        for tensor in value if isinstance(value, (list, tuple)) else [value]:
            if torch.is_tensor(tensor) and tensor.requires_grad:
                terms.append(tensor)
    assert terms
    return sum(terms)


def accumulate(rt, batches, step, inject_overflow=False):
    assert len(batches) == 2
    rng = rank_rng(rt)
    buffers = {n:v.detach().clone() for n,v in rt.bare.named_buffers()}
    failed_losses = None
    for attempt in range(5):
        if attempt:
            restore_rank_rng(rt, rng)
            with torch.no_grad():
                for n,v in rt.bare.named_buffers():
                    v.copy_(buffers[n])
        rt.optimizer.zero_grad(set_to_none=True)
        values = []
        for i,(batch, ident) in enumerate(batches):
            sample = base.core.C.processed(rt.satellite, batch, True)
            sync = rt.network.no_sync() if i == 0 else contextlib.nullcontext()
            with sync:
                with torch.autocast('cuda', dtype=torch.float16):
                    loss = objective(rt, sample, ident)
                finite = torch.isfinite(loss).to(dtype=torch.int32)
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                assert finite.item(), 'All ranks stop on nonfinite forward loss'
                values.append(float(loss.detach()))
                rt.scaler.scale(loss/2).backward()
            rt.clear()
        if inject_overflow and attempt == 0:
            # Exercise synchronized backoff; every rank must take the same retry.
            next(p for p in rt.network.parameters() if p.grad is not None).grad.fill_(float('inf'))
        succeeded = True
        try:
            norm = rt.update(step, allow_retry=True)
        except base.AmpOverflow:
            succeeded = False
        flag = torch.tensor(int(succeeded), device='cuda')
        low, high = flag.clone(), flag.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        assert low.item() == high.item(), 'Mixed AMP outcomes; refuse divergent optimizer replicas'
        if not succeeded:
            failed_losses = values
            if attempt == 4:
                raise RuntimeError('All-rank AMP backoff exhausted')
            continue
        total = torch.tensor(sum(values), dtype=torch.float64, device='cuda')
        dist.all_reduce(total)
        if inject_overflow:
            assert values == failed_losses, 'AMP retry changed prepared-input forward losses'
        return dict(loss_sum_global=float(total), local_losses=values, retries=attempt, preclip_norm=norm)
    raise AssertionError('unreachable')


def epoch_loader(rt, dataset, epoch, indices=None):
    reference = json.loads((base.ROOT/'rgbd'/f'epoch{epoch}.json').read_text())['sample_order']
    assert len(reference) == len(set(reference)) == 2136
    index_by_id = {int(s['image_id']):i for i,s in enumerate(dataset.samples)}
    ordered = [index_by_id[i] for i in reference] if indices is None else indices
    rank = dist.get_rank()
    local = [ordered[start+rank*2+offset] for start in range(0, len(ordered), 8) for offset in range(2)]
    return DataLoader(dataset, batch_size=1, sampler=local, num_workers=2,
        generator=rt.generator, collate_fn=base.core.C.rtmdet_collate_fn), reference


def save(rt, path, epoch, step, frozen, cpu):
    local_rng = rank_rng(rt)
    gathered_rng = [None]*4 if dist.get_rank() == 0 else None
    dist.gather_object(local_rng, gathered_rng, dst=0, group=cpu)
    if dist.get_rank() == 0:
        base.core.previous.atomic_save(dict(model=rt.bare.state_dict(), optimizer=rt.optimizer.state_dict(),
            scaler=rt.scaler.state_dict(), rank_rng=gathered_rng, epoch=epoch, step=step,
            contract=frozen, world_size=4, checkpoint_phase='after_validation'), path)
    dist.barrier()


def restore(rt, path, frozen, training):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    assert cp['contract'] == frozen and cp['world_size'] == 4
    assert cp['checkpoint_phase'] == 'after_validation'
    rt.bare.load_state_dict(cp['model'], strict=True)
    rt.set_epoch(max(1, cp['epoch']))
    if training:
        rt.optimizer.load_state_dict(cp['optimizer'])
        rt.scaler.load_state_dict(cp['scaler'])
        restore_rank_rng(rt, cp['rank_rng'][dist.get_rank()])
    return cp


def evaluate(rt, split, folder, cpu, checkpoint_sha=None, limit=None, mode='matched'):
    rt.bare.eval()
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, split)
    count = min(limit, len(ds)) if limit else len(ds)
    local, meta_rows = [], []
    donor_map = {}
    with torch.inference_mode():
        for i in range(dist.get_rank(), count, 4):
            ident = ds.samples[i]['image_id']
            batch = base.core.C.rtmdet_collate_fn([ds[i]])
            sample = base.core.C.processed(rt.satellite, batch)
            if rt.joint is not None:
                payload = rt.payload(ident)
                if mode == 'constant':
                    payload['rgb'][:] = rt.joint.uav.rgb_mean
                elif mode == 'donor':
                    donor = ds.samples[(i+len(ds)//2)%len(ds)]['image_id']
                    payload['rgb'] = rt.uav_inputs.load(donor, geometry_mode='measured', rgb_mode='correct')['rgb']
                    donor_map[str(ident)] = donor
                rt.joint.prepare_uav(**payload)
                adapter, geometry = rt.joint.uav, payload['geometry']
            else:
                adapter, geometry = base.SimpleNamespace(enabled=False), None
            features = rt.satellite.extract_feat(sample['inputs'])
            prefix, combined, accepted = base.core.inference_outputs(rt.satellite, adapter,
                sample, features, geometry)
            local.append(dict(index=i, image_id=ident,
                gt=base.core.C.serialized(base.core.C.gt_arrays(sample['data_samples'][0].gt_instances)),
                dt=base.core.C.serialized(combined), original_path=base.core.C.serialized(prefix),
                geometry_accepted=accepted))
            meta_rows.append((i, batch['img_metas'][0]))
            rt.clear()
    gathered = [None]*4 if dist.get_rank() == 0 else None
    dist.gather_object((local, meta_rows, donor_map), gathered, dst=0, group=cpu)
    metrics = None
    if dist.get_rank() == 0:
        rows = sorted([r for part,_,_ in gathered for r in part], key=lambda r:r['index'])
        assert [r.pop('index') for r in rows] == list(range(count))
        metas = [m for _,m in sorted([m for _,part,_ in gathered for m in part], key=lambda x:x[0])]
        gt = [{**r['gt'], 'bboxes':np.asarray(r['gt']['bboxes']), 'labels':np.asarray(r['gt']['labels'])} for r in rows]
        dt = [base.core.consumer.arrays(r['dt']) for r in rows]
        cg,cd = base.core.C.build_coco_gt_and_dt(gt, dt, metas, score_key='scores')
        metrics = {}
        for kind in ('segm', 'bbox'):
            metrics.update(base.core.C.run_coco_eval(cg, cd, kind, max_dets=[1,10,250]))
        metrics = {k.replace('AR@100', 'AR@250'):v for k,v in metrics.items()}
        base.core.C.dump(folder/'records.json', rows)
        base.core.C.dump(folder/'metrics.json', metrics)
        base.core.C.dump(folder/'complete.json', dict(complete=True, split=split, images=count,
            world_size=4, precision='FP32', checkpoint_sha256=checkpoint_sha, source_hashes=hashes(),
            mode=mode, donor_map={k:v for _,_,part in gathered for k,v in part.items()},
            purpose='engineering_subset' if limit else 'fixed_E10_full_split'))
    box = [metrics]
    dist.broadcast_object_list(box, src=0, group=cpu)
    return box[0]


def gate(rt, args, cpu):
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train')
    # Different images on each rank, including maximum GT and a sparse image.
    ordered = sorted(range(len(ds)), key=lambda i:len(ds.samples[i]['annotations']))
    chosen = [ordered[-1], ordered[0], ordered[-2], ordered[1], ordered[-3], ordered[2], ordered[-4], ordered[3]]
    rank = dist.get_rank()
    batches = [(base.core.C.rtmdet_collate_fn([ds[i]]), ds.samples[i]['image_id'])
               for i in chosen[rank*2:rank*2+2]]
    frozen_before = {n:p.detach().cpu().clone() for n,p in rt.bare.named_parameters() if not p.requires_grad}
    rt.network.train()
    rt.set_epoch(1)
    first = accumulate(rt, batches, 1, inject_overflow=True)
    assert first['retries'] == 1 and rt.scaler.get_scale() == 256
    for step in (2,3):
        accumulate(rt, batches, step)
    assert all(torch.equal(v, dict(rt.bare.named_parameters())[n].detach().cpu()) for n,v in frozen_before.items())
    # Verify replica equality from actual parameter bytes, not just rank metadata.
    digest = base.core.C.state_digest(rt.bare)
    all_digests = [None]*4
    dist.all_gather_object(all_digests, digest, group=cpu)
    assert len(set(all_digests)) == 1
    metrics = evaluate(rt, 'val', args.root/'gates'/args.variant/'evaluation', cpu, limit=8)
    with torch.inference_mode():
        sample = base.core.C.processed(rt.satellite, batches[0][0])
        if rt.joint is not None:
            rt.joint.prepare_uav(**rt.payload(batches[0][1]))
        features = rt.satellite.extract_feat(sample['inputs'])
        force = torch.tensor([[16.,16.,40.,40.]], device='cuda').repeat(250, 1)
        prediction = base.core.consumer.masks(rt.satellite.roi_head,
            rt.satellite.roi_head._inject_extra_pe(features[0]), features,
            sample['data_samples'][0].metainfo, force, torch.full((250,), .1, device='cuda'))
        assert len(prediction['scores']) == 250
        rt.clear()
    path = args.root/'gates'/args.variant/'restore.pth'
    frozen = identity(args)
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    save(rt, path, 0, 3, frozen, cpu)
    expected_rng = rank_rng(rt)
    restore_digest = base.core.C.state_digest(rt.bare)
    parameter = next(p for p in rt.bare.parameters() if p.requires_grad)
    moment = next(v['exp_avg'] for v in rt.optimizer.state.values() if 'exp_avg' in v)
    expected_moment = moment.clone()
    with torch.no_grad():
        parameter.add_(1)
        moment.zero_()
    torch.rand(5, generator=rt.generator)
    restore(rt, path, frozen, training=True)
    actual_rng = rank_rng(rt)
    assert torch.equal(expected_rng['loader_generator'], actual_rng['loader_generator'])
    assert expected_rng['python'] == actual_rng['python']
    assert expected_rng['numpy'][0] == actual_rng['numpy'][0]
    assert np.array_equal(expected_rng['numpy'][1], actual_rng['numpy'][1])
    assert expected_rng['numpy'][2:] == actual_rng['numpy'][2:]
    assert torch.equal(expected_rng['torch'], actual_rng['torch'])
    assert torch.equal(expected_rng['cuda'], actual_rng['cuda'])
    assert base.core.C.state_digest(rt.bare) == restore_digest
    actual_moment = next(v['exp_avg'] for v in rt.optimizer.state.values() if 'exp_avg' in v)
    assert torch.equal(actual_moment, expected_moment)
    rt.network.train()
    accumulate(rt, batches, 4)
    peaks = [None]*4
    dist.all_gather_object(peaks, dict(rank=rank, peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20, first_update=first,
        restored_all_five_rng=True, forced_decoder_instances=250), group=cpu)
    if rank == 0:
        path.unlink()
        base.core.C.dump(args.root/'gates'/args.variant/'complete.json', dict(passed=True,
            source_hashes=hashes(), world_size=4, successful_updates=4, replicas_identical=True,
            frozen_parameter_tensors=len(frozen_before), nonempty_optimizer_restore=True,
            global_batch=8, rank_peaks=peaks, evaluation_metrics=metrics))


def train(rt, args, cpu):
    rank = dist.get_rank()
    folder = args.root/args.variant/args.content
    frozen = identity(args)
    ready = json.loads((args.root/'engineering_complete.json').read_text())
    assert ready['passed'] and ready['source_hashes'] == hashes()
    epoch = step = 0
    if args.resume:
        cp = restore(rt, args.resume, frozen, training=True)
        epoch, step = cp['epoch'], cp['step']
        assert step == epoch*267
    else:
        assert not (folder/'last.pth').exists()
    if rank == 0:
        base.core.C.dump(folder/'contract.json', frozen)
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train', augment=True)
    for epoch in range(epoch+1, 11):
        rt.network.train()
        rt.set_epoch(epoch)
        loader, order = epoch_loader(rt, ds, epoch)
        prepared, total, retries, seen = [], 0., 0, []
        began = time.time()
        for batch in loader:
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            seen.append(ident)
            prepared.append((batch, ident))
            if len(prepared) == 2:
                row = accumulate(rt, prepared, step+1)
                step += 1
                prepared = []
                total += row['loss_sum_global']
                retries += row['retries']
                if rank == 0 and step % 25 == 0:
                    base.core.C.dump(folder/'progress.json', dict(epoch=epoch, update=step,
                        last_global_microbatch_mean=row['loss_sum_global']/8,
                        preclip_norm=row['preclip_norm'], AMP_retries_epoch=retries,
                        rank0_pid=os.getpid(), world_size=4, effective_batch=8))
                    print('DDP UPDATE', args.variant, args.content, epoch, step, flush=True)
        assert not prepared and len(seen) == 534 and len(set(seen)) == 534 and step == epoch*267
        all_seen = [None]*4
        dist.all_gather_object(all_seen, seen, group=cpu)
        reconstructed = [v for block in range(267) for r in range(4) for v in all_seen[r][block*2:block*2+2]]
        assert reconstructed == order
        metrics = evaluate(rt, 'val', folder/'validation_current', cpu)
        save(rt, folder/'last.pth', epoch, step, frozen, cpu)
        if rank == 0:
            if epoch in (1,3,10):
                shutil.copyfile(folder/'last.pth', folder/f'epoch{epoch}.pth')
            base.core.C.dump(folder/f'epoch{epoch}.json', dict(epoch=epoch, step=step,
                sample_order=order, rank_sample_order=all_seen, world_size=4, effective_batch=8,
                validation=metrics, mean_microbatch_loss=total/2136, AMP_retries_epoch=retries,
                seconds=time.time()-began))
        assert hashes() == frozen['source_hashes']
    if rank == 0:
        base.core.C.dump(folder/'training_complete.json', dict(complete=True, epoch=10, updates=2670,
            world_size=4, checkpoint_sha256=base.core.C.file_sha(folder/'epoch10.pth'), contract=frozen))
    dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['gate','train','eval'])
    p.add_argument('--variant', choices=['query','detail','both','baseline','satellite'], default='both')
    p.add_argument('--content', choices=['correct','constant','plane'], default='correct')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--split', choices=['val','test'], default='val')
    p.add_argument('--mode', choices=['matched','constant','donor'], default='matched')
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    rank,cpu = setup()
    launches = [None]*4 if rank == 0 else None
    dist.gather_object(dict(rank=rank, pid=os.getpid(), local_rank=int(os.environ['LOCAL_RANK']),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19]), launches, dst=0, group=cpu)
    if rank == 0:
        base.core.C.dump(args.root/'rank_launches'/('_'.join([args.action,args.variant,args.content,args.split,args.mode])+'.json'),
            dict(world_size=4, ranks=launches, observed=time.time()))
    rt = runtime(args.variant, args.content, training=args.action != 'eval')
    if args.action == 'gate':
        gate(rt, args, cpu)
    elif args.action == 'train':
        train(rt, args, cpu)
    else:
        if args.split == 'test':
            assert all((args.root/v/c/'training_complete.json').exists() for v,c in PROTOCOL['training_jobs'])
        folder = args.root/args.variant/args.content
        done = json.loads((folder/'training_complete.json').read_text())
        path = folder/'epoch10.pth'
        assert done['checkpoint_sha256'] == base.core.C.file_sha(path)
        restore(rt, path, identity(args), training=False)
        out = args.root/'evaluation'/args.split/args.variant/args.content/args.mode
        assert not (out/'complete.json').exists()
        evaluate(rt, args.split, out, cpu, checkpoint_sha=done['checkpoint_sha256'], mode=args.mode)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
