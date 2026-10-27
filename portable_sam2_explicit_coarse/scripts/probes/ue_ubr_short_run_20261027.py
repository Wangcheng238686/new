"""Four-GPU G/UBR diagnostic; requires reviewed engineering receipt, never schedules jobs.

All arms use fresh common foundation. Only the UBR geometry port is zeroed in
its matching control. Frozen numerical sources remain unchanged. Full-budget
training and test-guided model selection are outside this runner's contract.
"""
import argparse
from datetime import timedelta
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
REPOSITORY = WORK.parent
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.structures import InstanceData
import ue_ubr_runtime_20261027 as integration

previous = integration.previous
base, core = integration.base, integration.core
CONFIG = WORK / 'configs/ablations/ue_g_ubr_short_protocol_20261027.json'
PROTOCOL = json.loads(CONFIG.read_text())
ROOT = Path('/data/wangcheng/checkpoint/ue_g_ubr_short_20261027')
CRITICAL = [
    'scripts/probes/ue_ubr_runtime_20261027.py',
    'rsprompter/uav/ubr_detector_integration.py',
    'rsprompter/uav/ubr_coordinate_adapter.py',
    'rsprompter/uav/unified_boundary_refiner.py',
    'rsprompter/uav/ubr_prediction_context.py',
]


class TailPortModel(integration.GWithUBR):
    """Only UBR-port content differs; all training losses execute inside DDP."""
    def __init__(self, joint, controller, arm):
        super().__init__(joint, controller, arm != 'G_no_tail')
        self.arm = arm

    def forward(self, sample, geometry, rgb, index, anchor, instance_target, center_loss_fn):
        joint, satellite = self.joint, self.joint.satellite
        joint.prepare_uav(geometry, rgb, index, anchor)
        features = satellite.extract_feat(sample['inputs'])
        with core.C.cached_extract(satellite, features):
            losses = satellite.loss(sample['inputs'], sample['data_samples'])
        direct, _ = center_loss_fn(joint.uav.outputs, instance_target)
        losses = dict(losses, loss_uav_instances=direct)
        if self.ubr_enabled:
            meta = sample['data_samples'][0].metainfo
            prepared = self.controller.prepare(features, meta)
            pool = self.pool_callback(features, geometry, prepared)
            gt = sample['data_samples'][0].gt_instances
            target = gt.masks.to_tensor(dtype=torch.float32, device=geometry.device)
            port = torch.zeros_like(geometry) if self.arm == 'G_UBR_zero_geometry' else geometry
            losses.update(self.controller.training_loss(prepared, pool, gt.bboxes, target, port))
        return losses


def validate_protocol():
    bindings = {'lr': 'lr', 'weight_decay': 'weight_decay',
        'warmup_updates': 'warmup_updates', 'cosine_min_factor': 'cosine_min_factor',
        'clip': 'clip', 'amp_init_scale': 'amp_init_scale', 'seed': 'seed',
        'RPN_pre': 'RPN_pre', 'max_dets': 'max_dets'}
    for short, inherited in bindings.items():
        if PROTOCOL[short] != base.PROTOCOL[inherited]:
            raise RuntimeError('Inherited update/runtime protocol mismatch: ' + short)
    assert base.PROTOCOL['epochs'] == PROTOCOL['cosine_schedule_epochs'] == 100
    assert base.PROTOCOL['optimizer_updates_per_epoch'] == PROTOCOL['updates_per_epoch'] == 267
    assert PROTOCOL['world_size'] * PROTOCOL['per_rank_batch'] * PROTOCOL['per_rank_accumulation'] == 8


def setup(memory_cap_mib):
    if int(os.environ.get('WORLD_SIZE', 0)) != 4:
        raise RuntimeError('Every experiment requires torchrun --nproc_per_node=4')
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(2)
    if memory_cap_mib:
        total = torch.cuda.get_device_properties(rank).total_memory
        if not 0 < memory_cap_mib * 2**20 <= total:
            raise ValueError('allocator cap outside actual device memory')
        torch.cuda.set_per_process_memory_fraction(memory_cap_mib * 2**20 / total, rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=90))
    return dist.new_group(backend='gloo', timeout=timedelta(minutes=90))


def runtime(arm, training):
    rt = integration.build_runtime(content='correct', variant='detail', enabled=arm != 'G_no_tail')
    rt.bare = TailPortModel(rt.joint, rt.controller, arm)
    rt.bare.pool_callback = rt.select_pool
    rt.network = rt.bare
    rt.optimizer = torch.optim.AdamW([p for p in rt.bare.parameters() if p.requires_grad],
        lr=PROTOCOL['lr'], weight_decay=PROTOCOL['weight_decay'])
    if training:
        rt.network = DistributedDataParallel(rt.bare,
            device_ids=[torch.cuda.current_device()], output_device=torch.cuda.current_device(),
            find_unused_parameters=True, broadcast_buffers=False)
    rt.generator = torch.Generator().manual_seed(PROTOCOL['seed'] + dist.get_rank())
    core.C.seed(PROTOCOL['seed'] + dist.get_rank())
    return rt


def source_inventory():
    """After real model construction: actual local imports + declared numerical files."""
    paths = {Path(__file__).resolve(), CONFIG}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_file() and REPOSITORY in path.parents:
                paths.add(path)
    paths.update(WORK / name for name in previous.hashes())
    paths.update(WORK / name for name in CRITICAL)
    return {str(path.relative_to(REPOSITORY)): core.C.file_sha(path) for path in sorted(paths)}


def verify_sources(frozen):
    for name, expected in frozen.items():
        if core.C.file_sha(REPOSITORY / name) != expected:
            raise RuntimeError('Frozen source changed: ' + name)
    actual = source_inventory()
    if actual != frozen:
        raise RuntimeError('Loaded local source closure changed: ' + str(sorted(set(actual) ^ set(frozen))))


def verify_receipt(path, sources):
    gate = json.loads(path.read_text())
    if gate.get('passed') is not True or gate.get('world_size') != 4:
        raise RuntimeError('A genuinely PASSED four-rank full-detector gate is required')
    ranks = gate.get('ranks', [])
    if len(ranks) != 4 or not all(row.get('passed_rank_checks') for row in ranks):
        raise RuntimeError('Engineering receipt lacks four accepted real rank records')
    if max(row.get('max_natural_active', 0) for row in ranks) < 64:
        raise RuntimeError('Engineering receipt did not cover 64 real matched predictions')
    for row in ranks:
        known = row.get('sources_after', {})
        for name in CRITICAL:
            relative = str((WORK / name).relative_to(REPOSITORY))
            # Gate inventories use repository-relative paths; tolerate absolute keys.
            matches = [known.get(relative), known.get(str(WORK / name)), known.get(name)]
            expected = sources[relative]
            if expected not in matches:
                raise RuntimeError('Engineering receipt source mismatch: ' + relative)
    return core.C.file_sha(path)


def identity(rt, args, sources):
    purpose = 'engineering_one_epoch_only' if args.epochs == 1 else PROTOCOL['purpose']
    orders = {}
    for epoch in range(1, args.epochs + 1):
        path = base.ROOT / 'rgbd' / ('epoch%d.json' % epoch)
        order = json.loads(path.read_text())['sample_order']
        assert len(order) == len(set(order)) == 2136
        orders[str(path)] = core.C.file_sha(path)
    return dict(arm=args.arm, epochs=args.epochs, purpose=purpose, protocol=PROTOCOL,
        allocator_cap_mib=args.allocator_cap_mib, source_hashes=sources,
        initialization=rt.initialization_audit,
        data_identity=core.C.archive_identity(core.previous.DATA),
        input_cache_manifest_sha256=core.C.file_sha(core.OUT / 'input_cache_sha256.json'),
        measured_cache='/data/wangcheng/cache/ue_v1_native_geometry',
        anchor_cache='/data/wangcheng/cache/ue_v1_s2db_geometry_v2',
        source_order_sha256=orders, engineering_receipt_sha256=verify_receipt(args.engineering_receipt, sources))


def freeze_root(args, frozen):
    """Bind all three arms to one source/input identity; reject mixed experiment roots."""
    shared = {k: v for k, v in frozen.items() if k != 'arm'}
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / 'study.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.root / 'study_contract.json'
        if path.exists():
            receipt = json.loads(path.read_text())
            if receipt['shared'] != shared:
                raise RuntimeError('Root already binds different protocol/source/input contract')
            commit = receipt['git_commit']
        else:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
        for name, expected in shared['source_hashes'].items():
            blob = subprocess.check_output(['git', 'show', commit + ':' + name], cwd=REPOSITORY)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise RuntimeError('Commit numerical source before launch: ' + name)
        if not path.exists():
            core.C.dump(path, dict(shared=shared, git_commit=commit, created=time.time()))
            for name in shared['source_hashes']:
                target = args.root / 'frozen_source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((REPOSITORY / name).read_bytes())
    return commit


def prediction_arrays(pool, masks):
    inst = InstanceData(bboxes=pool.boxes.clone(), scores=pool.scores.clone(),
        labels=torch.zeros(len(pool.boxes), dtype=torch.long, device=pool.boxes.device), masks=masks)
    return core.C.gt_arrays(inst)


@torch.inference_mode()
def evaluate(rt, arm, split, out, cpu, checkpoint_sha=None):
    if torch.is_autocast_enabled():
        raise RuntimeError('Evaluation must use FP32')
    rt.bare.eval()
    ds = core.C.MinimalSatelliteDataset(core.previous.DATA, split)
    if len(ds) != 459:
        raise RuntimeError('Expected full459 split')
    rows, metas = [], []
    for i in range(dist.get_rank(), len(ds), 4):
        ident = ds.samples[i]['image_id']
        batch = core.C.rtmdet_collate_fn([ds[i]])
        sample = core.C.processed(rt.satellite, batch)
        payload = rt.payload(ident)
        rt.joint.prepare_uav(**payload)
        features = rt.satellite.extract_feat(sample['inputs'])
        prepared = rt.controller.prepare(features, sample['data_samples'][0].metainfo)
        pool = rt.select_pool(features, payload['geometry'], prepared)
        if arm == 'G_no_tail':
            context = rt.controller.prediction_context(prepared, pool)
            masks = context.retained_masks
        else:
            port = torch.zeros_like(payload['geometry']) if arm == 'G_UBR_zero_geometry' else payload['geometry']
            masks = rt.controller.predict(prepared, pool, port)['masks']
        pred = prediction_arrays(pool, masks)
        original = torch.tensor([s != 'uav' for s in pool.source_ids], dtype=torch.bool, device=pool.boxes.device)
        prefix = prediction_arrays(type(pool)(pool.boxes[original], pool.scores[original],
            pool.candidate_ids[original], tuple(s for s in pool.source_ids if s != 'uav')), masks[original])
        rows.append(dict(index=i, image_id=ident,
            gt=core.C.serialized(core.C.gt_arrays(sample['data_samples'][0].gt_instances)),
            dt=core.C.serialized(pred), original_path=core.C.serialized(prefix),
            source_ids=list(pool.source_ids)))
        metas.append((i, batch['img_metas'][0]))
        rt.clear()
    gathered = [None] * 4 if dist.get_rank() == 0 else None
    dist.gather_object((rows, metas), gathered, dst=0, group=cpu)
    metrics = None
    if dist.get_rank() == 0:
        records = sorted([r for part, _ in gathered for r in part], key=lambda r: r['index'])
        assert [r.pop('index') for r in records] == list(range(459))
        assert len({r['image_id'] for r in records}) == 459
        metadata = [m for _, m in sorted([m for _, part in gathered for m in part], key=lambda v: v[0])]
        gt = [{**r['gt'], 'bboxes': np.asarray(r['gt']['bboxes'], dtype=np.float32).reshape(-1, 4),
            'labels': np.asarray(r['gt']['labels'], dtype=np.int64)} for r in records]
        dt = [core.consumer.arrays(r['dt']) for r in records]
        cg, cd = core.C.build_coco_gt_and_dt(gt, dt, metadata, score_key='scores')
        metrics = {}
        for kind in ('segm', 'bbox'):
            metrics.update(core.C.run_coco_eval(cg, cd, kind, max_dets=[1, 10, 250]))
        metrics = {k.replace('AR@100', 'AR@250'): v for k, v in metrics.items()}
        core.C.dump(out / 'records.json', records)
        core.C.dump(out / 'metrics.json', metrics)
        core.C.dump(out / 'complete.json', dict(complete=True, split=split, images=459,
            world_size=4, precision='FP32', arm=arm, checkpoint_sha256=checkpoint_sha,
            scope='descriptive_development_test_not_selection' if split == 'test' else 'full_validation'))
    box = [metrics]
    dist.broadcast_object_list(box, src=0, group=cpu)
    return box[0]


def verify_orders(frozen):
    for path, expected in frozen['source_order_sha256'].items():
        if core.C.file_sha(path) != expected:
            raise RuntimeError('Original sample-order record changed: ' + path)


def save(rt, path, epoch, step, frozen, cpu, validation_performed=False):
    """Every epoch boundary, all five rank RNG families; no validation fiction."""
    local = previous.rank_rng(rt)
    states = [None] * 4 if dist.get_rank() == 0 else None
    dist.gather_object(local, states, dst=0, group=cpu)
    if dist.get_rank() == 0:
        core.previous.atomic_save(dict(model=rt.bare.state_dict(),
            optimizer=rt.optimizer.state_dict(), scaler=rt.scaler.state_dict(),
            rank_rng=states, epoch=epoch, step=step, contract=frozen, world_size=4,
            checkpoint_phase='after_epoch', validation_performed=validation_performed), path)
    dist.barrier()


def restore(rt, path, frozen, training):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if cp['contract'] != frozen or cp['world_size'] != 4 or cp['checkpoint_phase'] != 'after_epoch':
        raise RuntimeError('Checkpoint contract/world/phase mismatch')
    rt.bare.load_state_dict(cp['model'], strict=True)
    rt.set_epoch(max(1, cp['epoch']))
    if training:
        rt.optimizer.load_state_dict(cp['optimizer'])
        rt.scaler.load_state_dict(cp['scaler'])
        previous.restore_rank_rng(rt, cp['rank_rng'][dist.get_rank()])
    return cp


def rng_draw(rt):
    return dict(python=[random.random() for _ in range(5)],
        numpy=np.random.random(5).tolist(), torch=torch.rand(5).tolist(),
        cuda=torch.rand(5, device='cuda').cpu().tolist(),
        loader_generator=torch.rand(5, generator=rt.generator).tolist())


def resume_gate(rt, args, cpu, frozen):
    """Non-training save/restore gate. Synthetic moments are labelled explicitly."""
    folder = args.root / 'resume_gate' / args.arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        raise RuntimeError('Preserve completed resume gate; use a new root')
    gate_contract = dict(frozen, gate_only='RNG_and_checkpoint_restore_no_training')
    rt.set_epoch(1)
    parameter = next(p for p in rt.bare.parameters() if p.requires_grad)
    # Exercise nonempty optimizer serialization without any optimizer update or AP claim.
    rt.optimizer.state[parameter] = dict(step=torch.tensor(0.),
        exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
    before = core.C.state_digest(rt.bare)
    optimizer_before = rt.optimizer.state[parameter]['exp_avg'].clone()
    scaler_before = rt.scaler.state_dict()
    path = folder / 'restore_gate.pth'
    save(rt, path, 0, 0, gate_contract, cpu)
    expected = rng_draw(rt)
    rng_draw(rt)
    with torch.no_grad():
        parameter.add_(1.)
        rt.optimizer.state[parameter]['exp_avg'].add_(.5)
    changed_scaler = dict(scaler_before, scale=float(scaler_before['scale']) / 2)
    rt.scaler.load_state_dict(changed_scaler)
    restore(rt, path, gate_contract, training=True)
    actual = rng_draw(rt)
    same = {key: actual[key] == value for key, value in expected.items()}
    model_exact = core.C.state_digest(rt.bare) == before
    moment_exact = torch.equal(rt.optimizer.state[parameter]['exp_avg'], optimizer_before)
    scaler_exact = rt.scaler.state_dict() == scaler_before
    passed = all(same.values()) and model_exact and moment_exact and scaler_exact
    local = dict(rank=dist.get_rank(), passed=passed, rng_next_five_exact=same,
        model_exact=model_exact, optimizer_moment_exact=moment_exact, scaler_exact=scaler_exact,
        moment_origin='synthetic_zero_AdamW_state_no_training_update',
        scope='restoration mechanics; not real next-update trajectory equivalence')
    gathered = [None] * 4 if dist.get_rank() == 0 else None
    dist.gather_object(local, gathered, dst=0, group=cpu)
    if dist.get_rank() == 0:
        core.C.dump(folder / 'complete.json', dict(passed=all(r['passed'] for r in gathered),
            world_size=4, ranks=gathered, checkpoint_sha256=core.C.file_sha(path), contract=gate_contract))
    if not passed:
        raise RuntimeError('RNG/model/optimizer/scaler restoration gate failed')


def train(rt, args, cpu, frozen):
    folder = args.root / args.arm
    epoch = step = 0
    if args.resume:
        cp = restore(rt, args.resume, frozen, training=True)
        epoch, step = cp['epoch'], cp['step']
        assert step == epoch * 267 and 0 <= epoch <= args.epochs
    elif (folder / 'last.pth').exists():
        raise RuntimeError('Existing checkpoint requires explicit audited --resume')
    if dist.get_rank() == 0:
        core.C.dump(folder / 'contract.json', frozen)
    ds = core.C.MinimalSatelliteDataset(core.previous.DATA, 'train', augment=True)
    for epoch in range(epoch + 1, args.epochs + 1):
        verify_sources(frozen['source_hashes'])
        verify_orders(frozen)
        rt.network.train()
        rt.set_epoch(epoch)
        loader, reference = previous.epoch_loader(rt, ds, epoch)
        seen, batches, total, retries = [], [], 0., 0
        began = time.monotonic()
        for batch in loader:
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            seen.append(ident)
            batches.append((batch, ident))
            if len(batches) == 2:
                result = previous.accumulate(rt, batches, step + 1)
                step += 1
                total += result['loss_sum_global']
                retries += result['retries']
                batches = []
                if dist.get_rank() == 0 and step % 25 == 0:
                    core.C.dump(folder / 'progress.json', dict(epoch=epoch, update=step,
                        last_global_microbatch_mean=result['loss_sum_global'] / 8,
                        preclip_norm=result['preclip_norm'], AMP_retries_epoch=retries,
                        world_size=4, effective_batch=8, purpose=frozen['purpose']))
                    print('UBR UPDATE', args.arm, epoch, step, flush=True)
        assert not batches and len(seen) == len(set(seen)) == 534 and step == epoch * 267
        all_seen = [None] * 4
        dist.all_gather_object(all_seen, seen, group=cpu)
        reconstructed = [v for block in range(267) for rank in range(4)
                         for v in all_seen[rank][block * 2:block * 2 + 2]]
        assert reconstructed == reference
        metrics = None
        # Validation is sparse; checkpoint phase records epoch end honestly.
        if epoch in (1, args.epochs):
            metrics = evaluate(rt, args.arm, 'val', folder / ('validation_epoch%d' % epoch), cpu)
        save(rt, folder / 'last.pth', epoch, step, frozen, cpu, metrics is not None)
        if epoch in (1, args.epochs) and dist.get_rank() == 0:
            import shutil
            shutil.copyfile(folder / 'last.pth', folder / ('epoch%d.pth' % epoch))
        verify_sources(frozen['source_hashes'])
        if dist.get_rank() == 0:
            core.C.dump(folder / ('epoch%d.json' % epoch), dict(epoch=epoch, step=step,
                sample_order=reference, rank_sample_order=all_seen, world_size=4,
                mean_microbatch_loss=total / 2136, AMP_retries_epoch=retries,
                validation=metrics, checkpoint_retained=True, fixed_epoch_checkpoint=epoch in (1, args.epochs),
                seconds=time.monotonic() - began, purpose=frozen['purpose']))
    if dist.get_rank() == 0:
        path = folder / ('epoch%d.pth' % args.epochs)
        core.C.dump(folder / 'training_complete.json', dict(complete=True,
            epoch=args.epochs, updates=step, world_size=4, arm=args.arm,
            purpose=frozen['purpose'], checkpoint_sha256=core.C.file_sha(path), contract=frozen))
    dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'eval'])
    p.add_argument('--arm', required=True, choices=PROTOCOL['arms'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--epochs', type=int, choices=[1, 10], default=10)
    p.add_argument('--engineering-receipt', type=Path, required=True)
    p.add_argument('--allocator-cap-mib', type=int, default=0)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--resume', type=Path)
    p.add_argument('--gate-resume', action='store_true',
                   help='save/load and compare all five RNG streams; no training/evaluation')
    args = p.parse_args()
    if args.epochs == 1 and args.root.resolve() == ROOT.resolve():
        p.error('--epochs1 requires a separate engineering root')
    if args.epochs == 1 and args.split == 'test':
        p.error('one-epoch engineering runs may not evaluate test')
    validate_protocol()
    cpu = setup(args.allocator_cap_mib)
    try:
        rt = runtime(args.arm, args.action == 'train')
        sources = source_inventory()
        # Config tuples become lists in study_contract.json; canonicalize once
        # so subsequent arms/resume do not fail a Python tuple/list comparison.
        frozen = json.loads(json.dumps(identity(rt, args, sources)))
        commit = None
        if dist.get_rank() == 0:
            commit = freeze_root(args, frozen)
        data = [commit]
        dist.broadcast_object_list(data, src=0, group=cpu)
        frozen['git_commit'] = data[0]
        dist.barrier()
        verify_sources(sources)
        launches = [None] * 4 if dist.get_rank() == 0 else None
        dist.gather_object(dict(rank=dist.get_rank(), local_rank=int(os.environ['LOCAL_RANK']),
            pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19]),
            launches, dst=0, group=cpu)
        if dist.get_rank() == 0:
            core.C.dump(args.root / 'rank_launches' / ('%s_%s_%s.json' % (args.action, args.arm, args.split)),
                dict(world_size=4, ranks=launches, source_hashes=sources, git_commit=data[0]))
        if args.gate_resume:
            resume_gate(rt, args, cpu, frozen)
        elif args.action == 'train':
            train(rt, args, cpu, frozen)
        else:
            if args.resume:
                p.error('eval uses fixed final checkpoint, not --resume')
            if args.split == 'test':
                for arm in PROTOCOL['arms']:
                    done = json.loads((args.root / arm / 'training_complete.json').read_text())
                    if done['epoch'] != 10 or done['updates'] != 2670:
                        raise RuntimeError('All three fixedE10 trainings must finish before test')
            folder = args.root / args.arm
            done = json.loads((folder / 'training_complete.json').read_text())
            path = folder / ('epoch%d.pth' % args.epochs)
            if core.C.file_sha(path) != done['checkpoint_sha256']:
                raise RuntimeError('Fixed checkpoint SHA mismatch')
            restore(rt, path, frozen, training=False)
            out = args.root / 'evaluation' / args.split / args.arm
            if (out / 'complete.json').exists():
                raise RuntimeError('Evaluation already complete; preserve existing artifact')
            evaluate(rt, args.arm, args.split, out, cpu, done['checkpoint_sha256'])
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
