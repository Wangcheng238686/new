"""Full100 compact RGB-D, two independent four-rank seeds, generic init only."""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WORK), str(WORK / 'scripts/probes')]
import ue_candidate_detail_ddp_20261026 as ddp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

base = ddp.base
ORIGINAL = copy.deepcopy(base.PROTOCOL)
CONFIG = WORK / 'configs/ablations/ue_rgbd_two_seed_ddp_20261027.json'
PROTOCOL = json.loads(CONFIG.read_text())
ROOT = Path('/data/wangcheng/checkpoint/ue_rgbd_two_seed_ddp_20261027')
EXTRA = ['configs/ablations/ue_rgbd_two_seed_ddp_20261027.json',
         'scripts/probes/ue_rgbd_two_seed_ddp_20261027.py',
         'scripts/probes/ue_rgbd_two_seed_owner_20261027.py',
         'scripts/probes/ue_candidate_detail_ddp_20261026.py']


def hashes():
    return {**base.sources(), **{name: base.core.C.file_sha(WORK / name) for name in EXTRA}}


def configure(seed):
    assert seed in PROTOCOL['seeds']
    # Imported frozen functions get an explicit run-local seed override; no file edits.
    base.PROTOCOL = dict(ORIGINAL, seed=seed)
    ddp.PROTOCOL = PROTOCOL
    ddp.hashes = hashes
    ddp.identity = identity


def seed_folder(root, seed):
    return root / ('seed%d' % seed)


def identity(args):
    folder = seed_folder(args.root, args.seed)
    return dict(seed=args.seed, protocol=PROTOCOL, numerical_protocol=base.PROTOCOL,
        source_hashes=hashes(), initialization=json.loads((folder / 'initialization.json').read_text()),
        order_sha256=base.core.C.file_sha(folder / 'epoch_orders.json'),
        data_identity=base.core.C.archive_identity(base.core.previous.DATA),
        measured_manifest_sha256=base.core.C.file_sha(base.core.OUT / 'input_cache_sha256.json'),
        reference_build_sha256=base.core.C.file_sha(base.PLANE / 'BUILD_COMPLETE.json'))


def initialize(root, seed):
    configure(seed)
    folder = seed_folder(root, seed)
    assert not (folder / 'common_init.pth').exists()
    folder.mkdir(parents=True, exist_ok=True)
    actual_load = torch.load
    loaded = []
    def generic_only(path, *args, **kwargs):
        assert Path(path).resolve() == base.FOUNDATION.resolve(), path
        loaded.append(str(Path(path).resolve()))
        return actual_load(path, *args, **kwargs)
    base.core.C.seed(seed)
    with patch('torch.load', side_effect=generic_only):
        satellite, config = base.build_foundation_satellite(base.core.C._register_and_build,
            WORK / 'configs/ablations/ue_minimal_single.py', base.FOUNDATION)
        satellite_sha = base.core.C.state_digest(satellite)
        joint = base.CompactJoint(satellite, base.FOUNDATION, source_seed=2001 + seed)
    assert base.core.C.state_digest(satellite) == satellite_sha
    state = dict(satellite=satellite.state_dict(), uav=joint.uav.state_dict(), model_config=config,
        satellite_initial_sha=satellite_sha, uav_initial_sha=base.core.previous.tensor_digest(joint.uav.state_dict()),
        foundation_sha256=base.core.C.file_sha(base.FOUNDATION), source_hashes=base.sources(),
        protocol=base.PROTOCOL, task_weights_loaded=False, actual_torch_load_paths=loaded)
    base.core.previous.atomic_save(state, folder / 'common_init.pth')
    base.core.C.dump(folder / 'initialization.json', dict(common_init_sha256=base.core.C.file_sha(folder / 'common_init.pth'),
        satellite_initial_sha=satellite_sha, uav_initial_sha=state['uav_initial_sha'],
        foundation_sha256=state['foundation_sha256'], source_hashes=state['source_hashes'],
        protocol=base.PROTOCOL, task_weights_loaded=False, actual_torch_load_paths=loaded))
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train')
    assert len(ds) == 2142
    generator = torch.Generator().manual_seed(seed)
    orders = [[int(ds.samples[i]['image_id']) for i in torch.randperm(2142, generator=generator)[:2136].tolist()]
              for _ in range(100)]
    assert all(len(order) == len(set(order)) == 2136 for order in orders)
    base.core.C.dump(folder / 'epoch_orders.json', orders)
    del joint, satellite, state


def inventory(root):
    files = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        p = Path(filename).resolve()
        if p.suffix == '.py' and p.is_file() and WORK.parent in p.parents:
            files[str(p.relative_to(WORK.parent))] = base.core.C.file_sha(p)
    for rel, sha in hashes().items():
        files[str((WORK / rel).relative_to(WORK.parent))] = sha
    base.core.C.dump(root / 'runtime_sources.json', files)
    for rel in files:
        out = root / 'source' / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(WORK.parent / rel, out)
    return files


def prepare(root):
    assert not (root / 'runtime_sources.json').exists(), 'Preserve prior preparation'
    for seed in PROTOCOL['seeds']:
        initialize(root, seed)
    assert json.loads((root / 'seed44/initialization.json').read_text())['satellite_initial_sha'] != json.loads((root / 'seed45/initialization.json').read_text())['satellite_initial_sha']
    files = inventory(root)
    versions = sorted([dict(name=d.metadata.get('Name', 'UNKNOWN'), version=d.version)
                       for d in importlib.metadata.distributions()], key=lambda d: d['name'].lower())
    base.core.C.dump(root / 'environment.json', dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), packages=versions, torch=torch.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), captured=time.time(),
        gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,driver_version,memory.total',
                                      '--format=csv'], text=True)))
    base.core.C.dump(root / 'prepared.json', dict(prepared=True, source_hashes=hashes(),
        imported_source_files=len(files), task_weights_loaded=False, protocol=PROTOCOL))
    print('PREPARED', PROTOCOL['seeds'], len(files), flush=True)


def finalize(root):
    files = json.loads((root / 'runtime_sources.json').read_text())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=WORK, text=True).strip()
    for rel, sha in files.items():
        assert base.core.C.file_sha(WORK.parent / rel) == sha
        blob = subprocess.check_output(['git', 'show', commit + ':' + rel], cwd=WORK)
        assert hashlib.sha256(blob).hexdigest() == sha, rel
        assert base.core.C.file_sha(root / 'source' / rel) == sha
    reservation = 0
    for seed in PROTOCOL['seeds']:
        gate = json.loads((seed_folder(root, seed) / 'gates/baseline/complete.json').read_text())
        assert gate['passed'] and gate['source_hashes'] == hashes()
        assert gate['world_size'] == 4 and gate['successful_updates'] == 4
        assert gate['replicas_identical'] and gate['nonempty_optimizer_restore']
        for row in gate['rank_peaks']:
            assert row['restored_all_five_rng'] and row['forced_decoder_instances'] == 250
            reservation = max(reservation, int(row['peak_reserved_MiB']) + 1024)
    base.core.C.dump(root / 'engineering_complete.json', dict(passed=True, source_hashes=hashes(),
        per_GPU_reservation_MiB=reservation, imported_sources=files, git_commit=commit,
        environment_sha256=base.core.C.file_sha(root / 'environment.json'), protocol=PROTOCOL))
    print('FINALIZED', commit, 'per GPU per group', reservation, flush=True)


def runtime(args):
    rt = base.Runtime(seed_folder(args.root, args.seed), 'rgbd')
    rt.bare = rt.network
    if args.action != 'eval':
        rt.network = DistributedDataParallel(rt.bare, device_ids=[torch.cuda.current_device()],
            output_device=torch.cuda.current_device(), find_unused_parameters=True, broadcast_buffers=False)
    rt.generator = torch.Generator().manual_seed(1000 * args.seed + dist.get_rank())
    base.core.C.seed(1000 * args.seed + dist.get_rank())
    return rt


def evaluate(rt, split, out, cpu, checkpoint_sha=None):
    metrics = ddp.evaluate(rt, split, out, cpu, checkpoint_sha=checkpoint_sha)
    if dist.get_rank() == 0:
        receipt = json.loads((out / 'complete.json').read_text())
        receipt.update(purpose='full100_valbest_test' if split == 'test' else 'training_validation',
                       max_dets=250, optimizer_updates=0)
        base.core.C.dump(out / 'complete.json', receipt)
    dist.barrier()
    return metrics


def save(rt, folder, epoch, step, frozen, best, cpu, improved):
    rank_rng = [None] * 4 if dist.get_rank() == 0 else None
    dist.gather_object(ddp.rank_rng(rt), rank_rng, dst=0, group=cpu)
    if dist.get_rank() == 0:
        path = folder / ('epoch%d.pth' % epoch)
        assert not path.exists()
        base.core.previous.atomic_save(dict(model=rt.bare.state_dict(), optimizer=rt.optimizer.state_dict(),
            scaler=rt.scaler.state_dict(), rank_rng=rank_rng, epoch=epoch, step=step, best=best,
            contract=frozen, world_size=4, checkpoint_phase='after_validation'), path)
        for name in ['last.pth'] + (['best_model.pth'] if improved else []):
            tmp = folder / (name + '.linktmp')
            os.link(path, tmp)
            tmp.replace(folder / name)
    dist.barrier()


def train(rt, args, cpu):
    folder = seed_folder(args.root, args.seed)
    ready = json.loads((args.root / 'engineering_complete.json').read_text())
    assert ready['passed'] and ready['source_hashes'] == hashes()
    frozen = identity(args)
    best = dict(score=-1., epoch=None)
    epoch = step = 0
    if args.resume:
        cp = ddp.restore(rt, args.resume, frozen, training=True)
        epoch, step, best = cp['epoch'], cp['step'], cp['best']
        assert step == epoch * 267
        del cp
    else:
        assert not (folder / 'last.pth').exists()
    if dist.get_rank() == 0:
        base.core.C.dump(folder / 'contract.json', frozen)
    ds = base.core.C.MinimalSatelliteDataset(base.core.previous.DATA, 'train', augment=True)
    index_by_id = {int(s['image_id']): i for i, s in enumerate(ds.samples)}
    orders = json.loads((folder / 'epoch_orders.json').read_text())
    for epoch in range(epoch + 1, 101):
        rt.network.train()
        rt.set_epoch(epoch)
        order = orders[epoch - 1]
        local_ids = [order[start + dist.get_rank() * 2 + offset]
                     for start in range(0, 2136, 8) for offset in range(2)]
        loader = DataLoader(ds, batch_size=1, sampler=[index_by_id[i] for i in local_ids],
            num_workers=2, generator=rt.generator, collate_fn=base.core.C.rtmdet_collate_fn)
        batches, seen, total, retries = [], [], 0., 0
        began = time.time()
        for batch in loader:
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            seen.append(ident)
            batches.append((batch, ident))
            if len(batches) == 2:
                value = ddp.accumulate(rt, batches, step + 1)
                step += 1
                batches = []
                total += value['loss_sum_global']
                retries += value['retries']
                if dist.get_rank() == 0 and step % 25 == 0:
                    base.core.C.dump(folder / 'progress.json', dict(seed=args.seed, epoch=epoch,
                        update=step, rank0_pid=os.getpid(), world_size=4, effective_batch=8,
                        preclip_norm=value['preclip_norm'], AMP_retries_epoch=retries))
                    print('RGBD TWO SEED UPDATE', args.seed, epoch, step, flush=True)
        assert not batches and seen == local_ids and step == epoch * 267
        all_seen = [None] * 4
        dist.all_gather_object(all_seen, seen, group=cpu)
        assert [i for block in range(267) for rank in range(4)
                for i in all_seen[rank][block * 2:block * 2 + 2]] == order
        metrics = evaluate(rt, 'val', folder / 'validation_current', cpu)
        improved = metrics['segm/mAP'] > best['score']
        if improved:
            best = dict(score=metrics['segm/mAP'], epoch=epoch)
        save(rt, folder, epoch, step, frozen, best, cpu, improved)
        if dist.get_rank() == 0:
            if improved:
                base.core.C.dump(folder / 'best_validation_metrics.json', dict(epoch=epoch, metrics=metrics))
            base.core.C.dump(folder / ('epoch%d.json' % epoch), dict(epoch=epoch, step=step, best=best,
                seed=args.seed, sample_order=order, rank_sample_order=all_seen, validation=metrics,
                world_size=4, mean_microbatch_loss=total / 2136, AMP_retries_epoch=retries,
                seconds=time.time() - began))
        assert hashes() == frozen['source_hashes']
    if dist.get_rank() == 0:
        base.core.C.dump(folder / 'training_complete.json', dict(complete=True, epoch=100, updates=26700,
            seed=args.seed, world_size=4, best=best, contract=frozen,
            best_checkpoint_sha256=base.core.C.file_sha(folder / 'best_model.pth'),
            last_checkpoint_sha256=base.core.C.file_sha(folder / 'last.pth')))
    dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'finalize', 'gate', 'train', 'eval'])
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--seed', type=int, choices=PROTOCOL['seeds'], default=44)
    p.add_argument('--resume', type=Path)
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.root)
        return
    if args.action == 'finalize':
        finalize(args.root)
        return
    configure(args.seed)
    rank, cpu = ddp.setup()
    ranks = [None] * 4 if rank == 0 else None
    dist.gather_object(dict(rank=rank, local_rank=int(os.environ['LOCAL_RANK']), pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19]), ranks, dst=0, group=cpu)
    if rank == 0:
        base.core.C.dump(seed_folder(args.root, args.seed) / (args.action + '_launch.json'),
            dict(world_size=4, ranks=ranks, observed=time.time()))
    rt = runtime(args)
    if args.action == 'gate':
        gate_args = argparse.Namespace(root=seed_folder(args.root, args.seed), variant='baseline', content='correct')
        # The old gate asks identity(gate_args); retain this run's actual seed/root.
        ddp.identity = lambda unused: identity(args)
        ddp.gate(rt, gate_args, cpu)
    elif args.action == 'train':
        train(rt, args, cpu)
    else:
        folder = seed_folder(args.root, args.seed)
        done = json.loads((folder / 'training_complete.json').read_text())
        assert done['complete'] and done['epoch'] == 100 and done['updates'] == 26700
        path = folder / 'best_model.pth'
        assert base.core.C.file_sha(path) == done['best_checkpoint_sha256']
        ddp.restore(rt, path, identity(args), training=False)
        assert not (folder / 'test/complete.json').exists()
        evaluate(rt, 'test', folder / 'test', cpu, done['best_checkpoint_sha256'])
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
