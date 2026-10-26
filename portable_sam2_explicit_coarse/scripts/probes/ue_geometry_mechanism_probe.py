"""Independent geometry sampling x fusion x content factorial probe.

All arms start from one seed45 initialization and reseed each epoch identically.
Existing probe code and artifacts are read-only dependencies.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import ue_v1_adapter as common
from ue_v1_adapter import (ROOT, seed, dump, file_sha, MinimalSatelliteDataset, archive_identity,
    point_folder, processed, cached_extract, state_digest, gt_arrays, serialized,
    rtmdet_collate_fn, _register_and_build, _load_model_state, build_coco_gt_and_dt, run_coco_eval)
import numpy as np
import torch
from torch.utils.data import DataLoader
from rsprompter.uav.native_geometry_probe import prepare_native, VERSION
from rsprompter.uav.native_geometry_mechanism import GeometryMechanismAdapter

DATA = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
BASE = Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth')
FROZEN_SHA = 'c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'
SOURCES = ['scripts/probes/ue_geometry_mechanism_probe.py',
    'rsprompter/uav/native_geometry_mechanism.py', 'rsprompter/uav/native_geometry_probe.py',
    'scripts/probes/ue_v1_adapter.py', 'rsprompter/uav/local_adapter_probe.py',
    'data/ue_minimal_dataset.py', 'data/whu_instance_dataset.py']


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(obj, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp); tmp.replace(path)


@lru_cache(maxsize=4)
def load_constants(path):
    obj = json.loads(Path(path).read_text())
    assert obj['split'] == 'train', 'Constants must be computed on train only'
    values = np.asarray(obj['mean_fields'], dtype=np.float32)
    assert values.shape == (4,) and np.isfinite(values).all()
    return values


def constants(args):
    return load_constants(str(args.constants.resolve()))


def payload(args, folder, mode):
    # Explicit evaluation mode is independent of the arm's training content.
    assert mode in ('correct', 'constant', 'shuffle', 'zero')
    with np.load(prepare_native(folder, args.cache)) as a:
        support = a['support'].astype(bool)
        fields = a['fields'].astype(np.float32)
    if mode == 'constant':
        fields[:] = constants(args)[:, None, None] * support[None]
    elif mode == 'zero':
        fields.fill(0)
    elif mode == 'shuffle':
        vectors = fields[:, support].copy()
        perm = np.random.default_rng(20261022 + int(folder.name)).permutation(vectors.shape[1])
        fields[:, support] = vectors[:, perm]
    assert np.count_nonzero(fields[:, ~support]) == 0
    return (None, torch.from_numpy(support.astype(np.float32))[None, None].cuda(),
            torch.from_numpy(fields)[None].cuda())


def make_model(args):
    cp = torch.load(args.base, map_location='cpu', weights_only=False)
    snapshot = cp['config_snapshot']
    assert archive_identity(args.data_root) == snapshot['data_config']['archive_hashes']
    model = _register_and_build(snapshot['model_config'])
    report = _load_model_state(model, cp['model'], allow_nonstrict=False)
    assert not report['missing_keys'] and not report['unexpected_keys']
    del cp
    for p in model.parameters(): p.requires_grad_(False)
    model.eval().cuda()
    assert state_digest(model) == FROZEN_SHA
    seed(45)
    adapter = GeometryMechanismAdapter(sampling=args.sampling, fusion=args.fusion).cuda()
    init = torch.load(args.init, map_location='cpu', weights_only=False)
    adapter.load_state_dict(init['adapter'], strict=True)
    assert tensor_digest(adapter.state_dict()) == init['fingerprint']
    adapter.attach(model)
    seed(44)
    return model, adapter, snapshot


def bind(model, adapter, sample, data):
    with torch.no_grad(): features = model.extract_feat(sample['inputs'])
    adapter.encode(*data, features[3][0])
    return features


def rng_state(gen):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(), loader_generator=gen.get_state())


def restore_rng(rng, gen):
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
    torch.cuda.set_rng_state_all(rng['cuda']); gen.set_state(rng['loader_generator'])


def contract(args, snapshot, adapter):
    return dict(study='native_geometry_mechanism_factorial_v1',
        sampling=args.sampling, fusion=args.fusion, content=args.content,
        base_sha256=file_sha(args.base), satellite_state_sha256=FROZEN_SHA,
        archive_hashes=archive_identity(args.data_root), geometry_version=VERSION,
        source_hashes={s: file_sha(ROOT/s) for s in SOURCES},
        initial_fingerprint=tensor_digest(adapter.state_dict()), init_file_sha256=file_sha(args.init),
        constants_sha256=file_sha(args.constants), constant_fields=constants(args).tolist(),
        base_config_snapshot=snapshot, optimizer='AdamW', weight_decay=.01, clip=.5,
        initialization_seed=45, data_seed=44, epoch_seed_formula='1044000+epoch',
        epoch_reseed_from=1, epochs=10, batch=1, accum=8, updates=2670,
        lr=1e-4, warmup=50, cosine_min_factor=.1, amp=True, amp_init_scale=512,
        ema=False, selection='E10 last', satellite_frozen=True,
        eval_precision='fp32', max_dets=[1, 10, 250], test_primary=True,
        historical_comparison='Old probes use a different epoch RNG protocol; not single-factor controls',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})


def train(args, model, adapter, snapshot):
    ds = MinimalSatelliteDataset(args.data_root, 'train', augment=True)
    assert len(ds) == 2142 and not (args.out/'last.pth').exists()
    c = contract(args, snapshot, adapter)
    initial = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    gen = torch.Generator().manual_seed(44)
    start_epoch = step = 0
    history = args.out/'train.jsonl'
    if args.resume_from:
        parent = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        # Compare the complete mathematical/data/source contract, excluding only runtime paths.
        for key in c:
            if key != 'args': assert parent['contract'][key] == c[key], key
        adapter.load_state_dict(parent['adapter'], strict=True)
        optimizer.load_state_dict(parent['optimizer']); scaler.load_state_dict(parent['scaler'])
        start_epoch, step = int(parent['epoch']), int(parent['step'])
        assert step == start_epoch*267
        restore_rng(parent['rng_state'], gen)
        rows = [json.loads(s) for s in history.read_text().splitlines()]
        assert len(rows) == step and [r['update'] for r in rows] == list(range(1, step+1))
        dump(args.out/'resume_identity.json', dict(parent=str(args.resume_from),
             sha256=file_sha(args.resume_from), epoch=start_epoch, update=step, all_rng_restored=True))
        del parent
    else:
        assert not history.exists(), 'Refuse overwriting training history'
    dump(args.out/'frozen_training_contract.json', c)
    for path in SOURCES:
        dest = args.out/'source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/path).read_bytes())
    model.eval(); adapter.train()
    dump(args.out/'progress.json', dict(stage='train', epoch=start_epoch, update=step, pid=os.getpid()))
    with history.open('a' if args.resume_from else 'w') as log:
        for epoch in range(start_epoch+1, 11):
            epoch_seed = 1044000+epoch
            seed(epoch_seed); gen.manual_seed(epoch_seed)
            loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                                collate_fn=rtmdet_collate_fn, generator=gen)
            optimizer.zero_grad(set_to_none=True)
            started = time.time(); ids = []; total_loss = 0.
            for i, batch in enumerate(loader):
                if i >= 2136: break
                folder = Path(batch['img_metas'][0]['filename']).parent.parent
                ids.append(folder.name)
                sample = processed(model, batch, True)
                data = payload(args, folder, 'correct' if args.content == 'geometry' else 'constant')
                with torch.autocast('cuda', dtype=torch.float16):
                    features = bind(model, adapter, sample, data)
                    with cached_extract(model, features):
                        losses = model.loss(sample['inputs'], sample['data_samples'])
                    loss = sum((v for v in losses.values() if torch.is_tensor(v) and v.requires_grad), adapter.alpha*0)
                assert torch.isfinite(loss)
                scaler.scale(loss/8).backward()
                stats = dict(adapter.stats); adapter.context = None
                total_loss += float(loss.detach())
                if (i+1) % 8 == 0:
                    step += 1
                    lr = 1e-4*min(1., step/50)*(.1+.9*.5*(1+np.cos(np.pi*step/2670)))
                    for group in optimizer.param_groups: group['lr'] = lr
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), .5)
                    assert torch.isfinite(norm)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    row = dict(epoch=epoch, update=step, images=i+1, loss=float(loss.detach()), lr=lr,
                        grad_norm=float(norm), amp_scale=scaler.get_scale(), latest_forward=stats, epoch_seed=epoch_seed)
                    log.write(json.dumps(row)+'\n'); log.flush()
                    if step % 10 == 0: print('TRAIN', json.dumps(row), flush=True)
            assert len(ids) == 2136 and step == epoch*267
            assert all(p.grad is None for p in model.parameters()) and state_digest(model) == FROZEN_SHA
            dump(args.out/f'epoch{epoch}_summary.json', dict(epoch=epoch, update=step,
                seconds=time.time()-started, microbatches=len(ids), loss_microbatch_mean=total_loss/len(ids),
                sample_order=ids, sample_order_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                satellite_state_sha256=FROZEN_SHA))
            cp = dict(adapter=adapter.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                      epoch=epoch, step=step, contract=c, rng_state=rng_state(gen))
            atomic_save(cp, args.out/f'epoch{epoch}.pth')
            dump(args.out/'progress.json', dict(stage='train', epoch=epoch, update=step, pid=os.getpid()))
            print('EPOCH', epoch, time.time()-started, flush=True)
    assert step == 2670
    dump(args.out/'parameter_changes.json', dict(satellite_unchanged=True, satellite_state_sha256=FROZEN_SHA,
        adapter_changes={k: float((v.detach().cpu()-initial[k]).float().norm()) for k, v in adapter.state_dict().items()}))
    # Last includes optimizer/scaler/RNG too, so it is also a complete recovery artifact.
    atomic_save(cp, args.out/'last.pth')


def evaluate(args, model, adapter):
    assert args.mode is not None
    if args.adapter:
        cp = torch.load(args.adapter, map_location='cpu', weights_only=False)
        for key, value in (('sampling', args.sampling), ('fusion', args.fusion), ('content', args.content),
                           ('base_sha256', file_sha(args.base)), ('constants_sha256', file_sha(args.constants))):
            assert cp['contract'][key] == value, key
        assert cp['epoch'] == 10 and cp['step'] == 2670
        adapter.load_state_dict(cp['adapter'], strict=True); del cp
    else:
        assert args.mode == 'disabled', 'An enabled evaluation requires an E10 adapter checkpoint'
    seed(44); model.eval(); adapter.eval(); adapter.enabled = args.mode != 'disabled'
    ds = MinimalSatelliteDataset(args.data_root, args.eval_split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=rtmdet_collate_fn)
    all_gt, all_dt, metas, records = [], [], [], []
    out = args.out/args.mode
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if args.max_images and i >= args.max_images: break
            sample = processed(model, batch)
            if adapter.enabled:
                features = bind(model, adapter, sample, payload(args, point_folder(ds, i, args.data_root), args.mode))
            else:
                features = model.extract_feat(sample['inputs']); adapter.context = None
            with cached_extract(model, features):
                result = model.predict(sample['inputs'], sample['data_samples'], rescale=True)[0].pred_instances
            target, pred = gt_arrays(sample['data_samples'][0].gt_instances), gt_arrays(result)
            all_gt.append(target); all_dt.append(pred); metas.append(batch['img_metas'][0])
            records.append(dict(image_id=ds.samples[i]['image_id'], gt=serialized(target), dt=serialized(pred),
                                stats=dict(adapter.stats) if adapter.enabled else {}))
            adapter.context = None
            if (i+1) % 25 == 0: print('EVAL', args.mode, i+1, flush=True)
    cg, cd = build_coco_gt_and_dt(all_gt, all_dt, metas, score_key='scores')
    metrics = {}
    for kind in ('segm', 'bbox'): metrics.update(run_coco_eval(cg, cd, kind, max_dets=[1,10,250]))
    dump(out/'metrics.json', metrics); dump(out/'records.json', records)
    dump(out/'evaluation_identity.json', dict(split=args.eval_split, images=len(records), mode=args.mode,
        training_content=args.content, sampling=args.sampling, fusion=args.fusion,
        base_sha256=file_sha(args.base), adapter_sha256=file_sha(args.adapter) if args.adapter else None,
        constants_sha256=file_sha(args.constants), init_sha256=file_sha(args.init),
        source_hashes={s: file_sha(ROOT/s) for s in SOURCES},
        score_key='scores', max_dets=[1,10,250], inference_precision='fp32'))
    print('METRICS', args.mode, json.dumps(metrics), flush=True)


def memory_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free,memory.total',
            '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
        cards = result.stdout.strip().splitlines()
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'], text=True, capture_output=True, check=True)
        processes = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        cards = []; processes = []; error_message = str(error)
    else:
        error_message = None
    return dict(pid=os.getpid(), allocated_MiB=torch.cuda.memory_allocated()/2**20,
        reserved_MiB=torch.cuda.memory_reserved()/2**20, nvidia_gpu_rows=cards,
        nvidia_process_rows=processes, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        nvidia_error=error_message)


def gate(args, model, adapter):
    ds = MinimalSatelliteDataset(args.data_root, 'train')
    i = int(np.argmax([len(s['annotations']) for s in ds.samples]))
    sample = processed(model, rtmdet_collate_fn([ds[i]]), True)
    data = payload(args, point_folder(ds, i, args.data_root), 'correct' if args.content == 'geometry' else 'constant')
    before = state_digest(model); adapter.eval(); captured = []
    handle = model.roi_head.mask_head.register_forward_hook(lambda m, a, o: captured.append(o.mask_logits.detach().clone()))
    reports = {}
    def prediction(active, value, amp):
        captured.clear(); adapter.enabled = active
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            if active: features = bind(model, adapter, sample, value)
            else: features = model.extract_feat(sample['inputs']); adapter.context = None
            with cached_extract(model, features):
                pred = model.predict(sample['inputs'], sample['data_samples'], rescale=True)[0].pred_instances
        return pred, list(captured)
    def equal(a, b):
        assert len(a[1]) == len(b[1]) and len(a[1]) > 0
        assert all(torch.equal(x,y) for x,y in zip(a[1],b[1])), 'raw logits changed'
        for name in ('masks','bboxes','scores'): assert torch.equal(getattr(a[0],name), getattr(b[0],name)), name
    for amp in (False, True):
        adapter.alpha.data.zero_()
        baseline = prediction(False, data, amp)
        equal(baseline, prediction(True, data, amp))
        adapter.alpha.data.fill_(.01)
        empty = (None, torch.zeros_like(data[1]), torch.zeros_like(data[2]))
        equal(baseline, prediction(True, empty, amp))
        if args.fusion == 'difference':
            zero = (None, data[1], torch.zeros_like(data[2]))
            equal(baseline, prediction(True, zero, amp))
        reports['amp' if amp else 'fp32'] = dict(zero_alpha_logits_exact=True,
            no_support_nonzero_alpha_logits_exact=True, zero_fields_difference_identity=args.fusion == 'difference')
    handle.remove(); adapter.context = None
    del baseline, captured
    torch.cuda.empty_cache()
    # Gate training starts from the common initialization, never from gate mutations.
    adapter.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=False)['adapter'], strict=True)
    adapter.enabled = True; adapter.train(); seed(44)
    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    # Actual loss/backward warmup, without optimizer update, then discard grads.
    with torch.autocast('cuda', dtype=torch.float16):
        features = bind(model, adapter, sample, data)
        with cached_extract(model, features): warm_losses = model.loss(sample['inputs'], sample['data_samples'])
        warm_loss = sum((v for v in warm_losses.values() if torch.is_tensor(v) and v.requires_grad), adapter.alpha*0)
    scaler.scale(warm_loss).backward()
    opt.zero_grad(set_to_none=True); adapter.context = None
    del warm_loss, warm_losses, features
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); rows = []
    memory_before_train = memory_snapshot()
    for step in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            features = bind(model, adapter, sample, data)
            with cached_extract(model, features): losses = model.loss(sample['inputs'], sample['data_samples'])
            loss = sum((v for v in losses.values() if torch.is_tensor(v) and v.requires_grad), adapter.alpha*0)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        grads = {k:float(p.grad.float().norm()) for k,p in adapter.named_parameters() if p.grad is not None}
        assert np.isfinite(list(grads.values())).all()
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), .5); assert torch.isfinite(norm)
        scaler.step(opt); scaler.update(); adapter.context = None
        rows.append(dict(step=step, loss=float(loss.detach()), grads=grads, stats=dict(adapter.stats)))
    assert rows[0]['grads']['alpha'] > 0 and rows[-1]['grads']['mixer.0.weight'] > 0
    train_peak = dict(allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                      reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    assert all(p.grad is None for p in model.parameters()) and state_digest(model) == before
    adapter.eval(); opt.zero_grad(set_to_none=True); adapter.context = None
    # Prewarm the exact production mask path before measuring the full 250 RoIs.
    with torch.inference_mode():
        features = bind(model, adapter, sample, data)
        x, embedding, pe, high = features
        b = torch.tensor([[0.,0.,0.,512.,512.]], device='cuda')
        warm = model.roi_head._mask_forward(x,b,image_embeddings=embedding,
            image_positional_embeddings=pe,high_res_features=high)['mask_preds']
        del warm
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    memory_before_eval = memory_snapshot()
    with torch.inference_mode():
        features = bind(model, adapter, sample, data)
        x, embedding, pe, high = features
        boxes = torch.tensor([a['bbox'] for a in ds.samples[i]['annotations']], device='cuda', dtype=torch.float32)
        boxes[:, 2:] += boxes[:, :2]
        boxes = boxes.repeat((250+len(boxes)-1)//len(boxes),1)[:250]
        rois = torch.cat((torch.zeros(250,1,device='cuda'), boxes),1)
        count = 0
        for chunk in rois.split(32):
            masks = model.roi_head._mask_forward(x,chunk,image_embeddings=embedding,
                image_positional_embeddings=pe,high_res_features=high)['mask_preds']
            assert torch.isfinite(masks).all(); count += len(masks)
        assert count == 250
    torch.cuda.synchronize()
    resource = dict(allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
                    reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
    assert state_digest(model) == before
    dump(args.out/'gpu_gate.json', dict(identity=reports, satellite_state_sha256=before,
        satellite_parameters_buffers_exact=True, satellite_no_grad=True, gradient_steps=rows,
        training_amp_peak=train_peak, fp32_250roi_peak=resource, resource_rois=250,
        memory_before_train=memory_before_train, memory_before_eval=memory_before_eval,
        memory_after_eval=memory_snapshot(), gate_train_gt_instances=len(ds.samples[i]['annotations']),
        resource_chunk=32, memory_scope='per-process PyTorch allocator; not whole GPU capacity',
        image_id=ds.samples[i]['image_id'], source_hashes={s:file_sha(ROOT/s) for s in SOURCES}))
    print('GATE COMPLETE', args.sampling, args.fusion, args.content, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init','gate','train','eval'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=DATA); p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--cache', type=Path, default=Path('/data/wangcheng/cache/ue_v1_native_geometry'))
    p.add_argument('--sampling', choices=['bilinear','nearest'], default='bilinear')
    p.add_argument('--fusion', choices=['concat','difference'], default='concat')
    p.add_argument('--content', choices=['geometry','constant'], default='geometry')
    p.add_argument('--constants', type=Path); p.add_argument('--init', type=Path)
    p.add_argument('--adapter', type=Path); p.add_argument('--resume-from', type=Path)
    p.add_argument('--eval-split', choices=['val','test'], default='test')
    p.add_argument('--mode', choices=['correct','constant','shuffle','zero','disabled'])
    p.add_argument('--max-images', type=int, default=0)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.action == 'init':
        seed(45)
        adapter = GeometryMechanismAdapter(sampling='bilinear', fusion='concat')
        state = adapter.state_dict()
        path = args.out/'common_init.pth'
        assert not path.exists(), 'Refuse replacing existing common initialization'
        atomic_save(dict(adapter=state, fingerprint=tensor_digest(state), seed=45), path)
        dump(args.out/'common_init.json', dict(path=str(path), sha256=file_sha(path),
             fingerprint=tensor_digest(state), initialization_seed=45))
        print('INIT', path, tensor_digest(state), flush=True); return
    assert args.init and args.constants
    constants(args)
    if args.action != 'eval': assert args.mode is None and not args.max_images
    if args.action == 'eval': assert args.mode is not None and (args.adapter or args.mode == 'disabled')
    seed(44); model, adapter, snapshot = make_model(args)
    if args.action == 'train': train(args, model, adapter, snapshot)
    elif args.action == 'eval': evaluate(args, model, adapter)
    else:
        dump(args.out/'gate_contract.json', contract(args, snapshot, adapter))
        for path in SOURCES:
            dest = args.out/'source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((ROOT/path).read_bytes())
        gate(args, model, adapter)


if __name__ == '__main__': main()
