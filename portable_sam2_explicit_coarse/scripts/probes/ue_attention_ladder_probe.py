"""Independent normalized-attention L0..L6 single-factor probe.

All arms start from one seed45 initialization and reseed each epoch identically.
Existing probe code and artifacts are read-only dependencies.
"""
import argparse
import hashlib
import inspect
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
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from rsprompter.uav.s2db_geometry import prepare_point, VERSION
from rsprompter.uav.attention_ladder import AttentionLadderAdapter
from rsprompter.uav.local_adapter_probe import YAWS

DATA = Path('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
BASE = Path('/data/wangcheng/checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth')
FROZEN_SHA = 'c6921c30416595fb994facd3d4d927408be9bdd636399fd669cfbe5968329536'
SOURCES = ['scripts/probes/ue_attention_ladder_probe.py',
    'rsprompter/uav/attention_ladder.py', 'rsprompter/uav/s2db_geometry.py',
    'scripts/probes/ue_v1_adapter.py', 'rsprompter/uav/local_adapter_probe.py',
    'rsprompter/uav/local_adapter_variants.py',
    'data/ue_minimal_dataset.py', 'data/whu_instance_dataset.py']

class DecoderLowRank(nn.Module):
    """Additive rank4 branch; original linear weights and state names stay intact."""
    def __init__(self):
        super().__init__()
        self.A = nn.Linear(256, 4, bias=False)
        self.B = nn.Linear(4, 128, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=np.sqrt(5))
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.B(self.A(x))  # alpha/rank = 4/4 = 1, dropout = 0


def build_adapter(pretrained, direction):
    seed(45)
    adapter = AttentionLadderAdapter(pretrained, direction=direction)
    # Isolate decoder initialization from every shared source/adapter parameter.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4606)
        adapter.decoder_lora_q = DecoderLowRank()
        adapter.decoder_lora_v = DecoderLowRank()
    for branch in (adapter.decoder_lora_q, adapter.decoder_lora_v):
        branch.requires_grad_(direction == 'L6')
    adapter.observe = False
    return adapter


def forward_stats(adapter):
    return dict(adapter.stats)


def source_lattice_contract(adapter):
    patches = [(name,m) for name,m in adapter.encoder.named_modules()
               if 'patch_embed.proj' in name and 'lora_' not in name and isinstance(m, nn.Conv2d)]
    assert len(patches) == 1, [n for n,m in patches]
    name, patch = patches[0]
    assert patch.kernel_size == (7,7) and patch.stride == (4,4) and patch.padding == (3,3)
    necks = [(name,m) for name,m in adapter.encoder.named_modules() if hasattr(m, 'fpn_top_down_levels')]
    assert len(necks) == 1
    neckname, neck = necks[0]
    assert list(neck.fpn_top_down_levels) == [2,3] and 0 not in neck.fpn_top_down_levels
    assert all(m.kernel_size == (1,1) for m in neck.convs.modules() if isinstance(m, nn.Conv2d))
    patch_parent = [(n,m) for n,m in adapter.encoder.named_modules() if n.endswith('patch_embed')]
    assert len(patch_parent) == 1
    patch_source = Path(inspect.getfile(type(patch_parent[0][1]))).resolve()
    neck_source = Path(inspect.getfile(type(neck))).resolve()
    return dict(patch_module=name, patch_source=str(patch_source), patch_source_sha256=file_sha(patch_source),
        neck_source=str(neck_source),neck_source_sha256=file_sha(neck_source),
        kernel=list(patch.kernel_size),stride=list(patch.stride),
        padding=list(patch.padding),neck_module=neckname,top_down_levels=list(neck.fpn_top_down_levels),
        source_integer_center_formula='token_i centered at source integer pixel index 4*i',
        normalized_grid_formula='(source_uv+2)/256-1; align_corners=False; border padding',
        historical_contract='No old task or old cache changed')


def install_decoder_hooks(model, adapter, direction):
    if direction != 'L6': return
    attention = model.roi_head.mask_head.mask_decoder.decoder.transformer.layers[0].cross_attn_token_to_image
    for name, branch in [('q_proj', adapter.decoder_lora_q), ('v_proj', adapter.decoder_lora_v)]:
        projection = getattr(attention, name)
        assert isinstance(projection, nn.Linear)
        assert (projection.in_features, projection.out_features) == (256, 128)
        def hook(module, inputs, output, branch=branch):
            return output + branch(inputs[0]).to(output.dtype) if adapter.enabled else output
        adapter.handles.append(projection.register_forward_hook(hook))


def trainable(adapter):
    return [p for p in adapter.parameters() if p.requires_grad]


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_save(obj, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp); tmp.replace(path)


def payload(args, folder, mode, donor=None):
    assert mode in ('correct', 'constant', 'donor')
    path = prepare_point(folder, args.cache)
    with np.load(path) as a:
        assert str(a['version']) == VERSION
        index = a['index'].astype(np.int64)
        geom = a['geom'].astype(np.float32)
        anchor = a['anchor'].copy()
    assert anchor.dtype == np.float64, 'L1 requires FP64 projected metadata before coordinate mapping'
    rgbfolder = folder if donor is None else donor
    images = []
    for yaw in YAWS:
        image = cv2.imread(str(rgbfolder/f'uav/yaw_{yaw:03d}/rgb.png'))
        assert image is not None and image.shape == (512,512,3)
        images.append(image[:,:,::-1].transpose(2,0,1).copy())
    rgb = np.stack(images).astype(np.float32)
    if mode == 'constant':
        rgb[:] = np.array([123.675,116.28,103.53],np.float32)[None,:,None,None]
    return tuple(torch.from_numpy(v).cuda() for v in (rgb,index,geom,anchor))


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
    adapter = build_adapter(snapshot['runtime_config']['sam2_checkpoint'], args.direction).cuda()
    init = torch.load(args.init, map_location='cpu', weights_only=False)
    adapter.load_state_dict(init['adapter'], strict=True)
    assert tensor_digest(adapter.state_dict()) == init['fingerprint']
    adapter.attach(model)
    install_decoder_hooks(model, adapter, args.direction)
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
    return dict(study='normalized_attention_ladder_v1',
        direction=args.direction, content=args.content,
        base_sha256=file_sha(args.base), satellite_state_sha256=FROZEN_SHA,
        archive_hashes=archive_identity(args.data_root), geometry_version=VERSION,
        source_hashes={s: file_sha(ROOT/s) for s in SOURCES},
        initial_fingerprint=tensor_digest(adapter.state_dict()), init_file_sha256=file_sha(args.init),
        base_config_snapshot=snapshot, optimizer='AdamW', weight_decay=.01, clip=.5,
        initialization_seed=45, data_seed=44, epoch_seed_formula='1044000+epoch',
        epoch_reseed_from=1, epochs=10, batch=1, accum=8, updates=2670,
        lr=1e-4, warmup=50, cosine_min_factor=.1, amp=True, amp_init_scale=512,
        ema=False, selection='E10 last', satellite_frozen=args.direction != 'L6', original_satellite_parameters_frozen=True,
        decoder_lora=dict(active=args.direction == 'L6', rank=4, alpha=4, dropout=0,
            paths=['transformer.layers.0.cross_attn_token_to_image.q_proj',
                   'transformer.layers.0.cross_attn_token_to_image.v_proj']),
        donor_policy='cyclic next RGB only, target depth/poses/candidates unchanged',
        eval_precision='fp32', max_dets=[1, 10, 250], test_primary=True,
        historical_comparison='Old probes use a different epoch RNG protocol; not single-factor controls',
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})


def train(args, model, adapter, snapshot):
    ds = MinimalSatelliteDataset(args.data_root, 'train', augment=True)
    assert len(ds) == 2142 and not (args.out/'last.pth').exists()
    c = contract(args, snapshot, adapter)
    initial = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    optimizer = torch.optim.AdamW(trainable(adapter), lr=1e-4, weight_decay=.01)
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
                data = payload(args, folder, args.content)
                with torch.autocast('cuda', dtype=torch.float16):
                    features = bind(model, adapter, sample, data)
                    with cached_extract(model, features):
                        losses = model.loss(sample['inputs'], sample['data_samples'])
                    loss = sum((v for v in losses.values() if torch.is_tensor(v) and v.requires_grad), adapter.alpha*0)
                assert torch.isfinite(loss)
                scaler.scale(loss/8).backward()
                stats = forward_stats(adapter); adapter.context = None
                total_loss += float(loss.detach())
                if (i+1) % 8 == 0:
                    step += 1
                    lr = 1e-4*min(1., step/50)*(.1+.9*.5*(1+np.cos(np.pi*step/2670)))
                    for group in optimizer.param_groups: group['lr'] = lr
                    scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(trainable(adapter), .5)
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
        for key, value in (('direction', args.direction), ('content', args.content),
                           ('base_sha256', file_sha(args.base))):
            assert cp['contract'][key] == value, key
        assert cp['epoch'] == 10 and cp['step'] == 2670
        adapter.load_state_dict(cp['adapter'], strict=True); del cp
    else:
        assert args.mode == 'disabled', 'An enabled evaluation requires an E10 adapter checkpoint'
    seed(44); model.eval(); adapter.eval(); adapter.observe = True; adapter.enabled = args.mode != 'disabled'
    ds = MinimalSatelliteDataset(args.data_root, args.eval_split)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=rtmdet_collate_fn)
    all_gt, all_dt, metas, records = [], [], [], []
    out = args.out/args.mode
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if args.max_images and i >= args.max_images: break
            sample = processed(model, batch)
            if adapter.enabled:
                donor = point_folder(ds,(i+1)%len(ds),args.data_root) if args.mode == 'donor' else None
                features = bind(model, adapter, sample, payload(args, point_folder(ds, i, args.data_root), args.mode, donor))
            else:
                features = model.extract_feat(sample['inputs']); adapter.context = None
            with cached_extract(model, features):
                result = model.predict(sample['inputs'], sample['data_samples'], rescale=True)[0].pred_instances
            target, pred = gt_arrays(sample['data_samples'][0].gt_instances), gt_arrays(result)
            all_gt.append(target); all_dt.append(pred); metas.append(batch['img_metas'][0])
            records.append(dict(image_id=ds.samples[i]['image_id'], gt=serialized(target), dt=serialized(pred),
                                stats=forward_stats(adapter) if adapter.enabled else {},
                                read_observation=dict(adapter.observation) if adapter.enabled else {}))
            adapter.context = None
            if (i+1) % 25 == 0: print('EVAL', args.mode, i+1, flush=True)
    cg, cd = build_coco_gt_and_dt(all_gt, all_dt, metas, score_key='scores')
    metrics = {}
    for kind in ('segm', 'bbox'): metrics.update(run_coco_eval(cg, cd, kind, max_dets=[1,10,250]))
    dump(out/'metrics.json', metrics); dump(out/'records.json', records)
    dump(out/'evaluation_identity.json', dict(split=args.eval_split, images=len(records), mode=args.mode,
        training_content=args.content, direction=args.direction,
        base_sha256=file_sha(args.base), adapter_sha256=file_sha(args.adapter) if args.adapter else None,
        init_sha256=file_sha(args.init), donor_preserves_geometry=True,
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
    data = payload(args, point_folder(ds, i, args.data_root), args.content)
    lattice = source_lattice_contract(adapter)
    adapter.observe = True
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
        empty = (data[0], torch.full_like(data[1], -1), torch.zeros_like(data[2]), torch.zeros_like(data[3]))
        equal(baseline, prediction(True, empty, amp))
        reports['amp' if amp else 'fp32'] = dict(zero_alpha_logits_exact=True,
            no_support_nonzero_alpha_logits_exact=True,
            l6_identity_scope='initial zero-B decoder LoRA only; trained L6 can change satellite decoding' if args.direction == 'L6' else None)
    handle.remove(); adapter.context = None
    del baseline, captured
    torch.cuda.empty_cache()
    # Gate training starts from the common initialization, never from gate mutations.
    adapter.load_state_dict(torch.load(args.init, map_location='cpu', weights_only=False)['adapter'], strict=True)
    adapter.enabled = True; adapter.train(); seed(44)
    opt = torch.optim.AdamW(trainable(adapter), lr=1e-4, weight_decay=.01)
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
        norm = torch.nn.utils.clip_grad_norm_(trainable(adapter), .5); assert torch.isfinite(norm)
        scaler.step(opt); scaler.update(); adapter.context = None
        rows.append(dict(step=step, loss=float(loss.detach()), grads=grads, stats=forward_stats(adapter), read_observation=dict(adapter.observation)))
    assert rows[0]['grads']['alpha'] > 0
    assert any(v > 0 for k,v in rows[-1]['grads'].items() if k.startswith('source.'))
    assert any(v > 0 for k,v in rows[-1]['grads'].items() if 'lora_' in k and k.startswith('encoder.'))
    if args.direction == 'L3': assert rows[-1]['grads']['null_logit'] > 0
    if args.direction == 'L6':
        for prefix in ('decoder_lora_q.', 'decoder_lora_v.'):
            assert any(v > 0 for k,v in rows[-1]['grads'].items() if k.startswith(prefix))
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
    dump(args.out/'gpu_gate.json', dict(identity=reports, source_lattice=lattice, satellite_state_sha256=before,
        satellite_parameters_buffers_exact=True, satellite_no_grad=True, gradient_steps=rows,
        training_amp_peak=train_peak, fp32_250roi_peak=resource, resource_rois=250,
        memory_before_train=memory_before_train, memory_before_eval=memory_before_eval,
        memory_after_eval=memory_snapshot(), gate_train_gt_instances=len(ds.samples[i]['annotations']),
        resource_chunk=32, memory_scope='per-process PyTorch allocator; not whole GPU capacity',
        image_id=ds.samples[i]['image_id'], source_hashes={s:file_sha(ROOT/s) for s in SOURCES}))
    print('GATE COMPLETE', args.direction, args.content, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['init','gate','train','eval'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=DATA); p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--cache', type=Path, default=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    p.add_argument('--direction', choices=['L0','L1','L2','L3','L4','L5','L6'], default='L0')
    p.add_argument('--content', choices=['correct','constant'], default='correct')
    p.add_argument('--init', type=Path)
    p.add_argument('--adapter', type=Path); p.add_argument('--resume-from', type=Path)
    p.add_argument('--eval-split', choices=['val','test'], default='test')
    p.add_argument('--mode', choices=['correct','constant','donor','disabled'])
    p.add_argument('--max-images', type=int, default=0)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.action == 'init':
        cp = torch.load(args.base,map_location='cpu',weights_only=False)
        adapter = build_adapter(cp['config_snapshot']['runtime_config']['sam2_checkpoint'], 'L0')
        del cp
        state = adapter.state_dict()
        path = args.out/'common_init.pth'
        assert not path.exists(), 'Refuse replacing existing common initialization'
        atomic_save(dict(adapter=state, fingerprint=tensor_digest(state), seed=45), path)
        dump(args.out/'common_init.json', dict(path=str(path), sha256=file_sha(path),
             fingerprint=tensor_digest(state), initialization_seed=45))
        print('INIT', path, tensor_digest(state), flush=True); return
    assert args.init
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
