"""Independent instance-supervised Hiera interaction: init/gate/train/eval.

E73 is a frozen diagnostic base, not the initialization of a final joint model.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.utils.data import DataLoader
import ue_joint_encoding_probe as previous
import ue_geometry_proposal_pilot_20261024 as pilot
import ue_geometry_conditional_configured_20261024 as consumer
from rsprompter.uav.instance_interaction_probe import InstanceInteraction

C = previous.common
ROOT = C.ROOT
OUT = Path('/data/wangcheng/checkpoint/ue_instance_interaction_20261024')
SOURCES = list(dict.fromkeys(previous.SOURCES + [
    'rsprompter/uav/instance_interaction_probe.py',
    'scripts/probes/ue_instance_interaction_20261024.py',
    'scripts/probes/ue_geometry_proposal_pilot_20261024.py',
    'scripts/probes/ue_geometry_conditional_configured_20261024.py',
    'scripts/analysis/ue_geometry_conditional_complement_20261024.py',
    'scripts/probes/ue_detection_attrition_probe_20261023.py']))


def init_path(args):
    suffix = '_rgb' if args.family == 'rgb' else ''
    return args.root / f'init_seed{args.seed}{suffix}.pth'


def new_adapter(args, snapshot):
    adapter = InstanceInteraction()
    if args.family == 'rgb':
        adapter.add_rgb(snapshot['runtime_config']['sam2_checkpoint'], args.seed + 2001)
    return adapter


def bind_rgb(args, adapter, dataset, index, input_mode=None):
    if args.family != 'rgb':
        return
    folder = C.point_folder(dataset, index, previous.DATA)
    mode = args.rgb
    donor = None
    if input_mode == 'constant_rgb':
        mode = 'constant'
    elif input_mode == 'donor_rgb':
        mode = 'donor'
        donor = C.point_folder(dataset, (index + len(dataset) // 2) % len(dataset), previous.DATA)
    p = argparse.Namespace(cache=Path('/data/wangcheng/cache/ue_v1_s2db_geometry_v2'))
    rgb, indices, _, anchor = previous.payload(p, folder, mode, donor)
    adapter.rgb_context = (rgb, indices, anchor)


def source_hashes():
    return {name: C.file_sha(ROOT / name) for name in SOURCES}


def load_geometry(ident, device='cuda'):
    with np.load(pilot.CACHE / f'{ident:06d}.npz') as z:
        x = np.concatenate((z['fields'].astype(np.float32), z['support'][None].astype(np.float32)))
    assert x.shape == (5, 512, 512) and np.isfinite(x).all()
    return torch.from_numpy(x)[None].to(device)


def means(device='cuda'):
    values = json.loads(pilot.CONSTANTS.read_text())['mean_fields']
    return torch.tensor(values, device=device)[None, :, None, None]


def geometry_content(x, mode, ident, ids, constants):
    if mode == 'geometry':
        return x
    if mode == 'constant':
        return pilot.content(x, 'constant', constants)
    if mode == 'donor':
        index = ids.index(ident)
        return load_geometry(ids[(index + len(ids) // 2) % len(ids)], x.device)
    raise ValueError(mode)


def build(args):
    checkpoint = torch.load(previous.BASE, map_location='cpu', weights_only=False)
    snapshot = checkpoint['config_snapshot']
    assert C.archive_identity(previous.DATA) == snapshot['data_config']['archive_hashes']
    model = C._register_and_build(snapshot['model_config'])
    report = C._load_model_state(model, checkpoint['model'], allow_nonstrict=False)
    assert not report['missing_keys'] and not report['unexpected_keys']
    del checkpoint
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval().cuda()
    assert C.state_digest(model) == previous.FROZEN_SHA
    model.rpn_head.test_cfg.nms_pre = 2000
    adapter = new_adapter(args, snapshot).cuda()
    initial = torch.load(init_path(args), map_location='cpu', weights_only=False)
    adapter.load_state_dict(initial['adapter'], strict=True)
    assert previous.tensor_digest(adapter.state_dict()) == initial['fingerprint']
    adapter.internal = args.route == 'internal'
    adapter.attach(model)
    C.seed(args.seed)
    return model, adapter, snapshot, initial['fingerprint']


def forward(model, adapter, sample, x):
    adapter.encode(x)
    return model.extract_feat(sample['inputs'])


def losses(model, adapter, sample, x, target):
    features = forward(model, adapter, sample, x)
    with C.cached_extract(model, features):
        original = model.loss(sample['inputs'], sample['data_samples'])
    center, parts = pilot.loss_fn(adapter.outputs, target)
    primary = previous.objective(adapter, original)
    return primary + center, dict(original=float(primary.detach()), center=float(center.detach()), **parts)


def contract(args, initial):
    return dict(study='geometry_instance_internal_v1', route=args.route, content=args.content,
        family=args.family, rgb_content=args.rgb,
        seed=args.seed, initialization_seed=args.seed + 1, initial_fingerprint=initial,
        source_hashes=source_hashes(), base_sha256=C.file_sha(previous.BASE),
        frozen_state_sha256=previous.FROZEN_SHA, archive_hashes=C.archive_identity(previous.DATA),
        native_cache=str(pilot.CACHE), constants_sha256=C.file_sha(pilot.CONSTANTS),
        epochs=10, updates=2670, batch=1, accumulation=8, used_images_per_epoch=2136,
        optimizer='AdamW', lr=1e-4, weight_decay=.01, warmup_updates=50, cosine_min=.1,
        gradient_clip=.5, AMP=True, AMP_init_scale=512, EMA=False,
        epoch_seed_formula='1000000+seed*1000+epoch', selection='E10 last',
        center_loss_weight=1., center_loss='existing pilot Gaussian focal + offset smoothL1 + .1 logwh smoothL1',
        original_losses='all existing RPN/RoI/coarse/mask/UDPR losses; no decoder changes',
        support='real support retained for constant fields; no support = satellite fallback',
        route_difference='head-only uses S-prime only in center head; internal additionally passes it to block5 onward',
        precision_evaluation='FP32', max_dets=250, inference_RPN_pre=2000,
        test_role='development-exposed; never selects architecture/epoch/threshold',
        rgb=('five-view SAM2 foundation + LoRA, retained L1 continuous anchors, masked mean of max8 candidates'
             if args.family == 'rgb' else 'not used in geometry factorial'),
        rgb_source_training='only LoRA trainable; original source encoder eval; LoRA dropout .05 during training')


def write_sources(out, hashes):
    for name, expected in hashes.items():
        path = ROOT / name
        assert C.file_sha(path) == expected
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())


def train(args, model, adapter, initial):
    out = args.out
    frozen = contract(args, initial)
    write_sources(out, frozen['source_hashes'])
    C.dump(out / 'contract.json', frozen)
    dataset = C.MinimalSatelliteDataset(previous.DATA, 'train', augment=True)
    targets = pilot.Data('train')
    ids = [row['image_id'] for row in dataset.samples]
    assert ids == targets.ids and len(ids) == 2142
    lookup = {ident: index for index, ident in enumerate(ids)}
    constants = means()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    generator = torch.Generator().manual_seed(args.seed)
    start = step = 0
    if args.resume:
        cp = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert cp['contract'] == frozen
        adapter.load_state_dict(cp['adapter'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        scaler.load_state_dict(cp['scaler'])
        previous.restore_rng(cp['rng_state'], generator)
        start, step = cp['epoch'], cp['step']
        assert step == start * 267
        C.dump(out / 'resume.json', dict(path=str(args.resume), sha256=C.file_sha(args.resume),
                                      epoch=start, all_five_RNG_categories_restored=True))
    else:
        assert not (out / 'epoch1.pth').exists(), 'Refuse overwriting a prior run'
    model.eval()
    adapter.train()
    for epoch in range(start + 1, 11):
        epoch_seed = 1000000 + args.seed * 1000 + epoch
        C.seed(epoch_seed)
        generator.manual_seed(epoch_seed)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,
                            collate_fn=C.rtmdet_collate_fn, generator=generator)
        order = []
        sums = {}
        began = time.time()
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(loader):
            if index >= 2136:
                break
            ident = int(Path(batch['img_metas'][0]['filename']).parent.parent.name)
            order.append(ident)
            target = targets[lookup[ident]]
            target = {k: v[None].cuda() for k, v in target.items() if k in ('heat', 'reg', 'valid')}
            x = geometry_content(load_geometry(ident), args.content, ident, ids, constants)
            bind_rgb(args, adapter, dataset, lookup[ident])
            sample = C.processed(model, batch, True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss, parts = losses(model, adapter, sample, x, target)
            assert torch.isfinite(loss), (epoch, index, parts)
            scaler.scale(loss / 8).backward()
            adapter.clear()
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.) + value
            if (index + 1) % 8 == 0:
                step += 1
                lr = 1e-4 * min(1., step / 50) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / 2670)))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), .5)
                assert torch.isfinite(norm), (epoch, index, 'nonfinite gradient')
                before_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                assert scaler.get_scale() >= before_scale, 'AMP skipped update; abort to preserve exact budget'
                optimizer.zero_grad(set_to_none=True)
                if step % 25 == 0:
                    C.dump(out / 'progress.json', dict(stage='train', epoch=epoch, update=step,
                           last_microbatch=index + 1, pid=os.getpid(), seconds=time.time() - began))
                    print('UPDATE', args.route, args.content, step, parts, flush=True)
        assert len(order) == 2136 and step == epoch * 267
        assert all(p.grad is None for p in model.parameters())
        assert C.state_digest(model) == previous.FROZEN_SHA
        state = dict(adapter=adapter.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                     epoch=epoch, step=step, contract=frozen, rng_state=previous.rng_state(generator))
        previous.atomic_save(state, out / f'epoch{epoch}.pth')
        C.dump(out / f'epoch{epoch}.json', dict(epoch=epoch, step=step, sample_order=order,
               mean_microbatch_components={k: v / len(order) for k, v in sums.items()},
               seconds=time.time() - began, amp_scale=scaler.get_scale(), frozen_SHA=previous.FROZEN_SHA,
               peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20))
        print('EPOCH', args.route, args.content, epoch, time.time() - began, flush=True)
    assert step == 2670
    previous.atomic_save(state, out / 'last.pth')
    assert source_hashes() == frozen['source_hashes']
    C.dump(out / 'training_complete.json', dict(complete=True, epoch=10, updates=2670,
           fingerprint=initial, frozen_SHA=previous.FROZEN_SHA, checkpoint_sha256=C.file_sha(out / 'last.pth')))


def prediction(model, features, sample):
    with C.cached_extract(model, features):
        return model.predict(sample['inputs'], sample['data_samples'], rescale=True)[0].pred_instances


def inference_outputs(model, adapter, sample, features, x):
    """Shared formal path for resource/identity gates and complete evaluation."""
    head = model.roi_head
    xx = head._inject_extra_pe(features[0])
    meta = sample['data_samples'][0].metainfo
    cfg = head.test_cfg
    cls, reg = model.rpn_head(features[0])
    proposals = model.rpn_head.predict_by_feat(cls, reg, batch_img_metas=[meta], rescale=False)
    sat = head.predict_bbox(xx, [meta], proposals, cfg, rescale=False)[0]
    prefix = consumer.masks(head, xx, features, meta, sat.bboxes, sat.scores)
    rois = torch.cat((proposals[0].bboxes.new_zeros((len(proposals[0]), 1)), proposals[0].bboxes), 1)
    br = head._bbox_forward(xx, rois)
    raw = head.bbox_head._predict_by_feat_single(rois, br['cls_score'], br['bbox_pred'], meta, False, None)
    boxes, score = consumer.hypotheses(raw.bboxes, raw.scores[:, 0], sat.bboxes, cfg)
    prefix = consumer.merge(prefix, consumer.masks(head, xx, features, meta, boxes, cfg.score_thr * score.clamp(0, 1)))
    combined = copy.deepcopy(prefix)
    combined['scores'] = np.asarray(consumer.trusted_score(prefix['scores']), np.float64)
    accepted = 0
    if adapter.enabled:
        boxes, score = pilot.decode(adapter.outputs, x[:, 4:5])
        boxes, score = consumer.hypotheses(boxes, score, torch.tensor(prefix['bboxes'], device='cuda'), cfg)
        extra = consumer.masks(head, xx, features, meta, boxes, .25 * score.clamp(0, 1))
        combined = consumer.merge(combined, extra)
        accepted = len(boxes)
    assert len(combined['scores']) <= 250
    return prefix, combined, accepted


def evaluate(args, model, adapter):
    checkpoint = args.checkpoint or args.out / 'last.pth'
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['epoch'] == 10 and cp['step'] == 2670
    assert cp['contract']['route'] == args.route and cp['contract']['content'] == args.content
    assert cp['contract']['family'] == args.family and cp['contract']['rgb_content'] == args.rgb
    assert cp['contract']['source_hashes'] == source_hashes()
    adapter.load_state_dict(cp['adapter'], strict=True)
    model.eval()
    adapter.eval()
    adapter.enabled = args.input != 'disabled'
    adapter.observe = True
    dataset = C.MinimalSatelliteDataset(previous.DATA, args.split)
    ids = [row['image_id'] for row in dataset.samples]
    constants = means()
    records = {'satellite': [], 'combined': []}
    metas = []
    out = args.out / args.split / args.input
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = C.rtmdet_collate_fn([dataset[index]])
            sample = C.processed(model, batch)
            ident = ids[index]
            x = load_geometry(ident)
            if adapter.enabled:
                content_mode = 'geometry' if args.input in ('constant_rgb', 'donor_rgb') else args.input
                x = geometry_content(x, content_mode, ident, ids, constants)
                bind_rgb(args, adapter, dataset, index, args.input)
                features = forward(model, adapter, sample, x)
            else:
                adapter.clear()
                features = model.extract_feat(sample['inputs'])
            prefix, combined, accepted = inference_outputs(model, adapter, sample, features, x)
            target = C.serialized(C.gt_arrays(sample['data_samples'][0].gt_instances))
            for key, value in (('satellite', prefix), ('combined', combined)):
                records[key].append(dict(image_id=ident, gt=target, dt=C.serialized(value),
                    geometry_accepted=accepted if key == 'combined' else 0,
                    observation=adapter.observation if adapter.enabled else {}))
            adapter.clear()
            metas.append(batch['img_metas'][0])
            if (index + 1) % 25 == 0:
                C.dump(out / 'progress.json', dict(images=index + 1, pid=os.getpid()))
                print('EVAL', args.route, args.content, args.input, args.split, index + 1, flush=True)
    for key, rows in records.items():
        C.dump(out / f'{key}_records.json', rows)
        gt = [{**r['gt'], 'bboxes': np.asarray(r['gt']['bboxes'], np.float32),
               'labels': np.asarray(r['gt']['labels'], np.int64)} for r in rows]
        dt = [consumer.arrays(r['dt']) for r in rows]
        cg, cd = C.build_coco_gt_and_dt(gt, dt, metas, score_key='scores')
        metrics = {}
        for kind in ('segm', 'bbox'):
            metrics.update(C.run_coco_eval(cg, cd, kind, max_dets=[1, 10, 250]))
        C.dump(out / f'{key}_metrics.json', metrics)
        print('METRICS', key, metrics['segm/mAP'], metrics['bbox/mAP'], flush=True)
    assert C.state_digest(model) == previous.FROZEN_SHA
    C.dump(out / 'complete.json', dict(complete=True, images=len(dataset), split=args.split, input=args.input,
        checkpoint_sha256=C.file_sha(checkpoint), source_hashes=source_hashes(), frozen_SHA=previous.FROZEN_SHA,
        route=args.route, content=args.content, family=args.family, rgb=args.rgb,
        seed=args.seed, precision='FP32', max_dets=250,
        warning='internal changes satellite predictions; no protected-prefix guarantee against separately frozen SAT'))


def gate(args, model, adapter, initial):
    dataset = C.MinimalSatelliteDataset(previous.DATA, 'train')
    index = int(np.argmax([len(s['annotations']) for s in dataset.samples]))
    ident = dataset.samples[index]['image_id']
    batch = C.rtmdet_collate_fn([dataset[index]])
    sample = C.processed(model, batch, True)
    x = load_geometry(ident)
    bind_rgb(args, adapter, dataset, index)
    target = {k: v[None].cuda() for k, v in pilot.Data('train')[index].items() if k in ('heat', 'reg', 'valid')}
    assert pilot.Data('train').ids[index] == ident
    original = copy.deepcopy(adapter.state_dict())
    raw_logits = []
    h = model.roi_head.mask_head.register_forward_hook(lambda m, a, o: raw_logits.append(o.mask_logits.detach().clone()))

    def run(active, value, amp):
        adapter.enabled = active
        raw_logits.clear()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            features = forward(model, adapter, sample, value) if active else model.extract_feat(sample['inputs'])
            result = prediction(model, features, sample)
        adapter.clear()
        return result, list(raw_logits)

    def same(left, right):
        assert left[1] and len(left[1]) == len(right[1])
        assert all(torch.equal(a, b) for a, b in zip(left[1], right[1])), 'raw logits differ'
        for name in ('bboxes', 'scores', 'masks'):
            assert torch.equal(getattr(left[0], name), getattr(right[0], name)), name

    for amp in (False, True):
        adapter.load_state_dict(original)
        baseline = run(False, x, amp)
        same(baseline, run(False, x, amp))
        same(baseline, run(True, x, amp))
        with torch.no_grad():
            adapter.output.weight.fill_(.01)
            adapter.output.bias.fill_(.02)
        same(baseline, run(True, torch.zeros_like(x), amp))
    h.remove()
    del baseline, raw_logits
    adapter.load_state_dict(original)
    adapter.enabled = True
    adapter.train()
    adapter.observe = True
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    gradient_rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss, parts = losses(model, adapter, sample, x, target)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = {name: float(p.grad.float().norm()) for name, p in adapter.named_parameters() if p.grad is not None}
        assert all(np.isfinite(v) for v in gradients.values())
        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), .5)
        assert torch.isfinite(norm)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= scale
        gradient_rows.append(gradients)
        adapter.clear()
    for prefix in ('uq.', 'sk.', 'sv.', 'uout.', 'sq.', 'uk.', 'uv.', 'output.', 'a.', 'head_fusion.'):
        assert any(v > 0 for name, v in gradient_rows[-1].items() if name.startswith(prefix)), prefix
    # The direct center target itself, not only original detection losses,
    # must reach the satellite write-back projection and attention.
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.float16):
        forward(model, adapter, sample, x)
        direct_loss, _ = pilot.loss_fn(adapter.outputs, target)
    direct_loss.backward()
    direct_grads = {name: float(p.grad.float().norm()) for name, p in adapter.named_parameters() if p.grad is not None}
    assert direct_grads['output.weight'] > 0 and direct_grads['uk.weight'] > 0
    if args.family == 'rgb':
        assert any(value > 0 for name, value in direct_grads.items() if 'lora_' in name)
        assert all(not parameter.requires_grad for name, parameter in adapter.rgb_encoder.named_parameters()
                   if 'lora_' not in name)
    adapter.clear()
    optimizer.zero_grad(set_to_none=True)
    adapter.eval()
    with torch.inference_mode():
        features = forward(model, adapter, sample, x)
        meta = sample['data_samples'][0].metainfo
        boxes = torch.tensor([[0., 0., 512., 512.]], device='cuda').repeat(250, 1)
        masks = consumer.masks(model.roi_head, model.roi_head._inject_extra_pe(features[0]), features,
                               meta, boxes, torch.ones(250, device='cuda') * .1)
        assert len(masks['rles']) == 250
        empty, _ = pilot.decode(adapter.outputs, torch.zeros_like(x[:, 4:5]))
        assert len(empty) == 0
    assert C.state_digest(model) == previous.FROZEN_SHA
    assert all(p.grad is None for p in model.parameters())
    C.dump(args.out / 'gate.json', dict(passed=True, maximum_train_instances=len(sample['data_samples'][0].gt_instances),
        maximum_FP32_inference_instances=250, identity_FP32_AMP=True, no_support_identity=True,
        gradient_rows=gradient_rows, direct_center_gradient=direct_grads, initial_fingerprint=initial,
        satellite_SHA=previous.FROZEN_SHA, peak_reserved_MiB=torch.cuda.max_memory_reserved() / 2**20,
        peak_allocated_MiB=torch.cuda.max_memory_allocated() / 2**20, source_hashes=source_hashes()))
    print('GATE PASS', args.route, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['init', 'gate', 'train', 'eval'])
    parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, choices=[44, 45], default=44)
    parser.add_argument('--family', choices=['geom', 'rgb'], default='geom')
    parser.add_argument('--rgb', choices=['correct', 'constant'], default='correct')
    parser.add_argument('--route', choices=['head', 'internal'], default='internal')
    parser.add_argument('--content', choices=['geometry', 'constant'], default='geometry')
    parser.add_argument('--input', choices=['geometry', 'constant', 'donor', 'disabled', 'constant_rgb', 'donor_rgb'], default='geometry')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.action == 'init':
        C.seed(args.seed + 1)
        cp = torch.load(previous.BASE, map_location='cpu', weights_only=False) if args.family == 'rgb' else None
        adapter = new_adapter(args, cp['config_snapshot'] if cp else {})
        path = init_path(args)
        assert not path.exists()
        previous.atomic_save(dict(adapter=adapter.state_dict(), fingerprint=previous.tensor_digest(adapter.state_dict())), path)
        return
    args.out = args.out or args.root / f'seed{args.seed}_{args.family}_{args.route}_{args.content}_{args.rgb}'
    args.out.mkdir(parents=True, exist_ok=True)
    C.seed(args.seed)
    model, adapter, _, initial = build(args)
    if args.action == 'gate':
        gate(args, model, adapter, initial)
    elif args.action == 'train':
        g = json.loads((args.root / f'gate_{args.family}_{args.route}' / 'gate.json').read_text())
        assert g['passed'] and g['source_hashes'] == source_hashes()
        train(args, model, adapter, initial)
    else:
        evaluate(args, model, adapter)


if __name__ == '__main__':
    main()
