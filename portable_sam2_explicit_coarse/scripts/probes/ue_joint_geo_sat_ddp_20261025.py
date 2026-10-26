"""Joint satellite+geometry training, DDP2, full 100ep budget (2026-10-25).

PI rulings applied: full budget (no 10ep screen), hyperparams as running
screens (sat 5e-4 / geo 1e-3 / wd .05 / geo-lambda .1, warmup 100, cosine
to .1x, clip .5, AMP), from-SAM2-scratch satellite + common-init geometry
head. DDP: paired dataset (geo fields + satellite sample fetched in worker
processes - the single-process loop's in-loop I/O was the bottleneck),
per-rank batch 4 x 2 ranks = global 8 (matches the screen's effective
batch), one combined backward per step (single DDP sync), val on rank0.

Launch:
  torchrun --standalone --nproc_per_node=2 ue_joint_geo_sat_ddp_20261025.py train
Modes: init (rank0 writes common_init) | gate | train [--no-geo] [--max-batches N]
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).parent))
import ue_detection_attrition_probe_20261023 as d
import ue_s2db_probe as _s2
_s2.DATA = d.DATA
_s2.BASE = d.BASE
import ue_geometry_proposal_pilot_20261024 as pilot
import ue_geometry_mechanism_probe as g

_L = Path('/data2/wangcheng/remote-lthpc')
OUT = Path(os.environ.get('JOINT_OUT', str(_L / 'checkpoint/ue_joint_scratch100_20261025')))
pilot.OUT = OUT
pilot.CACHE = _L / 'cache/ue_v1_native_geometry'
pilot.CONSTANTS = _L / 'checkpoint/ue_geometry_mechanism_20261022/common_constants.json'
d.DATA = _L / 'dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
g.DATA = d.DATA
d.BASE = _L / 'checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth'

EPOCHS = int(os.environ.get('JOINT_EPOCHS', '100'))
PER_RANK = 4


def seed_all(seed):
    g.seed(seed)


class Paired(torch.utils.data.Dataset):
    """geo fields/targets + the satellite sample for the same point (I/O in
    DataLoader workers, not the train loop)."""

    def __init__(self, split, augment):
        self.geo = pilot.Data(split)
        self.sat = d.common.MinimalSatelliteDataset(d.DATA, split, augment=augment)
        assert len(self.geo) == len(self.sat), (len(self.geo), len(self.sat))

    def __len__(self):
        return len(self.geo)

    def __getitem__(self, i):
        b = self.geo[i]
        b['sat'] = self.sat[i]
        return b


def collate(batch):
    return dict(
        x=torch.stack([b['x'] for b in batch]),
        heat=torch.stack([b['heat'] for b in batch]),
        reg=torch.stack([b['reg'] for b in batch]),
        valid=torch.stack([b['valid'] for b in batch]),
        image_id=[b['image_id'] for b in batch],
        sat=[b['sat'] for b in batch],
    )


def build_scratch_sat():
    cp = torch.load(d.BASE, map_location='cpu', weights_only=False)
    snap = cp['config_snapshot']
    import json as _json
    _s = _json.dumps(snap)
    _s = _s.replace('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt',
                   os.environ.get('SAM2_CKPT', '/data2/wangcheng/remote-lthpc/sam2_hiera_base_plus.pt'))
    _s = _s.replace('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021', str(d.DATA))
    snap = _json.loads(_s)
    model = d.common._register_and_build(snap['model_config'])   # SAM2-pretrained ctor, heads random
    del cp
    model.cuda()
    # KEEP the ctor-native trainable set (backbone frozen + LoRA + neck +
    # all heads): the E73 recipe. The earlier blanket requires_grad_(True)
    # re-enabled the 69M backbone base params and destroyed the pretrained
    # features -> catastrophic overfit (train 0.2 / val 0.04).
    n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
    print(f'scratch trainable set (ctor-native): {n_train} params')
    model.train()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['init', 'gate', 'train'])
    p.add_argument('--no-geo', action='store_true')
    p.add_argument('--max-batches', type=int, default=0)
    p.add_argument('--resume-from', type=str, default='')
    a = p.parse_args()
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl') if world > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)

    net = None
    if a.mode == 'init':
        if rank == 0:
            seed_all(45)
            n = pilot.Net()
            OUT.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=n.state_dict(), fingerprint=g.tensor_digest(n.state_dict())),
                       OUT / 'common_init.pth')
            g.dump(OUT / 'protocol.json',
                   dict(epochs=EPOCHS, per_rank_batch=PER_RANK, world=world,
                        optimizer='AdamW sat 5e-4 / geo 1e-3, wd .05, warmup100, cosine to .1, clip .5, AMP',
                        init='scratch: SAM2-pretrained satellite ctor, heads random; geometry common-init seed45',
                        geo_lambda=0.1, budget='FULL (PI ruling 2026-10-25, no 10ep screen)'))
            print('INIT DONE')
        dist.destroy_process_group() if world > 1 else None
        return

    model = build_scratch_sat()
    if not a.no_geo:
        net = pilot.Net()
        net.load_state_dict(torch.load(OUT / 'common_init.pth', map_location='cpu', weights_only=False)['model'])
        net.cuda().train()
    # find_unused_parameters: RSPrompter heads (shape-miner/UDPR/conditional
    # paths) are not exercised on every batch; gate them out of the reducer
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True) if world > 1 else model
    net_ddp = DDP(net, device_ids=[local_rank]) if (world > 1 and net is not None) else net

    if a.mode == 'gate':
        b = next(iter(DataLoader(Paired('train', True), batch_size=PER_RANK, collate_fn=collate)))
        b = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
        sat_total = 0.
        for s in b['sat']:
            sb = d.common.rtmdet_collate_fn([s])
            smp = d.common.processed(model, sb, True)
            with torch.autocast('cuda', dtype=torch.float16):
                l = model_ddp(smp['inputs'], smp['data_samples'], mode='loss')
                sat_total = sat_total + sum(v for v in l.values() if torch.is_tensor(v) and v.requires_grad) / PER_RANK
        with torch.autocast('cuda', dtype=torch.float16):
            geo_loss, _ = pilot.loss_fn(net_ddp(b['x']), b) if net_ddp is not None else (None, None)
            loss = sat_total + (0.1 * geo_loss if geo_loss is not None else 0.)
        loss.backward()
        grads = [v.grad for v in model.parameters() if v.grad is not None] + \
                ([v.grad for v in net.parameters() if v.grad is not None] if net is not None else [])
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if rank == 0:
            g.dump(OUT / 'gate.json', dict(passed=True, peak_MiB=torch.cuda.max_memory_reserved() / 2 ** 20,
                                           sat_trainable=sum(q.numel() for q in model.parameters() if q.requires_grad)))
        print(f'[r{rank}] GATE PASS')
        dist.destroy_process_group() if world > 1 else None
        return

    assert json.loads((OUT / 'gate.json').read_text())['passed']
    sat_params = [q for q in model.parameters() if q.requires_grad]
    geo_params = list(net.parameters()) if net is not None else []
    sat_lr, geo_lr = 5e-4, 1e-3
    groups = [{'params': sat_params, 'lr': sat_lr}]
    if geo_params and not a.no_geo:
        groups.append({'params': geo_params, 'lr': geo_lr})
    opt = torch.optim.AdamW(groups, weight_decay=.05)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    ds = Paired('train', augment=True)
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if world > 1 else None
    loader = DataLoader(ds, batch_size=PER_RANK, sampler=sampler, shuffle=sampler is None,
                        drop_last=True, num_workers=4, collate_fn=collate, pin_memory=True,
                        persistent_workers=True)
    batches_per_ep = len(loader)
    total_steps = EPOCHS * batches_per_ep
    step = 0
    history = OUT / 'train.jsonl'
    sat_val = d.common.MinimalSatelliteDataset(d.DATA, 'val')
    start_epoch = 0
    if a.resume_from:
        cp = torch.load(a.resume_from, map_location='cpu', weights_only=False)
        model.load_state_dict(cp['model'])
        if net is not None and cp.get('net') is not None:
            net.load_state_dict(cp['net'])
        opt.load_state_dict(cp['optimizer'])
        scaler.load_state_dict(cp['scaler'])
        start_epoch = int(cp['epoch'])
        step = start_epoch * batches_per_ep
        if rank == 0:
            print('RESUMED from', a.resume_from, 'epoch', start_epoch, flush=True)
    if rank == 0 and history.exists() and not a.max_batches and not a.resume_from:
        raise SystemExit('refusing to overwrite train history')
    log = history.open('a') if rank == 0 else None
    for epoch in range(start_epoch + 1, EPOCHS + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        seed_all(1044000 + epoch)
        model.train()
        if net is not None:
            net.train()
        t0 = time.time(); loss_sum = 0.; nb = 0
        for batch in loader:
            if a.max_batches and nb >= a.max_batches:
                break
            geo_b = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()
                     if k != 'sat'}
            opt.zero_grad(set_to_none=True)
            sat_total = 0.
            for s in geo_b['image_id'] and batch['sat']:
                sb = d.common.rtmdet_collate_fn([s])
                smp = d.common.processed(model, sb, True)
                with torch.autocast('cuda', dtype=torch.float16):
                    l = model_ddp(smp['inputs'], smp['data_samples'], mode='loss')
                    sat_total = sat_total + sum(v for v in l.values() if torch.is_tensor(v) and v.requires_grad) / PER_RANK
            with torch.autocast('cuda', dtype=torch.float16):
                if net_ddp is not None:
                    geo_loss, _ = pilot.loss_fn(net_ddp(geo_b['x']), geo_b)
                    loss = sat_total + 0.1 * geo_loss
                else:
                    loss = sat_total
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            try:
                scaler.unscale_(opt)
            except RuntimeError:
                # rare scaler-cycle desync after an inf-skip race (killed the
                # baseline at ep66): reset the cycle and drop this batch
                opt.zero_grad(set_to_none=True)
                scaler.update()
                continue
            # clip each branch SEPARATELY: a fresh geo head's early large
            # gradients must not eat the sat branch's clip budget (the sat
            # dynamics then exactly match the --no-geo baseline arm)
            norm = torch.nn.utils.clip_grad_norm_(sat_params, .5)
            if geo_params:
                norm_g = torch.nn.utils.clip_grad_norm_(geo_params, .5)
                norm = norm + norm_g
            # norm non-finite here only via unscaled inf grads: GradScaler's
            # step() detects inf itself and skips the update while update()
            # halves the scale - keep calling both so the scaler state machine
            # stays consistent (manual zero_grad+continue broke unscale_)
            step += 1
            lr_mult = min(1., step / 100) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / total_steps)))
            for gi, grp in enumerate(opt.param_groups):
                grp['lr'] = (sat_lr if gi == 0 else geo_lr) * lr_mult
            scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach()); nb += 1
            if rank == 0 and nb % 40 == 0:
                log.write(json.dumps(dict(epoch=epoch, batches=nb, loss=loss_sum / nb,
                                          lr_sat=opt.param_groups[0]['lr'])) + '\n')
                log.flush()
        if a.max_batches:
            print(f'[r{rank}] SMOKE epoch fragment done {nb} batches loss {loss_sum/max(nb,1):.4f} '
                  f'({(time.time()-t0)/max(nb,1):.2f}s/batch)', flush=True)
            break
        if rank == 0:
            torch.save(dict(model=model.state_dict(),
                            net=(net.state_dict() if net is not None else None), epoch=epoch,
                            optimizer=opt.state_dict(), scaler=scaler.state_dict()),
                       OUT / 'last.pth.tmp')
            (OUT / 'last.pth.tmp').replace(OUT / 'last.pth')
            if epoch % 2 == 0:
                try:
                    mAP = quick_val(model, sat_val, 128)
                    log.write(json.dumps(dict(epoch=epoch, val_mask_ap=mAP)) + '\n'); log.flush()
                    print('VAL', epoch, mAP, flush=True)
                except Exception as e:
                    print('VAL FAILED (non-fatal)', repr(e)[:200], flush=True)
            log.write(json.dumps(dict(epoch=epoch, done=True, seconds=time.time() - t0,
                                      mean_loss=loss_sum / max(nb, 1))) + '\n')
            log.flush()
            print('EPOCH', epoch, loss_sum / max(nb, 1), time.time() - t0, flush=True)
    if rank == 0:
        print('JOINT TRAIN DONE')
    dist.destroy_process_group() if world > 1 else None


@torch.inference_mode()
def quick_val(model, val, n):
    """Per-image COCO AP subsample; image ids shared by GT and DT."""
    import numpy as _np
    from pycocotools import mask as _mu
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    model.eval()
    anns, dts = [], []
    for i in range(n):
        b = d.common.rtmdet_collate_fn([val[i]])
        smp = d.common.processed(model, b, False)
        with torch.autocast('cuda', dtype=torch.float16):
            o = model.predict(smp['inputs'], smp['data_samples'], rescale=True)[0].pred_instances
        for ann in val.samples[i]['annotations']:
            r = ann['segmentation']
            if isinstance(r, dict) and isinstance(r['counts'], list):
                r = _mu.frPyObjects(r, r['size'][0], r['size'][1])
            x, y, w, h = _mu.toBbox(r)
            anns.append(dict(id=len(anns) + 1, image_id=i, category_id=1, segmentation=r,
                             bbox=[float(x), float(y), float(w), float(h)],
                             area=float(_mu.area(r)), iscrowd=0))
        masks = o.masks.masks if hasattr(o.masks, 'masks') else o.masks.detach().cpu().numpy()
        for m, sc in zip(_np.asarray(masks).astype(bool), o.scores.detach().cpu().numpy()):
            r = _mu.encode(_np.asfortranarray(m.astype(_np.uint8)))
            x, y, w, h = _mu.toBbox(r)
            dts.append(dict(image_id=i, category_id=1, score=float(sc), segmentation=r,
                            bbox=[float(x), float(y), float(w), float(h)]))
    model.train()
    if not anns or not dts:
        return 0.0
    coco = COCO()
    coco.dataset = dict(images=[dict(id=i, height=512, width=512) for i in range(n)],
                        annotations=anns, categories=[dict(id=1, name='b')])
    coco.createIndex()
    dt = coco.loadRes(dts)
    ev = COCOeval(coco, dt, 'segm')
    ev.params.maxDets = [1, 10, 250]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    v = ev.eval['precision'][:, :, 0, 0, 2]
    v = v[v > -1]
    return float(v.mean()) if v.size else 0.0


if __name__ == '__main__':
    main()
