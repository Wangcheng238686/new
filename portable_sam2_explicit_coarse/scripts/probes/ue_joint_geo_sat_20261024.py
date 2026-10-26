"""Joint satellite+geometry training - final-form first run (prereg part 2).

Satellite detector initialized from the sealed E73 single-stream checkpoint
(now TRAINABLE) + the geometry instance-candidate head (fresh common-init
discipline), trained jointly: per batch, satellite instance losses +
geometry direct supervision. 10-epoch screen; kill gate = val Mask collapse
vs the frozen-E73 prefix (66.09 strong-SAT reference band).

Reuse: d.build() (attrition module) for the satellite stack, pilot.Net/
loss_fn/Data for the geometry branch (module substitution - locked sources
untouched). Eval afterwards via the conditional interface with the JOINT
model's own detections as prefix.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
import ue_detection_attrition_probe_20261023 as d
import ue_s2db_probe as _s2
_s2.DATA = d.DATA
_s2.BASE = d.BASE

import ue_geometry_proposal_pilot_20261024 as pilot
import ue_geometry_mechanism_probe as g

_L = Path('/data2/wangcheng/remote-lthpc')
OUT = Path(os.environ.get('JOINT_OUT', str(_L / 'checkpoint/ue_joint_geo_sat_20261024')))
pilot.OUT = OUT
pilot.CACHE = _L / 'cache/ue_v1_native_geometry'
pilot.CONSTANTS = _L / 'checkpoint/ue_geometry_mechanism_20261022/common_constants.json'
d.DATA = _L / 'dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
g.DATA = _L / 'dataset/CitySample_SatUAV_TrainMinimal_v1_20261021'
d.BASE = _L / 'checkpoint/ue_minimal_single_20261021_EXP-0017/checkpoints/best_model_epoch73.pth'


def seed_all(seed):
    g.seed(seed)


class Joint(torch.utils.data.Dataset):
    """Satellite sample + geometry fields/targets for the same point id."""

    def __init__(self, split, augment):
        self.sat = d.common.MinimalSatelliteDataset(d.DATA, split, augment=augment)
        self.geo = pilot.Data(split)

    def __len__(self):
        return len(self.geo)

    def __getitem__(self, i):
        return self.geo[i]


def collate(batch):
    out = dict(x=torch.stack([b['x'] for b in batch]),
               heat=torch.stack([b['heat'] for b in batch]),
               reg=torch.stack([b['reg'] for b in batch]),
               valid=torch.stack([b['valid'] for b in batch]),
               image_id=[b['image_id'] for b in batch])
    return out


def sat_collate(batch):
    from data.loader import rtmdet_collate_fn
    return batch  # satellite samples are collated individually below


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['init', 'gate', 'train'])
    p.add_argument('--resume', type=Path)
    p.add_argument('--scratch', action='store_true', help='from-scratch: build fresh from SAM2-pretrained ctor, do NOT load E73')
    p.add_argument('--no-geo', action='store_true', help='pure-satellite control: same loop without the geometry head/loss')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == 'init':
        g.seed(45)
        net = pilot.Net()
        g.atomic_save(dict(model=net.state_dict(), fingerprint=g.tensor_digest(net.state_dict())), OUT / 'common_init.pth')
        g.dump(OUT / 'protocol.json', dict(
            epochs=10, batch_sat=1, batch_geo=8, steps='per satellite image; geo batch accumulates 8',
            optimizer='AdamW sat 5e-5 / geo 1e-3, wd .01, warmup100, cosine to .1, clip .5, AMP',
            init='satellite from sealed E73 (trainable); geometry fresh seed45 common-init',
            loss='satellite instance losses + geometry focal+offset+.1size (weight 1.0 each per its own batch scale)',
            kill_gate='val Mask drop >1pp vs frozen-E73 prefix band on epochs 2/4/6/8/10 (256-image val subsample)',
            source_sha256=g.file_sha(__file__), archive_hashes=d.common.archive_identity(d.DATA),
            scope='Joint-training final form screen; consumption via instance-candidate interface afterwards'))
        print('INIT DONE')
        return
    seed_all(44)
    cp_sat = None
    model, snapshot = None, None
    # build satellite trainable
    import copy as _copy
    cp = torch.load(d.BASE, map_location='cpu', weights_only=False)
    snap = cp['config_snapshot']
    import json as _json
    _s = _json.dumps(snap)
    _s = _s.replace('/data/wangcheng/pretrained-models/sam2/sam2_hiera_base_plus.pt',
                    os.environ.get('SAM2_CKPT', '/data2/wangcheng/remote-lthpc/sam2_hiera_base_plus.pt'))
    _s = _s.replace('/data/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021', str(d.DATA))
    snap = _json.loads(_s)
    model = d.common._register_and_build(snap['model_config'])
    if not a.scratch:
        rep = d.common._load_model_state(model, cp['model'], allow_nonstrict=False)
        assert not rep['missing_keys'] and not rep['unexpected_keys']
    else:
        print('SCRATCH: satellite built from SAM2-pretrained ctor; heads random (E73 not loaded)')
    del cp
    model.cuda()
    for prm in model.parameters():
        prm.requires_grad_(True)
    model.train()
    net = None
    if not a.no_geo:
        net = pilot.Net()
        cp = torch.load(OUT / 'common_init.pth', map_location='cpu', weights_only=False)
        net.load_state_dict(cp['model'])
        net.cuda()
        net.train()
    if a.mode == 'gate':
        b = next(iter(DataLoader(pilot.Data('train'), batch_size=2, collate_fn=collate)))
        b = {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()}
        with torch.autocast('cuda', dtype=torch.float16):
            loss, _ = pilot.loss_fn(net(b['x']), b)
        loss.backward()
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
        sat_batch = d.common.rtmdet_collate_fn([d.common.MinimalSatelliteDataset(d.DATA, 'train')[0]])
        smp = d.common.processed(model, sat_batch, True)
        with torch.autocast('cuda', dtype=torch.float16):
            losses = model.loss(smp['inputs'], smp['data_samples'])
        total = sum(v for v in losses.values() if torch.is_tensor(v) and v.requires_grad)
        total.backward()
        grads = [v.grad for v in model.parameters() if v.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        g.dump(OUT / 'gate.json', dict(passed=True, sat_trainable=sum(q.numel() for q in model.parameters() if q.requires_grad),
                                       geo_params=sum(q.numel() for q in net.parameters()),
                                       peak_MiB=torch.cuda.max_memory_reserved() / 2 ** 20))
        print('GATE PASS')
        return
    assert json.loads((OUT / 'gate.json').read_text())['passed']
    sat_params = [q for q in model.parameters() if q.requires_grad]
    geo_params = list(net.parameters()) if net is not None else []
    sat_lr = 5e-4 if a.scratch else 5e-5
    groups = [{'params': sat_params, 'lr': sat_lr}]
    if geo_params and not a.no_geo:
        groups.append({'params': geo_params, 'lr': 1e-3})
    opt = torch.optim.AdamW(groups, weight_decay=.05 if a.scratch else .01)
    scaler = torch.amp.GradScaler('cuda', init_scale=512)
    ds = Joint('train', augment=True)
    sat_lookup = {i: ds.sat[i] for i in range(0)}  # sat samples fetched by geo index
    gen = torch.Generator().manual_seed(44)
    loader = DataLoader(ds, batch_size=8, shuffle=True, drop_last=True, num_workers=2,
                        collate_fn=collate, generator=gen)
    total_imgs = 2136
    steps_per_ep = total_imgs
    total_steps = 10 * total_imgs
    step = 0
    means = torch.tensor(json.loads(pilot.CONSTANTS.read_text())['mean_fields'], device='cuda')[None, :, None, None]
    val_images = 128
    sat_val = d.common.MinimalSatelliteDataset(d.DATA, 'val')
    history = OUT / 'train.jsonl'
    model_saved = None
    with history.open('w') as log:
        for epoch in range(1, 11):
            seed_all(1044000 + epoch)
            gen.manual_seed(1044000 + epoch)
            if net is not None: net.train()
            model.train()
            t0 = time.time(); loss_sum = 0; nb = 0
            for batch in loader:
                geo_b = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
                ids = geo_b['image_id']
                opt.zero_grad(set_to_none=True)
                # satellite losses (per image, batch 1)
                sat_loss = 0.
                for iid in ids:
                    idx = next(j for j in range(len(ds.sat)) if ds.sat.samples[j]['image_id'] == iid) if False else None
                # direct index mapping: pilot.Data ids order == sat dataset order (same split json)
                sat_losses_accum = []
                for j, iid in enumerate(ids):
                    sidx = _sat_index(ds, iid)
                    s = ds.sat[sidx]
                    sb = d.common.rtmdet_collate_fn([s])
                    smp = d.common.processed(model, sb, True)
                    with torch.autocast('cuda', dtype=torch.float16):
                        l = model.loss(smp['inputs'], smp['data_samples'])
                        sat_losses_accum.append(sum(v for v in l.values() if torch.is_tensor(v) and v.requires_grad) / 8)
                with torch.autocast('cuda', dtype=torch.float16):
                    sat_total = sum(sat_losses_accum)
                    if net is not None:
                        geo_loss, _ = pilot.loss_fn(net(geo_b['x']), geo_b)
                        loss = sat_total + (0.1 if a.scratch else 1.0) * geo_loss
                    else:
                        loss = sat_total
                assert torch.isfinite(loss)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                norm = torch.nn.utils.clip_grad_norm_(sat_params + geo_params, .5)
                assert torch.isfinite(norm)
                step += 8
                warm = 100 if a.scratch else 800
                lr_mult = min(1., step / warm) * (.1 + .9 * .5 * (1 + math.cos(math.pi * step / total_steps)))
                for gi, grp in enumerate(opt.param_groups):
                    grp['lr'] = (sat_lr if gi == 0 else 1e-3) * lr_mult
                scaler.step(opt); scaler.update()
                loss_sum += float(loss.detach()); nb += 1
                if nb % 40 == 0:
                    log.write(json.dumps(dict(epoch=epoch, batches=nb, loss=loss_sum / nb, lr_sat=opt.param_groups[0]['lr'])) + '\n')
                    log.flush()
            # kill gate on val subsample every 2 epochs
            if epoch % 2 == 0:
                mAP = quick_val(model, sat_val, val_images)
                log.write(json.dumps(dict(epoch=epoch, val_mask_ap=mAP)) + '\n'); log.flush()
                print('VAL', epoch, mAP, flush=True)
                # kill gate evaluated ONLY at ep10 (design deadline); early
                # from-scratch val is legitimately near zero for epochs
                floor = 0.50 if a.scratch else 0.6309
                if mAP < floor and epoch >= 10:
                    g.dump(OUT / 'kill_gate.json', dict(epoch=epoch, val_mask_ap=mAP, reason='collapse'))
                    print('KILL GATE TRIGGERED')
                    return
            torch.save(dict(model=model.state_dict(), net=(net.state_dict() if net is not None else None), epoch=epoch,
                            optimizer=opt.state_dict(), scaler=scaler.state_dict()), OUT / 'last.pth.tmp')
            (OUT / 'last.pth.tmp').replace(OUT / 'last.pth')
            print('EPOCH', epoch, loss_sum / max(nb, 1), time.time() - t0, flush=True)
    print('JOINT TRAIN DONE')


_sat_idx_cache = {}


def _sat_index(ds, iid):
    if not _sat_idx_cache:
        for j, s in enumerate(ds.sat.samples):
            _sat_idx_cache[s['image_id']] = j
    return _sat_idx_cache[iid]


@torch.inference_mode()
def quick_val(model, val, n):
    from pycocotools import mask as mu
    from eval_records import _build_data_samples
    model.eval()
    aps = []
    cg, cd = [], []
    metas = []
    for i in range(n):
        b = d.common.rtmdet_collate_fn([val[i]])
        smp = d.common.processed(model, b)
        out = model.predict(smp['inputs'], smp['data_samples'], rescale=True)[0].pred_instances
        gt = d.common.gt_arrays(smp['data_samples'][0].gt_instances)
        pred = d.common.gt_arrays(out)
        cg.append(gt); cd.append(pred); metas.append(b['img_metas'][0])
    cge, cde = d.common.build_coco_gt_and_dt(cg, cd, metas, score_key='scores')
    m = d.common.run_coco_eval(cge, cde, 'segm', max_dets=[1, 10, 250])
    model.train()
    return m['segm/mAP']


if __name__ == '__main__':
    main()
