#!/usr/bin/env python3
"""Additive UBR trainer for UBR-ADDPROBE-20261026 (prereg level 2).

Trains rsprompter/uav/ubr.py::UnifiedBoundaryRefiner on the frozen-inference
cache built by ue_ubr_addprobe_cache_20261026.py (frozen EXP-0025 v1best
masks + tapped decoder features; additive, serial chain). Per matched
instance, on its box+32px window:

  input 1  decoder feat crop    upscaled x4 (bilinear) to the window
  input 2  final mask logits    pasted post-UDPR prob crop (from cache)
  input 3  sdt                  ubr.signed_distance_transform of the
                                binarized final mask crop
  input 4  neighbors            sum of OTHER instances' binarized masks
  input 5  uav geometry         real fields+support, or the support-matched
                                constant arm (ubr.build_constant_geometry,
                                train-only mean_fields[0:4])

Loss: band focal BCE (gamma 2) + soft Dice inside |sdt| <= k only, vs the
matched GT crop; per-instance equal weight. AdamW lr 1e-3 wd 0.05, cosine
with 100-step warmup, AMP. Selection is FIXED-E{epochs} (all epochs
recorded; no best-picking per prereg).

Val per epoch: for each cached val instance, UBR runs on the crop window
and ubr.apply_band_reclassification writes the band decision back into the
binarized final mask. Application is crop-local: every pasted mask lies
inside its box (paste parity, cache builder) hence inside its window, and
band pixels sit >=24px inside the window while UBR's receptive-field radius
is 8px, so crop-local application == full-512 application (re-asserted per
instance via the mask-within-window invariant). Metrics follow
ue_ubr_snap_20261026.py: boundary_0.01 (7px) per-scene AP + global COCO
stats + standard segm AP via eval_citysample_boundary machinery.

Out: {out}/train.jsonl (step/epoch telemetry), {out}/final.json (fixed-E
numbers, orig baseline, final-epoch scene APs for the paired bootstrap that
runs at analysis time, content-delta placeholder), {out}/ubr_last.pth.
"""
import argparse
import json
import random
import time
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UE = ROOT / 'scripts' / 'ue'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(UE))

import eval_citysample_boundary as E  # noqa: E402  (boundary eval machinery)
from citysample_pair import decode_rle  # noqa: E402
from rsprompter.uav import ubr  # noqa: E402
from pycocotools.coco import COCO  # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402

CACHE = Path('/data2/wangcheng/remote-lthpc/checkpoint/ue_ubr_probe_20261026/cache')
GEOM = Path('/data2/wangcheng/remote-lthpc/cache/ue_v1_native_geometry')
GEOM_VERSION = 'native512_height_dx_dy_viewcount_v1'
CONSTANTS = Path('/data2/wangcheng/remote-lthpc/checkpoint/'
                 'ue_geometry_mechanism_20261022/common_constants.json')
DATA = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
TAG = 'UBR-ADDPROBE-20261026'
SEED = 20261021
DIL = 7  # boundary_0.01 band; evaluator contract (CITYSAMPLE-BOUNDARY-AP)
WARMUP_STEPS = 100
SMOKE_STEPS = 30


def mean_fields() -> torch.Tensor:
    vals = json.loads(CONSTANTS.read_text())['mean_fields']
    assert len(vals) == 4
    return torch.tensor(vals[:4], dtype=torch.float32)


class SceneStore:
    """Small LRU over scene npz caches and native geometry (per process)."""

    def __init__(self, cache_dir, capacity=6):
        self.dir = Path(cache_dir)
        self.capacity = capacity
        self._scenes = OrderedDict()
        self._geoms = OrderedDict()

    def _push(self, pool, key, val):
        pool[key] = val
        pool.move_to_end(key)
        while len(pool) > self.capacity:
            pool.popitem(last=False)

    def scene(self, sid: int) -> dict:
        if sid not in self._scenes:
            with np.load(self.dir / f'{sid:06d}.npz') as z:
                d = {k: z[k] for k in z.files}
            n = d['boxes'].shape[0]
            d['masks'] = (np.unpackbits(d['masks_pack'], axis=-1)
                          .reshape(n, 512, 512).astype(bool) if n
                          else np.zeros((0, 512, 512), bool))
            self._push(self._scenes, sid, d)
        return self._scenes[sid]

    def geom(self, sid: int):
        if sid not in self._geoms:
            with np.load(GEOM / f'{sid:06d}.npz') as z:
                assert str(z['version']) == GEOM_VERSION, f'geom ver {sid}'
                g = (z['fields'].astype(np.float32),
                     z['support'].astype(np.float32))
            self._push(self._geoms, sid, g)
        return self._geoms[sid]


def manifest_ids(cache_dir):
    """Scene ids in the cache manifest (dedup: last line per id wins)."""
    last = {}
    for line in (Path(cache_dir) / 'manifest.jsonl').read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            last[rec['id']] = rec
    return [last[k] for k in sorted(last)]


def assemble(store: SceneStore, sid: int, i: int, arm: str, means, k,
             want_gt=True):
    """Per-instance UBR inputs on its box+32 window (shared train/val)."""
    d = store.scene(sid)
    h0, w0, h, w = d['win'][i]
    win = (slice(h0, h0 + h), slice(w0, w0 + w))
    logit = d['log_buf'][d['log_off'][i]:d['log_off'][i + 1]].astype(
        np.float32).reshape(h, w)
    fh0, fw0, fh, fw = d['fwin'][i]
    feat = d['feat_buf'][d['feat_off'][i]:d['feat_off'][i + 1]].astype(
        np.float32).reshape(32, fh, fw)
    feat_up = F.interpolate(torch.from_numpy(feat)[None], size=(h, w),
                            mode='bilinear', align_corners=False)[0]
    mask = d['masks'][i][win]
    sdt = ubr.signed_distance_transform(
        torch.from_numpy(mask.astype(np.float32))[None, None])[0]  # [1,h,w]
    nbr = (d['masks'].sum(0)[win] - mask).astype(np.float32)
    fields, support = store.geom(sid)
    geom5 = np.concatenate([fields, support[None]])[(slice(None),) + win]
    if arm == 'constant':
        geom5 = ubr.build_constant_geometry(
            torch.from_numpy(np.ascontiguousarray(geom5))[None], means)
        geom5 = geom5[0].numpy()
    out = dict(
        feat=feat_up,
        logit=torch.from_numpy(logit)[None],
        sdt=sdt,
        nbr=torch.from_numpy(nbr)[None],
        geom=torch.from_numpy(np.ascontiguousarray(geom5)),
        band=ubr.band_mask(sdt, k),
    )
    if want_gt:
        g = np.unpackbits(d['gt_buf'][d['gt_off'][i]:d['gt_off'][i + 1]])
        gt = g[:h * w].reshape(h, w).astype(np.float32) \
            if d['match_gt'][i] >= 0 else np.zeros((h, w), np.float32)
        out['gt'] = torch.from_numpy(gt)[None]
    return out


class SceneMajorSampler(torch.utils.data.Sampler):
    """Shuffle scene order, keep a scene's instances contiguous.

    With random shuffling the SceneStore LRU thrashes (one ~5MB npz reload
    per sample -> 20x budget overrun, review finding); scene-major order
    hits the cache almost every sample while instance order stays shuffled
    within scene and scenes reshuffle every epoch.
    """

    def __init__(self, index, seed=SEED):
        self.pos = {(sid, i): j for j, (sid, i) in enumerate(index)}
        by_scene = {}
        for sid, i in index:
            by_scene.setdefault(sid, []).append(i)
        self.by_scene = {k: sorted(v) for k, v in by_scene.items()}
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        scenes = sorted(self.by_scene)
        rng.shuffle(scenes)
        for sid in scenes:
            insts = list(self.by_scene[sid])
            rng.shuffle(insts)
            for i in insts:
                yield self.pos[(sid, i)]

    def __len__(self):
        return sum(len(v) for v in self.by_scene.values())


class AddProbeDataset(Dataset):
    """Matched instances across the split cache (one sample per instance)."""

    def __init__(self, cache_dir, arm, k, means):
        self.store = SceneStore(cache_dir)
        self.arm, self.k, self.means = arm, k, means
        self.index = []
        for line in manifest_ids(cache_dir):
            d = self.store.scene(line['id'])
            for i in np.nonzero(d['match_gt'] >= 0)[0]:
                self.index.append((int(line['id']), int(i)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sid, i = self.index[idx]
        return assemble(self.store, sid, i, self.arm, self.means, self.k)


def pad_collate(batch):
    """Top-left zero-pad variable windows to the batch max (band pads 0)."""
    max_h = max(s['logit'].shape[-2] for s in batch)
    max_w = max(s['logit'].shape[-1] for s in batch)
    out = {}
    for key in batch[0]:
        t = torch.zeros(len(batch), *batch[0][key].shape[:-2], max_h, max_w,
                        dtype=batch[0][key].dtype)
        for b, s in enumerate(batch):
            x = s[key]
            t[b, ..., :x.shape[-2], :x.shape[-1]] = x
        out[key] = t
    return out


def band_loss(pred, gt, band, gamma=2.0):
    """Focal BCE + soft Dice inside the band; per-instance equal weight."""
    focals, dices = [], []
    for b in range(pred.shape[0]):
        m = band[b]
        if not m.any():
            focals.append(pred.new_zeros(()))
            dices.append(pred.new_zeros(()))
            continue
        lg, y = pred[b][m].float(), gt[b][m].float()
        bce = F.binary_cross_entropy_with_logits(lg, y, reduction='none')
        p = torch.sigmoid(lg)
        p_t = p * y + (1 - p) * (1 - y)
        focals.append(((1 - p_t) ** gamma * bce).mean())
        inter = (p * y).sum()
        dices.append(1 - 2 * inter / (p.sum() + y.sum() + 1e-6))
    f, dd = torch.stack(focals).mean(), torch.stack(dices).mean()
    return f + dd, f, dd


class BoundaryEvaluator:
    """boundary_0.01 + standard segm COCO stats on the cached val scenes."""

    def __init__(self, ann_path, ids):
        ids = set(ids)
        ann = json.load(open(ann_path))
        self.imgs = [{'id': e['id'], 'width': 512, 'height': 512}
                     for e in ann['images'] if e['id'] in ids]
        self.cats = [{'id': 1, 'name': 'building'}]
        self.gt_bnd, self.gt_std = {}, []
        for a in ann['annotations']:
            if a['image_id'] not in ids:
                continue
            m = decode_rle(a['segmentation'])
            self.gt_bnd.setdefault(a['image_id'], []).append(
                {'id': a['id'], 'image_id': a['image_id'], 'category_id': 1,
                 'area': a['area'], 'iscrowd': 0,
                 'segmentation': E.rle(E.mask_to_boundary(m, DIL))})
            self.gt_std.append(dict(a))
        self.bnd_flat = [x for v in self.gt_bnd.values() for x in v]

    @staticmethod
    def _coco_stats(coco, dets):
        ev = COCOeval(coco, coco.loadRes([dict(x) for x in dets]), 'segm')
        ev.params.maxDets = [1, 10, 250]
        ev.evaluate()
        ev.accumulate()
        return E.coco_stats(ev)

    def stats(self, dets_bnd, dets_std):
        coco = COCO()
        coco.dataset = {'images': self.imgs, 'annotations': self.bnd_flat,
                        'categories': self.cats}
        coco.createIndex()
        bnd = self._coco_stats(coco, dets_bnd)
        coco = COCO()
        coco.dataset = {'images': self.imgs, 'annotations': self.gt_std,
                        'categories': self.cats}
        coco.createIndex()
        return bnd, self._coco_stats(coco, dets_std)

    @staticmethod
    def scene_ap(gt_bnd, dets_bnd):
        if not dets_bnd:
            return 0.0
        gt1 = [dict(x, image_id=1) for x in gt_bnd]  # single-scene id space
        coco = COCO()
        coco.dataset = {'images': [{'id': 1, 'width': 512, 'height': 512}],
                        'annotations': gt1, 'categories': [{'id': 1}]}
        coco.createIndex()
        ev = COCOeval(coco, coco.loadRes([dict(d) for d in dets_bnd]), 'segm')
        ev.params.imgIds = [1]
        ev.params.maxDets = [1, 10, 250]
        ev.evaluate()
        ev.accumulate()
        prec = ev.eval['precision'][:, :, 0, 0, 2]
        v = prec[prec > -1]
        return float(v.mean()) if v.size else 0.0


def refine_scene(ubr_net, store, sid, arm, means, k, dev):
    """UBR band-reclassification on every instance of one val scene."""
    d = store.scene(sid)
    n = d['boxes'].shape[0]
    refined = list(d['masks'])
    if n == 0:
        return d, refined
    inputs = [assemble(store, sid, i, arm, means, k, want_gt=False)
              for i in range(n)]
    max_h = max(s['logit'].shape[-2] for s in inputs)
    max_w = max(s['logit'].shape[-1] for s in inputs)
    batch = {}
    for key in inputs[0]:
        t = torch.zeros(len(inputs), *inputs[0][key].shape[:-2], max_h, max_w,
                        dtype=inputs[0][key].dtype)
        for b, s in enumerate(inputs):
            t[b, ..., :s[key].shape[-2], :s[key].shape[-1]] = s[key]
        batch[key] = t.to(dev)
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = ubr_net(batch['feat'], batch['logit'], batch['sdt'],
                      batch['nbr'], batch['geom']).float().cpu()
    for i, s in enumerate(inputs):
        h0, w0, h, w = d['win'][i]
        mask = d['masks'][i]
        rest = mask.copy()  # mask-within-window invariant (cache builder)
        rest[h0:h0 + h, w0:w0 + w] = False
        assert not rest.any(), f'scene {sid} inst {i} outside window'
        rec = ubr.apply_band_reclassification(
            torch.from_numpy(mask[h0:h0 + h, w0:w0 + w].astype(np.float32))[
                None, None],
            out[i:i + 1, :, :h, :w], s['sdt'][None], k)
        refined[i] = mask.copy()
        refined[i][h0:h0 + h, w0:w0 + w] = rec[0, 0].numpy() > 0.5
    return d, refined


def val_pass(ubr_net, val_dir, arm, means, k, dev, orig=False,
             split='val'):
    """Full val sweep; returns per-scene APs + global boundary/segm stats."""
    store = SceneStore(val_dir)
    ids = [r['id'] for r in manifest_ids(val_dir)]
    ev = BoundaryEvaluator(DATA / f'annotations/{split}.json', ids)
    aps, dets_bnd, dets_std = [], [], []
    for sid in ids:
        if orig:
            d = store.scene(sid)
            refined = list(d['masks'])
        else:
            d, refined = refine_scene(ubr_net, store, sid, arm, means, k, dev)
        scene_dets = []
        for i in range(d['boxes'].shape[0]):
            x1, y1, x2, y2 = d['boxes'][i]
            rle_full = E.rle(refined[i].astype(np.uint8))
            rle_bnd = E.rle(E.mask_to_boundary(refined[i].astype(np.uint8), DIL))
            entry = {'image_id': sid, 'category_id': 1,
                     'score': float(d['scores'][i]),
                     'bbox': [float(x1), float(y1), float(x2 - x1),
                              float(y2 - y1)],
                     'segmentation': rle_full}
            dets_std.append(entry)
            dets_bnd.append(dict(entry, segmentation=rle_bnd))
            scene_dets.append({'image_id': 1, 'category_id': 1,
                               'score': float(d['scores'][i]),
                               'bbox': [0, 0, 1, 1],
                               'segmentation': rle_bnd})
        aps.append(BoundaryEvaluator.scene_ap(ev.gt_bnd.get(sid, []), scene_dets))
    bnd, std = ev.stats(dets_bnd, dets_std)
    return {'scene_ap': aps, 'boundary_0.01': bnd, 'standard_segm': std}


def train_epoch(ubr_net, loader, opt, sched, scaler, epoch, steps_cap, log):
    ubr_net.train()
    dev = next(ubr_net.parameters()).device
    parts = {'loss': [], 'focal': [], 'dice': []}
    log_every = max(1, steps_cap // 10)  # ~10 step lines per epoch
    t0 = time.time()
    for it, batch in enumerate(loader):
        if steps_cap and it >= steps_cap:
            break
        batch = {key: v.to(dev, non_blocking=True) for key, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            out = ubr_net(batch['feat'], batch['logit'], batch['sdt'],
                          batch['nbr'], batch['geom'])
            loss, focal, dice = band_loss(out, batch['gt'], batch['band'])
        assert torch.isfinite(loss), f'non-finite loss ep{epoch} it{it}'
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        for name, val in (('loss', loss), ('focal', focal), ('dice', dice)):
            parts[name].append(float(val))
        if len(parts['loss']) % log_every == 0:
            log({'type': 'step', 'epoch': epoch, 'it': len(parts['loss']),
                 'loss': float(loss), 'focal': float(focal),
                 'dice': float(dice),
                 'lr': float(sched.get_last_lr()[0])})
    n = max(len(parts['loss']), 1)
    return {key: sum(v) / n for key, v in parts.items()}, time.time() - t0


def lr_lambda(step, total_steps):
    if step < WARMUP_STEPS:
        return step / WARMUP_STEPS
    t = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
    return 0.5 * (1.0 + np.cos(np.pi * min(1.0, t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['real', 'constant'], required=True)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-dir', default=str(CACHE / 'train'))
    ap.add_argument('--val-dir', default=str(CACHE / 'val'))
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--limit-smoke', action='store_true',
                    help=f'cap steps/epoch at {SMOKE_STEPS} (smoke only)')
    a = ap.parse_args()
    out = Path(a.out)
    assert not out.exists(), f'out dir exists: {out}'
    out.mkdir(parents=True)
    dev = torch.device(a.device)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    means = mean_fields()
    mf = open(out / 'train.jsonl', 'a')

    def log(rec):
        rec['t'] = round(time.time(), 1)
        mf.write(json.dumps(rec) + '\n')
        mf.flush()

    ds = AddProbeDataset(a.train_dir, a.arm, a.k, means)
    print(f'arm={a.arm} train samples={len(ds)} from {a.train_dir}', flush=True)
    sampler = SceneMajorSampler(ds.index)
    loader = DataLoader(ds, batch_size=a.batch, sampler=sampler,
                        num_workers=a.num_workers, collate_fn=pad_collate,
                        drop_last=False)
    steps_per_epoch = len(loader) if not a.limit_smoke else min(len(loader),
                                                                SMOKE_STEPS)
    assert steps_per_epoch > 0, 'empty loader (fewer samples than batch?)'
    ubr_net = ubr.UnifiedBoundaryRefiner().to(dev)
    n_par = sum(p.numel() for p in ubr_net.parameters())
    total_steps = steps_per_epoch * a.epochs
    opt = torch.optim.AdamW(ubr_net.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total_steps))
    scaler = torch.amp.GradScaler('cuda')

    t_start = time.time()
    orig = val_pass(ubr_net, a.val_dir, a.arm, means, a.k, dev, orig=True)
    log({'type': 'val_orig', 'boundary_0.01': orig['boundary_0.01'],
         'standard_segm': orig['standard_segm'],
         'scene_ap_mean': float(np.mean(orig['scene_ap']))})
    print('orig:', json.dumps({k: round(v, 4) for k, v in
                               orig['boundary_0.01'].items()}), flush=True)

    epochs, last_val = [], orig
    for ep in range(a.epochs):
        sampler.set_epoch(ep)
        parts, tr_secs = train_epoch(ubr_net, loader, opt, sched, scaler, ep,
                                     steps_per_epoch, log)
        last_val = val_pass(ubr_net, a.val_dir, a.arm, means, a.k, dev)
        rec = {'type': 'epoch', 'epoch': ep, 'steps': steps_per_epoch,
               'loss': parts['loss'], 'loss_focal': parts['focal'],
               'loss_dice': parts['dice'], 'train_secs': round(tr_secs, 1),
               'boundary_0.01': last_val['boundary_0.01'],
               'standard_segm': last_val['standard_segm'],
               'scene_ap_mean': float(np.mean(last_val['scene_ap']))}
        log(rec)
        epochs.append(rec)
        print(f"ep{ep} loss={parts['loss']:.4f} "
              f"B-AP_l={last_val['boundary_0.01']['AP_l']:.4f} "
              f"AP={last_val['standard_segm']['AP']:.4f}", flush=True)
    torch.save({'state_dict': ubr_net.state_dict(), 'arm': a.arm, 'k': a.k,
                'epochs': a.epochs, 'seed': SEED,
                'mean_fields': means.tolist()}, out / 'ubr_last.pth')
    final = {'tag': TAG, 'arm': a.arm, 'k': a.k, 'epochs': a.epochs,
             'selection': f'fixed-E{a.epochs} (no best-picking)',
             'samples': len(ds), 'ubr_params': n_par, 'seed': SEED,
             'geometry_version': GEOM_VERSION, 'mean_fields': means.tolist(),
             'final_epoch': epochs[-1] if epochs else None,
             'orig_boundary_0.01': orig['boundary_0.01'],
             'orig_standard_segm': orig['standard_segm'],
             'orig_scene_ap': [float(x) for x in orig['scene_ap']],
             'scene_ap_final': [float(x) for x in last_val['scene_ap']],
             'content_delta': {
                 'status': 'deferred-to-analysis',
                 'method': 'paired bootstrap 500 draws seed 20261021 on '
                           'per-scene AP; runs at analysis time, not here',
                 'real_minus_constant_AP_l': None},
             'total_secs': round(time.time() - t_start, 1)}
    (out / 'final.json').write_text(json.dumps(final, indent=2))
    mf.close()
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
