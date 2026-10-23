#!/usr/bin/env python3
"""WARM-RR v2a (prereg docs/WARM-RR-V2_preregistration.md §1): zero-training
refinement of the recall-rescue supply.

v1 bottlenecks -> fixes:
  4px kron mask quantization  -> bilinear-upsampled 512 evidence, mask =
                                 (evi512 >= thr_m) & component & gates
  blobby merged components    -> peak detection + watershed split per comp
  constant score 0.3          -> PAVA isotonic calibration P(hit | mean evi)

Stages:
  calib   sweep (thr, thr_m, split) on cached calib scenes; fit isotonic;
          freeze params.json
  apply   one test read per variant in {main, constant, donor, noheight,
          noviews}; metrics + safety + miss-rescue diagnostics
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import cv2
from scipy import ndimage
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path('/home/wangcheng/project/new/.worktrees/dual-stream-dev/portable_sam2_explicit_coarse')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'inference')); sys.path.insert(0, str(ROOT / 'scripts' / 'ue'))
import ue.dual  # noqa
from ue.dual import _UAV_MEAN, _UAV_STD
from rsprompter.backbone.sam2_vision import RSSAM2VisionEncoder
from probe_evi_decoder_v2 import EviDecoder
from warm_rr import to_rle

DS = Path('/data1/wangcheng/dataset/CitySample_SatUAV_TrainMinimal_v1_20261021')
CACHE = Path('/data2/wangcheng/remote-lthpc/ue_v1_local_adapter_geometry')
DECODER = '/data2/wangcheng/checkpoint/surfbev_b0/probes/EVI-DECODER-20261021/decoder.pt'
OUT = Path('/data2/wangcheng/remote-lthpc/warm_rr_v2a')
YAWS = (0, 72, 144, 216, 288)


@torch.no_grad()
def sat_stack(enc, dec, pid, appearance='correct', donor_pid=None):
    """(evi128, height128, views128) via candidates; donor swaps RGB only."""
    rgb_pid = donor_pid or pid
    rgbs = []
    for yaw in YAWS:
        img = cv2.imread(str(DS / 'points' / rgb_pid / 'uav' / f'yaw_{yaw:03d}' / 'rgb.png'))
        assert img is not None
        rgbs.append(img[:, :, ::-1].transpose(2, 0, 1).copy())
    rgb = torch.from_numpy(np.stack(rgbs)).float().cuda()
    if appearance == 'constant':
        rgb[:] = torch.tensor([123.675, 116.28, 103.53], device='cuda').view(1, 3, 1, 1)
    mean = rgb.new_tensor(_UAV_MEAN)[:, None, None]
    std = rgb.new_tensor(_UAV_STD)[:, None, None]
    _, fpn = enc((rgb - mean) / std)
    desc = max(tuple(fpn[-4:]), key=lambda f: f.shape[-2] * f.shape[-1]).float()
    ev = torch.sigmoid(dec(desc))[:, 0].cpu().numpy()
    with np.load(CACHE / f'{pid}.npz') as a:
        index, geom = a['index'], a['geom'].astype(np.float32)
    tflat = index.reshape(-1)
    valid = tflat >= 0
    view, rem = np.divmod(tflat[valid].astype(np.int64), 128 * 128)
    v, u = np.divmod(rem, 128)
    vals = ev[view, v, u]
    cells = np.repeat(np.arange(16384), index.shape[1])[valid]
    sat = np.zeros(16384, np.float32); np.maximum.at(sat, cells, vals)
    height = np.zeros(16384, np.float32); np.maximum.at(height, cells, geom.reshape(-1, 6)[valid, 2])
    seen = np.zeros((16384, 5), bool); seen[cells, view] = True
    views = seen.sum(1).astype(np.float32)
    return (sat.reshape(128, 128), height.reshape(128, 128), views.reshape(128, 128))


def make_proposals(evi, height, views, thr, thr_m, hthr, min_views, min_cells,
                   split=True, peak_dist=10, min_area_px=48):
    evi512 = cv2.resize(evi, (512, 512), interpolation=cv2.INTER_LINEAR)
    h512 = cv2.resize(height, (512, 512), interpolation=cv2.INTER_LINEAR)
    gate128 = np.ones_like(evi, bool)
    if hthr > 0: gate128 &= height >= hthr
    if min_views > 1: gate128 &= views >= min_views
    lab, n = ndimage.label((evi >= thr) & gate128)
    out = []
    for c in range(1, n + 1):
        cells = lab == c
        if cells.sum() < min_cells:
            continue
        foot = np.kron(cells, np.ones((4, 4), np.uint8)).astype(bool)
        soft = (evi512 >= thr_m) & foot
        if soft.sum() < min_area_px:
            soft = foot  # keep tiny candidates as full footprint
        if not split or soft.sum() < 200:
            out.append(soft)
            continue
        # peak-based watershed inside the component
        d = ndimage.distance_transform_edt(soft)
        peaks = (d == ndimage.maximum_filter(d, size=peak_dist)) & (d >= peak_dist / 2)
        pl, npk = ndimage.label(peaks)
        if npk <= 1:
            out.append(soft)
            continue
        markers = np.where(soft, pl + 1, 0)
        wl = ndimage.watershed_ift(np.uint8(np.clip(255 * (1 - evi512), 0, 255)) * soft, markers)
        for r in range(1, npk + 1):
            m = (wl == r + 1) & soft
            if m.sum() >= min_area_px:
                out.append(m)
    return out


def pava_isotonic(xs, ys):
    """Pool-adjacent-violators; returns callable score(x)."""
    order = np.argsort(xs)
    x, y = np.asarray(xs)[order], np.asarray(ys)[order]
    val, cnt, lvl = [], [], []
    for xi, yi in zip(x, y):
        val.append(yi); cnt.append(1); lvl.append(xi)
        while len(val) > 1 and val[-2] > val[-1]:
            v2 = (val[-2] * cnt[-2] + val[-1] * cnt[-1]) / (cnt[-2] + cnt[-1])
            c2 = cnt[-2] + cnt[-1]
            val = val[:-2] + [v2]; cnt = cnt[:-2] + [c2]
    bounds, vals = [], []
    i = 0
    for v, c in zip(val, cnt):
        bounds.append(x[min(i + c - 1, len(x) - 1)]); vals.append(v); i += c
    xb = np.asarray(bounds); yv = np.asarray(vals)

    def f(q):
        q = np.asarray(q, float)
        idx = np.searchsorted(xb, q, side='right') - 1
        idx = np.clip(idx, 0, len(yv) - 1)
        return yv[idx]
    return f


def coco_ap(all_gt_masks, all_dt_masks, all_dt_scores):
    coco = COCO(); anns = []; dts = []
    for i, (g, d, s) in enumerate(zip(all_gt_masks, all_dt_masks, all_dt_scores)):
        iid = i + 1
        for m in g:
            r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            x, y, w, h = maskUtils.toBbox(r)
            anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': 1, 'segmentation': r,
                         'bbox': [float(x), float(y), float(w), float(h)], 'area': float(maskUtils.area(r)), 'iscrowd': 0})
        for m, s_ in zip(d, s):
            r = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
            x, y, w, h = maskUtils.toBbox(r)
            dts.append({'image_id': iid, 'category_id': 1, 'score': float(s_), 'segmentation': r,
                        'bbox': [float(x), float(y), float(w), float(h)]})
    coco.dataset = {'images': [{'id': i + 1, 'height': 512, 'width': 512} for i in range(len(all_gt_masks))],
                    'annotations': anns, 'categories': [{'id': 1, 'name': 'b'}]}
    coco.createIndex()
    dt = coco.loadRes(dts)
    ev = COCOeval(coco, dt, 'segm'); ev.params.maxDets = [1, 10, 250]
    ev.evaluate(); ev.accumulate(); ev.summarize()
    v = ev.eval['precision'][:, :, 0, 0, 2]; v = v[v > -1]
    return dict(AP=float(v.mean()), AP50=float(ev.stats[1]), AP75=float(ev.stats[2]),
                AP_s=float(ev.stats[3]), AP_m=float(ev.stats[4]), AP_l=float(ev.stats[5]))


def inject(dt_rles, dt_scores, props, prop_scores):
    add_r, add_s = [], []
    for m, s in zip(props, prop_scores):
        ys, xs = np.where(m)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        ok = True
        for r in dt_rles + add_r:
            g = maskUtils.toBbox(r)
            if g[0] < x2 and x1 < g[0] + g[2] and g[1] < y2 and y1 < g[1] + g[3]:
                if (m & maskUtils.decode(r).astype(bool)).sum() > 0.3 * min(m.sum(), maskUtils.area(r)):
                    ok = False; break
        if ok:
            add_r.append(to_rle(maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))))
            add_s.append(float(s))
    # NMS among added
    keep = []
    for i in range(len(add_r)):
        sup = False
        for j in keep:
            if maskUtils.iou([add_r[i]], [add_r[j]], [0])[0][0] > 0.5 and add_s[i] <= add_s[j]:
                sup = True; break
        if not sup:
            keep.append(i)
    add_r = [add_r[i] for i in keep]; add_s = [add_s[i] for i in keep]
    # cap 250
    if len(dt_rles) + len(add_r) > 250:
        add_r, add_s = add_r[:250 - len(dt_rles)], add_s[:250 - len(dt_rles)]
    return dt_rles + add_r, list(dt_scores) + add_s, len(add_r)


def run_scenes(scenes, params, iso=None):
    gts, dts, scores = [], [], []
    added = 0; hit = 0; total_prop = 0
    for sc in scenes:
        props = make_proposals(sc['evi'], sc['height'], sc['views'],
                               params['thr'], params['thr_m'], params.get('hthr', 0.03),
                               params.get('min_views', 2), params.get('min_cells', 3),
                               params.get('split', True))
        total_prop += len(props)
        if iso is not None:
            e512 = cv2.resize(sc['evi'], (512, 512), interpolation=cv2.INTER_LINEAR)
            feats = [float(e512[m].mean()) if m.sum() else 0.0 for m in props]
            ps = iso(feats).tolist() if feats else []
        else:
            ps = [params.get('score', 0.3)] * len(props)
        dt_r, dt_s, nadd = inject(sc['dt_rles'], sc['scores'], props, ps)
        added += nadd
        gts.append([maskUtils.decode(g).astype(bool) for g in sc['gt_rles']])
        dts.append([maskUtils.decode(d).astype(bool) for d in dt_r])
        scores.append(dt_s)
    m = coco_ap(gts, dts, scores)
    m['added'] = added; m['n_scenes'] = len(scenes)
    return m


def stage_calib():
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = torch.load('/data2/wangcheng/remote-lthpc/warm_rr/calib_scenes.pt', weights_only=False)
    # split half: sweep half / score-fit half (no leakage into test)
    rng = np.random.default_rng(20261014)
    perm = rng.permutation(len(scenes))
    sweep_set = [scenes[i] for i in perm[:len(scenes) // 2]]
    fit_set = [scenes[i] for i in perm[len(scenes) // 2:]]
    best = None
    for thr in (0.6, 0.7, 0.8):
        for thr_m in (0.5, 0.6, 0.7):
            for split in (True, False):
                params = dict(thr=thr, thr_m=thr_m, hthr=0.03, min_views=2,
                              min_cells=3, split=split, score=0.3)
                m = run_scenes(sweep_set, params)
                print(f'thr={thr} thr_m={thr_m} split={split}: AP={m["AP"]:.4f} AP50={m["AP50"]:.4f} added={m["added"]}', flush=True)
                if best is None or m['AP'] > best[1]['AP']:
                    best = (params, m)
    params = best[0]
    # isotonic score fit on the held-out fit half
    evi512_cache = {}
    xs, ys = [], []
    for sc in fit_set:
        props = make_proposals(sc['evi'], sc['height'], sc['views'], params['thr'], params['thr_m'],
                               params['hthr'], params['min_views'], params['min_cells'], params['split'])
        if not props:
            continue
        e512 = cv2.resize(sc['evi'], (512, 512), interpolation=cv2.INTER_LINEAR)
        gt_dec = [maskUtils.decode(g).astype(bool) for g in sc['gt_rles']]
        for m in props:
            xs.append(float(e512[m].mean()))
            best_iou = 0.0
            for g in gt_dec:
                inter = (m & g).sum()
                if inter:
                    iou = inter / (m.sum() + g.sum() - inter)
                    best_iou = max(best_iou, iou)
            ys.append(1.0 if best_iou >= 0.5 else 0.0)
    iso = pava_isotonic(xs, ys) if xs else (lambda q: np.full_like(np.asarray(q, float), 0.3))
    n_pos = sum(ys)
    print(f'isotonic fit: n={len(xs)} pos_rate={n_pos / max(len(xs), 1):.3f}', flush=True)
    m_fit = run_scenes(fit_set, params, iso)
    print('fit-half with calibrated scores:', json.dumps(m_fit), flush=True)
    json.dump(params, open(OUT / 'params.json', 'w'))
    np.save(OUT / 'iso_x.npy', np.asarray(xs)); np.save(OUT / 'iso_y.npy', np.asarray(ys))
    print('FROZEN params:', json.dumps(params), flush=True)


def stage_apply(variant):
    params = json.load(open(OUT / 'params.json'))
    xs = np.load(OUT / 'iso_x.npy'); ys = np.load(OUT / 'iso_y.npy')
    iso = pava_isotonic(xs.tolist(), ys.tolist())
    enc = RSSAM2VisionEncoder(checkpoint_path='/data1/wangcheng/pretrained/sam2_hiera_base_plus.pt',
                              model_size='base_plus', freeze_backbone=True,
                              use_gradient_checkpointing=False, use_lora=False).cuda().eval()
    dec = EviDecoder().cuda().eval()
    dec.load_state_dict(torch.load(DECODER, map_location='cpu', weights_only=False)['state_dict'])
    recs = json.load(open('/data2/wangcheng/remote-lthpc/evals/base_disabled/disabled/records.json'))
    recs_sorted = sorted(recs, key=lambda r: r['image_id'])
    gts, dts, scores = [], [], []
    added = 0
    # miss-rescue diagnostics
    rescued, total_prop, hits = 0, 0, 0
    for i, rec in enumerate(recs_sorted):
        pid = f"{rec['image_id']:06d}"
        if not (CACHE / f'{pid}.npz').exists():
            gts.append([maskUtils.decode(g).astype(bool) for g in []]); dts.append([]); scores.append([])
            continue
        appearance = 'constant' if variant == 'constant' else 'correct'
        donor = None
        if variant == 'donor':
            donor = f"{recs_sorted[(i + 1) % len(recs_sorted)]['image_id']:06d}"
        evi, height, views = sat_stack(enc, dec, pid, appearance, donor)
        if variant == 'noheight':
            params_use = dict(params); params_use['hthr'] = 0.0
        elif variant == 'noviews':
            params_use = dict(params); params_use['min_views'] = 1
        else:
            params_use = params
        props = make_proposals(evi, height, views, params_use['thr'], params_use['thr_m'],
                               params_use.get('hthr', 0.03), params_use.get('min_views', 2),
                               params_use.get('min_cells', 3), params_use.get('split', True))
        total_prop += len(props)
        e512 = cv2.resize(evi, (512, 512), interpolation=cv2.INTER_LINEAR)
        ps = iso([float(e512[m].mean()) for m in props]).tolist() if props else []
        dt_rles = [to_rle(e) for e in rec['dt']['rles']]
        dt_scores = list(rec['dt']['scores'])
        dt_r, dt_s, nadd = inject(dt_rles, dt_scores, props, ps)
        added += nadd
        # diagnostics: do added proposals hit unmatched GT?
        gt_dec = [maskUtils.decode(g).astype(bool) for g in rec['gt']['rles']]
        dt_dec = [maskUtils.decode(r).astype(bool) for r in dt_rles]   # decode once
        base_cover = np.zeros(len(gt_dec), bool)
        for gi, g in enumerate(gt_dec):
            gs = g.sum()
            for dmask in dt_dec:
                inter = (g & dmask).sum()
                if inter and inter / (gs + dmask.sum() - inter) >= 0.5:
                    base_cover[gi] = True
                    break
        for m in props:
            best_iou, best_gi = 0.0, -1
            for gi, g in enumerate(gt_dec):
                inter = (m & g).sum()
                if inter:
                    iou = inter / (m.sum() + g.sum() - inter)
                    if iou > best_iou:
                        best_iou, best_gi = iou, gi
            if best_iou >= 0.5:
                hits += 1
                if best_gi >= 0 and not base_cover[best_gi]:
                    rescued += 1
        gts.append(gt_dec)
        dts.append([maskUtils.decode(d).astype(bool) for d in dt_r])
        scores.append(dt_s)
        if (i + 1) % 100 == 0:
            print('APPLY', variant, i + 1, flush=True)
    m = coco_ap(gts, dts, scores)
    m.update(variant=variant, added=added, proposals=total_prop, prop_hits=hits,
             prop_hit_rate=hits / max(total_prop, 1), rescued_miss_gts=rescued)
    print(json.dumps(m, indent=2), flush=True)
    json.dump(m, open(OUT / f'{variant}_result.json', 'w'))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument('--stage', choices=['calib', 'main', 'constant', 'donor', 'noheight', 'noviews'], required=True)
    a = ap_.parse_args()
    if a.stage == 'calib':
        stage_calib()
    else:
        stage_apply(a.stage)


if __name__ == '__main__':
    main()
